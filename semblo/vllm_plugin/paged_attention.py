from __future__ import annotations

"""Hopper sparse attention over vLLM's native paged KV cache.

The semblo index stores only logical token positions.  This module
expands ranked product buckets into compact work tiles and maps every selected
logical position through vLLM's block table inside the attention kernel.  No
secondary K/V cache and no gathered K/V tensor are materialized.
"""

from dataclasses import dataclass
import functools
import math
import os

import torch

try:  # Optional on CPU-only development machines.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

from semblo.vllm_plugin.validation import (
    DEFERRED_CHECKS,
    SLOT_MAX_RANGES,
    SLOT_MAX_SELECTED,
    SLOT_MAX_TILES,
    SLOT_MIN_SELECTED,
    SLOT_SHORTFALL,
    SLOT_SUFFIX_BREAK,
)


@dataclass(frozen=True)
class PostingTiles:
    starts: torch.Tensor  # [Q,Hkv,T], offsets into posting_positions[Hkv]
    lengths: torch.Tensor  # [Q,Hkv,T]
    counts: torch.Tensor  # [Q,Hkv]
    selected_tokens: torch.Tensor  # [Q,Hkv]


@dataclass(frozen=True)
class PagedSparseAttentionResult:
    output: torch.Tensor  # [Q,Hq,D]
    logsumexp: torch.Tensor | None  # [Q,Hq]


def merge_paged_attention_results(
    left: PagedSparseAttentionResult,
    right: PagedSparseAttentionResult,
) -> PagedSparseAttentionResult:
    """Exactly merge attention computed over two disjoint key sets.

    Both inputs must carry per-head log-sum-exp values.  Decode uses this to
    combine sparse attention over the prompt with dense attention over the
    generated suffix.
    """

    if left.logsumexp is None or right.logsumexp is None:
        raise ValueError("attention-state merge requires logsumexp from both inputs")
    if left.output.shape != right.output.shape:
        raise ValueError("attention outputs must have identical shapes")
    if left.logsumexp.shape != right.logsumexp.shape:
        raise ValueError("attention logsumexp tensors must have identical shapes")
    if left.output.is_cuda and triton is not None:
        # One fused kernel instead of a dozen elementwise ops: decode is bound
        # by the number of launches, not by their cost.
        return _merge_attention_results_on_device(left, right)
    merged_lse = torch.logaddexp(left.logsumexp.float(), right.logsumexp.float())
    left_weight = torch.exp(left.logsumexp.float() - merged_lse)
    right_weight = torch.exp(right.logsumexp.float() - merged_lse)
    output = (
        left.output.float() * left_weight.unsqueeze(-1)
        + right.output.float() * right_weight.unsqueeze(-1)
    ).to(left.output.dtype)
    return PagedSparseAttentionResult(output=output, logsumexp=merged_lse)


if triton is not None:

    @triton.jit
    def _merge_attention_kernel(
        out_ptr,        # [Q,Hq,D]
        lse_ptr,        # [Q,Hq] float32
        left_out_ptr,
        left_lse_ptr,
        right_out_ptr,
        right_lse_ptr,
        stride_o0: tl.int64, stride_o1: tl.int64, stride_o2: tl.int64,
        stride_l0: tl.int64, stride_l1: tl.int64,
        HEAD_DIM: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Combine two attention states over disjoint key sets, exactly."""

        query = tl.program_id(0)
        head = tl.program_id(1)
        offs = tl.arange(0, BLOCK_D)
        valid = offs < HEAD_DIM
        scalar = query * stride_l0 + head * stride_l1
        left_lse = tl.load(left_lse_ptr + scalar).to(tl.float32)
        right_lse = tl.load(right_lse_ptr + scalar).to(tl.float32)
        peak = tl.maximum(left_lse, right_lse)
        merged = peak + tl.log(tl.exp(left_lse - peak) + tl.exp(right_lse - peak))
        base = query * stride_o0 + head * stride_o1 + offs * stride_o2
        left = tl.load(left_out_ptr + base, mask=valid, other=0.0).to(tl.float32)
        right = tl.load(right_out_ptr + base, mask=valid, other=0.0).to(tl.float32)
        combined = left * tl.exp(left_lse - merged) + right * tl.exp(right_lse - merged)
        tl.store(out_ptr + base, combined.to(out_ptr.dtype.element_ty), mask=valid)
        tl.store(lse_ptr + scalar, merged)


def _merge_attention_results_on_device(
    left: PagedSparseAttentionResult, right: PagedSparseAttentionResult
) -> PagedSparseAttentionResult:
    """CUDA path of :func:`merge_paged_attention_results`."""

    q_count, num_heads, head_dim = map(int, left.output.shape)
    output = torch.empty_like(left.output)
    merged_lse = torch.empty_like(left.logsumexp, dtype=torch.float32)
    _merge_attention_kernel[(q_count, num_heads)](
        output, merged_lse,
        left.output, left.logsumexp, right.output, right.logsumexp,
        output.stride(0), output.stride(1), output.stride(2),
        merged_lse.stride(0), merged_lse.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_D=_next_power_of_two(head_dim),
        num_warps=4,
        num_stages=1,
    )
    return PagedSparseAttentionResult(output=output, logsumexp=merged_lse)


def posting_positions_to_tiles(
    positions: torch.Tensor,
    *,
    num_queries: int,
    num_kv_heads: int,
    tile_size: int,
) -> tuple[torch.Tensor, PostingTiles]:
    """Tile one logical-position list without changing its order.

    ``positions`` may be contiguous or sparse.  Each tile contains consecutive
    offsets in the posting list; the logical token IDs stored at those offsets
    may be arbitrary.  This is the representation consumed by the paged-KV
    attention kernel.
    """

    if positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if positions.numel() == 0:
        raise ValueError("cannot build posting tiles for an empty position list")
    if num_queries <= 0 or num_kv_heads <= 0 or tile_size <= 0:
        raise ValueError("query/head counts and tile_size must be positive")

    posting = positions.to(torch.int32).repeat(num_kv_heads, 1).contiguous()
    tile_count = math.ceil(positions.numel() / tile_size)
    starts_1d = torch.arange(
        tile_count, device=positions.device, dtype=torch.int32
    ) * int(tile_size)
    lengths_1d = torch.full(
        (tile_count,), int(tile_size), device=positions.device, dtype=torch.int32
    )
    tail = positions.numel() - (tile_count - 1) * int(tile_size)
    lengths_1d[-1] = int(tail)
    starts = (
        starts_1d.view(1, 1, -1)
        .expand(num_queries, num_kv_heads, -1)
        .contiguous()
    )
    lengths = lengths_1d.view(1, 1, -1).expand_as(starts).contiguous()
    counts = torch.full(
        (num_queries, num_kv_heads),
        tile_count,
        device=positions.device,
        dtype=torch.int32,
    )
    selected_tokens = torch.full_like(counts, int(positions.numel()))
    return posting, PostingTiles(starts, lengths, counts, selected_tokens)


def contiguous_positions_to_posting_tiles(
    positions: torch.Tensor,
    *,
    num_queries: int,
    num_kv_heads: int,
    tile_size: int,
) -> tuple[torch.Tensor, PostingTiles]:
    """Tile one contiguous logical-position range.

    This stricter helper is used for the generated suffix.  Sparse prompt
    postings use :func:`posting_positions_to_tiles` or bucket-derived tiles.
    """

    if positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    if positions.numel() > 1:
        broken = positions[1:] != positions[:-1] + 1
        if DEFERRED_CHECKS.active:
            DEFERRED_CHECKS.record_violations(
                SLOT_SUFFIX_BREAK, broken,
                message="suffix positions must form one contiguous range",
            )
        elif bool(broken.any()):
            raise ValueError("suffix positions must form one contiguous range")
    return posting_positions_to_tiles(
        positions,
        num_queries=num_queries,
        num_kv_heads=num_kv_heads,
        tile_size=tile_size,
    )


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _require_triton() -> None:
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for the H200 paged sparse backend")


@functools.lru_cache(maxsize=None)
def _capability_major(device_index: int | None) -> int:
    return int(torch.cuda.get_device_capability(device_index)[0])


def _validate_hopper(device: torch.device) -> None:
    # The capability lookup is cached because this runs in every layer on
    # every decode step.
    if device.type != "cuda":
        raise RuntimeError("paged sparse attention requires CUDA")
    if _capability_major(device.index) != 9:
        raise RuntimeError("the production kernel currently targets SM90/H100/H200")


def _ranked_buckets_to_posting_tiles_reference(
    ranked_bucket_ids: torch.Tensor,
    ranked_bucket_counts: torch.Tensor,
    bucket_indptr: torch.Tensor,
    posting_positions: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    target_tokens: int,
    token_capacity: int | None,
    tile_size: int,
    max_tiles: int,
) -> PostingTiles:
    q_count, num_kv_heads, max_ranked = ranked_bucket_ids.shape
    starts = torch.full(
        (q_count, num_kv_heads, max_tiles),
        -1,
        dtype=torch.int32,
        device=ranked_bucket_ids.device,
    )
    lengths = torch.zeros_like(starts)
    counts = torch.zeros((q_count, num_kv_heads), dtype=torch.int32, device=starts.device)
    selected_tokens = torch.zeros_like(counts)
    for q in range(q_count):
        q_pos = int(query_positions[q])
        for kv_head in range(num_kv_heads):
            written = 0
            token_count = 0
            ranked_count = min(int(ranked_bucket_counts[q, kv_head]), max_ranked)
            for slot in range(ranked_count):
                bucket = int(ranked_bucket_ids[q, kv_head, slot])
                if bucket < 0:
                    continue
                begin = int(bucket_indptr[kv_head, bucket])
                end = int(bucket_indptr[kv_head, bucket + 1])
                valid = int(
                    torch.searchsorted(
                        posting_positions[kv_head, begin:end],
                        posting_positions.new_tensor(q_pos),
                        right=True,
                    )
                )
                if valid <= 0:
                    continue
                cursor = begin
                bucket_end = begin + valid
                while cursor < bucket_end:
                    if written >= max_tiles:
                        raise RuntimeError(
                            f"selected buckets require more than {max_tiles} posting tiles; "
                            f"capacity={max_tiles}"
                        )
                    length = min(tile_size, bucket_end - cursor)
                    starts[q, kv_head, written] = cursor
                    lengths[q, kv_head, written] = length
                    written += 1
                    cursor += length
                token_count += valid
                if token_count >= target_tokens:
                    break
            if token_count == 0:
                raise RuntimeError("semblo search selected no causal token")
            if token_capacity is not None and token_count > token_capacity:
                raise RuntimeError(
                    f"selected buckets contain {token_count} causal tokens, "
                    f"capacity={token_capacity}"
                )
            counts[q, kv_head] = written
            selected_tokens[q, kv_head] = token_count
    return PostingTiles(starts, lengths, counts, selected_tokens)


if triton is not None:

    @triton.jit
    def _bucket_valid_lengths_kernel(
        out_ptr,
        ranked_ids_ptr,
        ranked_counts_ptr,
        indptr_ptr,
        posting_positions_ptr,
        query_positions_ptr,
        stride_rank_q: tl.int64,
        stride_rank_h: tl.int64,
        stride_rank_s: tl.int64,
        stride_count_q: tl.int64,
        stride_count_h: tl.int64,
        stride_indptr_h: tl.int64,
        stride_post_h: tl.int64,
        NUM_KV_HEADS: tl.constexpr,
        MAX_RANKED: tl.constexpr,
    ):
        flat = tl.program_id(0)
        slot = flat % MAX_RANKED
        tmp = flat // MAX_RANKED
        kv_head = tmp % NUM_KV_HEADS
        query = tmp // NUM_KV_HEADS
        ranked_count = tl.load(
            ranked_counts_ptr + query * stride_count_q + kv_head * stride_count_h
        )
        valid = slot < ranked_count
        bucket = tl.load(
            ranked_ids_ptr
            + query * stride_rank_q
            + kv_head * stride_rank_h
            + slot * stride_rank_s,
            mask=valid,
            other=-1,
        ).to(tl.int64)
        valid &= bucket >= 0
        begin = tl.load(
            indptr_ptr + kv_head * stride_indptr_h + bucket,
            mask=valid,
            other=0,
        ).to(tl.int64)
        end = tl.load(
            indptr_ptr + kv_head * stride_indptr_h + bucket + 1,
            mask=valid,
            other=0,
        ).to(tl.int64)
        length = end - begin
        q_pos = tl.load(query_positions_ptr + query)
        lo = tl.zeros((), dtype=tl.int64)
        hi = length
        while lo < hi:
            mid = (lo + hi) // 2
            pos = tl.load(
                posting_positions_ptr + kv_head * stride_post_h + begin + mid
            )
            move_right = pos <= q_pos
            lo = tl.where(move_right, mid + 1, lo)
            hi = tl.where(move_right, hi, mid)
        tl.store(out_ptr + flat, tl.where(valid, lo, 0))


    @triton.jit
    def _write_posting_tiles_kernel(
        starts_ptr,
        lengths_ptr,
        valid_lengths_ptr,
        tile_offsets_ptr,
        keep_ptr,
        ranked_ids_ptr,
        indptr_ptr,
        stride_rank_q: tl.int64,
        stride_rank_h: tl.int64,
        stride_rank_s: tl.int64,
        stride_indptr_h: tl.int64,
        NUM_KV_HEADS: tl.constexpr,
        MAX_RANKED: tl.constexpr,
        MAX_TILES: tl.constexpr,
        TILE_SIZE: tl.constexpr,
    ):
        flat = tl.program_id(0)
        slot = flat % MAX_RANKED
        tmp = flat // MAX_RANKED
        kv_head = tmp % NUM_KV_HEADS
        query = tmp // NUM_KV_HEADS
        keep = tl.load(keep_ptr + flat) != 0
        valid_length = tl.load(valid_lengths_ptr + flat).to(tl.int64)
        offset = tl.load(tile_offsets_ptr + flat).to(tl.int64)
        bucket = tl.load(
            ranked_ids_ptr
            + query * stride_rank_q
            + kv_head * stride_rank_h
            + slot * stride_rank_s,
            mask=keep,
            other=-1,
        ).to(tl.int64)
        begin = tl.load(
            indptr_ptr + kv_head * stride_indptr_h + bucket,
            mask=keep,
            other=0,
        ).to(tl.int64)
        num_tiles = (valid_length + TILE_SIZE - 1) // TILE_SIZE
        for tile_idx in range(0, num_tiles):
            destination = offset + tile_idx
            if keep and destination < MAX_TILES:
                consumed = tile_idx * TILE_SIZE
                base = (query * NUM_KV_HEADS + kv_head) * MAX_TILES + destination
                tl.store(starts_ptr + base, begin + consumed)
                tl.store(
                    lengths_ptr + base,
                    tl.minimum(TILE_SIZE, valid_length - consumed),
                )


    @triton.jit
    def _softcap(scores, cap):
        return cap * (2.0 * tl.sigmoid(2.0 * scores / cap) - 1.0)


    @triton.jit
    def _softmax_update(scores, valid, m_i, l_i, acc, values):
        masked = tl.where(valid, scores, -float("inf"))
        block_max = tl.max(masked, axis=1)
        m_new = tl.maximum(m_i, block_max)
        safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - safe))
        probs = tl.where(valid, tl.exp(masked - safe[:, None]), 0.0)
        l_new = l_i * alpha + tl.sum(probs, axis=1)
        acc = acc * alpha[:, None] + tl.dot(probs.to(values.dtype), values)
        return m_new, l_new, acc


    @triton.jit
    def _paged_posting_attention_kernel(
        out_ptr,
        lse_ptr,
        q_ptr,
        kv_cache_ptr,
        posting_positions_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        tile_counts_ptr,
        query_positions_ptr,
        query_request_ids_ptr,
        block_table_ptr,
        sm_scale,
        softcap,
        stride_q0: tl.int64,
        stride_q1: tl.int64,
        stride_q2: tl.int64,
        stride_cache_kind: tl.int64,
        stride_cache_block: tl.int64,
        stride_cache_token: tl.int64,
        stride_cache_head: tl.int64,
        stride_cache_dim: tl.int64,
        stride_post_h: tl.int64,
        stride_tile_q: tl.int64,
        stride_tile_h: tl.int64,
        stride_tile_s: tl.int64,
        stride_count_q: tl.int64,
        stride_count_h: tl.int64,
        stride_block_table_q: tl.int64,
        stride_block_table_b: tl.int64,
        stride_out0: tl.int64,
        stride_out1: tl.int64,
        stride_out2: tl.int64,
        stride_lse0: tl.int64,
        stride_lse1: tl.int64,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        Q_PER_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_DIM_PADDED: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TILE_CAPACITY: tl.constexpr,
        TOKEN_TILE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        USE_SOFTCAP: tl.constexpr,
        WRITE_LSE: tl.constexpr,
    ):
        query_idx = tl.program_id(0)
        kv_head = tl.program_id(1)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM_PADDED)
        q_head = kv_head * Q_PER_KV + offs_m
        q_valid = (offs_m < Q_PER_KV) & (q_head < NUM_Q_HEADS)
        d_valid = offs_d < HEAD_DIM
        q = tl.load(
            q_ptr
            + query_idx * stride_q0
            + q_head[:, None] * stride_q1
            + offs_d[None, :] * stride_q2,
            mask=q_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM_PADDED), tl.float32)
        tile_count = tl.load(
            tile_counts_ptr + query_idx * stride_count_q + kv_head * stride_count_h
        )
        q_position = tl.load(query_positions_ptr + query_idx)
        request_id = tl.load(query_request_ids_ptr + query_idx).to(tl.int64)
        for tile_slot in range(0, tile_count):
            posting_base = tl.load(
                tile_starts_ptr
                + query_idx * stride_tile_q
                + kv_head * stride_tile_h
                + tile_slot * stride_tile_s
            ).to(tl.int64)
            valid_length = tl.load(
                tile_lengths_ptr
                + query_idx * stride_tile_q
                + kv_head * stride_tile_h
                + tile_slot * stride_tile_s
            )
            for offset in tl.static_range(0, TILE_CAPACITY, TOKEN_TILE):
                offs_n = offset + tl.arange(0, TOKEN_TILE)
                posting_idx = posting_base + offs_n
                token_valid = offs_n < valid_length
                logical_pos = tl.load(
                    posting_positions_ptr + kv_head * stride_post_h + posting_idx,
                    mask=token_valid,
                    other=-1,
                ).to(tl.int64)
                token_valid &= (logical_pos >= 0) & (logical_pos <= q_position)
                logical_block = logical_pos // BLOCK_SIZE
                physical_block = tl.load(
                    block_table_ptr
                    + request_id * stride_block_table_q
                    + logical_block * stride_block_table_b,
                    mask=token_valid,
                    other=0,
                ).to(tl.int64)
                block_offset = logical_pos - logical_block * BLOCK_SIZE
                k = tl.load(
                    kv_cache_ptr
                    + 0 * stride_cache_kind
                    + physical_block[None, :] * stride_cache_block
                    + block_offset[None, :] * stride_cache_token
                    + kv_head * stride_cache_head
                    + offs_d[:, None] * stride_cache_dim,
                    mask=d_valid[:, None] & token_valid[None, :],
                    other=0.0,
                )
                v = tl.load(
                    kv_cache_ptr
                    + 1 * stride_cache_kind
                    + physical_block[:, None] * stride_cache_block
                    + block_offset[:, None] * stride_cache_token
                    + kv_head * stride_cache_head
                    + offs_d[None, :] * stride_cache_dim,
                    mask=token_valid[:, None] & d_valid[None, :],
                    other=0.0,
                )
                scores = sm_scale * tl.dot(q, k)
                if USE_SOFTCAP:
                    scores = _softcap(scores, softcap)
                valid = q_valid[:, None] & token_valid[None, :]
                m_i, l_i, acc = _softmax_update(scores, valid, m_i, l_i, acc, v)
        denom = tl.where(l_i > 0, l_i, 1.0)
        result = tl.where(l_i[:, None] > 0, acc / denom[:, None], 0.0)
        tl.store(
            out_ptr
            + query_idx * stride_out0
            + q_head[:, None] * stride_out1
            + offs_d[None, :] * stride_out2,
            result,
            mask=q_valid[:, None] & d_valid[None, :],
        )
        if WRITE_LSE:
            lse = tl.where(l_i > 0, m_i + tl.log(l_i), -float("inf"))
            tl.store(
                lse_ptr + query_idx * stride_lse0 + q_head * stride_lse1,
                lse,
                mask=q_valid,
            )


    @triton.jit
    def _paged_posting_attention_split_kernel(
        partial_acc_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        q_ptr,
        kv_cache_ptr,
        posting_positions_ptr,
        tile_starts_ptr,
        tile_lengths_ptr,
        tile_counts_ptr,
        query_positions_ptr,
        query_request_ids_ptr,
        block_table_ptr,
        sm_scale,
        softcap,
        stride_q0: tl.int64,
        stride_q1: tl.int64,
        stride_q2: tl.int64,
        stride_cache_kind: tl.int64,
        stride_cache_block: tl.int64,
        stride_cache_token: tl.int64,
        stride_cache_head: tl.int64,
        stride_cache_dim: tl.int64,
        stride_post_h: tl.int64,
        stride_tile_q: tl.int64,
        stride_tile_h: tl.int64,
        stride_tile_s: tl.int64,
        stride_count_q: tl.int64,
        stride_count_h: tl.int64,
        stride_block_table_q: tl.int64,
        stride_block_table_b: tl.int64,
        stride_pa0: tl.int64,
        stride_pa1: tl.int64,
        stride_pa2: tl.int64,
        stride_pa3: tl.int64,
        stride_pa4: tl.int64,
        stride_pm0: tl.int64,
        stride_pm1: tl.int64,
        stride_pm2: tl.int64,
        stride_pm3: tl.int64,
        NUM_Q_HEADS: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        Q_PER_KV: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_DIM_PADDED: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        TILE_CAPACITY: tl.constexpr,
        TOKEN_TILE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        USE_SOFTCAP: tl.constexpr,
    ):
        query_idx = tl.program_id(0)
        kv_head = tl.program_id(1)
        split = tl.program_id(2)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM_PADDED)
        q_head = kv_head * Q_PER_KV + offs_m
        q_valid = (offs_m < Q_PER_KV) & (q_head < NUM_Q_HEADS)
        d_valid = offs_d < HEAD_DIM
        q = tl.load(
            q_ptr
            + query_idx * stride_q0
            + q_head[:, None] * stride_q1
            + offs_d[None, :] * stride_q2,
            mask=q_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, HEAD_DIM_PADDED), tl.float32)
        tile_count = tl.load(
            tile_counts_ptr + query_idx * stride_count_q + kv_head * stride_count_h
        )
        per_split = (tile_count + NUM_SPLITS - 1) // NUM_SPLITS
        begin = split * per_split
        end = tl.minimum(begin + per_split, tile_count)
        q_position = tl.load(query_positions_ptr + query_idx)
        request_id = tl.load(query_request_ids_ptr + query_idx).to(tl.int64)
        for tile_slot in range(begin, end):
            posting_base = tl.load(
                tile_starts_ptr
                + query_idx * stride_tile_q
                + kv_head * stride_tile_h
                + tile_slot * stride_tile_s
            ).to(tl.int64)
            valid_length = tl.load(
                tile_lengths_ptr
                + query_idx * stride_tile_q
                + kv_head * stride_tile_h
                + tile_slot * stride_tile_s
            )
            for offset in tl.static_range(0, TILE_CAPACITY, TOKEN_TILE):
                offs_n = offset + tl.arange(0, TOKEN_TILE)
                posting_idx = posting_base + offs_n
                token_valid = offs_n < valid_length
                logical_pos = tl.load(
                    posting_positions_ptr + kv_head * stride_post_h + posting_idx,
                    mask=token_valid,
                    other=-1,
                ).to(tl.int64)
                token_valid &= (logical_pos >= 0) & (logical_pos <= q_position)
                logical_block = logical_pos // BLOCK_SIZE
                physical_block = tl.load(
                    block_table_ptr
                    + request_id * stride_block_table_q
                    + logical_block * stride_block_table_b,
                    mask=token_valid,
                    other=0,
                ).to(tl.int64)
                block_offset = logical_pos - logical_block * BLOCK_SIZE
                k = tl.load(
                    kv_cache_ptr
                    + physical_block[None, :] * stride_cache_block
                    + block_offset[None, :] * stride_cache_token
                    + kv_head * stride_cache_head
                    + offs_d[:, None] * stride_cache_dim,
                    mask=d_valid[:, None] & token_valid[None, :],
                    other=0.0,
                )
                v = tl.load(
                    kv_cache_ptr
                    + stride_cache_kind
                    + physical_block[:, None] * stride_cache_block
                    + block_offset[:, None] * stride_cache_token
                    + kv_head * stride_cache_head
                    + offs_d[None, :] * stride_cache_dim,
                    mask=token_valid[:, None] & d_valid[None, :],
                    other=0.0,
                )
                scores = sm_scale * tl.dot(q, k)
                if USE_SOFTCAP:
                    scores = _softcap(scores, softcap)
                valid = q_valid[:, None] & token_valid[None, :]
                m_i, l_i, acc = _softmax_update(scores, valid, m_i, l_i, acc, v)
        pa = (
            query_idx * stride_pa0
            + kv_head * stride_pa1
            + split * stride_pa2
            + offs_m[:, None] * stride_pa3
            + offs_d[None, :] * stride_pa4
        )
        pm = (
            query_idx * stride_pm0
            + kv_head * stride_pm1
            + split * stride_pm2
            + offs_m * stride_pm3
        )
        tl.store(partial_acc_ptr + pa, acc, mask=q_valid[:, None] & d_valid[None, :])
        tl.store(partial_max_ptr + pm, m_i, mask=q_valid)
        tl.store(partial_sum_ptr + pm, l_i, mask=q_valid)


    @triton.jit
    def _reduce_split_kernel(
        out_ptr,
        lse_ptr,
        partial_acc_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        stride_out0: tl.int64,
        stride_out1: tl.int64,
        stride_out2: tl.int64,
        stride_lse0: tl.int64,
        stride_lse1: tl.int64,
        stride_pa0: tl.int64,
        stride_pa1: tl.int64,
        stride_pa2: tl.int64,
        stride_pa3: tl.int64,
        stride_pa4: tl.int64,
        stride_pm0: tl.int64,
        stride_pm1: tl.int64,
        stride_pm2: tl.int64,
        stride_pm3: tl.int64,
        NUM_Q_HEADS: tl.constexpr,
        Q_PER_KV: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_DIM_PADDED: tl.constexpr,
        BLOCK_D: tl.constexpr,
        WRITE_LSE: tl.constexpr,
    ):
        query_idx = tl.program_id(0)
        q_head = tl.program_id(1)
        d_block = tl.program_id(2)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        d_valid = offs_d < HEAD_DIM
        kv_head = q_head // Q_PER_KV
        local_head = q_head - kv_head * Q_PER_KV
        split = tl.arange(0, NUM_SPLITS)
        pm = (
            query_idx * stride_pm0
            + kv_head * stride_pm1
            + split * stride_pm2
            + local_head * stride_pm3
        )
        maxima = tl.load(partial_max_ptr + pm)
        sums = tl.load(partial_sum_ptr + pm)
        global_max = tl.max(maxima)
        safe_max = tl.where(global_max == -float("inf"), 0.0, global_max)
        weights = tl.where(sums > 0, tl.exp(maxima - safe_max), 0.0)
        denominator = tl.sum(weights * sums)
        pa = (
            query_idx * stride_pa0
            + kv_head * stride_pa1
            + split[:, None] * stride_pa2
            + local_head * stride_pa3
            + offs_d[None, :] * stride_pa4
        )
        partial = tl.load(partial_acc_ptr + pa, mask=d_valid[None, :], other=0.0)
        numerator = tl.sum(partial * weights[:, None], axis=0)
        output = tl.where(denominator > 0, numerator / denominator, 0.0)
        tl.store(
            out_ptr
            + query_idx * stride_out0
            + q_head * stride_out1
            + offs_d * stride_out2,
            output,
            mask=d_valid,
        )
        if WRITE_LSE and d_block == 0:
            lse = tl.where(denominator > 0, safe_max + tl.log(denominator), -float("inf"))
            tl.store(lse_ptr + query_idx * stride_lse0 + q_head * stride_lse1, lse)


def ranked_buckets_to_posting_tiles(
    ranked_bucket_ids: torch.Tensor,
    ranked_bucket_counts: torch.Tensor,
    bucket_indptr: torch.Tensor,
    posting_positions: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    target_tokens: int,
    token_capacity: int | None = None,
    tile_size: int,
    max_tiles: int,
    validate_capacity: bool = True,
) -> PostingTiles:
    if ranked_bucket_ids.ndim != 3:
        raise ValueError("ranked_bucket_ids must have shape [Q,Hkv,max_ranked]")
    q_count, num_kv_heads, max_ranked = map(int, ranked_bucket_ids.shape)
    if ranked_bucket_counts.shape != (q_count, num_kv_heads):
        raise ValueError("ranked_bucket_counts must have shape [Q,Hkv]")
    if bucket_indptr.ndim != 2 or bucket_indptr.shape[0] != num_kv_heads:
        raise ValueError("bucket_indptr must have shape [Hkv,num_buckets+1]")
    if posting_positions.ndim != 2 or posting_positions.shape[0] != num_kv_heads:
        raise ValueError("posting_positions must have shape [Hkv,L]")
    if query_positions.shape != (q_count,):
        raise ValueError("query_positions must have shape [Q]")
    if target_tokens <= 0 or tile_size <= 0 or max_tiles <= 0:
        raise ValueError("target_tokens, tile_size, and max_tiles must be positive")
    if token_capacity is not None and token_capacity < target_tokens:
        raise ValueError("token_capacity must be at least target_tokens")

    if not ranked_bucket_ids.is_cuda:
        return _ranked_buckets_to_posting_tiles_reference(
            ranked_bucket_ids,
            ranked_bucket_counts,
            bucket_indptr,
            posting_positions,
            query_positions,
            target_tokens=target_tokens,
            token_capacity=token_capacity,
            tile_size=tile_size,
            max_tiles=max_tiles,
        )

    _require_triton()
    if ranked_bucket_ids.dtype != torch.int32 or ranked_bucket_counts.dtype != torch.int32:
        raise ValueError("ranked bucket tensors must be int32")
    flat_count = q_count * num_kv_heads * max_ranked
    valid_lengths = torch.empty(flat_count, dtype=torch.int64, device=ranked_bucket_ids.device)
    _bucket_valid_lengths_kernel[(flat_count,)](
        valid_lengths,
        ranked_bucket_ids,
        ranked_bucket_counts,
        bucket_indptr,
        posting_positions,
        query_positions.to(torch.int32),
        ranked_bucket_ids.stride(0),
        ranked_bucket_ids.stride(1),
        ranked_bucket_ids.stride(2),
        ranked_bucket_counts.stride(0),
        ranked_bucket_counts.stride(1),
        bucket_indptr.stride(0),
        posting_positions.stride(0),
        NUM_KV_HEADS=num_kv_heads,
        MAX_RANKED=max_ranked,
        num_warps=1,
    )
    lengths3 = valid_lengths.view(q_count, num_kv_heads, max_ranked)
    cumulative = lengths3.cumsum(dim=-1)
    before = cumulative - lengths3
    rank = torch.arange(max_ranked, device=ranked_bucket_ids.device).view(1, 1, -1)
    valid_rank = rank < ranked_bucket_counts[..., None]
    keep = valid_rank & (lengths3 > 0) & (before < int(target_tokens))
    kept_lengths = torch.where(keep, lengths3, torch.zeros_like(lengths3))
    selected_tokens = kept_lengths.sum(dim=-1).to(torch.int32)
    if DEFERRED_CHECKS.active:
        DEFERRED_CHECKS.record_min(
            SLOT_MIN_SELECTED, selected_tokens,
            message="semblo search selected no causal token", limit=1,
        )
        if token_capacity is not None:
            DEFERRED_CHECKS.record_max(
                SLOT_MAX_SELECTED, selected_tokens,
                message="selected buckets hold more causal tokens than the capacity",
                limit=int(token_capacity),
            )
    elif not DEFERRED_CHECKS.off:
        if bool((selected_tokens == 0).any()):
            raise RuntimeError("semblo search selected no causal token")
        if token_capacity is not None:
            largest_selection = int(selected_tokens.max().item())
            if largest_selection > token_capacity:
                raise RuntimeError(
                    f"selected buckets contain {largest_selection} causal tokens, "
                    f"capacity={token_capacity}"
                )
    tile_counts_per_bucket = torch.div(
        kept_lengths + int(tile_size) - 1,
        int(tile_size),
        rounding_mode="floor",
    )
    tile_offsets = tile_counts_per_bucket.cumsum(dim=-1) - tile_counts_per_bucket
    tile_counts = tile_counts_per_bucket.sum(dim=-1).to(torch.int32)
    if validate_capacity and not DEFERRED_CHECKS.off:
        if DEFERRED_CHECKS.active:
            DEFERRED_CHECKS.record_max(
                SLOT_MAX_TILES, tile_counts,
                message="selected buckets require more posting tiles than the capacity",
                limit=int(max_tiles),
            )
        else:
            required = int(tile_counts.max().item())
            if required > max_tiles:
                raise RuntimeError(
                    f"selected buckets require {required} posting tiles, "
                    f"capacity={max_tiles}"
                )
    starts = torch.full(
        (q_count, num_kv_heads, max_tiles),
        -1,
        dtype=torch.int32,
        device=ranked_bucket_ids.device,
    )
    out_lengths = torch.zeros_like(starts)
    _write_posting_tiles_kernel[(flat_count,)](
        starts,
        out_lengths,
        valid_lengths,
        tile_offsets.reshape(-1),
        keep.reshape(-1),
        ranked_bucket_ids,
        bucket_indptr,
        ranked_bucket_ids.stride(0),
        ranked_bucket_ids.stride(1),
        ranked_bucket_ids.stride(2),
        bucket_indptr.stride(0),
        NUM_KV_HEADS=num_kv_heads,
        MAX_RANKED=max_ranked,
        MAX_TILES=max_tiles,
        TILE_SIZE=tile_size,
        num_warps=1,
    )
    return PostingTiles(starts, out_lengths, tile_counts, selected_tokens)


def paged_posting_attention(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    posting_positions: torch.Tensor,
    tiles: PostingTiles,
    query_positions: torch.Tensor,
    query_request_ids: torch.Tensor,
    block_table: torch.Tensor,
    *,
    block_size: int,
    sm_scale: float,
    softcap: float | None,
    token_tile: int = 16,
    num_splits: int = 1,
    return_logsumexp: bool = False,
) -> PagedSparseAttentionResult:
    _require_triton()
    _validate_hopper(q.device)
    if q.ndim != 3:
        raise ValueError("q must have shape [Q,Hq,D]")
    if kv_cache.ndim != 5 or kv_cache.shape[0] != 2:
        raise ValueError("vLLM KV cache must have shape [2,num_blocks,block,Hkv,D]")
    if posting_positions.ndim != 2:
        raise ValueError("posting_positions must have shape [Hkv,L]")
    q_count, num_q_heads, head_dim = map(int, q.shape)
    num_kv_heads = int(kv_cache.shape[3])
    if posting_positions.shape[0] != num_kv_heads:
        raise ValueError("posting KV-head count differs from the cache")
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if query_positions.shape != (q_count,) or query_request_ids.shape != (q_count,):
        raise ValueError("query metadata must have shape [Q]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be two-dimensional")
    if int(kv_cache.shape[2]) != int(block_size):
        raise ValueError("block_size does not match the vLLM cache")
    if int(kv_cache.shape[-1]) != head_dim:
        raise ValueError("KV-cache head dimension differs from q")
    if token_tile not in {8, 16, 32}:
        raise ValueError("token_tile must be 8, 16, or 32")
    if num_splits not in {1, 2, 4, 8, 16, 32, 64}:
        raise ValueError("invalid split count")

    q = q.contiguous()
    output = torch.empty_like(q)
    lse = (
        torch.empty((q_count, num_q_heads), dtype=torch.float32, device=q.device)
        if return_logsumexp
        else torch.empty(1, dtype=torch.float32, device=q.device)
    )
    q_per_kv = num_q_heads // num_kv_heads
    padded = _next_power_of_two(head_dim)
    block_m = max(16, _next_power_of_two(q_per_kv))
    tile_capacity = int(tiles.starts.shape[-1] and token_tile)
    # Each work tile is created with token_tile capacity by the caller.
    common = dict(
        q_ptr=q,
        kv_cache_ptr=kv_cache,
        posting_positions_ptr=posting_positions,
        tile_starts_ptr=tiles.starts,
        tile_lengths_ptr=tiles.lengths,
        tile_counts_ptr=tiles.counts,
        query_positions_ptr=query_positions.to(torch.int32),
        query_request_ids_ptr=query_request_ids.to(torch.int32),
        block_table_ptr=block_table,
        sm_scale=float(sm_scale),
        softcap=0.0 if softcap is None else float(softcap),
        stride_q0=q.stride(0),
        stride_q1=q.stride(1),
        stride_q2=q.stride(2),
        stride_cache_kind=kv_cache.stride(0),
        stride_cache_block=kv_cache.stride(1),
        stride_cache_token=kv_cache.stride(2),
        stride_cache_head=kv_cache.stride(3),
        stride_cache_dim=kv_cache.stride(4),
        stride_post_h=posting_positions.stride(0),
        stride_tile_q=tiles.starts.stride(0),
        stride_tile_h=tiles.starts.stride(1),
        stride_tile_s=tiles.starts.stride(2),
        stride_count_q=tiles.counts.stride(0),
        stride_count_h=tiles.counts.stride(1),
        stride_block_table_q=block_table.stride(0),
        stride_block_table_b=block_table.stride(1),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        Q_PER_KV=q_per_kv,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=padded,
        BLOCK_SIZE=block_size,
        TILE_CAPACITY=token_tile,
        TOKEN_TILE=token_tile,
        BLOCK_M=block_m,
        USE_SOFTCAP=softcap is not None and softcap > 0,
        num_warps=4,
        num_stages=2,
    )
    if num_splits == 1:
        _paged_posting_attention_kernel[(q_count, num_kv_heads)](
            out_ptr=output,
            lse_ptr=lse,
            stride_out0=output.stride(0),
            stride_out1=output.stride(1),
            stride_out2=output.stride(2),
            stride_lse0=lse.stride(0),
            stride_lse1=lse.stride(1) if lse.ndim == 2 else 0,
            WRITE_LSE=return_logsumexp,
            **common,
        )
        return PagedSparseAttentionResult(output, lse if return_logsumexp else None)

    partial_acc = torch.empty(
        (q_count, num_kv_heads, num_splits, q_per_kv, padded),
        dtype=torch.float32,
        device=q.device,
    )
    partial_max = torch.empty(
        (q_count, num_kv_heads, num_splits, q_per_kv),
        dtype=torch.float32,
        device=q.device,
    )
    partial_sum = torch.empty_like(partial_max)
    _paged_posting_attention_split_kernel[(q_count, num_kv_heads, num_splits)](
        partial_acc_ptr=partial_acc,
        partial_max_ptr=partial_max,
        partial_sum_ptr=partial_sum,
        stride_pa0=partial_acc.stride(0),
        stride_pa1=partial_acc.stride(1),
        stride_pa2=partial_acc.stride(2),
        stride_pa3=partial_acc.stride(3),
        stride_pa4=partial_acc.stride(4),
        stride_pm0=partial_max.stride(0),
        stride_pm1=partial_max.stride(1),
        stride_pm2=partial_max.stride(2),
        stride_pm3=partial_max.stride(3),
        NUM_SPLITS=num_splits,
        **common,
    )
    block_d = 128
    _reduce_split_kernel[(q_count, num_q_heads, triton.cdiv(head_dim, block_d))](
        out_ptr=output,
        lse_ptr=lse,
        partial_acc_ptr=partial_acc,
        partial_max_ptr=partial_max,
        partial_sum_ptr=partial_sum,
        stride_out0=output.stride(0),
        stride_out1=output.stride(1),
        stride_out2=output.stride(2),
        stride_lse0=lse.stride(0),
        stride_lse1=lse.stride(1) if lse.ndim == 2 else 0,
        stride_pa0=partial_acc.stride(0),
        stride_pa1=partial_acc.stride(1),
        stride_pa2=partial_acc.stride(2),
        stride_pa3=partial_acc.stride(3),
        stride_pa4=partial_acc.stride(4),
        stride_pm0=partial_max.stride(0),
        stride_pm1=partial_max.stride(1),
        stride_pm2=partial_max.stride(2),
        stride_pm3=partial_max.stride(3),
        NUM_Q_HEADS=num_q_heads,
        Q_PER_KV=q_per_kv,
        NUM_SPLITS=num_splits,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=padded,
        BLOCK_D=block_d,
        WRITE_LSE=return_logsumexp,
        num_warps=4,
    )
    return PagedSparseAttentionResult(output, lse if return_logsumexp else None)


def reference_paged_posting_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    posting_positions: torch.Tensor,
    tiles: PostingTiles,
    query_positions: torch.Tensor,
    query_request_ids: torch.Tensor,
    block_table: torch.Tensor,
    *,
    sm_scale: float,
    softcap: float | None,
) -> torch.Tensor:
    """Slow exact reference for correctness tests."""
    q_count, num_q_heads, _ = q.shape
    num_kv_heads = int(k_cache.shape[2])
    q_per_kv = num_q_heads // num_kv_heads
    block_size = int(k_cache.shape[1])
    out = torch.zeros_like(q)
    for qi in range(q_count):
        req = int(query_request_ids[qi])
        for kv_head in range(num_kv_heads):
            positions: list[torch.Tensor] = []
            for tile in range(int(tiles.counts[qi, kv_head])):
                start = int(tiles.starts[qi, kv_head, tile])
                length = int(tiles.lengths[qi, kv_head, tile])
                positions.append(posting_positions[kv_head, start : start + length])
            logical = torch.cat(positions).long()
            logical = logical[logical <= int(query_positions[qi])]
            if logical.numel() == 0:
                raise RuntimeError("reference sparse attention selected no causal token")
            physical_blocks = block_table[req, logical // block_size].long()
            offsets = logical % block_size
            keys = k_cache[physical_blocks, offsets, kv_head]
            values = v_cache[physical_blocks, offsets, kv_head]
            start_head = kv_head * q_per_kv
            end_head = start_head + q_per_kv
            logits = q[qi, start_head:end_head].float() @ keys.float().T
            logits *= float(sm_scale)
            if softcap is not None:
                logits = float(softcap) * torch.tanh(logits / float(softcap))
            probs = logits.softmax(dim=-1)
            out[qi, start_head:end_head] = (probs @ values.float()).to(out.dtype)
    return out


if triton is not None:

    @triton.jit
    def _packed_range_attention_kernel(
        out_ptr,
        lse_ptr,
        q_ptr,
        packed_kv_ptr,
        range_starts_ptr,
        range_lengths_ptr,
        range_rows_ptr,
        range_counts_ptr,
        tile_order_ptr,
        sm_scale,
        softcap,
        num_queries,
        stride_out0, stride_out1, stride_out2,
        stride_lse0, stride_lse1,
        stride_q0, stride_q1, stride_q2,
        stride_kv0, stride_kv1, stride_kv2, stride_kv3,
        stride_r0, stride_r1, stride_r2,
        stride_w0, stride_w1, stride_w2, stride_w3,
        stride_c0, stride_c1,
        partial_acc_ptr,
        partial_max_ptr,
        partial_sum_ptr,
        stride_pa0: tl.int64, stride_pa1: tl.int64, stride_pa2: tl.int64,
        stride_pa3: tl.int64, stride_pa4: tl.int64,
        stride_pm0: tl.int64, stride_pm1: tl.int64,
        stride_pm2: tl.int64, stride_pm3: tl.int64,
        QUERIES_PER_TILE: tl.constexpr,
        ORDERED: tl.constexpr,
        NUM_KV_HEADS: tl.constexpr,
        Q_PER_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_DIM_PADDED: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        USE_SOFTCAP: tl.constexpr,
        WRITE_LSE: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        tile = tl.program_id(0)
        if ORDERED:
            tile = tl.load(tile_order_ptr + tile)
        kv_head = tl.program_id(1)
        split = tl.program_id(2)

        offs_d = tl.arange(0, HEAD_DIM_PADDED)
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        d_valid = offs_d < HEAD_DIM
        # Row m is head (m % Q_PER_KV) of query (tile * QUERIES_PER_TILE +
        # m // Q_PER_KV).  With two queries per tile every row is used and each
        # key is read once for both queries.
        member = offs_m // Q_PER_KV
        row_query = tile * QUERIES_PER_TILE + member
        head = kv_head * Q_PER_KV + offs_m % Q_PER_KV
        m_valid = (offs_m < QUERIES_PER_TILE * Q_PER_KV) & (row_query < num_queries)

        q = tl.load(
            q_ptr
            + row_query[:, None].to(tl.int64) * stride_q0
            + head[:, None] * stride_q1
            + offs_d[None, :] * stride_q2,
            mask=m_valid[:, None] & d_valid[None, :],
            other=0.0,
        )

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], tl.float32)

        # The trip count is the query's own range count.  A compile-time bound
        # would spin through the unused ranks of every query and would also
        # recompile the kernel for each distinct range width.
        count = tl.load(range_counts_ptr + tile * stride_c0 + kv_head * stride_c1)

        # With one query the grid is a single block per KV head, which leaves
        # most SMs idle.  Splitting divides the key chunks of the whole
        # selection into equal slices, one program each, and a second pass
        # combines the partial softmaxes.  Slices are taken over the
        # concatenated chunks rather than over ranges, because range lengths
        # differ by an order of magnitude.
        if NUM_SPLITS > 1:
            total_chunks = 0
            for rank in range(count):
                total_chunks += tl.cdiv(
                    tl.load(
                        range_lengths_ptr
                        + tile * stride_r0 + kv_head * stride_r1 + rank * stride_r2
                    ),
                    BLOCK_N,
                )
            per_split = tl.cdiv(total_chunks, NUM_SPLITS)
            first_chunk = split * per_split
            last_chunk = tl.minimum(first_chunk + per_split, total_chunks)
        else:
            first_chunk = 0
            last_chunk = 0

        seen_chunks = 0
        for rank in range(count):
            base = tile * stride_r0 + kv_head * stride_r1 + rank * stride_r2
            start = tl.load(range_starts_ptr + base)
            length = tl.load(range_lengths_ptr + base)
            # How far into this range each row actually reaches.  The postings
            # inside a bucket are position ordered, so a query's causal part is
            # a prefix and the cut is a count, not a position comparison.
            if QUERIES_PER_TILE > 1:
                row_length = tl.load(
                    range_rows_ptr
                    + tile * stride_w0
                    + kv_head * stride_w1
                    + rank * stride_w2
                    + member * stride_w3,
                    mask=m_valid,
                    other=0,
                )
            else:
                row_length = tl.full([BLOCK_M], 0, tl.int32) + length
            chunks = tl.cdiv(length, BLOCK_N)
            if NUM_SPLITS > 1:
                begin = tl.maximum(first_chunk - seen_chunks, 0)
                finish = tl.minimum(last_chunk - seen_chunks, chunks)
            else:
                begin = 0
                finish = chunks
            seen_chunks += chunks
            for chunk in range(begin, finish):
                taken = chunk * BLOCK_N + offs_n
                valid = taken < length
                pos = start + taken
                # One contiguous read per tile.  The caller has already
                # placed K/V in bucket order and truncated each range at the
                # query position, so there is no per-token address
                # translation and no causal mask inside the kernel.
                k = tl.load(
                    packed_kv_ptr
                    + 0 * stride_kv0
                    + pos[None, :] * stride_kv1
                    + kv_head * stride_kv2
                    + offs_d[:, None] * stride_kv3,
                    mask=d_valid[:, None] & valid[None, :],
                    other=0.0,
                )
                v = tl.load(
                    packed_kv_ptr
                    + 1 * stride_kv0
                    + pos[:, None] * stride_kv1
                    + kv_head * stride_kv2
                    + offs_d[None, :] * stride_kv3,
                    mask=valid[:, None] & d_valid[None, :],
                    other=0.0,
                )
                scores = tl.dot(q, k).to(tl.float32) * sm_scale
                if USE_SOFTCAP:
                    scores = _softcap(scores, softcap)
                keep = (taken[None, :] < row_length[:, None]) & m_valid[:, None]
                m_i, l_i, acc = _softmax_update(
                    scores, keep, m_i, l_i, acc, v
                )

        if NUM_SPLITS > 1:
            # Unnormalised, so the reduction can rescale against the maximum
            # across splits.  Same layout the posting path's reducer expects:
            # query, KV head, split, head within the group, dimension.
            local_head = offs_m % Q_PER_KV
            tl.store(
                partial_acc_ptr
                + row_query[:, None].to(tl.int64) * stride_pa0
                + kv_head * stride_pa1
                + split * stride_pa2
                + local_head[:, None] * stride_pa3
                + offs_d[None, :] * stride_pa4,
                acc,
                mask=m_valid[:, None] & d_valid[None, :],
            )
            partial_index = (
                row_query.to(tl.int64) * stride_pm0
                + kv_head * stride_pm1
                + split * stride_pm2
                + local_head * stride_pm3
            )
            tl.store(partial_max_ptr + partial_index, m_i, mask=m_valid)
            tl.store(partial_sum_ptr + partial_index, l_i, mask=m_valid)
        else:
            safe = tl.where(l_i == 0.0, 1.0, l_i)
            out = acc / safe[:, None]
            tl.store(
                out_ptr
                + row_query[:, None].to(tl.int64) * stride_out0
                + head[:, None] * stride_out1
                + offs_d[None, :] * stride_out2,
                out.to(out_ptr.dtype.element_ty),
                mask=m_valid[:, None] & d_valid[None, :],
            )
            if WRITE_LSE:
                finite = m_i != -float("inf")
                lse = tl.where(
                    finite, m_i + tl.log(tl.where(finite, l_i, 1.0)), -float("inf")
                )
                tl.store(
                    lse_ptr + row_query.to(tl.int64) * stride_lse0 + head * stride_lse1,
                    lse,
                    mask=m_valid,
                )


def packed_range_attention(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    range_starts: torch.Tensor,
    range_lengths: torch.Tensor,
    range_counts: torch.Tensor,
    *,
    sm_scale: float,
    softcap: float | None,
    block_n: int = 32,
    return_logsumexp: bool = False,
    num_warps: int = 4,
    num_stages: int = 2,
    range_row_lengths: torch.Tensor | None = None,
    queries_per_tile: int = 1,
    tile_order: torch.Tensor | None = None,
    num_splits: int = 1,
) -> PagedSparseAttentionResult:
    """Attend over contiguous runs of a bucket-packed K/V array.

    Selected tokens are scattered across paged KV blocks but contiguous once
    K/V are stored in bucket order, so a query reads a handful of long runs
    rather than hundreds of single positions.  Ranges must already be truncated
    at the query position; causality is the caller's responsibility, which keeps
    the inner loop free of a position comparison.

    ``block_n`` is the column tile.  At head dimension 512 the accumulator
    leaves room for 32; wider tiles run out of registers.

    ``num_splits`` divides the selection's column tiles between that many
    programs, whose partial softmaxes are then combined.  Prefill does not
    need it, since many query tiles already fill the grid.  A single decode
    query makes the grid one block per KV head, which leaves most SMs idle.
    vLLM's Triton attention splits in the same way when its grid is small.
    """

    _require_triton()
    _validate_hopper(q.device)
    if q.ndim != 3:
        raise ValueError("q must have shape [Q,Hq,D]")
    if packed_kv.ndim != 4 or packed_kv.shape[0] != 2:
        raise ValueError("packed_kv must have shape [2,N,Hkv,D]")
    q_count, num_q_heads, head_dim = map(int, q.shape)
    num_kv_heads = int(packed_kv.shape[2])
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if int(packed_kv.shape[-1]) != head_dim:
        raise ValueError("packed K/V head dimension differs from q")
    group = int(queries_per_tile)
    if group < 1:
        raise ValueError("queries_per_tile must be positive")
    tiles = (q_count + group - 1) // group
    expected = (tiles, num_kv_heads)
    if range_counts.shape != expected:
        raise ValueError("range_counts must have shape [tiles,Hkv]")
    if range_starts.shape != range_lengths.shape:
        raise ValueError("range starts and lengths must have the same shape")
    if range_starts.shape[:2] != expected:
        raise ValueError("range tensors must have shape [tiles,Hkv,R]")
    if group > 1:
        if range_row_lengths is None:
            raise ValueError("a query tile needs a per-query length for each range")
        if range_row_lengths.shape != (*range_starts.shape, group):
            raise ValueError("range_row_lengths must have shape [tiles,Hkv,R,G]")
    rows = range_row_lengths if range_row_lengths is not None else range_lengths
    if block_n not in {16, 32, 64}:
        raise ValueError("block_n must be 16, 32, or 64")

    if num_splits not in {1, 2, 4, 8, 16, 32, 64}:
        raise ValueError("num_splits must be one of 1,2,4,8,16,32,64")

    q = q.contiguous()
    output = torch.empty_like(q)
    lse = (
        torch.empty((q_count, num_q_heads), dtype=torch.float32, device=q.device)
        if return_logsumexp
        else torch.empty(1, dtype=torch.float32, device=q.device)
    )
    q_per_kv = num_q_heads // num_kv_heads
    padded = _next_power_of_two(head_dim)
    if num_splits > 1:
        partial_acc = torch.empty(
            (q_count, num_kv_heads, num_splits, q_per_kv, padded),
            dtype=torch.float32, device=q.device,
        )
        partial_max = torch.empty(
            (q_count, num_kv_heads, num_splits, q_per_kv),
            dtype=torch.float32, device=q.device,
        )
        partial_sum = torch.empty_like(partial_max)
    else:
        # The kernel never reads these when it is not splitting; a real tensor
        # is still needed for the pointer argument.
        partial_acc = lse
        partial_max = lse
        partial_sum = lse
    _packed_range_attention_kernel[(tiles, num_kv_heads, num_splits)](
        out_ptr=output,
        lse_ptr=lse,
        q_ptr=q,
        packed_kv_ptr=packed_kv,
        range_starts_ptr=range_starts,
        range_lengths_ptr=range_lengths,
        range_rows_ptr=rows,
        range_counts_ptr=range_counts,
        tile_order_ptr=tile_order if tile_order is not None else range_counts,
        sm_scale=float(sm_scale),
        softcap=0.0 if softcap is None else float(softcap),
        num_queries=q_count,
        stride_out0=output.stride(0),
        stride_out1=output.stride(1),
        stride_out2=output.stride(2),
        stride_lse0=lse.stride(0) if return_logsumexp else 0,
        stride_lse1=lse.stride(1) if return_logsumexp else 0,
        stride_q0=q.stride(0),
        stride_q1=q.stride(1),
        stride_q2=q.stride(2),
        stride_kv0=packed_kv.stride(0),
        stride_kv1=packed_kv.stride(1),
        stride_kv2=packed_kv.stride(2),
        stride_kv3=packed_kv.stride(3),
        stride_r0=range_starts.stride(0),
        stride_r1=range_starts.stride(1),
        stride_r2=range_starts.stride(2),
        stride_w0=rows.stride(0),
        stride_w1=rows.stride(1),
        stride_w2=rows.stride(2),
        stride_w3=rows.stride(3) if group > 1 else 0,
        stride_c0=range_counts.stride(0),
        stride_c1=range_counts.stride(1),
        partial_acc_ptr=partial_acc,
        partial_max_ptr=partial_max,
        partial_sum_ptr=partial_sum,
        stride_pa0=partial_acc.stride(0) if num_splits > 1 else 0,
        stride_pa1=partial_acc.stride(1) if num_splits > 1 else 0,
        stride_pa2=partial_acc.stride(2) if num_splits > 1 else 0,
        stride_pa3=partial_acc.stride(3) if num_splits > 1 else 0,
        stride_pa4=partial_acc.stride(4) if num_splits > 1 else 0,
        stride_pm0=partial_max.stride(0) if num_splits > 1 else 0,
        stride_pm1=partial_max.stride(1) if num_splits > 1 else 0,
        stride_pm2=partial_max.stride(2) if num_splits > 1 else 0,
        stride_pm3=partial_max.stride(3) if num_splits > 1 else 0,
        QUERIES_PER_TILE=group,
        ORDERED=tile_order is not None,
        NUM_KV_HEADS=num_kv_heads,
        Q_PER_KV=q_per_kv,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=padded,
        BLOCK_N=int(block_n),
        BLOCK_M=max(16, _next_power_of_two(group * q_per_kv)),
        USE_SOFTCAP=softcap is not None and softcap > 0,
        WRITE_LSE=bool(return_logsumexp),
        NUM_SPLITS=int(num_splits),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    if num_splits > 1:
        block_d = 128
        _reduce_split_kernel[
            (q_count, num_q_heads, triton.cdiv(head_dim, block_d))
        ](
            out_ptr=output,
            lse_ptr=lse,
            partial_acc_ptr=partial_acc,
            partial_max_ptr=partial_max,
            partial_sum_ptr=partial_sum,
            stride_out0=output.stride(0),
            stride_out1=output.stride(1),
            stride_out2=output.stride(2),
            stride_lse0=lse.stride(0) if return_logsumexp else 0,
            stride_lse1=lse.stride(1) if return_logsumexp else 0,
            stride_pa0=partial_acc.stride(0),
            stride_pa1=partial_acc.stride(1),
            stride_pa2=partial_acc.stride(2),
            stride_pa3=partial_acc.stride(3),
            stride_pa4=partial_acc.stride(4),
            stride_pm0=partial_max.stride(0),
            stride_pm1=partial_max.stride(1),
            stride_pm2=partial_max.stride(2),
            stride_pm3=partial_max.stride(3),
            NUM_Q_HEADS=num_q_heads,
            Q_PER_KV=q_per_kv,
            NUM_SPLITS=int(num_splits),
            HEAD_DIM=head_dim,
            HEAD_DIM_PADDED=padded,
            BLOCK_D=block_d,
            WRITE_LSE=bool(return_logsumexp),
            num_warps=4,
        )
    return PagedSparseAttentionResult(
        output=output,
        logsumexp=lse if return_logsumexp else None,
    )


def reference_packed_range_attention(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    range_starts: torch.Tensor,
    range_lengths: torch.Tensor,
    range_counts: torch.Tensor,
    *,
    sm_scale: float,
    softcap: float | None,
) -> torch.Tensor:
    """Plain PyTorch equivalent of :func:`packed_range_attention`."""

    q_count, num_q_heads, head_dim = map(int, q.shape)
    num_kv_heads = int(packed_kv.shape[2])
    q_per_kv = num_q_heads // num_kv_heads
    output = torch.zeros_like(q, dtype=torch.float32)
    for query in range(q_count):
        for kv_head in range(num_kv_heads):
            positions: list[int] = []
            for rank in range(int(range_counts[query, kv_head].item())):
                start = int(range_starts[query, kv_head, rank].item())
                length = int(range_lengths[query, kv_head, rank].item())
                positions.extend(range(start, start + length))
            if not positions:
                continue
            index = torch.tensor(positions, device=q.device, dtype=torch.long)
            keys = packed_kv[0, index, kv_head].float()
            values = packed_kv[1, index, kv_head].float()
            for offset in range(q_per_kv):
                head = kv_head * q_per_kv + offset
                scores = (q[query, head].float() @ keys.T) * float(sm_scale)
                if softcap is not None and softcap > 0:
                    scores = float(softcap) * torch.tanh(scores / float(softcap))
                weights = torch.softmax(scores, dim=-1)
                output[query, head] = weights @ values
    return output.to(q.dtype)


@dataclass(frozen=True)
class PackedRanges:
    starts: torch.Tensor  # [Q,Hkv,R], offsets into the bucket-packed K/V
    lengths: torch.Tensor  # [Q,Hkv,R]
    counts: torch.Tensor  # [Q,Hkv]
    selected_tokens: torch.Tensor  # [Q,Hkv]
    max_kept_rank: torch.Tensor  # [Q,Hkv], deepest ranked bucket used
    bucket_ids: torch.Tensor | None = None  # [Q,Hkv,R], -1 where unused
    tile_lengths: torch.Tensor | None = None  # [P,Hkv,slots,G], filled when tiling


def _bucket_causal_lengths(
    ranked_bucket_ids: torch.Tensor,
    ranked_bucket_counts: torch.Tensor,
    bucket_indptr: torch.Tensor,
    posting_positions: torch.Tensor,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    """Positions of each ranked bucket that lie at or before the query.

    Within a bucket the posting list keeps original position order, so the
    causal part is a prefix and a binary search finds its length.
    """

    q_count, num_kv_heads, max_ranked = map(int, ranked_bucket_ids.shape)
    if ranked_bucket_ids.is_cuda:
        _require_triton()
        flat = q_count * num_kv_heads * max_ranked
        lengths = torch.empty(flat, dtype=torch.int64, device=ranked_bucket_ids.device)
        _bucket_valid_lengths_kernel[(flat,)](
            lengths,
            ranked_bucket_ids,
            ranked_bucket_counts,
            bucket_indptr,
            posting_positions,
            query_positions.to(torch.int32),
            ranked_bucket_ids.stride(0),
            ranked_bucket_ids.stride(1),
            ranked_bucket_ids.stride(2),
            ranked_bucket_counts.stride(0),
            ranked_bucket_counts.stride(1),
            bucket_indptr.stride(0),
            posting_positions.stride(0),
            NUM_KV_HEADS=num_kv_heads,
            MAX_RANKED=max_ranked,
            num_warps=1,
        )
        return lengths.view(q_count, num_kv_heads, max_ranked)

    lengths = torch.zeros(
        q_count, num_kv_heads, max_ranked, dtype=torch.int64, device=ranked_bucket_ids.device
    )
    for query in range(q_count):
        limit = int(query_positions[query].item())
        for kv_head in range(num_kv_heads):
            for rank in range(int(ranked_bucket_counts[query, kv_head].item())):
                bucket = int(ranked_bucket_ids[query, kv_head, rank].item())
                if bucket < 0:
                    continue
                begin = int(bucket_indptr[kv_head, bucket].item())
                finish = int(bucket_indptr[kv_head, bucket + 1].item())
                interval = posting_positions[kv_head, begin:finish]
                lengths[query, kv_head, rank] = int(
                    torch.searchsorted(
                        interval.contiguous(),
                        torch.tensor(limit, device=interval.device, dtype=interval.dtype),
                        right=True,
                    ).item()
                )
    return lengths


if triton is not None:

    @triton.jit
    def _ranked_prefix_ranges_kernel(
        starts_ptr,           # [Q,Hkv,MAX_RANGES] int32
        lengths_ptr,          # [Q,Hkv,MAX_RANGES] int32
        buckets_ptr,          # [Q,Hkv,MAX_RANGES] int32, -1 where unused
        tile_lengths_ptr,     # [P,Hkv,slots,GROUP] int32, zeroed by the caller
        bucket_slot_ptr,      # [Hkv,num_buckets] int32, -1 for empty buckets
        counts_ptr,           # [Q,Hkv] int32
        selected_ptr,         # [Q,Hkv] int32
        deepest_ptr,          # [Q,Hkv] int32
        overflow_ptr,         # [Q,Hkv] int32, ranges past MAX_RANGES
        ranked_ids_ptr,       # [Q,Hkv,max_ranked] int32
        ranked_counts_ptr,    # [Q,Hkv] int32
        indptr_ptr,           # [Hkv,num_buckets+1]
        positions_ptr,        # [Hkv,N] int32
        query_positions_ptr,  # [Q] int32
        target_tokens_ptr,    # [Q] int32, per-query budget (adaptive)
        stride_rank_q: tl.int64,
        stride_rank_h: tl.int64,
        stride_rank_s: tl.int64,
        stride_count_q: tl.int64,
        stride_count_h: tl.int64,
        stride_indptr_h: tl.int64,
        stride_post_h: tl.int64,
        stride_out_q: tl.int64,
        stride_out_h: tl.int64,
        stride_out_r: tl.int64,
        stride_scalar_q: tl.int64,
        stride_scalar_h: tl.int64,
        stride_tile_p: tl.int64,
        stride_tile_h: tl.int64,
        stride_tile_s: tl.int64,
        stride_tile_g: tl.int64,
        stride_slot_h: tl.int64,
        GROUP: tl.constexpr,
        MAX_RANGES: tl.constexpr,
        BLOCK_R: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
    ):
        """Walk one query's ranked list and stop where the budget is reached.

        Rank order is the order the budget is applied in, so the walk can end
        early: late in a long prompt a query fills its budget after a few dozen
        of the several hundred ranked buckets.  A tile of ``BLOCK_R`` ranks is
        handled at a time, one lane per bucket, so the binary searches run
        side by side instead of one program each.
        """

        query = tl.program_id(0)
        kv_head = tl.program_id(1)

        ranked_count = tl.load(
            ranked_counts_ptr + query * stride_count_q + kv_head * stride_count_h
        )
        q_pos = tl.load(query_positions_ptr + query)
        target_tokens = tl.load(target_tokens_ptr + query)

        offs = tl.arange(0, BLOCK_R)
        taken = 0        # causal tokens in every rank walked so far
        kept_tokens = 0  # only the buckets actually admitted
        written = 0
        deepest = -1
        base_rank = 0

        while (base_rank < ranked_count) and (taken < target_tokens):
            rank = base_rank + offs
            valid = rank < ranked_count
            bucket = tl.load(
                ranked_ids_ptr
                + query * stride_rank_q
                + kv_head * stride_rank_h
                + rank.to(tl.int64) * stride_rank_s,
                mask=valid,
                other=-1,
            ).to(tl.int64)
            valid &= bucket >= 0
            begin = tl.load(
                indptr_ptr + kv_head * stride_indptr_h + bucket, mask=valid, other=0
            )
            finish = tl.load(
                indptr_ptr + kv_head * stride_indptr_h + bucket + 1, mask=valid, other=0
            )

            # The posting list keeps position order inside a bucket, so the
            # causal part is a prefix and a binary search finds its length.
            lo = tl.zeros([BLOCK_R], tl.int64)
            hi = tl.where(valid, finish - begin, 0)
            for _ in tl.static_range(SEARCH_STEPS):
                mid = (lo + hi) // 2
                active = lo < hi
                pos = tl.load(
                    positions_ptr + kv_head * stride_post_h + begin + mid,
                    mask=active,
                    other=0,
                )
                right = pos <= q_pos
                lo = tl.where(active & right, mid + 1, lo)
                hi = tl.where(active & (right == 0), mid, hi)
            length = tl.where(valid, lo, 0).to(tl.int32)

            before = taken + tl.cumsum(length, axis=0) - length
            keep = valid & (length > 0) & (before < target_tokens)
            keep_int = keep.to(tl.int32)
            destination = written + tl.cumsum(keep_int, axis=0) - keep_int

            store_mask = keep & (destination < MAX_RANGES)
            offset = (
                query * stride_out_q
                + kv_head * stride_out_h
                + destination.to(tl.int64) * stride_out_r
            )
            tl.store(starts_ptr + offset, begin.to(tl.int32), mask=store_mask)
            tl.store(lengths_ptr + offset, length, mask=store_mask)
            # Pairing queries needs to know which bucket a range is, so that
            # two queries that chose the same one share its read.
            tl.store(buckets_ptr + offset, bucket.to(tl.int32), mask=store_mask)
            if GROUP > 1:
                # Lay the length down in the tile's own column.  No two queries
                # write the same address, so the merge is a later max, not a
                # race.
                slot = tl.load(
                    bucket_slot_ptr + kv_head * stride_slot_h + bucket,
                    mask=keep, other=-1,
                ).to(tl.int64)
                tl.store(
                    tile_lengths_ptr
                    + (query // GROUP) * stride_tile_p
                    + kv_head * stride_tile_h
                    + slot * stride_tile_s
                    + (query % GROUP) * stride_tile_g,
                    length,
                    mask=keep & (slot >= 0),
                )

            written += tl.sum(keep_int, axis=0)
            taken += tl.sum(tl.where(valid, length, 0), axis=0)
            kept_tokens += tl.sum(tl.where(keep, length, 0), axis=0)
            deepest = tl.maximum(deepest, tl.max(tl.where(keep, rank, -1), axis=0))
            base_rank += BLOCK_R

        scalar = query * stride_scalar_q + kv_head * stride_scalar_h
        tl.store(counts_ptr + scalar, tl.minimum(written, MAX_RANGES))
        tl.store(selected_ptr + scalar, kept_tokens)
        tl.store(deepest_ptr + scalar, deepest)
        tl.store(overflow_ptr + scalar, tl.maximum(written - MAX_RANGES, 0))


@dataclass(frozen=True)
class PairedRanges:
    """Ranges for a tile of neighbouring queries that share one K/V read.

    ``lengths`` is the union: how many packed positions the tile must read.
    ``row_lengths`` is how many of those each query in the tile actually
    attends to.  A bucket's postings are in position order, so a query's causal
    part is a prefix and the per-query cut is a column count, not a mask over
    positions.  Two adjacent queries differ by at most one token in any bucket.
    """

    starts: torch.Tensor       # [P,Hkv,U] int32
    lengths: torch.Tensor      # [P,Hkv,U] int32, union over the tile
    row_lengths: torch.Tensor  # [P,Hkv,U,G] int32, per query in the tile
    counts: torch.Tensor       # [P,Hkv] int32
    num_queries: int
    tokens: torch.Tensor | None = None  # [P,Hkv] int32, union tokens per tile
    order: torch.Tensor | None = None   # [P] int32, heaviest tile first


def pair_packed_ranges(
    ranges: PackedRanges,
    slot_start: torch.Tensor,
    *,
    queries_per_tile: int = 2,
    block_slots: int = 256,
) -> PairedRanges:
    """Merge each tile of ``queries_per_tile`` queries onto shared ranges.

    Dense attention reads one K/V tile for two query positions because every
    query wants the same keys.  Here each query chooses its own buckets, so the
    tile reads the union of their choices and cuts each row separately.  The
    merge is exact: no query gains or loses a token.

    The lengths were already laid out per tile by the range kernel, so all that
    is left is to drop the slots nobody chose.
    """

    if ranges.tile_lengths is None:
        raise ValueError("packed ranges were not built for a query tile")
    _require_triton()
    tiles, num_kv_heads, slots, group = map(int, ranges.tile_lengths.shape)
    if group != int(queries_per_tile):
        raise ValueError("the ranges were built for a different tile size")
    device = ranges.tile_lengths.device
    # A tile's queries draw from the same non-empty bucket list, so their union
    # can never need more entries than there are non-empty buckets.
    starts = torch.zeros(tiles, num_kv_heads, slots, dtype=torch.int32, device=device)
    lengths = torch.zeros_like(starts)
    row_lengths = torch.zeros(
        tiles, num_kv_heads, slots, group, dtype=torch.int32, device=device
    )
    counts = torch.zeros(tiles, num_kv_heads, dtype=torch.int32, device=device)
    tokens = torch.zeros_like(counts)
    _compact_union_ranges_kernel[(tiles, num_kv_heads)](
        starts, lengths, row_lengths, counts, tokens,
        ranges.tile_lengths, slot_start, slots,
        ranges.tile_lengths.stride(0), ranges.tile_lengths.stride(1),
        ranges.tile_lengths.stride(2), ranges.tile_lengths.stride(3),
        slot_start.stride(0),
        starts.stride(0), starts.stride(1), starts.stride(2),
        row_lengths.stride(0), row_lengths.stride(1),
        row_lengths.stride(2), row_lengths.stride(3),
        counts.stride(0), counts.stride(1),
        GROUP=group,
        BLOCK_S=int(block_slots),
        num_warps=4,
        num_stages=1,
    )
    # Tiles differ by more than 5x in how much they read.  Launching the
    # heaviest first keeps them from forming the tail of the grid.
    order = torch.argsort(tokens.sum(dim=1), descending=True).to(torch.int32)
    return PairedRanges(
        starts=starts,
        lengths=lengths,
        row_lengths=row_lengths,
        counts=counts,
        num_queries=int(ranges.lengths.shape[0]),
        tokens=tokens,
        order=order.contiguous(),
    )



if triton is not None:

    @triton.jit
    def _compact_union_ranges_kernel(
        starts_ptr,        # [P,Hkv,U] int32
        lengths_ptr,       # [P,Hkv,U] int32, union over the tile
        rows_ptr,          # [P,Hkv,U,GROUP] int32
        counts_ptr,        # [P,Hkv] int32
        tokens_ptr,        # [P,Hkv] int32, union tokens the tile must read
        tile_lengths_ptr,  # [P,Hkv,slots,GROUP] int32
        slot_start_ptr,    # [Hkv,slots] int32
        num_slots,
        stride_tile_p: tl.int64,
        stride_tile_h: tl.int64,
        stride_tile_s: tl.int64,
        stride_tile_g: tl.int64,
        stride_ss_h: tl.int64,
        stride_out_p: tl.int64,
        stride_out_h: tl.int64,
        stride_out_u: tl.int64,
        stride_row_p: tl.int64,
        stride_row_h: tl.int64,
        stride_row_u: tl.int64,
        stride_row_g: tl.int64,
        stride_c0: tl.int64,
        stride_c1: tl.int64,
        GROUP: tl.constexpr,
        BLOCK_S: tl.constexpr,
    ):
        """Drop the slots nobody chose and number what is left."""

        tile = tl.program_id(0)
        kv_head = tl.program_id(1)
        offs = tl.arange(0, BLOCK_S)
        written = 0
        tokens = 0
        base = 0
        while base < num_slots:
            slot = base + offs
            valid = slot < num_slots
            source = (
                tile_lengths_ptr
                + tile * stride_tile_p
                + kv_head * stride_tile_h
                + slot.to(tl.int64) * stride_tile_s
            )
            union = tl.zeros([BLOCK_S], tl.int32)
            for member in tl.static_range(GROUP):
                union = tl.maximum(
                    union,
                    tl.load(source + member * stride_tile_g, mask=valid, other=0),
                )
            keep = valid & (union > 0)
            keep_int = keep.to(tl.int32)
            destination = written + tl.cumsum(keep_int, axis=0) - keep_int
            out = (
                tile * stride_out_p
                + kv_head * stride_out_h
                + destination.to(tl.int64) * stride_out_u
            )
            tl.store(
                starts_ptr + out,
                tl.load(
                    slot_start_ptr + kv_head * stride_ss_h + slot, mask=keep, other=0
                ),
                mask=keep,
            )
            tl.store(lengths_ptr + out, union, mask=keep)
            row = (
                tile * stride_row_p
                + kv_head * stride_row_h
                + destination.to(tl.int64) * stride_row_u
            )
            for member in tl.static_range(GROUP):
                tl.store(
                    rows_ptr + row + member * stride_row_g,
                    tl.load(source + member * stride_tile_g, mask=valid, other=0),
                    mask=keep,
                )
            written += tl.sum(keep_int, axis=0)
            tokens += tl.sum(tl.where(keep, union, 0), axis=0)
            base += BLOCK_S
        tl.store(counts_ptr + tile * stride_c0 + kv_head * stride_c1, written)
        tl.store(tokens_ptr + tile * stride_c0 + kv_head * stride_c1, tokens)


def _compare_packed_range_paths(
    ranked_bucket_ids: torch.Tensor,
    ranked_bucket_counts: torch.Tensor,
    bucket_indptr: torch.Tensor,
    posting_positions: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    target_tokens: int,
) -> None:
    """Debug hook: assert the kernel agrees with the grid-wide formulation.

    Enabled by ``SEMBLO_VERIFY_RANGES``.  Both paths apply the same rule, so any
    disagreement on real prompts is a defect in one of them.
    """

    q_count, num_kv_heads, max_ranked = map(int, ranked_bucket_ids.shape)
    lengths = _bucket_causal_lengths(
        ranked_bucket_ids,
        ranked_bucket_counts,
        bucket_indptr,
        posting_positions,
        query_positions,
    )
    if isinstance(target_tokens, torch.Tensor):
        target_col = (
            target_tokens.to(device=lengths.device, dtype=lengths.dtype)
            .view(-1, 1, 1)
        )
    else:
        target_col = torch.full(
            (1, 1, 1), int(target_tokens), dtype=lengths.dtype, device=lengths.device
        )
    rank = torch.arange(max_ranked, device=lengths.device).view(1, 1, -1)
    before = lengths.cumsum(dim=-1) - lengths
    keep = (
        (rank < ranked_bucket_counts[..., None])
        & (lengths > 0)
        & (before < target_col)
    )
    expected = torch.where(keep, lengths, torch.zeros_like(lengths)).sum(dim=-1)

    actual = _packed_ranges_on_device(
        ranked_bucket_ids,
        ranked_bucket_counts,
        bucket_indptr,
        posting_positions,
        query_positions,
        target_tokens=target_tokens,
        token_capacity=None,
        max_ranges=None,
        ranked_list_is_complete=True,
    )
    difference = actual.selected_tokens.to(torch.int64) - expected
    if bool((difference != 0).any()):
        worst = int(difference.abs().argmax())
        query, kv_head = divmod(worst, num_kv_heads)
        count = int(ranked_bucket_counts[query, kv_head])
        raise RuntimeError(
            "packed-range paths disagree: "
            f"query={query} head={kv_head} position={int(query_positions[query])} "
            f"ranked_count={count} kept_ranks={int(keep[query, kv_head].sum())} "
            f"reference={int(expected[query, kv_head])} "
            f"kernel={int(actual.selected_tokens[query, kv_head])} "
            f"first_lengths={lengths[query, kv_head, :8].tolist()} "
            f"max_length={int(lengths[query, kv_head].max())}"
        )


def _packed_ranges_on_device(
    ranked_bucket_ids: torch.Tensor,
    ranked_bucket_counts: torch.Tensor,
    bucket_indptr: torch.Tensor,
    posting_positions: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    target_tokens: int | torch.Tensor,
    token_capacity: int | None,
    max_ranges: int | None,
    ranked_list_is_complete: bool,
    block_ranks: int = 128,
    queries_per_tile: int = 1,
    bucket_slot: torch.Tensor | None = None,
    num_slots: int = 0,
) -> PackedRanges:
    """CUDA path of :func:`ranked_buckets_to_packed_ranges`.

    With ``queries_per_tile`` above one the kernel also lays each query's kept
    lengths into its tile's column, which is what the merge in
    :func:`pair_packed_ranges` reads.  Doing it here costs one store per range
    and saves a second walk of the ranked list.
    """

    _require_triton()
    q_count, num_kv_heads, max_ranked = map(int, ranked_bucket_ids.shape)
    device = ranked_bucket_ids.device
    width = max_ranked if max_ranges is None else int(max_ranges)
    # Left uninitialised to save kernel launches on the decode path.  Nothing
    # reads an element the kernel does not write: the scalars are stored
    # unconditionally, and the ranges are read only up to their count.
    starts = torch.empty(q_count, num_kv_heads, width, dtype=torch.int32, device=device)
    lengths = torch.empty_like(starts)
    buckets = torch.empty_like(starts)
    counts = torch.empty(q_count, num_kv_heads, dtype=torch.int32, device=device)
    selected = torch.empty_like(counts)
    deepest = torch.empty_like(counts)
    overflow = torch.empty_like(counts)

    group = int(queries_per_tile)
    if group > 1:
        if bucket_slot is None or num_slots <= 0:
            raise ValueError("query tiles need the bucket-to-slot table")
        tiles = (q_count + group - 1) // group
        tile_lengths = torch.zeros(
            tiles, num_kv_heads, int(num_slots), group,
            dtype=torch.int32, device=device,
        )
    else:
        tile_lengths = torch.empty(1, 1, 1, 1, dtype=torch.int32, device=device)
        bucket_slot = tile_lengths.view(1, 1)

    # Per-query budgets (adaptive) and the uniform budget share one kernel:
    # a [Q] vector is materialised either way and the kernel loads one int.
    if isinstance(target_tokens, torch.Tensor):
        target_vec = target_tokens.to(device=device, dtype=torch.int32).contiguous()
        if target_vec.shape != (q_count,):
            raise ValueError("per-query target_tokens must have shape [Q]")
        target_hi = target_vec
    else:
        target_vec = torch.full(
            (q_count,), int(target_tokens), dtype=torch.int32, device=device
        )
        target_hi = None

    # The search never needs more steps than the widest bucket, and the posting
    # list bounds that from above.
    search_steps = int(posting_positions.shape[-1]).bit_length() + 1
    _ranked_prefix_ranges_kernel[(q_count, num_kv_heads)](
        starts, lengths, buckets, tile_lengths, bucket_slot,
        counts, selected, deepest, overflow,
        ranked_bucket_ids, ranked_bucket_counts,
        bucket_indptr, posting_positions, query_positions.to(torch.int32),
        target_vec,
        ranked_bucket_ids.stride(0),
        ranked_bucket_ids.stride(1),
        ranked_bucket_ids.stride(2),
        ranked_bucket_counts.stride(0),
        ranked_bucket_counts.stride(1),
        bucket_indptr.stride(0),
        posting_positions.stride(0),
        starts.stride(0), starts.stride(1), starts.stride(2),
        counts.stride(0), counts.stride(1),
        tile_lengths.stride(0), tile_lengths.stride(1),
        tile_lengths.stride(2), tile_lengths.stride(3),
        bucket_slot.stride(0),
        GROUP=group,
        MAX_RANGES=width,
        BLOCK_R=int(block_ranks),
        SEARCH_STEPS=search_steps,
        num_warps=max(1, int(block_ranks) // 32),
        num_stages=1,
    )

    if DEFERRED_CHECKS.off or DEFERRED_CHECKS.active:
        if DEFERRED_CHECKS.active:
            # Reading the results back here would synchronise with the host,
            # so the checks are recorded on the device.  See validation.py.
            DEFERRED_CHECKS.record_min(
                SLOT_MIN_SELECTED, selected,
                message="semblo search selected no causal token", limit=1,
            )
            if token_capacity is not None:
                DEFERRED_CHECKS.record_max(
                    SLOT_MAX_SELECTED, selected,
                    message="selected buckets hold more causal tokens than the capacity",
                    limit=int(token_capacity),
                )
            DEFERRED_CHECKS.record_max(
                SLOT_MAX_RANGES, overflow,
                message=f"selected buckets need more than {width} ranges", limit=0,
            )
            if not ranked_list_is_complete:
                DEFERRED_CHECKS.record_violations(
                    SLOT_SHORTFALL,
                    (selected < target_vec[:, None]) & (ranked_bucket_counts >= max_ranked),
                    message=(
                        "the ranked bucket list ran out before the token budget was "
                        f"reached; raise max_probe_buckets above {max_ranked}"
                    ),
                )
        return PackedRanges(
            starts=starts,
            lengths=lengths,
            counts=counts,
            selected_tokens=selected,
            max_kept_rank=deepest,
            bucket_ids=buckets,
            tile_lengths=tile_lengths if group > 1 else None,
        )

    # All four checks are read back in a single device-to-host transfer.
    shortfall = (
        torch.zeros((), dtype=torch.int32, device=device)
        if ranked_list_is_complete
        else (
            (selected < target_vec[:, None]) & (ranked_bucket_counts >= max_ranked)
        ).sum().to(torch.int32)
    )
    status = torch.stack(
        (
            selected.min().reshape(()),
            selected.max().reshape(()),
            overflow.max().reshape(()),
            shortfall,
        )
    ).tolist()
    smallest, largest, spilled, cut_short = (int(value) for value in status)

    if smallest == 0:
        raise RuntimeError("semblo search selected no causal token")
    if token_capacity is not None and largest > token_capacity:
        raise RuntimeError(
            f"selected buckets contain {largest} causal tokens, "
            f"capacity={token_capacity}"
        )
    if spilled > 0:
        raise RuntimeError(
            f"selected buckets need {width + spilled} ranges, capacity={width}"
        )
    if cut_short:
        raise RuntimeError(
            "the ranked bucket list ran out before the token budget was reached; "
            f"raise max_probe_buckets above {max_ranked}"
        )
    return PackedRanges(
        starts=starts,
        lengths=lengths,
        counts=counts,
        selected_tokens=selected,
        max_kept_rank=deepest,
        bucket_ids=buckets,
        tile_lengths=tile_lengths if group > 1 else None,
    )


def ranked_buckets_to_packed_ranges(
    ranked_bucket_ids: torch.Tensor,
    ranked_bucket_counts: torch.Tensor,
    bucket_indptr: torch.Tensor,
    posting_positions: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    target_tokens: int | torch.Tensor,
    token_capacity: int | None = None,
    max_ranges: int | None = None,
    ranked_list_is_complete: bool = False,
    queries_per_tile: int = 1,
    bucket_slot: torch.Tensor | None = None,
    num_slots: int = 0,
) -> PackedRanges:
    """Turn ranked buckets into contiguous ranges of the bucket-packed K/V.

    A bucket occupies one interval of the packed array, so its causal part is
    one range and no tiling is needed.  Buckets are taken whole in rank order
    until the token budget is reached, which is the same rule the work-tile
    builder applies.

    Set ``ranked_list_is_complete`` when the ranking covered every bucket.  A
    query that then falls short of the budget simply has less causal context,
    whereas a query that exhausts a capped list has silently lost candidates and
    is treated as an error.

    On CUDA one kernel walks the ranked list and stops at the budget; the code
    below is the CPU reference for the same rule.  The two differ only in the
    padded width of ``starts`` and ``lengths``, which the reference trims to the
    widest query and the kernel leaves at the ranked-list width.
    """

    if ranked_bucket_ids.ndim != 3:
        raise ValueError("ranked_bucket_ids must have shape [Q,Hkv,max_ranked]")
    q_count, num_kv_heads, max_ranked = map(int, ranked_bucket_ids.shape)
    if ranked_bucket_counts.shape != (q_count, num_kv_heads):
        raise ValueError("ranked_bucket_counts must have shape [Q,Hkv]")
    if bucket_indptr.ndim != 2 or bucket_indptr.shape[0] != num_kv_heads:
        raise ValueError("bucket_indptr must have shape [Hkv,num_buckets+1]")
    if query_positions.shape != (q_count,):
        raise ValueError("query_positions must have shape [Q]")
    if isinstance(target_tokens, torch.Tensor):
        if target_tokens.shape != (q_count,):
            raise ValueError("per-query target_tokens must have shape [Q]")
        if bool((target_tokens <= 0).any()):
            raise ValueError("target_tokens must be positive")
    elif target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    if ranked_bucket_ids.is_cuda and os.environ.get("SEMBLO_VERIFY_RANGES"):
        _compare_packed_range_paths(
            ranked_bucket_ids,
            ranked_bucket_counts,
            bucket_indptr,
            posting_positions,
            query_positions,
            target_tokens=target_tokens,
        )
    if ranked_bucket_ids.is_cuda:
        return _packed_ranges_on_device(
            ranked_bucket_ids,
            ranked_bucket_counts,
            bucket_indptr,
            posting_positions,
            query_positions,
            target_tokens=target_tokens,
            token_capacity=token_capacity,
            max_ranges=max_ranges,
            ranked_list_is_complete=ranked_list_is_complete,
            queries_per_tile=queries_per_tile,
            bucket_slot=bucket_slot,
            num_slots=num_slots,
        )

    lengths = _bucket_causal_lengths(
        ranked_bucket_ids,
        ranked_bucket_counts,
        bucket_indptr,
        posting_positions,
        query_positions,
    )
    if isinstance(target_tokens, torch.Tensor):
        target_col = (
            target_tokens.to(device=lengths.device, dtype=lengths.dtype)
            .view(-1, 1, 1)
        )
    else:
        target_col = torch.full(
            (1, 1, 1), int(target_tokens), dtype=lengths.dtype, device=lengths.device
        )
    rank = torch.arange(max_ranked, device=lengths.device).view(1, 1, -1)
    before = lengths.cumsum(dim=-1) - lengths
    keep = (
        (rank < ranked_bucket_counts[..., None])
        & (lengths > 0)
        & (before < target_col)
    )
    kept = torch.where(keep, lengths, torch.zeros_like(lengths))
    selected = kept.sum(dim=-1).to(torch.int32)
    if bool((selected == 0).any()):
        raise RuntimeError("semblo search selected no causal token")
    if token_capacity is not None:
        largest = int(selected.max().item())
        if largest > token_capacity:
            raise RuntimeError(
                f"selected buckets contain {largest} causal tokens, "
                f"capacity={token_capacity}"
            )

    # A query that misses the budget only because the ranked list ran out has
    # silently attended to fewer tokens than asked for.  Early queries legitimately
    # have less causal context, but a list that hit its cap means real candidates
    # were cut off.
    exhausted = (
        (not ranked_list_is_complete)
        and bool(
            (
                (selected < target_col.squeeze(-1))
                & (ranked_bucket_counts >= max_ranked)
            ).any()
        )
    )
    if exhausted:
        raise RuntimeError(
            "the ranked bucket list ran out before the token budget was reached; "
            f"raise max_probe_buckets above {max_ranked}"
        )

    counts = keep.sum(dim=-1)
    max_kept_rank = torch.where(keep, rank.expand_as(keep), torch.full_like(lengths, -1)).amax(dim=-1)
    width = int(counts.max().item())
    if max_ranges is not None:
        if width > max_ranges:
            raise RuntimeError(
                f"selected buckets need {width} ranges, capacity={max_ranges}"
            )
        width = int(max_ranges)

    starts_all = torch.gather(
        bucket_indptr[:, :-1].unsqueeze(0).expand(q_count, -1, -1).to(torch.int64),
        2,
        ranked_bucket_ids.clamp_min(0).to(torch.int64),
    )
    # A dropped bucket shares the destination of the previous kept one, so it
    # must not take part in the scatter.  Send every dropped entry to a slot
    # past the end and cut it off afterwards.
    destination = torch.where(
        keep,
        (keep.cumsum(dim=-1) - 1).clamp_min(0),
        torch.full_like(lengths, width),
    )
    starts = torch.zeros(
        q_count, num_kv_heads, width + 1, dtype=torch.int32, device=lengths.device
    )
    out_lengths = torch.zeros_like(starts)
    starts.scatter_(2, destination, starts_all.to(torch.int32))
    out_lengths.scatter_(2, destination, kept.to(torch.int32))
    starts = starts[:, :, :width]
    out_lengths = out_lengths[:, :, :width]
    return PackedRanges(
        starts=starts.contiguous(),
        lengths=out_lengths.contiguous(),
        counts=counts.to(torch.int32).contiguous(),
        selected_tokens=selected,
        max_kept_rank=max_kept_rank.to(torch.int32),
    )


if triton is not None:

    @triton.jit
    def _topk_index_attention_kernel(
        out_ptr,
        lse_ptr,
        q_ptr,
        packed_kv_ptr,
        indices_ptr,
        sm_scale,
        softcap,
        num_selected,
        stride_out0, stride_out1, stride_out2,
        stride_lse0, stride_lse1,
        stride_q0, stride_q1, stride_q2,
        stride_kv0, stride_kv1, stride_kv2, stride_kv3,
        stride_i0, stride_i1, stride_i2,
        Q_PER_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        HEAD_DIM_PADDED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_SOFTCAP: tl.constexpr,
        WRITE_LSE: tl.constexpr,
    ):
        # One program per (query, KV head).  All heads of the KV group share
        # the selected token list, so each gathered K/V block serves the whole
        # group.  The row axis of the tile is the group's query heads.
        query_idx = tl.program_id(0)
        kv_head = tl.program_id(1)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM_PADDED)
        d_valid = offs_d < HEAD_DIM
        q_head = kv_head * Q_PER_KV + offs_m
        q_valid = offs_m < Q_PER_KV

        q = tl.load(
            q_ptr
            + query_idx.to(tl.int64) * stride_q0
            + q_head[:, None] * stride_q1
            + offs_d[None, :] * stride_q2,
            mask=q_valid[:, None] & d_valid[None, :],
            other=0.0,
        )

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], tl.float32)

        for n_start in range(0, num_selected, BLOCK_N):
            n = n_start + offs_n
            n_ok = n < num_selected
            idx = tl.load(
                indices_ptr
                + query_idx.to(tl.int64) * stride_i0
                + kv_head * stride_i1
                + n * stride_i2,
                mask=n_ok,
                other=-1,
            )
            # Queries whose causal pool slice is smaller than the budget pad
            # their index list with -1; those columns contribute nothing.
            token_valid = n_ok & (idx >= 0)
            safe_idx = tl.where(token_valid, idx, 0).to(tl.int64)
            k = tl.load(
                packed_kv_ptr
                + 0 * stride_kv0
                + safe_idx[None, :] * stride_kv1
                + kv_head * stride_kv2
                + offs_d[:, None] * stride_kv3,
                mask=d_valid[:, None] & token_valid[None, :],
                other=0.0,
            )
            v = tl.load(
                packed_kv_ptr
                + 1 * stride_kv0
                + safe_idx[:, None] * stride_kv1
                + kv_head * stride_kv2
                + offs_d[None, :] * stride_kv3,
                mask=token_valid[:, None] & d_valid[None, :],
                other=0.0,
            )
            scores = sm_scale * tl.dot(q, k)
            if USE_SOFTCAP:
                scores = _softcap(scores, softcap)
            valid = q_valid[:, None] & token_valid[None, :]
            m_i, l_i, acc = _softmax_update(scores, valid, m_i, l_i, acc, v)

        denom = tl.where(l_i > 0, l_i, 1.0)
        result = tl.where(l_i[:, None] > 0, acc / denom[:, None], 0.0)
        tl.store(
            out_ptr
            + query_idx.to(tl.int64) * stride_out0
            + q_head[:, None] * stride_out1
            + offs_d[None, :] * stride_out2,
            result,
            mask=q_valid[:, None] & d_valid[None, :],
        )
        if WRITE_LSE:
            lse = tl.where(l_i > 0, m_i + tl.log(l_i), -float("inf"))
            tl.store(
                lse_ptr + query_idx.to(tl.int64) * stride_lse0
                + q_head * stride_lse1,
                lse,
                mask=q_valid,
            )


def topk_index_attention(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    softcap: float | None,
    return_logsumexp: bool = False,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> PagedSparseAttentionResult:
    """Attend over per-query selected rows of the bucket-packed K/V array.

    Unlike :func:`packed_range_attention` there are no contiguous runs: every
    loaded column is an individually selected token.

    ``indices`` is ``[Q, Hkv, K]`` with row indices into the second axis of
    ``packed_kv``.  Entries of -1 are padding and are skipped.  All query heads
    of a KV group share the same list.
    """

    _require_triton()
    _validate_hopper(q.device)
    if q.ndim != 3:
        raise ValueError("q must have shape [Q,Hq,D]")
    if packed_kv.ndim != 4 or packed_kv.shape[0] != 2:
        raise ValueError("packed_kv must have shape [2,N,Hkv,D]")
    q_count, num_q_heads, head_dim = map(int, q.shape)
    num_kv_heads = int(packed_kv.shape[2])
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if indices.shape[:2] != (q_count, num_kv_heads):
        raise ValueError("indices must have shape [Q,Hkv,K]")
    if block_n not in {16, 32, 64}:
        raise ValueError("block_n must be 16, 32, or 64")
    q = q.contiguous()
    indices = indices.to(torch.int32).contiguous()
    output = torch.empty_like(q)
    lse = (
        torch.empty((q_count, num_q_heads), dtype=torch.float32, device=q.device)
        if return_logsumexp
        else torch.empty(1, dtype=torch.float32, device=q.device)
    )
    q_per_kv = num_q_heads // num_kv_heads
    block_m = max(16, _next_power_of_two(q_per_kv))
    grid = (q_count, num_kv_heads)
    _topk_index_attention_kernel[grid](
        output,
        lse,
        q,
        packed_kv,
        indices,
        float(sm_scale),
        float(softcap) if softcap is not None else 0.0,
        int(indices.shape[2]),
        *output.stride(),
        *( lse.stride() if return_logsumexp else (0, 0) ),
        *q.stride(),
        *packed_kv.stride(),
        *indices.stride(),
        Q_PER_KV=q_per_kv,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=_next_power_of_two(head_dim),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        USE_SOFTCAP=softcap is not None,
        WRITE_LSE=return_logsumexp,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return PagedSparseAttentionResult(
        output=output, logsumexp=lse if return_logsumexp else None
    )


def reference_topk_index_attention(
    q: torch.Tensor,
    packed_kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    sm_scale: float,
    softcap: float | None,
    return_logsumexp: bool = False,
) -> PagedSparseAttentionResult:
    """Torch mirror of ``topk_index_attention`` for tests and CPU checks."""

    q_count, num_q_heads, head_dim = map(int, q.shape)
    num_kv_heads = int(packed_kv.shape[2])
    q_per_kv = num_q_heads // num_kv_heads
    output = torch.zeros_like(q)
    lse = (
        torch.full((q_count, num_q_heads), -float("inf"), dtype=torch.float32)
        if return_logsumexp
        else None
    )
    keys = packed_kv[0].float()
    values = packed_kv[1].float()
    for query_idx in range(q_count):
        for kv_head in range(num_kv_heads):
            idx = indices[query_idx, kv_head].to(torch.int64)
            valid = idx >= 0
            rows = idx[valid]
            if rows.numel() == 0:
                continue
            k = keys[rows, kv_head]
            v = values[rows, kv_head]
            for member in range(q_per_kv):
                head = kv_head * q_per_kv + member
                scores = (q[query_idx, head].float() @ k.T) * sm_scale
                if softcap is not None:
                    scores = softcap * torch.tanh(scores / softcap)
                weights = torch.softmax(scores, dim=-1)
                output[query_idx, head] = (weights @ v).to(q.dtype)
                if lse is not None:
                    lse[query_idx, head] = torch.logsumexp(scores, dim=-1)
    return PagedSparseAttentionResult(output=output, logsumexp=lse)
