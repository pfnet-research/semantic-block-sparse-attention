"""The decode step of one sparse layer in a few kernels (SEMBLO_FUSED_DECODE=3).

Decode is bound by kernel launches rather than by the work itself (a small
codebook read, a threshold search and a gather-attention), so the arena decode
step is done as

  0. _suffix_write_kernel     the token's K/V into the suffix store
  1. _trivial_router_kernel   the routing query (trivial_router.py)
  2. _decode_select_kernel    scores, GQA sum, size penalty, bisection for the
                              threshold, and the selected clusters as CSR ranges
  3. _decode_attend_kernel    attention over the selected prompt rows, read
                              through the CSR, with the suffix tiles as extra
                              splits of the same grid
  4. _decode_merge_kernel     one LSE merge over every split, written directly
                              into the op's output

Everything runs on the device with no host reads and no allocations after the
first step (the workspace is cached per layer and shape), so vLLM's FULL decode
CUDA graph can capture it.  The selection rule and the attention match
fused_select.py, fused_body.py and paged_attention.py.
"""
from __future__ import annotations

import os

import torch

try:  # pragma: no cover
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _suffix_write_kernel(
        kv_ptr,            # [2, NBLK, BS, Hkv, D] suffix store
        table_ptr,         # [1, TABLE] int32 block table (logical block -> physical)
        pos_ptr,           # [1] int32/int64 the token's position
        k_ptr, v_ptr,      # [Hkv, D]
        stride_kv0, stride_kv1, stride_kv2, stride_kv3,
        stride_k0,
        HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    ):
        pos = tl.load(pos_ptr).to(tl.int64)
        block = pos // BS
        physical = tl.load(table_ptr + block).to(tl.int64)
        offset = pos - block * BS
        offs_h = tl.arange(0, HKV)
        offs_d = tl.arange(0, D)
        src = offs_h[:, None] * stride_k0 + offs_d[None, :]
        k = tl.load(k_ptr + src)
        v = tl.load(v_ptr + src)
        dst = (physical * stride_kv1 + offset * stride_kv2
               + offs_h[:, None] * stride_kv3 + offs_d[None, :])
        tl.store(kv_ptr + dst, k.to(kv_ptr.dtype.element_ty))
        tl.store(kv_ptr + stride_kv0 + dst, v.to(kv_ptr.dtype.element_ty))

    @triton.jit
    def _suffix_kpre_write_kernel(
        kpre_ptr,          # [NBLK, BS, Hkv, D] pre-RoPE keys of generated tokens (for the suffix flush)
        table_ptr, pos_ptr, k_ptr,
        stride_p1, stride_p2, stride_p3,
        stride_k0,
        HKV: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
    ):
        pos = tl.load(pos_ptr).to(tl.int64)
        block = pos // BS
        physical = tl.load(table_ptr + block).to(tl.int64)
        offset = pos - block * BS
        offs_h = tl.arange(0, HKV)
        offs_d = tl.arange(0, D)
        k = tl.load(k_ptr + offs_h[:, None] * stride_k0 + offs_d[None, :])
        dst = physical * stride_p1 + offset * stride_p2 + offs_h[:, None] * stride_p3 + offs_d[None, :]
        tl.store(kpre_ptr + dst, k.to(kpre_ptr.dtype.element_ty))

    @triton.jit
    def _decode_select_kernel(
        rstart_ptr, rlen_ptr, rcum_ptr,   # [Hkv, C] int32 out
        rcount_ptr, rtokens_ptr,          # [Hkv] int32 out
        prob_ws_ptr,                      # [Hkv, C] fp32 scratch, one row per program
        q_ptr,            # [Hkv, T, 2, DF] bf16 routing vectors
        cb_ptr,           # [Hkv, 2, C, DF] bf16
        sizes_ptr,        # [Hkv, C] int32
        indptr_ptr,       # [Hkv, C + 1] int32
        inv_temperature, beta, target_tokens,
        sink_tokens,      # > 0: cluster 0 is the attention sink (state._build_flat), always read
        stride_q0, stride_q1, stride_q2, stride_q3,
        stride_c0, stride_c1, stride_c2, stride_c3,
        stride_z0, stride_z1,
        stride_i0, stride_i1,
        stride_r0, stride_r1,
        T: tl.constexpr,          # scoring components per KV head (G x R)
        ROWS: tl.constexpr,       # T padded to the dot's minimum of 16
        C: tl.constexpr,
        DF: tl.constexpr,
        TILE_C: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        """One program per KV head: the whole selection of one decode query.

        Scores match fused_select._flat_scores_qtiled_kernel with a uniform
        mixture: per component t, s_t(c) = q_t . cb_c / temperature,
        normalised by log-sum-exp over c, and the cluster probability is the
        sum over the T components.  Selection matches
        _streaming_select_kernel: the score is log-probability minus
        beta log(1 + size), the threshold is the largest theta whose
        above-set still reaches the budget (found by bisection), and every
        live cluster at or above it is selected.  The output matches
        _union_compact_kernel's CSR: packed-K/V start, causal length and the
        exclusive token prefix of each selected cluster, in cluster order.

        With a sink (``sink_tokens`` > 0) the rule matches fused_prefill's
        _hold_out_sink and _force_sink_bit: the threshold search sees cluster
        0 with size zero and the budget less the sink, and cluster 0 is then
        selected with its real size.
        """
        kv_head = tl.program_id(0)
        offs_r = tl.arange(0, ROWS)
        offs_d = tl.arange(0, DF)
        offs_c = tl.arange(0, TILE_C)
        row_live = offs_r < T
        t_idx = tl.where(row_live, offs_r, 0)
        qbase = kv_head * stride_q0 + t_idx * stride_q1
        q0 = tl.load(q_ptr + qbase[:, None] + 0 * stride_q2 + offs_d[None, :] * stride_q3,
                     mask=row_live[:, None], other=0.0)
        q1 = tl.load(q_ptr + qbase[:, None] + 1 * stride_q2 + offs_d[None, :] * stride_q3,
                     mask=row_live[:, None], other=0.0)
        # pass 1: the per-component log-sum-exp over every cluster
        m_r = tl.full([ROWS], -float("inf"), tl.float32)
        l_r = tl.zeros([ROWS], tl.float32)
        for c0 in tl.range(0, C, TILE_C):
            cols = c0 + offs_c
            cb0 = tl.load(cb_ptr + kv_head * stride_c0 + 0 * stride_c1
                          + cols[:, None] * stride_c2 + offs_d[None, :] * stride_c3)
            cb1 = tl.load(cb_ptr + kv_head * stride_c0 + 1 * stride_c1
                          + cols[:, None] * stride_c2 + offs_d[None, :] * stride_c3)
            sc = (tl.dot(q0, tl.trans(cb0)) + tl.dot(q1, tl.trans(cb1))) * inv_temperature
            m_new = tl.maximum(m_r, tl.max(sc, axis=1))
            l_r = l_r * tl.exp(m_r - m_new) + tl.sum(tl.exp(sc - m_new[:, None]), axis=1)
            m_r = m_new
        lse = m_r + tl.log(l_r)
        # pass 2: the cluster probability, summed over the components, stored
        # in this program's scratch row so the threshold search can load all [C]
        for c0 in tl.range(0, C, TILE_C):
            cols = c0 + offs_c
            cb0 = tl.load(cb_ptr + kv_head * stride_c0 + 0 * stride_c1
                          + cols[:, None] * stride_c2 + offs_d[None, :] * stride_c3)
            cb1 = tl.load(cb_ptr + kv_head * stride_c0 + 1 * stride_c1
                          + cols[:, None] * stride_c2 + offs_d[None, :] * stride_c3)
            sc = (tl.dot(q0, tl.trans(cb0)) + tl.dot(q1, tl.trans(cb1))) * inv_temperature
            w = tl.exp(sc - lse[:, None])
            w = tl.where(row_live[:, None], w, 0.0)
            tl.store(prob_ws_ptr + kv_head * stride_r0 + cols * stride_r1, tl.sum(w, axis=0))
        tl.debug_barrier()
        # the threshold search on the resident [C] vectors
        offs_all = tl.arange(0, C)
        prob = tl.load(prob_ws_ptr + kv_head * stride_r0 + offs_all * stride_r1)
        real_sizes = tl.load(sizes_ptr + kv_head * stride_z0 + offs_all * stride_z1).to(tl.float32)
        fzero = tl.zeros([1], tl.float32)
        is_sink = (offs_all == 0) & (sink_tokens > 0)
        sizes = tl.where(is_sink, fzero, real_sizes)          # what the threshold search sees
        budget = tl.maximum(target_tokens - sink_tokens, 0).to(tl.float32)
        live = sizes > 0.0
        neg_inf = tl.full([1], -3.4e38, tl.float32)
        big = tl.full([1], 3.4e38, tl.float32)
        tiny = tl.full([1], 1e-38, tl.float32)
        fone = tl.full([1], 1.0, tl.float32)
        score = tl.log(tl.maximum(prob, tiny))
        if beta != 0.0:
            score = score - beta * tl.log(fone + sizes)
        score = tl.where(live, score, neg_inf)
        hi = tl.max(score) + 1e-6
        lo = tl.min(tl.where(live, score, big)) - 1e-6
        lo = tl.minimum(lo, hi - 1e-5)
        all_fit = tl.sum(tl.where(live, sizes, fzero)) < budget
        for _it in tl.static_range(ITERS):
            mid = (lo + hi) * 0.5
            above = tl.sum(tl.where(score >= mid, sizes, fzero))
            reach = above >= budget
            lo = tl.where(reach, mid, lo)
            hi = tl.where(reach, hi, mid)
        theta = tl.where(all_fit, -3.4e38, lo)
        sel = (live & (score >= theta)) | (is_sink & (real_sizes > 0.0))
        # the CSR emission in cluster order, with the real sizes
        sel_i = sel.to(tl.int32)
        sel_len = tl.where(sel, real_sizes, fzero).to(tl.int32)
        pos = tl.cumsum(sel_i, axis=0) - sel_i
        cum = tl.cumsum(sel_len, axis=0) - sel_len
        bstart = tl.load(indptr_ptr + kv_head * stride_i0 + offs_all * stride_i1, mask=sel, other=0)
        out_base = kv_head * stride_r0
        tl.store(rstart_ptr + out_base + pos * stride_r1, bstart, mask=sel)
        tl.store(rlen_ptr + out_base + pos * stride_r1, sel_len, mask=sel)
        tl.store(rcum_ptr + out_base + pos * stride_r1, cum, mask=sel)
        tl.store(rcount_ptr + kv_head, tl.sum(sel_i))
        tl.store(rtokens_ptr + kv_head, tl.sum(sel_len))

    @triton.jit
    def _decode_attend_kernel(
        pacc_ptr, pm_ptr, pl_ptr,     # [Hq, S, D] fp32, [Hq, S] fp32, [Hq, S] fp32
        q_ptr,                        # [Hq, D] bf16
        kv_ptr,                       # [2, W, Hkv, D] packed prompt K/V (cluster order)
        pos_ptr,                      # [Hkv, W] int32 packed row -> position
        rstart_ptr, rlen_ptr, rcum_ptr, rcount_ptr, rtokens_ptr,   # the CSR
        suffix_kv_ptr,                # [2, NBLK, BS, Hkv, D] suffix store
        table_ptr,                    # [1, TABLE] int32
        qpos_ptr,                     # [1] int32 the query's position
        plen_ptr,                     # [1] int32 the prompt length
        sm_scale, umax,
        stride_q0, stride_q1,
        stride_kv0, stride_kv1, stride_kv2, stride_kv3,
        stride_p0, stride_p1,
        stride_r0, stride_r1,
        stride_s0, stride_s1, stride_s2, stride_s3, stride_s4,
        stride_pa0, stride_pa1, stride_pa2,
        stride_pm0, stride_pm1,
        G: tl.constexpr,              # query heads per KV head
        BLOCK_M: tl.constexpr,        # G padded to 16
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SPLITS: tl.constexpr,     # prompt stripes
        SUFFIX_SPLITS: tl.constexpr,  # suffix stripes
        SEARCH_STEPS: tl.constexpr,
        TILE: tl.constexpr,           # suffix tokens per tile (16)
        MAX_TILES: tl.constexpr,      # tiles the reservation holds
        BS: tl.constexpr,             # suffix block size (256)
    ):
        """One program per (KV head, split).

        Prompt splits read the selected clusters' packed rows through the CSR,
        with a binary search over the token prefix per key.  Suffix splits read
        the generated tokens through the block table.  Both write an
        unnormalised (acc, m, l) partial, and the merge kernel combines all
        splits of a head.  At decode every prompt key precedes the query, so
        the causal mask is only needed on the suffix."""
        kv_head = tl.program_id(0)
        split = tl.program_id(1)
        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        head = kv_head * G + offs_m
        m_valid = offs_m < G
        q = tl.load(q_ptr + head[:, None] * stride_q0 + offs_d[None, :] * stride_q1,
                    mask=m_valid[:, None], other=0.0)
        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        q_pos = tl.load(qpos_ptr)
        r_base = kv_head * stride_r0
        if split < NUM_SPLITS:
            offs_n = tl.arange(0, BLOCK_N)
            count = tl.minimum(tl.load(rtokens_ptr + kv_head), umax)
            rcount = tl.load(rcount_ptr + kv_head)
            per_split = tl.cdiv(tl.cdiv(count, BLOCK_N), NUM_SPLITS) * BLOCK_N
            lo_n = split * per_split
            hi_n = tl.minimum(lo_n + per_split, count)
            for n_start in range(lo_n, hi_n, BLOCK_N):
                o = n_start + offs_n
                n_ok = o < hi_n
                # the cluster of slot o: the largest r with rcum[r] <= o
                lo_r = tl.zeros([BLOCK_N], tl.int32)
                hi_r = tl.zeros([BLOCK_N], tl.int32) + rcount
                for _ in tl.static_range(SEARCH_STEPS):
                    mid = (lo_r + hi_r) // 2
                    probe = n_ok & (mid < rcount)
                    cm = tl.load(rcum_ptr + r_base + mid * stride_r1, mask=probe, other=0)
                    right = probe & (cm <= o)
                    lo_r = tl.where(right, mid + 1, lo_r)
                    hi_r = tl.where(right, hi_r, mid)
                r = tl.maximum(lo_r - 1, 0)
                start = tl.load(rstart_ptr + r_base + r * stride_r1, mask=n_ok, other=0)
                cum = tl.load(rcum_ptr + r_base + r * stride_r1, mask=n_ok, other=0)
                idx = (start + (o - cum)).to(tl.int64)
                k = tl.load(kv_ptr + idx[None, :] * stride_kv1 + kv_head * stride_kv2
                            + offs_d[:, None] * stride_kv3,
                            mask=n_ok[None, :], other=0.0)
                key_pos = tl.load(pos_ptr + kv_head * stride_p0 + idx * stride_p1,
                                  mask=n_ok, other=2147483647)
                scores = tl.dot(q, k).to(tl.float32) * sm_scale
                keep = n_ok[None, :] & (key_pos[None, :] <= q_pos) & m_valid[:, None]
                scores = tl.where(keep, scores, -float("inf"))
                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
                alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))
                p = tl.exp(scores - m_safe[:, None])
                p = tl.where(keep, p, 0.0)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]
                v = tl.load(kv_ptr + stride_kv0 + idx[:, None] * stride_kv1 + kv_head * stride_kv2
                            + offs_d[None, :] * stride_kv3,
                            mask=n_ok[:, None], other=0.0)
                acc += tl.dot(p.to(v.dtype), v).to(tl.float32)
                m_i = m_new
        else:
            offs_t = tl.arange(0, TILE)
            plen = tl.load(plen_ptr)
            live_tiles = tl.minimum(tl.maximum((q_pos - plen) // TILE + 1, 1), MAX_TILES)
            s = split - NUM_SPLITS
            per = tl.cdiv(live_tiles, SUFFIX_SPLITS)
            t_lo = s * per
            t_hi = tl.minimum(t_lo + per, live_tiles)
            for t in range(t_lo, t_hi):
                pos = (plen + t * TILE + offs_t).to(tl.int64)
                ok = pos <= q_pos
                block = pos // BS
                physical = tl.load(table_ptr + block, mask=ok, other=0).to(tl.int64)
                off = pos - block * BS
                src = physical * stride_s1 + off * stride_s2 + kv_head * stride_s3
                k = tl.load(suffix_kv_ptr + src[None, :] + offs_d[:, None] * stride_s4,
                            mask=ok[None, :], other=0.0)
                scores = tl.dot(q, k).to(tl.float32) * sm_scale
                keep = ok[None, :] & m_valid[:, None]
                scores = tl.where(keep, scores, -float("inf"))
                m_new = tl.maximum(m_i, tl.max(scores, axis=1))
                m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
                alpha = tl.where(m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe))
                p = tl.exp(scores - m_safe[:, None])
                p = tl.where(keep, p, 0.0)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]
                v = tl.load(suffix_kv_ptr + stride_s0 + src[:, None] + offs_d[None, :] * stride_s4,
                            mask=ok[:, None], other=0.0)
                acc += tl.dot(p.to(v.dtype), v).to(tl.float32)
                m_i = m_new
        tl.store(pacc_ptr + head[:, None] * stride_pa0 + split * stride_pa1 + offs_d[None, :] * stride_pa2,
                 acc, mask=m_valid[:, None])
        pm = head * stride_pm0 + split * stride_pm1
        tl.store(pm_ptr + pm, m_i, mask=m_valid)
        tl.store(pl_ptr + pm, l_i, mask=m_valid)

    @triton.jit
    def _decode_merge_kernel(
        out_ptr,                     # [Hq, D] the op's output
        pacc_ptr, pm_ptr, pl_ptr,
        stride_o0, stride_o1,
        stride_pa0, stride_pa1, stride_pa2,
        stride_pm0, stride_pm1,
        D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr,
    ):
        head = tl.program_id(0)
        offs_s = tl.arange(0, S_PAD)
        s_ok = offs_s < S
        offs_d = tl.arange(0, D)
        pm = head * stride_pm0 + offs_s * stride_pm1
        m_s = tl.load(pm_ptr + pm, mask=s_ok, other=-float("inf"))
        l_s = tl.load(pl_ptr + pm, mask=s_ok, other=0.0)
        m_all = tl.max(m_s)
        m_safe = tl.where(m_all == -float("inf"), 0.0, m_all)
        scale = tl.where(m_s == -float("inf"), 0.0, tl.exp(m_s - m_safe))
        acc = tl.load(pacc_ptr + head * stride_pa0 + offs_s[:, None] * stride_pa1 + offs_d[None, :] * stride_pa2,
                      mask=s_ok[:, None], other=0.0)
        num = tl.sum(acc * scale[:, None], axis=0)
        den = tl.sum(l_s * scale)
        den_safe = tl.where(den == 0.0, 1.0, den)
        tl.store(out_ptr + head * stride_o0 + offs_d * stride_o1,
                 (num / den_safe).to(out_ptr.dtype.element_ty))


def _pow2_at_least(n: int, floor: int = 16) -> int:
    return max(floor, 1 << (int(n) - 1).bit_length())


class FusedDecodeWorkspace:
    """The step's buffers, allocated once per layer and shape so CUDA graph replay sees fixed addresses."""

    def __init__(self, *, num_q_heads: int, num_kv_heads: int, head_dim: int, num_clusters: int,
                 index_dim: int, num_splits: int, suffix_splits: int, device: torch.device) -> None:
        self.shape_key = (num_q_heads, num_kv_heads, head_dim, num_clusters, index_dim, num_splits, suffix_splits)
        hq, hkv, c = num_q_heads, num_kv_heads, num_clusters
        i32 = torch.int32
        self.rstart = torch.zeros(hkv, c, dtype=i32, device=device)
        self.rlen = torch.zeros(hkv, c, dtype=i32, device=device)
        self.rcum = torch.zeros(hkv, c, dtype=i32, device=device)
        self.rcount = torch.zeros(hkv, dtype=i32, device=device)
        self.rtokens = torch.zeros(hkv, dtype=i32, device=device)
        self.prob_ws = torch.zeros(hkv, c, dtype=torch.float32, device=device)
        self.vectors = torch.zeros(hq, index_dim, dtype=torch.bfloat16, device=device)
        splits = num_splits + suffix_splits
        self.pacc = torch.zeros(hq, splits, head_dim, dtype=torch.float32, device=device)
        self.pm = torch.zeros(hq, splits, dtype=torch.float32, device=device)
        self.pl = torch.zeros(hq, splits, dtype=torch.float32, device=device)
        self.num_splits = num_splits
        self.suffix_splits = suffix_splits


def _splits_for(num_q_heads: int, requested: int) -> int:
    """A power of two so the merge's tl.arange is legal; the layer asks for twice its packed split count."""
    return _pow2_at_least(max(int(requested), 1), floor=1)


def fused_decode_step(
    layer,                    # the Semblo*Attention module (source layer)
    arena,                    # DecodeArena
    *,
    q_post: torch.Tensor,     # [1, Hq * D] (post-rope) or [1, Hq, D]
    q_pre: torch.Tensor,      # [1, Hq, D] pre-rope (the router's input)
    positions: torch.Tensor,  # [1]
    k_post: torch.Tensor | None,   # [1, Hkv, D] (physical layers)
    v_states: torch.Tensor | None,
    k_pre: torch.Tensor | None = None,   # [1, Hkv, D] pre-rope key, kept for the suffix flush
    output: torch.Tensor,     # [1, Hq * D] the op's output buffer
    temperature: float,
    beta: float,
    target_tokens: int,
    sm_scale: float,
    num_splits: int,
    iters: int = 16,
) -> None:
    """The whole decode step of one layer, four launches after the router."""
    # Same sink rule as fused_prefill.fused_decode_attention.
    from semblo.vllm_plugin.fused_prefill import _fitted_sink_tokens
    sink = _fitted_sink_tokens()
    if triton is None:
        raise RuntimeError("the fused decode step needs triton")
    index = arena.index
    hq = int(layer.num_heads)
    hkv = int(layer.num_kv_heads)
    d = int(layer.head_dim)
    g = hq // hkv
    c = int(arena.num_clusters)
    codebooks = arena.codebooks                      # [Hkv, 2, C, DF]
    df = int(codebooks.shape[-1])
    index_dim = 2 * df
    if index_dim != d:
        raise RuntimeError(f"the fused decode step routes with the query itself (index_dim {index_dim} must equal head_dim {d})")
    splits = _splits_for(hq, num_splits)
    suffix_splits = int(os.environ.get("SEMBLO_DECODE_SUFFIX_SPLITS", "8"))
    suffix_splits = _pow2_at_least(suffix_splits, floor=1)
    ws = getattr(arena, "fused_decode_ws", None)
    key = (hq, hkv, d, c, index_dim, splits, suffix_splits)
    if ws is None or ws.shape_key != key:
        ws = FusedDecodeWorkspace(num_q_heads=hq, num_kv_heads=hkv, head_dim=d, num_clusters=c,
                                  index_dim=index_dim, num_splits=splits, suffix_splits=suffix_splits,
                                  device=q_pre.device)
        arena.fused_decode_ws = ws
    device = q_pre.device
    suffix_kv = arena.suffix_kv                      # [2, NBLK, BS, Hkv, D]
    table = arena.suffix_table                       # [1, TABLE]
    bs = int(arena.suffix_block)
    pos32 = positions[:1]
    # 0. the token's K/V into the suffix store
    if k_post is not None and v_states is not None:
        kk = k_post.view(-1, hkv, d)[0]
        vv = v_states.view(-1, hkv, d)[0]
        if not kk.is_contiguous():
            kk = kk.contiguous()
        if not vv.is_contiguous():
            vv = vv.contiguous()
        _suffix_write_kernel[(1,)](
            suffix_kv, table, pos32, kk, vv,
            suffix_kv.stride(0), suffix_kv.stride(1), suffix_kv.stride(2), suffix_kv.stride(3),
            kk.stride(0),
            HKV=hkv, D=d, BS=bs, num_warps=1)
        kpre_store = getattr(arena, "suffix_kpre", None)
        if kpre_store is not None and k_pre is not None:
            kp = k_pre.view(-1, hkv, d)[0]
            if not kp.is_contiguous():
                kp = kp.contiguous()
            _suffix_kpre_write_kernel[(1,)](
                kpre_store, table, pos32, kp,
                kpre_store.stride(0), kpre_store.stride(1), kpre_store.stride(2),
                kp.stride(0),
                HKV=hkv, D=d, BS=bs, num_warps=1)
    # 1. the routing query: the whole query vector, normalised
    from semblo.vllm_plugin.trivial_router import trivial_router_vectors
    qp = q_pre.reshape(1, hq, d)
    trivial_router_vectors(qp, 1, out=ws.vectors.view(1, hq, d))
    # 2. selection
    q5 = ws.vectors.view(hkv, g, 2, df)
    # The select kernel keeps a codebook tile (two halves of TILE_C x DF bf16)
    # in shared memory.  With DF=128 a 256-cluster tile and three pipeline
    # stages exceed the H200's 232 KB, so it uses one stage and a smaller tile.
    tile_c = min(c, int(os.environ.get("SEMBLO_DECODE_SELECT_TILE_C", "256" if df <= 64 else "128")))
    rows = _pow2_at_least(g, floor=16)
    _decode_select_kernel[(hkv,)](
        ws.rstart, ws.rlen, ws.rcum, ws.rcount, ws.rtokens, ws.prob_ws,
        q5, codebooks, arena.sizes, arena.indptr,
        1.0 / float(temperature), float(beta), int(target_tokens), int(sink),
        q5.stride(0), q5.stride(1), q5.stride(2), q5.stride(3),
        codebooks.stride(0), codebooks.stride(1), codebooks.stride(2), codebooks.stride(3),
        arena.sizes.stride(0), arena.sizes.stride(1),
        arena.indptr.stride(0), arena.indptr.stride(1),
        ws.rstart.stride(0), ws.rstart.stride(1),
        T=g, ROWS=rows, C=c, DF=df, TILE_C=tile_c, ITERS=int(iters),
        num_warps=4, num_stages=1)
    # 3. attention over the selected prompt rows and the suffix
    q = q_post.view(hq, d)
    if not q.is_contiguous():
        q = q.contiguous()
    packed = arena.packed_kv                          # [2, W, Hkv, D]
    posting = arena.posting                          # [Hkv, W]
    umax = int(posting.shape[1])
    tile = int(layer.runtime_config.triton_token_tile)
    max_tiles = -(-int(arena.reserve) // tile)
    block_n = int(os.environ.get("SEMBLO_DECODE_BLOCK_N", "128"))
    _decode_attend_kernel[(hkv, splits + suffix_splits)](
        ws.pacc, ws.pm, ws.pl, q, packed, posting,
        ws.rstart, ws.rlen, ws.rcum, ws.rcount, ws.rtokens,
        suffix_kv, table, pos32, arena.prompt_length_t,
        float(sm_scale), umax,
        q.stride(0), q.stride(1),
        packed.stride(0), packed.stride(1), packed.stride(2), packed.stride(3),
        posting.stride(0), posting.stride(1),
        ws.rstart.stride(0), ws.rstart.stride(1),
        suffix_kv.stride(0), suffix_kv.stride(1), suffix_kv.stride(2), suffix_kv.stride(3), suffix_kv.stride(4),
        ws.pacc.stride(0), ws.pacc.stride(1), ws.pacc.stride(2),
        ws.pm.stride(0), ws.pm.stride(1),
        G=g, BLOCK_M=_pow2_at_least(g, 16), D=d, BLOCK_N=block_n,
        NUM_SPLITS=splits, SUFFIX_SPLITS=suffix_splits,
        SEARCH_STEPS=max(1, int(c).bit_length()), TILE=tile, MAX_TILES=max_tiles, BS=bs,
        num_warps=4, num_stages=2)
    # 4. the merge, straight into the op's output
    out = output.view(hq, d)
    _decode_merge_kernel[(hq,)](
        out, ws.pacc, ws.pm, ws.pl,
        out.stride(0), out.stride(1),
        ws.pacc.stride(0), ws.pacc.stride(1), ws.pacc.stride(2),
        ws.pm.stride(0), ws.pm.stride(1),
        D=d, S=splits + suffix_splits, S_PAD=_pow2_at_least(splits + suffix_splits, floor=2), num_warps=4)
