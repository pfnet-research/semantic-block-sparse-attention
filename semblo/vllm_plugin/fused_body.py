"""Union block-sparse attention body (gather form), used for prefill.

- One tile serves G_Q queries (x Q_PER_KV group heads = BLOCK_M rows), so
  each K/V read is shared across them as in FlashAttention.  The tile reads
  the union of the selected tokens of its queries.
- Columns come from a flat slot-index list (the union of the packed slots of
  the clusters the block selected), so BLOCK_N tiles are dense.
- Causality is enforced per row with the key's position, gathered alongside K.

By default every row attends to the whole block union, so a query can attend
tokens selected by a neighbouring query in its block.  With STRICT
(``slot_rowmask``, or ``slot_bucket`` with ``query_sel_mask``) each row keeps
only keys whose cluster is in its own selection.  Traffic and tiling stay
union-shaped, but the attended set is exactly per-query, which is the
method's semantics.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _union_block_attention_kernel(
        out_ptr,        # [Q, Hq, D] bf16
        lse_ptr,        # [Q, Hq] fp32 (WRITE_LSE, non-split path)
        q_ptr,          # [Q, Hq, D] bf16
        kv_ptr,         # [2, L, Hkv, D] bf16 (or fp8e4m3 when FP8)
        kvscale_ptr,    # [2, L, Hkv] fp32 per-token dequant scales (FP8)
        pos_ptr,        # [Hkv, L] int32 (position of each packed slot)
        idx_ptr,        # [NB, Hkv, U] int32 (union slot list)
        count_ptr,      # [NB, Hkv] int32
        qpos_ptr,       # [Q] int32
        sbucket_ptr,    # [Hkv, L] int32 (STRICT: bucket id of a packed slot)
        qmask_ptr,      # [Q, Hkv, W] int32 (STRICT: per-query selected set)
        qmask_words,    # W = B/32 (runtime)
        srow_ptr,       # [NB, Hkv, U] int32 (ROWMASK: rows that selected the
                        # slot's bucket, one bit per row of the block)
        sm_scale,
        num_queries,
        stride_o0, stride_o1, stride_o2,
        stride_q0, stride_q1, stride_q2,
        stride_kv0, stride_kv1, stride_kv2, stride_kv3,
        stride_s0, stride_s1, stride_s2,
        stride_p0, stride_p1,
        stride_i0, stride_i1, stride_i2,
        stride_c0, stride_c1,
        stride_sb0, stride_sb1,
        partial_acc_ptr,  # [Q, Hq, S, D] fp32 (splits > 1)
        partial_m_ptr,    # [Q, Hq, S] fp32
        partial_l_ptr,    # [Q, Hq, S] fp32
        stride_pa0, stride_pa1, stride_pa2, stride_pa3,
        stride_pm0, stride_pm1, stride_pm2,
        G_Q: tl.constexpr,
        Q_PER_KV: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        FP8: tl.constexpr,
        WRITE_LSE: tl.constexpr,
        STRICT: tl.constexpr,
        ROWMASK: tl.constexpr,
    ):
        # The union size varies by an order of magnitude across (block, head)
        # and the (NB, H) grid is a single wave, so without splitting the
        # call is bound by its largest union.  NUM_SPLITS > 1 stripes the
        # token axis and a second kernel merges the partial softmaxes.
        block = tl.program_id(0)
        kv_head = tl.program_id(1)
        split = tl.program_id(2)

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)
        member = offs_m // Q_PER_KV
        row_query = block * G_Q + member
        head = kv_head * Q_PER_KV + offs_m % Q_PER_KV
        m_valid = (offs_m < G_Q * Q_PER_KV) & (row_query < num_queries)

        q = tl.load(
            q_ptr
            + row_query[:, None].to(tl.int64) * stride_q0
            + head[:, None] * stride_q1
            + offs_d[None, :] * stride_q2,
            mask=m_valid[:, None],
            other=0.0,
        )
        row_pos = tl.load(qpos_ptr + row_query, mask=m_valid, other=0)

        m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)

        count = tl.load(count_ptr + block * stride_c0 + kv_head * stride_c1)
        if NUM_SPLITS > 1:
            per_split = tl.cdiv(tl.cdiv(count, BLOCK_N), NUM_SPLITS) * BLOCK_N
            lo_n = split * per_split
            hi_n = tl.minimum(lo_n + per_split, count)
        else:
            lo_n = 0
            hi_n = count
        for n_start in range(lo_n, hi_n, BLOCK_N):
            n = n_start + offs_n
            n_ok = n < hi_n
            idx = tl.load(
                idx_ptr
                + block.to(tl.int64) * stride_i0
                + kv_head * stride_i1
                + n * stride_i2,
                mask=n_ok, other=0,
            ).to(tl.int64)
            k = tl.load(
                kv_ptr + 0 * stride_kv0 + idx[None, :] * stride_kv1
                + kv_head * stride_kv2 + offs_d[:, None] * stride_kv3,
                mask=n_ok[None, :], other=0.0,
            )
            key_pos = tl.load(
                pos_ptr + kv_head * stride_p0 + idx * stride_p1,
                mask=n_ok, other=2147483647,
            )
            if FP8:
                # k_true = k8 * scale[slot]; fold the scale into the scores
                kscale = tl.load(
                    kvscale_ptr + 0 * stride_s0 + idx * stride_s1
                    + kv_head * stride_s2,
                    mask=n_ok, other=0.0,
                )
                scores = tl.dot(q, k.to(q.dtype)).to(tl.float32)
                scores = scores * kscale[None, :] * sm_scale
            else:
                scores = tl.dot(q, k).to(tl.float32) * sm_scale
            keep = n_ok[None, :] & (key_pos[None, :] <= row_pos[:, None]) \
                & m_valid[:, None]
            if STRICT:
                # a row keeps a key only if the row's own selection contains
                # the key's cluster; masked keys never enter the softmax
                if ROWMASK:
                    # One contiguous BLOCK_N load of the rows that selected
                    # each slot's cluster, recorded by the union build.  The
                    # branch below re-derives it per (row, key) with a
                    # BLOCK_M x BLOCK_N scattered gather, which is much slower.
                    rw = tl.load(
                        srow_ptr + block.to(tl.int64) * stride_i0
                        + kv_head * stride_i1 + n * stride_i2,
                        mask=n_ok, other=0,
                    )
                    keep = keep & (
                        ((rw[None, :] >> member[:, None]) & 1) > 0)
                else:
                    bkt = tl.load(
                        sbucket_ptr + kv_head * stride_sb0 + idx * stride_sb1,
                        mask=n_ok, other=0,
                    )
                    w = tl.load(
                        qmask_ptr
                        + (row_query[:, None].to(tl.int64)
                           * tl.num_programs(1) + kv_head) * qmask_words
                        + (bkt >> 5)[None, :],
                        mask=m_valid[:, None] & n_ok[None, :], other=0,
                    )
                    keep = keep & (((w >> (bkt & 31)[None, :]) & 1) > 0)
            scores = tl.where(keep, scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            alpha = tl.where(
                m_i == -float("inf"), 0.0, tl.exp(m_i - m_safe)
            )
            p = tl.exp(scores - m_safe[:, None])
            p = tl.where(keep, p, 0.0)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]
            v = tl.load(
                kv_ptr + 1 * stride_kv0 + idx[:, None] * stride_kv1
                + kv_head * stride_kv2 + offs_d[None, :] * stride_kv3,
                mask=n_ok[:, None], other=0.0,
            )
            if FP8:
                vscale = tl.load(
                    kvscale_ptr + 1 * stride_s0 + idx * stride_s1
                    + kv_head * stride_s2,
                    mask=n_ok, other=0.0,
                )
                pv = (p * vscale[None, :]).to(q.dtype)
                acc += tl.dot(pv, v.to(q.dtype)).to(tl.float32)
            else:
                acc += tl.dot(p.to(v.dtype), v).to(tl.float32)
            m_i = m_new

        if NUM_SPLITS > 1:
            # unnormalised partials; the merge kernel rescales across splits
            tl.store(
                partial_acc_ptr
                + row_query[:, None].to(tl.int64) * stride_pa0
                + head[:, None] * stride_pa1
                + split * stride_pa2
                + offs_d[None, :] * stride_pa3,
                acc,
                mask=m_valid[:, None],
            )
            pm = (row_query.to(tl.int64) * stride_pm0 + head * stride_pm1
                  + split * stride_pm2)
            tl.store(partial_m_ptr + pm, m_i, mask=m_valid)
            tl.store(partial_l_ptr + pm, l_i, mask=m_valid)
        else:
            safe = tl.where(l_i == 0.0, 1.0, l_i)
            out = acc / safe[:, None]
            tl.store(
                out_ptr
                + row_query[:, None].to(tl.int64) * stride_o0
                + head[:, None] * stride_o1
                + offs_d[None, :] * stride_o2,
                out.to(out_ptr.dtype.element_ty),
                mask=m_valid[:, None],
            )
            if WRITE_LSE:
                finite = m_i != -float("inf")
                lse = tl.where(
                    finite, m_i + tl.log(tl.where(finite, l_i, 1.0)),
                    -float("inf"),
                )
                tl.store(
                    lse_ptr + row_query.to(tl.int64) * (tl.num_programs(1)
                    * Q_PER_KV) + head,
                    lse, mask=m_valid,
                )


    @triton.jit
    def _union_split_merge_kernel(
        out_ptr,          # [Q, Hq, D] bf16
        lse_ptr,          # [Q, Hq] fp32 (WRITE_LSE)
        partial_acc_ptr,  # [Q, Hq, S, D] fp32
        partial_m_ptr,    # [Q, Hq, S] fp32
        partial_l_ptr,    # [Q, Hq, S] fp32
        stride_o0, stride_o1, stride_o2,
        stride_pa0, stride_pa1, stride_pa2, stride_pa3,
        stride_pm0, stride_pm1, stride_pm2,
        HEAD_DIM: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        WRITE_LSE: tl.constexpr,
    ):
        query = tl.program_id(0).to(tl.int64)
        head = tl.program_id(1)
        offs_s = tl.arange(0, NUM_SPLITS)
        offs_d = tl.arange(0, HEAD_DIM)
        pm = query * stride_pm0 + head * stride_pm1 + offs_s * stride_pm2
        m_s = tl.load(partial_m_ptr + pm)
        l_s = tl.load(partial_l_ptr + pm)
        m_all = tl.max(m_s)
        m_safe = tl.where(m_all == -float("inf"), 0.0, m_all)
        scale = tl.where(m_s == -float("inf"), 0.0, tl.exp(m_s - m_safe))
        acc = tl.load(
            partial_acc_ptr + query * stride_pa0 + head * stride_pa1
            + offs_s[:, None] * stride_pa2 + offs_d[None, :] * stride_pa3,
        )
        num = tl.sum(acc * scale[:, None], axis=0)
        den = tl.sum(l_s * scale)
        den_safe = tl.where(den == 0.0, 1.0, den)
        tl.store(
            out_ptr + query * stride_o0 + head * stride_o1
            + offs_d * stride_o2,
            (num / den_safe).to(out_ptr.dtype.element_ty),
        )
        if WRITE_LSE:
            finite = m_all != -float("inf")
            lse = tl.where(
                finite,
                m_safe + tl.log(tl.where(den > 0.0, den, 1.0)),
                -float("inf"),
            )
            tl.store(lse_ptr + query * tl.num_programs(1) + head, lse)


def union_block_attention(
    q: torch.Tensor,          # [Q, Hq, D] bf16
    packed_kv: torch.Tensor,  # [2, L, Hkv, D] bf16
    positions: torch.Tensor,  # [Hkv, L] int32
    union_idx: torch.Tensor,  # [NB, Hkv, U] int32
    union_count: torch.Tensor,  # [NB, Hkv] int32
    query_pos: torch.Tensor,  # [Q] int32
    *,
    queries_per_block: int,
    sm_scale: float,
    block_n: int = 64,
    num_warps: int = 8,
    num_stages: int = 3,
    num_splits: int = 1,
    kv_scales: torch.Tensor | None = None,  # [2, L, Hkv] fp32 => fp8 path
    return_logsumexp: bool = False,
    slot_bucket: torch.Tensor | None = None,     # [Hkv, L] int32
    query_sel_mask: torch.Tensor | None = None,  # [Q, Hkv, B/32] int32
    slot_rowmask: torch.Tensor | None = None,    # [NB, Hkv, U] int32
):
    if triton is None:
        raise RuntimeError("union_block_attention needs triton")
    q_count, num_q_heads, head_dim = q.shape
    _, _, num_kv_heads, _ = packed_kv.shape
    q_per_kv = num_q_heads // num_kv_heads
    num_blocks = union_idx.shape[0]
    # tl.dot needs >= 16 rows; decode (1 query x 4 group heads) pads up
    block_m = max(16, queries_per_block * q_per_kv)
    out = torch.empty_like(q)
    lse = (torch.empty(q_count, num_q_heads, dtype=torch.float32,
                       device=q.device) if return_logsumexp else
           torch.empty(1, 1, dtype=torch.float32, device=q.device))
    if num_splits > 1:
        pacc = torch.empty(q_count, num_q_heads, num_splits, head_dim,
                           dtype=torch.float32, device=q.device)
        pm = torch.empty(q_count, num_q_heads, num_splits,
                         dtype=torch.float32, device=q.device)
        pl = torch.empty_like(pm)
    else:
        pacc = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=q.device)
        pm = torch.empty(1, 1, 1, dtype=torch.float32, device=q.device)
        pl = pm
    if kv_scales is None:
        kv_scales = torch.empty(1, 1, 1, dtype=torch.float32,
                                device=q.device)
    strict = query_sel_mask is not None or slot_rowmask is not None
    use_rowmask = slot_rowmask is not None
    if use_rowmask:
        assert slot_rowmask.shape == union_idx.shape, \
            (slot_rowmask.shape, union_idx.shape)
        assert slot_rowmask.dtype == torch.int32
        assert queries_per_block <= 32, \
            "the row bitmask holds one bit per row of a block"
        qmask_words = 1
        srow = slot_rowmask
        query_sel_mask = torch.empty(1, 1, 1, dtype=torch.int32,
                                     device=q.device)
        sbucket = torch.empty(1, 1, dtype=torch.int32, device=q.device)
    elif strict:
        srow = union_idx
        assert slot_bucket is not None, "strict mask needs slot_bucket"
        # slot_bucket is a per-packed-slot table [Hkv, L], not the compact
        # per-non-empty-cluster index.slot_bucket.  The packed buffer's token
        # axis is page-padded (state._PACK_ALIGN) so the FA3 body can view it
        # as pages.  The padding rows belong to no cluster, so no union slot
        # names them and slot_bucket need not cover them.
        assert slot_bucket.shape[0] == num_kv_heads \
            and slot_bucket.shape[1] + 256 >= packed_kv.shape[1], \
            (slot_bucket.shape, packed_kv.shape)
        assert query_sel_mask.dtype == torch.int32 \
            and query_sel_mask.is_contiguous()
        assert query_sel_mask.shape[:2] == (q_count, num_kv_heads), \
            query_sel_mask.shape
        qmask_words = query_sel_mask.shape[2]
        sbucket = slot_bucket
    else:
        srow = union_idx
        qmask_words = 1
        query_sel_mask = torch.empty(1, 1, 1, dtype=torch.int32,
                                     device=q.device)
        sbucket = torch.empty(1, 1, dtype=torch.int32, device=q.device)
    grid = (num_blocks, num_kv_heads, num_splits)
    _union_block_attention_kernel[grid](
        out, lse, q, packed_kv, kv_scales, positions, union_idx, union_count,
        query_pos, sbucket, query_sel_mask, qmask_words, srow,
        float(sm_scale), q_count,
        out.stride(0), out.stride(1), out.stride(2),
        q.stride(0), q.stride(1), q.stride(2),
        packed_kv.stride(0), packed_kv.stride(1), packed_kv.stride(2),
        packed_kv.stride(3),
        kv_scales.stride(0), kv_scales.stride(1), kv_scales.stride(2),
        positions.stride(0), positions.stride(1),
        union_idx.stride(0), union_idx.stride(1), union_idx.stride(2),
        union_count.stride(0), union_count.stride(1),
        sbucket.stride(0), sbucket.stride(1),
        pacc, pm, pl,
        pacc.stride(0), pacc.stride(1), pacc.stride(2), pacc.stride(3),
        pm.stride(0), pm.stride(1), pm.stride(2),
        G_Q=queries_per_block, Q_PER_KV=q_per_kv, HEAD_DIM=head_dim,
        BLOCK_M=block_m, BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
        FP8=(packed_kv.dtype == torch.float8_e4m3fn),
        WRITE_LSE=(return_logsumexp and num_splits == 1),
        STRICT=strict,
        ROWMASK=use_rowmask,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if num_splits > 1:
        _union_split_merge_kernel[(q_count, num_q_heads)](
            out, lse, pacc, pm, pl,
            out.stride(0), out.stride(1), out.stride(2),
            pacc.stride(0), pacc.stride(1), pacc.stride(2), pacc.stride(3),
            pm.stride(0), pm.stride(1), pm.stride(2),
            HEAD_DIM=head_dim, NUM_SPLITS=num_splits,
            WRITE_LSE=return_logsumexp,
            num_warps=4,
        )
    if return_logsumexp:
        return out, lse
    return out
