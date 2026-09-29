"""Fused bucket selection without sorting (the theta-select kernels).

Downstream attention only needs the selected set of buckets, not their rank
order.  "Take buckets in score order until the token budget is reached" is
equivalent to "take every bucket whose score is above a threshold theta".
Theta is found on chip by bisection over the score range (ITERS masked
reductions, resolution 2^-ITERS of the range), weighting each score by the
bucket's causal token count.  Buckets whose score falls inside the final
bisection window are included whole, so the selection can overshoot the
sorted cut by at most that window's causal tokens (typically at most one
bucket).

For the product partition, scoring matches _bucket_scores_matmul_kernel
(selector.py): per component t (KV-group head x routing sub-query), factor
scores are log-softmax-normalised over M, the mixture log-weight is added,
and buckets score log sum_t exp(f0[t,i] + logmix[t] + f1[t,j]), computed as
a matrix product.  The size penalty -beta*log1p(size) is applied inline.
The flat partition (theta_select_bitmask_flat) uses the same rule
restricted to one centroid per cluster.

Causal sizes: the caller passes per-bucket causal token counts (a per-chunk
snapshot).  Selection budgeting uses the snapshot; exact causality is
enforced later by the attention body's position mask, so a snapshot at the
chunk boundary changes only which marginal buckets are budgeted in.
"""

from __future__ import annotations

import os

import torch

try:  # pragma: no cover - exercised only without triton
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _theta_select_ranges_kernel(
        starts_out_ptr,   # [Q, Hkv, W] int32
        lengths_out_ptr,  # [Q, Hkv, W] int32
        buckets_out_ptr,  # [Q, Hkv, W] int32
        counts_out_ptr,   # [Q, Hkv] int32 (selected bucket count)
        tokens_out_ptr,   # [Q, Hkv] int32 (selected causal tokens)
        bitmask_ptr,      # [ceil(Q/G_UNION), Hkv, M*M/32] int32 (atomic OR)
        q_ptr,            # [Q, Hq, R, 2, DF]
        cb_ptr,           # [Hkv, 2, M, DF]
        logmix_ptr,       # [Q, Hq, R]
        sizes_ptr,        # [Hkv, M*M] int32, causal snapshot
        indptr_ptr,       # [Hkv, M*M+1] int32, packed-KV bucket offsets
        inv_temperature,
        beta,
        target_tokens,
        max_out,
        stride_so0, stride_so1, stride_so2,
        stride_q0, stride_q1, stride_q2, stride_q3, stride_q4,
        stride_c0, stride_c1, stride_c2, stride_c3,
        stride_l0, stride_l1, stride_l2,
        stride_z0, stride_z1,
        stride_i0, stride_i1,
        HEADS_PER_KV: tl.constexpr,
        R: tl.constexpr,
        T: tl.constexpr,
        M: tl.constexpr,
        DF: tl.constexpr,
        ITERS: tl.constexpr,
        DO_EMIT: tl.constexpr,
        DO_BITMASK: tl.constexpr,
        G_UNION: tl.constexpr,
        IEEE: tl.constexpr,
    ):
        query = tl.program_id(0)
        kv_head = tl.program_id(1)

        offs_m = tl.arange(0, M)
        offs_d = tl.arange(0, DF)
        offs_t = tl.arange(0, T)
        head_t = kv_head * HEADS_PER_KV + offs_t // R
        sub_t = offs_t % R
        component = head_t * stride_q1 + sub_t * stride_q2
        base = query * stride_q0

        # --- scores, same math as _bucket_scores_matmul_kernel ---
        c0 = tl.load(
            cb_ptr + kv_head * stride_c0 + 0 * stride_c1
            + offs_m[:, None] * stride_c2 + offs_d[None, :] * stride_c3
        ).to(tl.float32)
        q0 = tl.load(
            q_ptr + base + component[None, :] + 0 * stride_q3
            + offs_d[:, None] * stride_q4
        ).to(tl.float32)
        if IEEE:
            scores0 = tl.dot(c0, q0, input_precision="ieee") * inv_temperature
        else:
            scores0 = tl.dot(c0, q0) * inv_temperature
        peak0 = tl.max(scores0, axis=0)
        norm0 = peak0 + tl.log(tl.sum(tl.exp(scores0 - peak0[None, :]), axis=0))
        logmix = tl.load(
            logmix_ptr + query * stride_l0 + head_t * stride_l1
            + sub_t * stride_l2
        ).to(tl.float32)
        left = tl.exp(scores0 - norm0[None, :] + logmix[None, :])

        c1 = tl.load(
            cb_ptr + kv_head * stride_c0 + 1 * stride_c1
            + offs_m[None, :] * stride_c2 + offs_d[:, None] * stride_c3
        ).to(tl.float32)
        q1 = tl.load(
            q_ptr + base + component[:, None] + 1 * stride_q3
            + offs_d[None, :] * stride_q4
        ).to(tl.float32)
        if IEEE:
            scores1 = tl.dot(q1, c1, input_precision="ieee") * inv_temperature
        else:
            scores1 = tl.dot(q1, c1) * inv_temperature
        peak1 = tl.max(scores1, axis=1)
        norm1 = peak1 + tl.log(tl.sum(tl.exp(scores1 - peak1[:, None]), axis=1))
        right = tl.exp(scores1 - norm1[:, None])

        if IEEE:
            total = tl.dot(left, right, input_precision="ieee")
        else:
            total = tl.dot(left, right)
        score = tl.log(tl.maximum(total, 1e-38))

        # --- sizes, penalty ---
        # Keep only two [M,M] arrays live (score, sizes); everything else is
        # recomputed from them so the kernel stays out of register spills.
        sizes = tl.load(
            sizes_ptr + kv_head * stride_z0
            + (offs_m[:, None] * M + offs_m[None, :]) * stride_z1
        ).to(tl.float32)
        if beta != 0.0:
            score = score - beta * tl.log(1.0 + sizes)
        neg_inf = -3.4e38
        score = tl.where(sizes > 0.0, score, neg_inf)

        # --- theta search: bisection on the budget threshold ---
        # "Walk buckets in score order until the budget" == "select every
        # bucket with score above the largest theta whose above-set still
        # reaches the budget".  Each iteration costs one masked reduction.
        total_all = tl.sum(tl.where(sizes > 0.0, sizes, 0.0))
        all_fit = total_all < target_tokens

        hi = tl.max(score) + 1e-6
        lo = tl.min(tl.where(sizes > 0.0, score, -neg_inf)) - 1e-6
        lo = tl.minimum(lo, hi - 1e-5)
        for _ in tl.static_range(ITERS):
            mid = 0.5 * (lo + hi)
            above = tl.sum(tl.where(score >= mid, sizes, 0.0))
            reach = above >= target_tokens
            lo = tl.where(reach, mid, lo)
            hi = tl.where(reach, hi, mid)

        theta = tl.where(all_fit, neg_inf, lo)
        selected = (sizes > 0.0) & (score >= theta)

        # --- compacted emission in bucket-id order ---
        # The flat index of the row-major [M,M] grid is the bucket id.
        sel_i32 = selected.to(tl.int32)
        flat_sel = tl.reshape(sel_i32, [M * M])
        if DO_EMIT:
            flat_ids = tl.arange(0, M * M)
            flat_len = tl.reshape(sizes, [M * M]).to(tl.int32)
            pos = tl.cumsum(flat_sel, axis=0) - flat_sel
            keep = (flat_sel > 0) & (pos < max_out)
            bstart = tl.load(
                indptr_ptr + kv_head * stride_i0 + flat_ids * stride_i1,
                mask=keep, other=0,
            )
            out_base = query * stride_so0 + kv_head * stride_so1
            tl.store(starts_out_ptr + out_base + pos * stride_so2, bstart,
                     mask=keep)
            tl.store(lengths_out_ptr + out_base + pos * stride_so2, flat_len,
                     mask=keep)
            tl.store(buckets_out_ptr + out_base + pos * stride_so2, flat_ids,
                     mask=keep)
        if DO_BITMASK:
            # OR this query's selected-bucket bits into its union block's
            # bitmask.  Bits are disjoint within a word, so SUM == OR when
            # packing.
            words = tl.reshape(flat_sel, [(M * M) // 32, 32])
            shifts = tl.arange(0, 32)[None, :]
            packed = tl.sum(words << shifts, axis=1)
            union_block = query // G_UNION
            n_words = (M * M) // 32
            tl.atomic_or(
                bitmask_ptr
                + (union_block * tl.num_programs(1) + kv_head) * n_words
                + tl.arange(0, (M * M) // 32),
                packed,
            )
        n_sel = tl.sum(flat_sel)
        # counts/tokens are contiguous [Q,Hkv]
        tl.store(counts_out_ptr + query * tl.num_programs(1) + kv_head,
                 tl.minimum(n_sel, max_out))
        tl.store(tokens_out_ptr + query * tl.num_programs(1) + kv_head,
                 tl.sum(tl.where(selected, sizes, 0.0)).to(tl.int32))


def theta_select_ranges(
    vectors: torch.Tensor,        # [Q, Hq, R, I] with I = 2*DF (router layout)
    logmix: torch.Tensor,         # [Q, Hq, R] float32 (log mixture weights)
    codebooks: torch.Tensor,      # [Hkv, 2, M, DF]
    causal_sizes: torch.Tensor,   # [Hkv, B] int32
    bucket_indptr: torch.Tensor,  # [Hkv, B+1] int32
    *,
    temperature: float,
    beta: float,
    target_tokens: int,
    max_out: int,
    iters: int = 24,
    num_warps: int = 8,
    do_emit: bool = True,
    bitmask: torch.Tensor | None = None,
    g_union: int = 32,
    ieee_dots: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the fused selection. Returns (starts, lengths, buckets, counts,
    tokens); ranges are emitted in bucket-id order, not score order."""
    if triton is None:
        raise RuntimeError("theta_select_ranges needs triton")
    q_count, num_q_heads, r_dim, index_dim = vectors.shape
    num_kv_heads, two, m_dim, df = codebooks.shape
    assert two == 2 and index_dim == 2 * df
    heads_per_kv = num_q_heads // num_kv_heads
    t_dim = heads_per_kv * r_dim
    q5 = vectors.view(q_count, num_q_heads, r_dim, 2, df)

    device = vectors.device
    # Only [:count] entries are meaningful, and counts/tokens are fully
    # written by the kernel, so nothing needs zero-init.
    starts = torch.empty(q_count, num_kv_heads, max_out, dtype=torch.int32,
                         device=device)
    lengths = torch.empty_like(starts)
    buckets = torch.empty_like(starts)
    counts = torch.empty(q_count, num_kv_heads, dtype=torch.int32,
                         device=device)
    tokens = torch.empty_like(counts)

    if bitmask is None:
        bitmask_arg = torch.empty(1, dtype=torch.int32, device=device)
        do_bitmask = False
    else:
        assert bitmask.dtype == torch.int32 and bitmask.is_contiguous()
        assert bitmask.shape == (
            (q_count + g_union - 1) // g_union, num_kv_heads,
            (m_dim * m_dim) // 32,
        ), bitmask.shape
        bitmask_arg = bitmask
        do_bitmask = True

    grid = (q_count, num_kv_heads)
    _theta_select_ranges_kernel[grid](
        starts, lengths, buckets, counts, tokens, bitmask_arg,
        q5, codebooks, logmix, causal_sizes, bucket_indptr,
        1.0 / float(temperature), float(beta),
        float(target_tokens), int(max_out),
        starts.stride(0), starts.stride(1), starts.stride(2),
        q5.stride(0), q5.stride(1), q5.stride(2), q5.stride(3), q5.stride(4),
        codebooks.stride(0), codebooks.stride(1), codebooks.stride(2),
        codebooks.stride(3),
        logmix.stride(0), logmix.stride(1), logmix.stride(2),
        causal_sizes.stride(0), causal_sizes.stride(1),
        bucket_indptr.stride(0), bucket_indptr.stride(1),
        HEADS_PER_KV=heads_per_kv, R=r_dim, T=t_dim, M=m_dim, DF=df,
        ITERS=iters, DO_EMIT=do_emit,
        DO_BITMASK=do_bitmask, G_UNION=g_union,
        IEEE=ieee_dots,
        num_warps=num_warps,
    )
    return starts, lengths, buckets, counts, tokens


def theta_select_reference(
    scores: torch.Tensor,        # [Q, Hkv, B] float32, penalty applied
    causal_sizes: torch.Tensor,  # [Hkv, B] int32
    target_tokens: int,
) -> torch.Tensor:
    """Reference selected-set mask [Q,Hkv,B] by sorting: walk buckets in
    descending score order, stop once the cumulative causal tokens reach the
    budget (the crossing bucket is included)."""
    q_count, num_kv_heads, b_dim = scores.shape
    mask = torch.zeros_like(scores, dtype=torch.bool)
    for h in range(num_kv_heads):
        sizes = causal_sizes[h].to(scores.dtype)
        nonempty = sizes > 0
        s = scores[:, h].masked_fill(~nonempty, float("-inf"))
        order = s.argsort(dim=-1, descending=True)
        sorted_sizes = sizes[order]
        csum = sorted_sizes.cumsum(dim=-1)
        take = (csum - sorted_sizes) < target_tokens
        take &= s.gather(-1, order) > float("-inf")
        mask[:, h] = torch.zeros_like(take).scatter(-1, order, take)
    return mask


if triton is not None:

    @triton.jit
    def _theta_select_from_factors_kernel(
        counts_out_ptr,   # [Q, Hkv] int32
        tokens_out_ptr,   # [Q, Hkv] int32
        bitmask_ptr,      # [ceil(Q/G_UNION), Hkv, M*M/32] int32
        qmask_ptr,        # [Q, Hkv, M*M/32] int32 (DO_QMASK: per-query set)
        cand_ptr,         # [Q, Hkv, M*M/32] int32 (DO_CAND: candidate set)
        left_ptr,         # [Q, Hkv, T, M] fp32  f0 + logmix (log domain)
        right_ptr,        # [Q, Hkv, T, M] fp32  f1 (log domain)
        cleft_ptr,        # [NB, Hkv, T, M] fp32 block-mean f0+logmix (USE_COH)
        cright_ptr,       # [NB, Hkv, T, M] fp32 block-mean f1 (USE_COH)
        sizes_ptr,        # [NB, Hkv, M*M] int32 (per-block causal snapshot)
        beta,
        coh_lambda,
        target_tokens,
        stride_f0, stride_f1, stride_f2, stride_f3,
        stride_zb, stride_z0, stride_z1,
        T: tl.constexpr,
        M: tl.constexpr,
        ITERS: tl.constexpr,
        G_UNION: tl.constexpr,
        USE_COH: tl.constexpr,
        DO_QMASK: tl.constexpr,
        DO_CAND: tl.constexpr,
    ):
        """theta-select with the factor products precomputed by cuBLAS.

        The two factor dots and their log-softmax normalisations run as one
        batched fp32 GEMM + log_softmax on the torch side; this kernel takes
        the [M,T]x[T,M] product, applies the penalty, bisects theta, and ORs
        the selected set into the union bitmask.  The scoring math is that
        of _theta_select_ranges_kernel (fp32; reduction order differs only
        inside the GEMM).
        """
        query = tl.program_id(0)
        kv_head = tl.program_id(1)
        union_block = query // G_UNION
        offs_m = tl.arange(0, M)
        offs_t = tl.arange(0, T)
        base = query.to(tl.int64) * stride_f0 + kv_head * stride_f1
        # exp on chip rather than as separate elementwise passes over
        # [Q,H,T,M].
        left = tl.exp(tl.load(
            left_ptr + base + offs_t[None, :] * stride_f2
            + offs_m[:, None] * stride_f3
        ))
        right = tl.exp(tl.load(
            right_ptr + base + offs_t[:, None] * stride_f2
            + offs_m[None, :] * stride_f3
        ))
        total = tl.dot(left, right, input_precision="ieee")
        score = tl.log(tl.maximum(total, 1e-38))
        if USE_COH:
            # Blend toward the block-consensus score: coherent selections
            # shrink the union the body has to read.  lambda=1 is fully
            # block-shared selection.  cleft/cright are contiguous
            # [NB, H, T, M]; their strides equal left's.
            cbase = (union_block.to(tl.int64) * stride_f0
                     + kv_head * stride_f1)
            cl = tl.exp(tl.load(
                cleft_ptr + cbase + offs_t[None, :] * stride_f2
                + offs_m[:, None] * stride_f3
            ))
            cr = tl.exp(tl.load(
                cright_ptr + cbase + offs_t[:, None] * stride_f2
                + offs_m[None, :] * stride_f3
            ))
            ctotal = tl.dot(cl, cr, input_precision="ieee")
            cscore = tl.log(tl.maximum(ctotal, 1e-38))
            score = (1.0 - coh_lambda) * score + coh_lambda * cscore

        sizes = tl.load(
            sizes_ptr + union_block.to(tl.int64) * stride_zb
            + kv_head * stride_z0
            + (offs_m[:, None] * M + offs_m[None, :]) * stride_z1
        ).to(tl.float32)
        if beta != 0.0:
            score = score - beta * tl.log(1.0 + sizes)
        neg_inf = -3.4e38
        score = tl.where(sizes > 0.0, score, neg_inf)
        if DO_CAND:
            # Candidate-limited selection (P << M product search): buckets
            # outside the caller-provided candidate set never enter the
            # budget; sizes are zeroed too so theta budgets on candidates.
            cw = tl.load(
                cand_ptr
                + (query.to(tl.int64) * tl.num_programs(1) + kv_head)
                * ((M * M) // 32) + tl.arange(0, (M * M) // 32)
            )
            cbits = tl.reshape(
                (cw[:, None] >> tl.arange(0, 32)[None, :]) & 1, [M, M]
            )
            score = tl.where(cbits > 0, score, neg_inf)
            sizes = tl.where(cbits > 0, sizes, 0.0)

        total_all = tl.sum(tl.where(sizes > 0.0, sizes, 0.0))
        all_fit = total_all < target_tokens
        hi = tl.max(score) + 1e-6
        lo = tl.min(tl.where(sizes > 0.0, score, -neg_inf)) - 1e-6
        lo = tl.minimum(lo, hi - 1e-5)
        for _ in tl.static_range(ITERS):
            mid = 0.5 * (lo + hi)
            above = tl.sum(tl.where(score >= mid, sizes, 0.0))
            reach = above >= target_tokens
            lo = tl.where(reach, mid, lo)
            hi = tl.where(reach, hi, mid)
        theta = tl.where(all_fit, neg_inf, lo)
        selected = (sizes > 0.0) & (score >= theta)

        sel_i32 = selected.to(tl.int32)
        flat_sel = tl.reshape(sel_i32, [M * M])
        words = tl.reshape(flat_sel, [(M * M) // 32, 32])
        packed = tl.sum(words << tl.arange(0, 32)[None, :], axis=1)
        n_words = (M * M) // 32
        tl.atomic_or(
            bitmask_ptr
            + (union_block * tl.num_programs(1) + kv_head) * n_words
            + tl.arange(0, (M * M) // 32),
            packed,
        )
        if DO_QMASK:
            # Per-query selected set, single writer per (query, head): the
            # strict-mask and KV-outer bodies read it, and probes dump it.
            tl.store(
                qmask_ptr
                + (query.to(tl.int64) * tl.num_programs(1) + kv_head)
                * n_words
                + tl.arange(0, (M * M) // 32),
                packed,
            )
        tl.store(counts_out_ptr + query * tl.num_programs(1) + kv_head,
                 tl.sum(flat_sel))
        tl.store(tokens_out_ptr + query * tl.num_programs(1) + kv_head,
                 tl.sum(tl.where(selected, sizes, 0.0)).to(tl.int32))


def theta_select_bitmask_precomputed(
    vectors: torch.Tensor,        # [Q, Hq, R, I]
    logmix: torch.Tensor,         # [Q, Hq, R] float32 (log)
    codebooks: torch.Tensor,      # [Hkv, 2, M, DF]
    causal_sizes: torch.Tensor,   # [NB, Hkv, B] int32 (per union block)
    bitmask: torch.Tensor,        # [NB, Hkv, B/32] int32, pre-zeroed
    *,
    temperature: float,
    beta: float,
    target_tokens: int,
    g_union: int,
    iters: int = 16,
    num_warps: int = 16,
    coherence_lambda: float = 0.0,
    qmask: torch.Tensor | None = None,  # [Q, Hkv, B/32] int32 out
    top_p: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GEMM-precomputed variant. Returns (counts, tokens) [Q,Hkv] int32.

    top_p > 0 restricts the budgeted set to the per-component factor
    top-P Cartesian product (P << M product search).  The candidate mask is
    built torch-side and applied inside the full-scan kernel, so it changes
    the selection but not the selector cost."""
    q_count, num_q_heads, r_dim, index_dim = vectors.shape
    num_kv_heads, _, m_dim, df = codebooks.shape
    heads_per_kv = num_q_heads // num_kv_heads
    t_dim = heads_per_kv * r_dim
    device = vectors.device

    # fp32 GEMMs must not silently use tf32: its score jitter decorrelates
    # neighbouring selections and grows the union.  Guard the global flag.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q5 = (vectors.view(q_count, num_kv_heads, t_dim, 2, df)
              .float())
        cb = codebooks.float()
        inv_t = 1.0 / float(temperature)
        s0 = torch.einsum("qhtd,hmd->qhtm", q5[:, :, :, 0], cb[:, 0]) * inv_t
        s1 = torch.einsum("qhtd,hmd->qhtm", q5[:, :, :, 1], cb[:, 1]) * inv_t
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    lm = logmix.view(q_count, num_kv_heads, t_dim, 1)
    left = (s0.log_softmax(dim=-1) + lm).contiguous()
    right = s1.log_softmax(dim=-1).contiguous()

    use_coh = coherence_lambda > 0.0
    if use_coh:
        nb = (q_count + g_union - 1) // g_union
        pad = nb * g_union - q_count
        def blockmean(x):
            if pad:
                x = torch.cat([x, x[-1:].expand(pad, -1, -1, -1)], dim=0)
            return x.view(nb, g_union, num_kv_heads, t_dim, m_dim)                     .mean(dim=1).contiguous()
        cleft = blockmean(left)
        cright = blockmean(right)
    else:
        cleft = left[:1]
        cright = right[:1]

    counts = torch.empty(q_count, num_kv_heads, dtype=torch.int32,
                         device=device)
    tokens = torch.empty_like(counts)
    b_dim = m_dim * m_dim
    if top_p and top_p < m_dim:
        idx_l = left.topk(top_p, dim=-1).indices    # [Q, Hkv, T, P]
        idx_r = right.topk(top_p, dim=-1).indices
        cand = torch.zeros(q_count, num_kv_heads, m_dim, m_dim,
                           dtype=torch.bool, device=device)
        bq = torch.arange(q_count, device=device)[:, None, None, None]
        bh = torch.arange(num_kv_heads, device=device)[None, :, None, None]
        for t in range(t_dim):
            cand[bq, bh, idx_l[:, :, t, :, None], idx_r[:, :, t, None, :]] \
                = True
        shifts = torch.arange(32, device=device, dtype=torch.int32)
        cand_arg = (cand.view(q_count, num_kv_heads, b_dim // 32, 32).int()
                    << shifts).sum(dim=-1).to(torch.int32).contiguous()
        do_cand = True
        del cand
    else:
        cand_arg = torch.empty(1, dtype=torch.int32, device=device)
        do_cand = False
    if qmask is None:
        qmask_arg = torch.empty(1, dtype=torch.int32, device=device)
        do_qmask = False
    else:
        assert qmask.dtype == torch.int32 and qmask.is_contiguous()
        assert qmask.shape == (q_count, num_kv_heads, b_dim // 32), \
            qmask.shape
        qmask_arg = qmask
        do_qmask = True
    _theta_select_from_factors_kernel[(q_count, num_kv_heads)](
        counts, tokens, bitmask, qmask_arg, cand_arg,
        left, right, cleft, cright, causal_sizes,
        float(beta), float(coherence_lambda), float(target_tokens),
        left.stride(0), left.stride(1), left.stride(2), left.stride(3),
        causal_sizes.stride(0), causal_sizes.stride(1),
        causal_sizes.stride(2),
        T=t_dim, M=m_dim, ITERS=iters, G_UNION=g_union,
        USE_COH=use_coh,
        DO_QMASK=do_qmask,
        DO_CAND=do_cand,
        num_warps=num_warps,
    )
    return counts, tokens


if triton is not None:

    @triton.jit
    def _streaming_select_kernel(
        prob_ptr,         # [Q, Hkv, B] fp32 bucket probs (bmm out);
                          # overwritten with penalized log scores in sweep 1
        sizes_ptr,        # [NB, Hkv, B] int32 per-block causal snapshot
        bitmask_ptr,      # [NB, Hkv, B/32] int32 (atomic OR)
        qmask_ptr,        # [Q, Hkv, B/32] int32 (DO_QMASK: per-query set)
        counts_out_ptr,   # [Q, Hkv] int32
        tokens_out_ptr,   # [Q, Hkv] int32
        beta,
        target_tokens,
        stride_p0, stride_p1, stride_p2,
        stride_zb, stride_z0, stride_z1,
        B: tl.constexpr,
        TILE: tl.constexpr,
        ITERS: tl.constexpr,
        G_UNION: tl.constexpr,
        DO_QMASK: tl.constexpr,
        RESIDENT: tl.constexpr,
        WIDE: tl.constexpr,
    ):
        """Budget-threshold selection over materialised bucket scores.

        RESIDENT keeps the whole [B] slice in registers so the bisection
        reads it once instead of ITERS+2 times; this is the right form for
        the flat partition, where B = C is a few thousand at most.  The
        streaming form suits large B (the product path, B = M*M), where one
        (query, KV head) slice is L2-hot after sweep 1.

        The scores come from upstream (a cuBLAS GEMM or the query-tiled
        kernel); this kernel only transforms, bisects theta and emits the
        union bitmask.
        """
        query = tl.program_id(0)
        kv_head = tl.program_id(1)
        union_block = query // G_UNION
        # int32 offsets unless a chunk's arrays need 64-bit addressing.
        if WIDE:
            qw = query.to(tl.int64)
            uw = union_block.to(tl.int64)
        else:
            qw = query
            uw = union_block
        p_base = qw * stride_p0 + kv_head * stride_p1
        z_base = uw * stride_zb + kv_head * stride_z0
        # fp32 constants: bare literals promote to fp64 and change the type
        # of runtime-loop carried variables.
        neg_inf = tl.full([1], -3.4e38, tl.float32)
        big = tl.full([1], 3.4e38, tl.float32)
        tiny = tl.full([1], 1e-38, tl.float32)
        fzero = tl.zeros([1], tl.float32)
        fone = tl.full([1], 1.0, tl.float32)
        half = tl.full([1], 0.5, tl.float32)

        if RESIDENT:
            offs = tl.arange(0, TILE)
            prob_r = tl.load(prob_ptr + p_base + offs * stride_p2)
            sizes_r = tl.load(sizes_ptr + z_base
                              + offs * stride_z1).to(tl.float32)
            live_r = sizes_r > fzero
            score_r = tl.log(tl.maximum(prob_r, tiny))
            if beta != 0.0:
                score_r = score_r - beta * tl.log(fone + sizes_r)
            score_r = tl.where(live_r, score_r, neg_inf)
            hi_r = tl.max(score_r) + 1e-6
            lo_r = tl.min(tl.where(live_r, score_r, big)) - 1e-6
            lo_r = tl.minimum(lo_r, hi_r - 1e-5)
            all_fit_r = tl.sum(tl.where(live_r, sizes_r, fzero)) \
                < target_tokens
            for _it in tl.static_range(ITERS):
                mid = (lo_r + hi_r) * 0.5
                above = tl.sum(tl.where(score_r >= mid, sizes_r, fzero))
                reach = above >= target_tokens
                lo_r = tl.where(reach, mid, lo_r)
                hi_r = tl.where(reach, hi_r, mid)
            theta_r = tl.where(all_fit_r, -3.4e38, lo_r)
            sel = live_r & (score_r >= theta_r)
            n_words_r: tl.constexpr = TILE // 32
            words = tl.reshape(sel.to(tl.int32), [n_words_r, 32])
            packed = tl.sum(words << tl.arange(0, 32)[None, :], axis=1)
            tl.atomic_or(
                bitmask_ptr
                + (uw * tl.num_programs(1) + kv_head) * n_words_r
                + tl.arange(0, n_words_r),
                packed,
            )
            if DO_QMASK:
                tl.store(
                    qmask_ptr
                    + (qw * tl.num_programs(1) + kv_head) * n_words_r
                    + tl.arange(0, n_words_r),
                    packed,
                )
            tl.store(counts_out_ptr + query * tl.num_programs(1) + kv_head,
                     tl.sum(sel.to(tl.int32)))
            tl.store(tokens_out_ptr + query * tl.num_programs(1) + kv_head,
                     tl.sum(tl.where(sel, sizes_r, fzero)).to(tl.int32))
            return

        # Sweep 1: log + penalty in place, min/max, total causal tokens.
        lo = tl.full([1], 3.4e38, tl.float32)
        hi = tl.full([1], -3.4e38, tl.float32)
        total_all = tl.zeros([1], tl.float32)
        for t in tl.range(0, B // TILE):
            offs = t * TILE + tl.arange(0, TILE)
            prob = tl.load(prob_ptr + p_base + offs * stride_p2)
            sizes = tl.load(sizes_ptr + z_base + offs * stride_z1) \
                .to(tl.float32)
            score = tl.log(tl.maximum(prob, tiny))
            if beta != 0.0:
                score = score - beta * tl.log(fone + sizes)
            score = tl.where(sizes > fzero, score, neg_inf)
            tl.store(prob_ptr + p_base + offs * stride_p2, score)
            hi = tl.maximum(hi, tl.max(score))
            lo = tl.minimum(lo, tl.min(tl.where(sizes > fzero, score, big)))
            total_all += tl.sum(tl.where(sizes > fzero, sizes, fzero))
        all_fit = total_all < target_tokens
        eps6 = tl.full([1], 1e-6, tl.float32)
        eps5 = tl.full([1], 1e-5, tl.float32)
        hi = hi + eps6
        lo = tl.minimum(lo - eps6, hi - eps5)

        # Bisection sweeps over the (L2-resident) score slice.
        for _it in tl.static_range(ITERS):
            mid = (lo + hi) * half
            above = tl.zeros([1], tl.float32)
            for t in tl.range(0, B // TILE):
                offs = t * TILE + tl.arange(0, TILE)
                score = tl.load(prob_ptr + p_base + offs * stride_p2)
                sizes = tl.load(sizes_ptr + z_base
                                + offs * stride_z1).to(tl.float32)
                above += tl.sum(tl.where(score >= mid, sizes, fzero))
            reach = above >= target_tokens
            lo = tl.where(reach, mid, lo)
            hi = tl.where(reach, hi, mid)
        theta = tl.where(all_fit, neg_inf, lo)

        # Final sweep: select, bitmask, counts.
        n_sel = tl.zeros([1], tl.float32)
        n_tok = tl.zeros([1], tl.float32)
        n_words: tl.constexpr = B // 32
        for t in tl.range(0, B // TILE):
            offs = t * TILE + tl.arange(0, TILE)
            score = tl.load(prob_ptr + p_base + offs * stride_p2)
            sizes = tl.load(sizes_ptr + z_base
                            + offs * stride_z1).to(tl.float32)
            sel = (score > neg_inf) & (score >= theta)
            n_sel += tl.sum(sel.to(tl.float32))
            n_tok += tl.sum(tl.where(sel, sizes, fzero))
            words = tl.reshape(sel.to(tl.int32), [TILE // 32, 32])
            packed = tl.sum(words << tl.arange(0, 32)[None, :], axis=1)
            tl.atomic_or(
                bitmask_ptr
                + (uw * tl.num_programs(1) + kv_head) * n_words
                + t * (TILE // 32) + tl.arange(0, TILE // 32),
                packed,
            )
            if DO_QMASK:
                tl.store(
                    qmask_ptr
                    + (qw * tl.num_programs(1) + kv_head) * n_words
                    + t * (TILE // 32) + tl.arange(0, TILE // 32),
                    packed,
                )
        tl.store(counts_out_ptr + query * tl.num_programs(1) + kv_head,
                 tl.sum(n_sel).to(tl.int32))
        tl.store(tokens_out_ptr + query * tl.num_programs(1) + kv_head,
                 tl.sum(n_tok).to(tl.int32))


def theta_select_bitmask_gemm(
    vectors: torch.Tensor,        # [Q, Hq, R, I]
    logmix: torch.Tensor,         # [Q, Hq, R] float32 (log)
    codebooks: torch.Tensor,      # [Hkv, 2, M, DF]
    causal_sizes: torch.Tensor,   # [NB, Hkv, B] int32
    bitmask: torch.Tensor,        # [NB, Hkv, B/32] int32, pre-zeroed
    *,
    temperature: float,
    beta: float,
    target_tokens: int,
    g_union: int,
    tile: int = 512,
    iters: int = 16,
    qmask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GEMM-materialised scores + streaming threshold selection."""
    q_count, num_q_heads, r_dim, index_dim = vectors.shape
    num_kv_heads, _, m_dim, df = codebooks.shape
    heads_per_kv = num_q_heads // num_kv_heads
    t_dim = heads_per_kv * r_dim
    b_dim = m_dim * m_dim
    device = vectors.device

    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        q5 = vectors.view(q_count, num_kv_heads, t_dim, 2, df).float()
        cb = codebooks.float()
        inv_t = 1.0 / float(temperature)
        s0 = torch.einsum("qhtd,hmd->qhtm", q5[:, :, :, 0], cb[:, 0]) * inv_t
        s1 = torch.einsum("qhtd,hmd->qhtm", q5[:, :, :, 1], cb[:, 1]) * inv_t
        lm = logmix.view(q_count, num_kv_heads, t_dim, 1)
        pl = (s0.log_softmax(dim=-1) + lm).exp()
        pr = s1.log_softmax(dim=-1).exp()
        prob = torch.einsum("qhtm,qhtn->qhmn", pl, pr)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    prob = prob.reshape(q_count, num_kv_heads, b_dim).contiguous()

    counts = torch.empty(q_count, num_kv_heads, dtype=torch.int32,
                         device=device)
    tokens = torch.empty_like(counts)
    _run_streaming_select(prob, causal_sizes, bitmask, counts, tokens,
                          beta=beta, target_tokens=target_tokens,
                          b_dim=b_dim, tile=tile, iters=iters,
                          g_union=g_union, qmask=qmask)
    return counts, tokens


def _run_streaming_select(prob, causal_sizes, bitmask, counts, tokens, *,
                          beta, target_tokens, b_dim, tile, iters, g_union,
                          qmask=None):
    if qmask is None:
        qmask_arg = torch.empty(1, dtype=torch.int32, device=prob.device)
        do_qmask = False
    else:
        assert qmask.dtype == torch.int32 and qmask.is_contiguous()
        assert qmask.shape == (prob.shape[0], prob.shape[1], b_dim // 32), \
            qmask.shape
        qmask_arg = qmask
        do_qmask = True
    # One tile covering the whole slice lets the bisection stay in registers.
    resident = (os.environ.get("SEMBLO_SELECT_RESIDENT", "1") == "1"
                and b_dim <= int(os.environ.get("SEMBLO_SELECT_RESIDENT_MAX",
                                                "2048")))
    if resident:
        tile = b_dim
    _streaming_select_kernel[(prob.shape[0], prob.shape[1])](
        prob, causal_sizes, bitmask, qmask_arg, counts, tokens,
        float(beta), float(target_tokens),
        prob.stride(0), prob.stride(1), prob.stride(2),
        causal_sizes.stride(0), causal_sizes.stride(1),
        causal_sizes.stride(2),
        B=b_dim, TILE=tile, ITERS=iters, G_UNION=g_union,
        DO_QMASK=do_qmask, RESIDENT=resident,
        WIDE=(prob.shape[0] * max(prob.stride(0), causal_sizes.stride(0))
              >= (1 << 31)),
        # Few warps: each of the ~20 block-wide reductions of the bisection
        # crosses every warp through shared memory, so at B = 1024 one warp
        # holding the whole slice is fastest (about 3.5x faster than 16).
        # The default scales with the slice size.
        num_warps=int(os.environ.get(
            "SEMBLO_SELECT_WARPS", max(1, b_dim // 1024))),
    )


def theta_select_bitmask_flat(
    vectors: torch.Tensor,        # [Q, Hq, R, I]
    logmix: torch.Tensor,         # [Q, Hq, R] float32 (log)
    codebooks: torch.Tensor,      # [Hkv, 2, C, DF] cluster descriptors
    causal_sizes: torch.Tensor,   # [NB, Hkv, C] int32
    bitmask: torch.Tensor,        # [NB, Hkv, C/32] int32, pre-zeroed
    *,
    group_desc: torch.Tensor | None = None,  # [Hkv, 2, G, DF]
    # The cluster descriptor.  All three are given together or not at all.
    cluster_descriptors: torch.Tensor | None = None,  # [Hkv, C, Dm] float32
    query_descriptors: torch.Tensor | None = None,    # [Q, Hq, R, Dm]
    descriptor_gamma: torch.Tensor | None = None,     # [Hq]
    temperature: float,
    beta: float,
    target_tokens: int,
    g_union: int,
    tile: int = 512,
    iters: int = 16,
    qmask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Selection over a flat partition (one centroid per cluster).

    The product path scores a bucket as log sum_t exp(f0[t,i] + logmix[t]
    + f1[t,j]) with a log-softmax per factor over M.  With flat clusters
    there is no lattice: component t scores cluster c by the same half-wise
    metric, q_t0.c0 + q_t1.c1, log-softmax-normalised over the C clusters,
    and the components are combined by the same logsumexp."""
    q_count, num_q_heads, r_dim, index_dim = vectors.shape
    num_kv_heads, two, c_dim, df = codebooks.shape
    assert two == 2 and index_dim == 2 * df, (codebooks.shape, vectors.shape)
    heads_per_kv = num_q_heads // num_kv_heads
    t_dim = heads_per_kv * r_dim

    with_descriptor = cluster_descriptors is not None
    if with_descriptor and (query_descriptors is None or descriptor_gamma is None):
        raise ValueError(
            "the cluster descriptor needs all three of cluster_descriptors, "
            "query_descriptors and descriptor_gamma")
    if os.environ.get("SEMBLO_FLAT_TORCH_SCORES") != "1":
        # Query-tiled tensor-core kernel, which never materialises the
        # [Q, Hkv, T, C] score volume.  It includes the descriptor term.
        # SEMBLO_FLAT_TORCH_SCORES=1 selects the torch form below instead.
        # The launch shape is set in flat_mixture_scores_qtiled.
        prob = flat_mixture_scores_qtiled(
            vectors, logmix, codebooks, temperature=temperature,
            group_desc=group_desc,
            cluster_descriptors=cluster_descriptors,
            query_descriptors=query_descriptors,
            descriptor_gamma=descriptor_gamma,
        )
        counts = torch.empty(q_count, num_kv_heads, dtype=torch.int32,
                             device=vectors.device)
        tokens = torch.empty_like(counts)
        _run_streaming_select(prob, causal_sizes, bitmask, counts, tokens,
                              beta=beta, target_tokens=target_tokens,
                              b_dim=c_dim, tile=tile, iters=iters,
                              g_union=g_union, qmask=qmask)
        return counts, tokens

    # Torch form.  [Q, H, T, C] can take gigabytes, so the component axis
    # is reduced in query slabs rather than materialised whole.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    prob = torch.empty(q_count, num_kv_heads, c_dim, dtype=torch.float32,
                       device=vectors.device)
    slab = int(os.environ.get("SEMBLO_FLAT_SELECT_SLAB", "1024"))
    try:
        cb = codebooks.float()
        inv_t = 1.0 / float(temperature)
        for lo in range(0, q_count, slab):
            hi = min(lo + slab, q_count)
            n = hi - lo
            q5 = vectors[lo:hi].view(n, num_kv_heads, t_dim, 2, df).float()
            lm = logmix[lo:hi].view(n, num_kv_heads, t_dim, 1)
            if with_descriptor:
                # [n, Hkv, T, Dm]: the same (head, sub-query) flattening the
                # routing vectors use, so component t lines up with q5[:, h, t]
                phi = query_descriptors[lo:hi].reshape(
                    n, num_kv_heads, t_dim, -1).float()
                gam = descriptor_gamma.view(num_kv_heads, heads_per_kv, 1)
                gam = gam.expand(num_kv_heads, heads_per_kv, r_dim).reshape(
                    num_kv_heads, t_dim)
            for h in range(num_kv_heads):
                # One plain [n*T, DF] x [DF, C] GEMM per half rather than a
                # 4-d einsum, which leaves cuBLAS with tiny per-batch problems.
                a0 = q5[:, h, :, 0].reshape(n * t_dim, df)
                a1 = q5[:, h, :, 1].reshape(n * t_dim, df)
                sc = (a0 @ cb[h, 0].T + a1 @ cb[h, 1].T).view(
                    n, t_dim, c_dim)
                sc *= inv_t
                if with_descriptor:
                    # gamma * log <phi(u), g_c>, added to the same quantity
                    # the log-softmax normalises.  softplus rather than a
                    # clamp: with a linear psi the sum can be negative, and a
                    # clamp would put those on one floor with no ordering.
                    mass = phi[:, h] @ cluster_descriptors[h].T.float()
                    sc += gam[h].view(1, t_dim, 1) * torch.log(
                        torch.nn.functional.softplus(mass) + 1e-12)
                prob[lo:hi, h] = (
                    sc.log_softmax(dim=-1) + lm[:, h]
                ).exp_().sum(dim=1)
                del sc, a0, a1
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    prob = prob.contiguous()

    counts = torch.empty(q_count, num_kv_heads, dtype=torch.int32,
                         device=vectors.device)
    tokens = torch.empty_like(counts)
    _run_streaming_select(prob, causal_sizes, bitmask, counts, tokens,
                          beta=beta, target_tokens=target_tokens,
                          b_dim=c_dim, tile=tile, iters=iters,
                          g_union=g_union, qmask=qmask)
    return counts, tokens

if triton is not None:

    @triton.jit
    def _flat_scores_kernel(
        prob_ptr,      # [Q, Hkv, C] fp32 out (mixture probability)
        q_ptr,         # [Q, Hkv, T, 2, DF] fp32/bf16 routing vectors
        cb_ptr,        # [Hkv, 2, C, DF] cluster descriptors
        logmix_ptr,    # [Q, Hkv, T] fp32
        inv_temperature,
        stride_p0, stride_p1,
        stride_q0, stride_q1, stride_q2, stride_q3, stride_q4,
        stride_c0, stride_c1, stride_c2, stride_c3,
        stride_l0, stride_l1, stride_l2,
        T: tl.constexpr,
        C: tl.constexpr,
        DF: tl.constexpr,
        TILE_C: tl.constexpr,
    ):
        """Mixture score over a flat cluster set, without materialising the
        component axis.

        One program owns one (query, KV head).  It streams the C axis twice,
        once for the per-component log-softmax normaliser (online max/sum)
        and once to accumulate sum_t exp(s[t,c] - lse_t + logmix_t), and
        writes only the [C] result.  The scoring math is that of
        theta_select_bitmask_flat."""
        query = tl.program_id(0)
        kv_head = tl.program_id(1)
        offs_t = tl.arange(0, T)
        offs_d = tl.arange(0, DF)
        offs_c = tl.arange(0, TILE_C)

        qbase = query.to(tl.int64) * stride_q0 + kv_head * stride_q1
        q0 = tl.load(q_ptr + qbase + offs_t[:, None] * stride_q2
                     + 0 * stride_q3 + offs_d[None, :] * stride_q4).to(
            tl.float32)
        q1 = tl.load(q_ptr + qbase + offs_t[:, None] * stride_q2
                     + 1 * stride_q3 + offs_d[None, :] * stride_q4).to(
            tl.float32)
        logmix = tl.load(logmix_ptr + query.to(tl.int64) * stride_l0
                         + kv_head * stride_l1 + offs_t * stride_l2).to(
            tl.float32)

        m_t = tl.full([T], -float("inf"), tl.float32)
        l_t = tl.zeros([T], tl.float32)
        for c0 in tl.range(0, C, TILE_C):
            cols = c0 + offs_c
            cb0 = tl.load(cb_ptr + kv_head * stride_c0 + 0 * stride_c1
                          + cols[:, None] * stride_c2
                          + offs_d[None, :] * stride_c3).to(tl.float32)
            cb1 = tl.load(cb_ptr + kv_head * stride_c0 + 1 * stride_c1
                          + cols[:, None] * stride_c2
                          + offs_d[None, :] * stride_c3).to(tl.float32)
            sc = (tl.dot(q0, tl.trans(cb0), input_precision="ieee")
                  + tl.dot(q1, tl.trans(cb1), input_precision="ieee"))
            sc = sc * inv_temperature
            m_new = tl.maximum(m_t, tl.max(sc, axis=1))
            l_t = l_t * tl.exp(m_t - m_new) + tl.sum(
                tl.exp(sc - m_new[:, None]), axis=1)
            m_t = m_new
        lse = m_t + tl.log(l_t)

        pbase = query.to(tl.int64) * stride_p0 + kv_head * stride_p1
        for c0 in tl.range(0, C, TILE_C):
            cols = c0 + offs_c
            cb0 = tl.load(cb_ptr + kv_head * stride_c0 + 0 * stride_c1
                          + cols[:, None] * stride_c2
                          + offs_d[None, :] * stride_c3).to(tl.float32)
            cb1 = tl.load(cb_ptr + kv_head * stride_c0 + 1 * stride_c1
                          + cols[:, None] * stride_c2
                          + offs_d[None, :] * stride_c3).to(tl.float32)
            sc = (tl.dot(q0, tl.trans(cb0), input_precision="ieee")
                  + tl.dot(q1, tl.trans(cb1), input_precision="ieee"))
            sc = sc * inv_temperature
            prob = tl.sum(tl.exp(sc - lse[:, None] + logmix[:, None]), axis=0)
            tl.store(prob_ptr + pbase + cols, prob)


def flat_mixture_scores(
    vectors: torch.Tensor,    # [Q, Hq, R, I]
    logmix: torch.Tensor,     # [Q, Hq, R]
    codebooks: torch.Tensor,  # [Hkv, 2, C, DF]
    *,
    temperature: float,
    tile_c: int = 128,
    num_warps: int = 4,
    num_stages: int = 1,
) -> torch.Tensor:
    """[Q, Hkv, C] mixture probability, computed without the T axis."""
    q_count, num_q_heads, r_dim, index_dim = vectors.shape
    num_kv_heads, two, c_dim, df = codebooks.shape
    assert two == 2 and index_dim == 2 * df
    t_dim = (num_q_heads // num_kv_heads) * r_dim
    q5 = vectors.view(q_count, num_kv_heads, t_dim, 2, df)
    lm = logmix.view(q_count, num_kv_heads, t_dim).contiguous()
    prob = torch.empty(q_count, num_kv_heads, c_dim, dtype=torch.float32,
                       device=vectors.device)
    _flat_scores_kernel[(q_count, num_kv_heads)](
        prob, q5, codebooks, lm,
        1.0 / float(temperature),
        prob.stride(0), prob.stride(1),
        q5.stride(0), q5.stride(1), q5.stride(2), q5.stride(3), q5.stride(4),
        codebooks.stride(0), codebooks.stride(1), codebooks.stride(2),
        codebooks.stride(3),
        lm.stride(0), lm.stride(1), lm.stride(2),
        T=t_dim, C=c_dim, DF=df, TILE_C=tile_c,
        num_warps=num_warps, num_stages=num_stages,
    )
    return prob

if triton is not None:

    @triton.jit
    def _log_nonneg(x):
        """log(x) for x >= 0, the descriptor readout for a non-negative psi.

        The general readout is log(softplus(x)), which costs an exp on every
        (row, cluster) entry.  With a non-negative psi (softplus, exp) both
        phi(u) and g_c are non-negative, so their inner product is too, and
        for the large values it takes in practice softplus(x) = x to fp32
        precision.  The general path remains for a linear psi, whose sums can
        be negative.
        """
        return tl.log(tl.maximum(x, 0.0) + 1e-12)

    @triton.jit
    def _log_softplus(x):
        """log(softplus(x) + 1e-12), the descriptor readout.

        softplus rather than a clamp because with a linear psi the cluster
        sum can be negative, and log(max(x, eps)) would put every negative
        pair on one floor with no ordering and no gradient.

        Written as max(x,0) + log(1 + exp(-|x|)) rather than log(1 + exp(x)):
        the direct form overflows to inf for x above ~88 in fp32, and the
        cluster sums here are sums over many keys, so large values are common.
        """
        sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
        return tl.log(sp + 1e-12)

    @triton.jit
    def _flat_scores_qtiled_kernel(
        prob_ptr,      # [Q, Hkv, C] fp32 out
        q_ptr,         # [Q, Hkv, T, 2, DF]
        cb_ptr,        # [Hkv, 2, C, DF]
        logmix_ptr,    # [Q, Hkv, T] fp32
        gb_ptr,        # [Hkv, 2, G, DF] group descriptors (GNORM)
        # The cluster descriptor.  Unused pointers are passed as the codebook
        # so Triton always has something to specialise on; DESC gates the code.
        gd_ptr,        # [Hkv, C, DM]   cluster descriptors g_c
        phi_ptr,       # [Q, Hkv, T, DM] query descriptors phi(u)
        gam_ptr,       # [Hkv, T]        gamma, already expanded per component
        inv_temperature,
        num_queries,
        stride_p0, stride_p1,
        stride_q0, stride_q1, stride_q2, stride_q3, stride_q4,
        stride_c0, stride_c1, stride_c2, stride_c3,
        stride_l0, stride_l1, stride_l2,
        stride_g0, stride_g1, stride_g2, stride_g3,
        stride_d0, stride_d1, stride_d2,
        stride_f0, stride_f1, stride_f2, stride_f3,
        stride_m0, stride_m1,
        BLOCK_Q: tl.constexpr,
        T: tl.constexpr,
        C: tl.constexpr,
        DF: tl.constexpr,
        TILE_C: tl.constexpr,
        BF16_DOT: tl.constexpr,
        GNORM: tl.constexpr,
        G: tl.constexpr,
        WIDE: tl.constexpr,
        DESC: tl.constexpr,
        DM: tl.constexpr,
        DESC_NONNEG: tl.constexpr,
        LOG_SPAN,
    ):
        """Flat mixture scores, tiled over queries so the dot is a GEMM.

        One program owns BLOCK_Q queries, so the dot is
        [BLOCK_Q*T, DF] x [DF, TILE_C], a shape suited to tensor cores,
        without materialising the [Q, Hkv, T, C] score volume.

        Two streams over C: one for the per-(query, component) log-softmax
        normaliser (online max/sum), one to accumulate
        sum_t exp(s - lse + logmix) into the [BLOCK_Q, TILE_C] output."""
        qb = tl.program_id(0)
        kv_head = tl.program_id(1)
        rows: tl.constexpr = BLOCK_Q * T
        offs_r = tl.arange(0, rows)
        offs_d = tl.arange(0, DF)
        offs_c = tl.arange(0, TILE_C)
        q_in_tile = offs_r // T
        t_idx = offs_r % T
        qi = qb * BLOCK_Q + q_in_tile
        live = qi < num_queries
        qi_safe = tl.where(live, qi, 0)
        # int32 offsets are much cheaper than 64-bit ones; WIDE is set from
        # the largest real offset when a chunk needs 64-bit addressing.
        if WIDE:
            qi_safe = qi_safe.to(tl.int64)

        qbase = qi_safe * stride_q0 + kv_head * stride_q1 \
            + t_idx * stride_q2
        # Keeping the dot operands in bfloat16 halves what the C loop holds
        # in shared memory and puts the dot on the bf16 tensor cores rather
        # than tf32, at twice the rate.
        q0 = tl.load(q_ptr + qbase[:, None] + 0 * stride_q3
                     + offs_d[None, :] * stride_q4,
                     mask=live[:, None], other=0.0)
        q1 = tl.load(q_ptr + qbase[:, None] + 1 * stride_q3
                     + offs_d[None, :] * stride_q4,
                     mask=live[:, None], other=0.0)
        if not BF16_DOT:
            q0 = q0.to(tl.float32)
            q1 = q1.to(tl.float32)
        logmix = tl.load(
            logmix_ptr + qi_safe * stride_l0
            + kv_head * stride_l1 + t_idx * stride_l2,
            mask=live, other=0.0).to(tl.float32)

        if DESC:
            # phi(u) is per (query, component) like q0/q1, so it is loaded
            # once here and reused by both passes over C.
            offs_m = tl.arange(0, DM)
            phi = tl.load(
                phi_ptr + qi_safe[:, None] * stride_f0
                + kv_head * stride_f1 + t_idx[:, None] * stride_f2
                + offs_m[None, :] * stride_f3,
                mask=live[:, None], other=0.0)
            if not BF16_DOT:
                phi = phi.to(tl.float32)
            gam = tl.load(gam_ptr + kv_head * stride_m0 + t_idx * stride_m1
                          ).to(tl.float32)

        if GNORM:
            # The normaliser sweep is a second full pass over C that only
            # produces one scalar per (query, component).  Taking it from the
            # coarse level instead (logsumexp over the C/span group
            # descriptors, plus log(span) for the members each one stands
            # for) halves the work.  Every cluster is still scored; only the
            # relative weight of the components is approximated.
            gb0 = tl.load(gb_ptr + kv_head * stride_g0 + 0 * stride_g1
                          + tl.arange(0, G)[:, None] * stride_g2
                          + offs_d[None, :] * stride_g3)
            gb1 = tl.load(gb_ptr + kv_head * stride_g0 + 1 * stride_g1
                          + tl.arange(0, G)[:, None] * stride_g2
                          + offs_d[None, :] * stride_g3)
            if not BF16_DOT:
                gb0 = gb0.to(tl.float32)
                gb1 = gb1.to(tl.float32)
            sg = (tl.dot(q0, tl.trans(gb0))
                  + tl.dot(q1, tl.trans(gb1))) * inv_temperature
            gm = tl.max(sg, axis=1)
            lse = gm + tl.log(tl.sum(tl.exp(sg - gm[:, None]), axis=1)) \
                + LOG_SPAN
        else:
            m_r = tl.full([rows], -float("inf"), tl.float32)
            l_r = tl.zeros([rows], tl.float32)
            for c0 in tl.range(0, C, TILE_C):
                cols = c0 + offs_c
                cb0 = tl.load(cb_ptr + kv_head * stride_c0 + 0 * stride_c1
                              + cols[:, None] * stride_c2
                              + offs_d[None, :] * stride_c3)
                cb1 = tl.load(cb_ptr + kv_head * stride_c0 + 1 * stride_c1
                              + cols[:, None] * stride_c2
                              + offs_d[None, :] * stride_c3)
                if not BF16_DOT:
                    cb0 = cb0.to(tl.float32)
                    cb1 = cb1.to(tl.float32)
                # Tensor-core dots: exact fp32 ("ieee") runs on CUDA cores
                # and is several times slower, and a budget-thresholded
                # selection does not need that exactness.
                sc = (tl.dot(q0, tl.trans(cb0)) + tl.dot(q1, tl.trans(cb1)))
                sc = sc * inv_temperature
                if DESC:
                    gd = tl.load(gd_ptr + kv_head * stride_d0
                                 + cols[:, None] * stride_d1
                                 + tl.arange(0, DM)[None, :] * stride_d2)
                    if not BF16_DOT:
                        gd = gd.to(tl.float32)
                    mass = tl.dot(phi, tl.trans(gd))
                    if DESC_NONNEG:
                        sc += gam[:, None] * _log_nonneg(mass)
                    else:
                        sc += gam[:, None] * _log_softplus(mass)
                m_new = tl.maximum(m_r, tl.max(sc, axis=1))
                l_r = l_r * tl.exp(m_r - m_new) + tl.sum(
                    tl.exp(sc - m_new[:, None]), axis=1)
                m_r = m_new
            lse = m_r + tl.log(l_r)

        out_q = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
        out_live = out_q < num_queries
        if WIDE:
            out_q = out_q.to(tl.int64)
        for c0 in tl.range(0, C, TILE_C):
            cols = c0 + offs_c
            cb0 = tl.load(cb_ptr + kv_head * stride_c0 + 0 * stride_c1
                          + cols[:, None] * stride_c2
                          + offs_d[None, :] * stride_c3)
            cb1 = tl.load(cb_ptr + kv_head * stride_c0 + 1 * stride_c1
                          + cols[:, None] * stride_c2
                          + offs_d[None, :] * stride_c3)
            if not BF16_DOT:
                cb0 = cb0.to(tl.float32)
                cb1 = cb1.to(tl.float32)
            sc = (tl.dot(q0, tl.trans(cb0)) + tl.dot(q1, tl.trans(cb1)))
            sc = sc * inv_temperature
            if DESC:
                gd = tl.load(gd_ptr + kv_head * stride_d0
                             + cols[:, None] * stride_d1
                             + tl.arange(0, DM)[None, :] * stride_d2)
                if not BF16_DOT:
                    gd = gd.to(tl.float32)
                mass = tl.dot(phi, tl.trans(gd))
                if DESC_NONNEG:
                    sc += gam[:, None] * _log_nonneg(mass)
                else:
                    sc += gam[:, None] * _log_softplus(mass)
            w = tl.exp(sc - lse[:, None] + logmix[:, None])
            w = tl.where(live[:, None], w, 0.0)
            pr = tl.sum(tl.reshape(w, [BLOCK_Q, T, TILE_C]), axis=1)
            tl.store(
                prob_ptr + out_q[:, None] * stride_p0
                + kv_head * stride_p1 + cols[None, :],
                pr, mask=out_live[:, None],
            )


def flat_mixture_scores_qtiled(
    vectors: torch.Tensor,    # [Q, Hq, R, I]
    logmix: torch.Tensor,     # [Q, Hq, R]
    codebooks: torch.Tensor,  # [Hkv, 2, C, DF]
    *,
    temperature: float,
    group_desc: torch.Tensor | None = None,  # [Hkv, 2, G, DF] (SEMBLO_SCORE_GNORM)
    cluster_descriptors: torch.Tensor | None = None,  # [Hkv, C, DM]
    query_descriptors: torch.Tensor | None = None,    # [Q, Hq, R, DM]
    descriptor_gamma: torch.Tensor | None = None,     # [Hq]
    block_q: int = 8,
    tile_c: int = 128,
    num_warps: int = 8,
    num_stages: int = 1,
) -> torch.Tensor:
    """[Q, Hkv, C] mixture probability, query-tiled so the dots are GEMMs."""
    # Launch shape.  block_q and num_stages cannot change any output (block_q
    # only groups rows, num_stages only prefetches); tile_c changes the
    # online-softmax accumulation order but, in practice, not the selection.
    # A wide query tile fills the dot's M dimension, and a narrow cluster
    # tile keeps that in registers; larger tiles run out of registers.  The
    # descriptor path holds two more tiles (phi across the whole C loop, and
    # the mass inside it), so it needs a smaller shape, and a shape that
    # spills registers is an order of magnitude slower.
    # SEMBLO_SCORE_BLOCK_Q, _TILE_C, _WARPS and _STAGES override the defaults.
    _desc = cluster_descriptors is not None
    block_q = int(os.environ.get("SEMBLO_SCORE_BLOCK_Q", 8 if _desc else 16))
    tile_c = int(os.environ.get("SEMBLO_SCORE_TILE_C", 32 if _desc else 128))
    num_warps = int(os.environ.get("SEMBLO_SCORE_WARPS", 8 if _desc else 16))
    num_stages = int(os.environ.get("SEMBLO_SCORE_STAGES", num_stages))
    q_count, num_q_heads, r_dim, index_dim = vectors.shape
    num_kv_heads, two, c_dim, df = codebooks.shape
    assert two == 2 and index_dim == 2 * df
    t_dim = (num_q_heads // num_kv_heads) * r_dim
    q5 = vectors.view(q_count, num_kv_heads, t_dim, 2, df)
    lm = logmix.view(q_count, num_kv_heads, t_dim).contiguous()
    prob = torch.empty(q_count, num_kv_heads, c_dim, dtype=torch.float32,
                       device=vectors.device)
    import math
    gnorm = (group_desc is not None
             and os.environ.get("SEMBLO_SCORE_GNORM", "0") == "1")
    gb = group_desc if gnorm else codebooks
    n_groups = int(gb.shape[2])

    desc = _desc
    if desc and gnorm:
        # GNORM takes the log-softmax normaliser from the coarse level, and
        # the group descriptors carry no descriptor term, so the normaliser
        # would not match the scores it normalises.
        raise ValueError(
            "SEMBLO_SCORE_GNORM approximates the normaliser from group "
            "descriptors, which the cluster descriptor does not have; "
            "run one or the other")
    if desc:
        gd = cluster_descriptors
        # [Q, Hkv, T, DM], the same (head, sub-query) flattening the routing
        # vectors use, so component t lines up with q5[:, h, t].
        phi = query_descriptors.reshape(
            q_count, num_kv_heads, t_dim, -1).contiguous()
        # gamma is per query head; a row of the kernel is a (query, component)
        # pair, and components run heads_per_kv x r_dim, so it is expanded here
        # rather than indexed with a division inside the loop.
        gam = descriptor_gamma.view(num_kv_heads, num_q_heads // num_kv_heads,
                                    1).expand(
            num_kv_heads, num_q_heads // num_kv_heads, r_dim
        ).reshape(num_kv_heads, t_dim).contiguous().float()
        dm = int(gd.shape[-1])
        if gd.dtype != phi.dtype:
            phi = phi.to(gd.dtype)
        # Which readout the kernel may use.  A non-negative psi (softplus,
        # exp) makes every entry of g_c and phi(u) non-negative, so their
        # inner product is too and softplus can be dropped.  Checked on the
        # tensors rather than assumed from the config.
        nonneg = bool(gd.min() >= 0) and bool(phi.min() >= 0)
    else:
        gd = codebooks
        phi = q5
        gam = lm
        dm = 1
        nonneg = False

    grid = ((q_count + block_q - 1) // block_q, num_kv_heads)
    _flat_scores_qtiled_kernel[grid](
        prob, q5, codebooks, lm, gb, gd, phi, gam,
        1.0 / float(temperature), q_count,
        prob.stride(0), prob.stride(1),
        q5.stride(0), q5.stride(1), q5.stride(2), q5.stride(3), q5.stride(4),
        codebooks.stride(0), codebooks.stride(1), codebooks.stride(2),
        codebooks.stride(3),
        lm.stride(0), lm.stride(1), lm.stride(2),
        gb.stride(0), gb.stride(1), gb.stride(2), gb.stride(3),
        gd.stride(0), gd.stride(-2), gd.stride(-1),
        phi.stride(0), phi.stride(1), phi.stride(2), phi.stride(-1),
        gam.stride(0), gam.stride(-1),
        BLOCK_Q=block_q, T=t_dim, C=c_dim, DF=df, TILE_C=tile_c,
        # bf16 dot operands: `vectors` and `codebooks` are stored in bf16,
        # so widening to fp32 adds no information and moves the dot onto
        # tf32 at half the rate.  Not bit-identical to the fp32 form: a
        # marginal cluster can occasionally flip.
        BF16_DOT=os.environ.get("SEMBLO_SCORE_BF16", "1") == "1",
        GNORM=gnorm, G=n_groups if gnorm else 1,
        WIDE=(q_count * max(q5.stride(0), prob.stride(0), lm.stride(0))
              >= (1 << 31)),
        DESC=desc, DM=dm, DESC_NONNEG=nonneg,
        LOG_SPAN=math.log(c_dim / n_groups) if gnorm else 0.0,
        num_warps=num_warps, num_stages=num_stages,
    )
    return prob
