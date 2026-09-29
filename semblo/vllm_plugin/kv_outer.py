"""KV-outer attention: the reverse index, and the log-sum-exp combine.

The attention itself is FA3's: one varlen sequence per (block, KV head),
whose query rows are the queries that selected that block.  Each query is
charged only its own selection, not the union of a query tile's selections.

The two pieces around the attention call are Triton kernels:

  _rev_count_kernel / _qrank_kernel / _rev_fill_kernel
      selection bitmask -> CSR of query ids per (KV head, block), and the
      inverse list of entry ids per query.  The boolean matrix is never
      materialised (one 32-bit word carries 32 blocks), and the query lists
      come out sorted and deterministic.

  _combine_kernel / _combine_grouped_kernel
      each query's partials -> one output, by online log-sum-exp.  A query
      holds about budget / block size partials.

In this module "cluster" and "block" are the same thing: one k-means cluster
of prompt keys.
"""

from __future__ import annotations

import contextlib
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
    def _rev_count_kernel(
        counts_ptr,    # [NBQ, H, C] int32 out: this query block's share
        qmask_ptr,     # [Q, H, W] int32
        live_ptr,      # [H, C] int32: 1 if the cluster contributes a sequence
        num_queries,
        n_words,
        stride_q, stride_h,
        BQ: tl.constexpr,
        WIDE: tl.constexpr,
    ):
        """Queries per cluster, counted per query block.

        Splitting the query axis into blocks and prefix-summing the per-block
        counts afterwards builds the CSR in parallel and deterministically,
        without atomics.
        """
        word = tl.program_id(0)
        head = tl.program_id(1)
        qb = tl.program_id(2)
        heads = tl.num_programs(1)
        lane = tl.arange(0, 32)
        cluster = word * 32 + lane
        live = tl.load(live_ptr + head * (n_words * 32) + cluster)
        qs = qb * BQ + tl.arange(0, BQ)
        ok = qs < num_queries
        qw = qs.to(tl.int64) if WIDE else qs
        w = tl.load(qmask_ptr + qw * stride_q + head * stride_h + word,
                    mask=ok, other=0)
        bits = ((w[:, None] >> lane[None, :]) & 1) * (live[None, :] > 0)
        tl.store(counts_ptr + (qb * heads + head) * (n_words * 32) + cluster,
                 tl.sum(bits, axis=0))

    @triton.jit
    def _popc(x):
        """Population count of an int32, without a 32-wide bit axis.

        The bit-expansion form ((x >> lane) & 1) needs a [.., .., 32] tensor
        and runs out of registers.  Every mask below clears the bits the
        arithmetic shift would sign-extend into, so a negative word (bit 31
        set) is counted correctly.
        """
        x = x - ((x >> 1) & 0x55555555)
        x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
        x = (x + (x >> 4)) & 0x0f0f0f0f
        x = x + (x >> 8)
        x = x + (x >> 16)
        return x & 0x3f

    @triton.jit
    def _qrank_kernel(
        pref_ptr,      # [Q, H, W] int32 out: set bits in words before this one
        perq_ptr,      # [Q, H] int32 out: this query's selected cluster count
        qmask_ptr,     # [Q, H, W] int32
        livew_ptr,     # [H, W] int32: live clusters, packed like qmask
        num_queries,
        stride_q, stride_h,
        BQ: tl.constexpr,
        W: tl.constexpr,
        WIDE: tl.constexpr,
    ):
        """The reverse index, read the other way round.

        build_reverse_index lists the queries of each cluster; the log-sum-exp
        merge needs the clusters of each query.  Both come from the same
        bitmask: a query's clusters, in cluster order, are its set bits, so
        the rank of cluster c within query q is the number of set bits below
        c.  This kernel materialises that prefix popcount over the words.
        """
        qb = tl.program_id(0)
        head = tl.program_id(1)
        heads = tl.num_programs(1)
        qs = qb * BQ + tl.arange(0, BQ)
        ok = qs < num_queries
        ws = tl.arange(0, W)
        qw = qs.to(tl.int64) if WIDE else qs
        w = tl.load(qmask_ptr + qw[:, None] * stride_q
                    + head * stride_h + ws[None, :],
                    mask=ok[:, None], other=0)
        lw = tl.load(livew_ptr + head * W + ws)
        pop = _popc(w & lw[None, :])
        pref = tl.cumsum(pop, axis=1) - pop
        tl.store(pref_ptr + (qw[:, None] * heads + head) * W
                 + ws[None, :], pref, mask=ok[:, None])
        tl.store(perq_ptr + qw * heads + head, tl.sum(pop, axis=1), mask=ok)

    @triton.jit
    def _rev_fill_kernel(
        qvals_ptr,     # [nnz] int32 out: query ids, cluster-major
        order_ptr,     # [nnz] int32 out: entry ids, query-major
        pref_ptr,      # [Q, H, W] int32 (from _qrank_kernel)
        startq_ptr,    # [H, Q] int32: this query's first query-major slot
        base_ptr,      # [NBQ, H, C] int32: run start for this query block
        qmask_ptr, live_ptr,
        num_queries,
        n_words,
        stride_q, stride_h,
        BQ: tl.constexpr,
        W: tl.constexpr,
        WIDE: tl.constexpr,
    ):
        word = tl.program_id(0)
        head = tl.program_id(1)
        qb = tl.program_id(2)
        heads = tl.num_programs(1)
        lane = tl.arange(0, 32)
        cluster = word * 32 + lane
        live = tl.load(live_ptr + head * (n_words * 32) + cluster)
        # Where this block's entries start inside each cluster's run: the
        # cluster's own start plus what the earlier query blocks contributed.
        off = tl.load(base_ptr + (qb * heads + head) * (n_words * 32)
                      + cluster)
        qs = qb * BQ + tl.arange(0, BQ)
        ok = qs < num_queries
        qw = qs.to(tl.int64) if WIDE else qs
        w = tl.load(qmask_ptr + qw * stride_q + head * stride_h + word,
                    mask=ok, other=0)
        bits = ((w[:, None] >> lane[None, :]) & 1) * (live[None, :] > 0)
        # Exclusive prefix within this block, so writes stay in query order.
        prefix = tl.cumsum(bits, axis=0) - bits
        pos = off[None, :] + prefix
        keep = (bits > 0) & ok[:, None]
        tl.store(qvals_ptr + (pos.to(tl.int64) if WIDE else pos),
                 tl.broadcast_to(qs[:, None], [BQ, 32]), mask=keep)
        # The same entry, written again at its query-major slot: this
        # query's base, plus the clusters it selected in earlier words,
        # plus the ones below this lane in this word.
        within = tl.cumsum(bits, axis=1) - bits
        pr = tl.load(pref_ptr + (qw * heads + head) * W + word,
                     mask=ok, other=0)
        sq = tl.load(startq_ptr + head * num_queries + qs, mask=ok, other=0)
        slot = sq[:, None] + pr[:, None] + within
        tl.store(order_ptr + (slot.to(tl.int64) if WIDE else slot),
                 pos.to(tl.int32), mask=keep)

    @triton.jit
    def _combine_kernel(
        out_ptr,       # [Q, Hq, D] out (bf16)
        lse_out_ptr,   # [Q, Hq] float32 out: the merged log-sum-exp
        o_ptr,         # [nnz, QPK, D] partials (bf16), entry-major
        lse_ptr,       # [nnz * QPK] float32
        order_ptr,     # [nnz] int32: entry ids sorted by query (global ids)
        start_ptr,     # [Q] int32: where this query's entries begin
        cnt_ptr,       # [Q] int32: how many entries this query has
        lo_ptr,        # [Q, Hq, D] (bf16) the chunk's own causal window (HAS_LOCAL)
        ll_ptr,        # [Q, Hq] float32 its log-sum-exp (HAS_LOCAL)
        kv_head, entry_base, stride_out_q, stride_out_h,
        stride_lse_q, stride_lse_h,
        stride_lo_q, stride_lo_h, stride_ll_q, stride_ll_h,
        QPK: tl.constexpr,
        D: tl.constexpr,
        WANT_LSE: tl.constexpr,
        WIDE: tl.constexpr,
        HAS_LOCAL: tl.constexpr,
    ):
        q = tl.program_id(0)
        j = tl.program_id(1)
        s = tl.load(start_ptr + q)
        n = tl.load(cnt_ptr + q)
        d = tl.arange(0, D)
        acc = tl.zeros([D], tl.float32)
        m = float("-inf")
        denom = 0.0
        if HAS_LOCAL:
            # The chunk's own causal window is one more partial, folded in
            # first so the combine order is fixed: local, then the entries in
            # query order.
            qw0 = q.to(tl.int64) if WIDE else q
            hh = kv_head * QPK + j
            m = tl.load(ll_ptr + qw0 * stride_ll_q + hh * stride_ll_h)
            acc = tl.load(lo_ptr + qw0 * stride_lo_q + hh * stride_lo_h + d
                          ).to(tl.float32)
            denom = 1.0
        i = 0
        while i < n:
            e = tl.load(order_ptr + s + i) - entry_base
            if WIDE:
                e = e.to(tl.int64)
            le = tl.load(lse_ptr + e * QPK + j)
            o = tl.load(o_ptr + (e * QPK + j) * D + d).to(tl.float32)
            new_m = tl.maximum(m, le)
            scale = tl.exp(m - new_m)
            w = tl.exp(le - new_m)
            acc = acc * scale + o * w
            denom = denom * scale + w
            m = new_m
            i += 1
        res = acc / tl.maximum(denom, 1e-30)
        qw = q.to(tl.int64) if WIDE else q
        tl.store(out_ptr + qw * stride_out_q
                 + (kv_head * QPK + j) * stride_out_h + d,
                 res.to(out_ptr.dtype.element_ty),
                 mask=(n > 0) | HAS_LOCAL)
        if WANT_LSE:
            # A query with no partials has an empty sparse set; -inf makes a
            # downstream merge drop it.
            tl.store(lse_out_ptr + qw * stride_lse_q
                     + (kv_head * QPK + j) * stride_lse_h,
                     tl.where((n > 0) | HAS_LOCAL, m + tl.log(denom),
                              float("-inf")))



# Profiler labels for the three FlashAttention-3 call sites
# (SEMBLO_PROFILE_ATTEND_PARTS=1).  Off by default because record_function adds
# a small per-call cost.
_ATTEND_PARTS = os.environ.get("SEMBLO_PROFILE_ATTEND_PARTS") == "1"


def _attend_part(name: str):
    if not _ATTEND_PARTS:
        return contextlib.nullcontext()
    return torch.profiler.record_function(name)

def build_reverse_index(qmask, live, *, num_queries, block_q=64, capacity=None):
    """Selection bitmask -> both directions of the query/cluster relation.

    qmask: [Q, H, C/32] int32, live: [H, C] int32 (1 = cluster is a sequence).
    Returns (qvals, counts, indptr, order, start_q, per_q):
      qvals [nnz]    query ids, cluster-major, query-sorted within a cluster
      counts [H, C]  queries per cluster;  indptr [H, C]  its run start
      order [nnz]    entry ids, query-major (the inverse of the above)
      start_q [H, Q] where a query's run of entry ids begins in `order`
      per_q [Q, H]   how many entries a query has
      pref [Q, H, W] entries in the clusters below each 32-cluster word, so
                     that a batch of clusters starting on a word boundary
                     has its per-query offset directly

    ``capacity``: the number of entries to allocate for ``qvals`` and
    ``order``.  With it, the entry count is not read to the host: a query
    selects at most C clusters, so ``num_queries * H * C`` entries hold any
    selection, and the caller slices per head with bounds it reads in its own
    sync.  Only the first nnz entries are then meaningful.  Without it (None),
    the count is read to size the buffers exactly, at the cost of one host
    sync.
    """
    if triton is None:
        raise RuntimeError("build_reverse_index needs triton")
    q_all, num_heads, n_words = qmask.shape
    c_dim = n_words * 32
    device = qmask.device
    # int32 addressing unless a chunk indexes past 2^31.
    wide = (num_queries * max(qmask.stride(0), num_heads * n_words)
            >= (1 << 31))
    n_qb = (num_queries + block_q - 1) // block_q
    grid = (n_words, num_heads, n_qb)
    # [NBQ, H, C]: how many of this query block's queries picked each cluster
    block_counts = torch.empty(n_qb, num_heads, c_dim, dtype=torch.int32,
                               device=device)
    _rev_count_kernel[grid](
        block_counts, qmask, live, num_queries, n_words,
        qmask.stride(0), qmask.stride(1), BQ=block_q, WIDE=wide, num_warps=4)
    counts = block_counts.sum(dim=0)
    flat = counts.reshape(-1)
    indptr = (torch.cumsum(flat, 0) - flat).to(torch.int32).view_as(counts)
    # The entry count: a host read only when no capacity was given.
    nnz = int(flat.sum()) if capacity is None else int(capacity)
    # Each block's start inside each cluster's run: the cluster's start plus
    # what the earlier blocks contributed.
    block_base = (indptr[None].to(torch.int64)
                  + (block_counts.cumsum(dim=0) - block_counts)
                  ).to(torch.int32).contiguous()

    # `live`, packed the way qmask is, so the rank kernel can mask in one AND.
    shifts = torch.arange(32, device=device, dtype=torch.int32)
    livew = (((live.view(num_heads, n_words, 32) > 0).to(torch.int32)
              << shifts).sum(dim=-1)).to(torch.int32).contiguous()
    pref = torch.empty(num_queries, num_heads, n_words, dtype=torch.int32,
                       device=device)
    per_q = torch.empty(num_queries, num_heads, dtype=torch.int32,
                        device=device)
    rank_bq = 32
    _qrank_kernel[((num_queries + rank_bq - 1) // rank_bq, num_heads)](
        pref, per_q, qmask, livew, num_queries,
        qmask.stride(0), qmask.stride(1),
        BQ=rank_bq, W=n_words, WIDE=wide, num_warps=4)
    # A query's slots are global, so `order` needs no per-head slicing.  The
    # head's first slot is the same head_lo the entry ids are based on.
    pqt = per_q.t().contiguous().to(torch.int64)
    start_q = (indptr[:, 0:1].to(torch.int64)
               + (pqt.cumsum(dim=1) - pqt)).to(torch.int32)

    qvals = torch.empty(max(nnz, 1), dtype=torch.int32, device=device)
    order = torch.empty(max(nnz, 1), dtype=torch.int32, device=device)
    # The fill also indexes by entry, so its bound is nnz as well.
    _rev_fill_kernel[grid](
        qvals, order, pref, start_q, block_base, qmask, live, num_queries,
        n_words, qmask.stride(0), qmask.stride(1),
        BQ=block_q, W=n_words, WIDE=wide or nnz >= (1 << 31), num_warps=4)
    return qvals[:nnz], counts, indptr, order[:nnz], start_q, per_q, pref


def _cluster_batches(c_dim: int, cbatch: int):
    """Cluster ranges the body processes in one FA3 call.

    Edges are multiples of 32 so that `pref` (the per-query count of entries
    below each 32-cluster word, built by the reverse index) gives each
    batch's per-query offset directly.
    """
    if cbatch <= 0 or cbatch >= c_dim:
        return [(0, c_dim)]
    return [(c0, min(c0 + cbatch, c_dim)) for c0 in range(0, c_dim, cbatch)]


def _merge_into(out, lse, part_out, part_lse, head_slice):
    """Fold one batch's (output, log-sum-exp) into the running pair.

    Both sides are rescaled to a common maximum and renormalised.  A batch
    that contributed nothing to a query carries -inf, which drops out.
    """
    ls = lse[:, head_slice]
    lp = part_lse[:, head_slice]
    live_s = torch.isfinite(ls)
    live_p = torch.isfinite(lp)
    m = torch.where(live_s, torch.maximum(ls, torch.where(live_p, lp, ls)),
                    torch.where(live_p, lp, ls))
    ws = torch.where(live_s, (ls - m).exp(), torch.zeros_like(ls))
    wp = torch.where(live_p, (lp - m).exp(), torch.zeros_like(lp))
    denom = (ws + wp).clamp_min(1e-30)
    merged = (out[:, head_slice, :].float() * ws[..., None]
              + part_out[:, head_slice, :].float() * wp[..., None]) / \
        denom[..., None]
    out[:, head_slice, :] = merged.to(out.dtype)
    lse[:, head_slice] = torch.where(
        live_s | live_p, m + denom.log(), torch.full_like(m, float("-inf")))


if triton is not None:

    @triton.jit
    def _combine_grouped_kernel(
        out_ptr, lse_out_ptr, o_ptr, lse_ptr, order_ptr, start_ptr, cnt_ptr,
        lo_ptr, ll_ptr,
        kv_head, entry_base, stride_out_q, stride_out_h,
        stride_lse_q, stride_lse_h,
        stride_lo_q, stride_lo_h, stride_ll_q, stride_ll_h,
        QPK: tl.constexpr,
        D: tl.constexpr,
        WANT_LSE: tl.constexpr,
        WIDE: tl.constexpr,
        HAS_LOCAL: tl.constexpr,
    ):
        """The same merge as `_combine_kernel`, one program per query.

        An entry's partials for the QPK heads of a KV group are contiguous, so
        this loads them as one [QPK, D] block and walks a query's entries once
        instead of QPK times.  The accumulation order per head is the same as
        in `_combine_kernel`, so the result is bit-identical.
        """
        q = tl.program_id(0)
        s = tl.load(start_ptr + q)
        n = tl.load(cnt_ptr + q)
        j = tl.arange(0, QPK)
        d = tl.arange(0, D)
        qw = q.to(tl.int64) if WIDE else q
        hh = kv_head * QPK + j
        acc = tl.zeros([QPK, D], tl.float32)
        m = tl.full([QPK], float("-inf"), tl.float32)
        denom = tl.zeros([QPK], tl.float32)
        if HAS_LOCAL:
            m = tl.load(ll_ptr + qw * stride_ll_q + hh * stride_ll_h)
            acc = tl.load(lo_ptr + qw * stride_lo_q
                          + hh[:, None] * stride_lo_h + d[None, :]
                          ).to(tl.float32)
            denom = tl.full([QPK], 1.0, tl.float32)
        i = 0
        while i < n:
            e = tl.load(order_ptr + s + i) - entry_base
            if WIDE:
                e = e.to(tl.int64)
            le = tl.load(lse_ptr + e * QPK + j)
            o = tl.load(o_ptr + (e * QPK + j)[:, None] * D + d[None, :]
                        ).to(tl.float32)
            new_m = tl.maximum(m, le)
            scale = tl.exp(m - new_m)
            w = tl.exp(le - new_m)
            acc = acc * scale[:, None] + o * w[:, None]
            denom = denom * scale + w
            m = new_m
            i += 1
        res = acc / tl.maximum(denom, 1e-30)[:, None]
        tl.store(out_ptr + qw * stride_out_q + hh[:, None] * stride_out_h
                 + d[None, :],
                 res.to(out_ptr.dtype.element_ty),
                 mask=(n > 0) | HAS_LOCAL)
        if WANT_LSE:
            tl.store(lse_out_ptr + qw * stride_lse_q + hh * stride_lse_h,
                     tl.where((n > 0) | HAS_LOCAL, m + tl.log(denom),
                              float("-inf")))


def combine_partials(out, partial, lse, order, start, count, *, kv_head,
                     q_per_kv, lse_out=None, entry_base=0,
                     local_o=None, local_lse=None):
    """Merge each query's per-cluster partials into `out` by log-sum-exp.

    The grouped kernel (one program per query) is the default.
    SEMBLO_COMBINE_GROUPED=0 selects the per-head kernel, which gives
    bit-identical results.
    """
    if triton is None:
        raise RuntimeError("combine_partials needs triton")
    q_count = start.numel()
    head_dim = partial.shape[-1]
    want_lse = lse_out is not None
    has_local = local_o is not None
    if has_local and local_lse is None:
        raise ValueError("a local partial needs its log-sum-exp")
    lo = local_o if has_local else out
    ll = local_lse if has_local else lse
    if os.environ.get("SEMBLO_COMBINE_GROUPED", "1") == "1":
        _combine_grouped_kernel[(q_count,)](
            out, lse_out if want_lse else out, partial, lse, order, start,
            count, lo, ll,
            kv_head, int(entry_base), out.stride(0), out.stride(1),
            lse_out.stride(0) if want_lse else 0,
            lse_out.stride(1) if want_lse else 0,
            lo.stride(0) if has_local else 0, lo.stride(1) if has_local else 0,
            ll.stride(0) if has_local else 0, ll.stride(1) if has_local else 0,
            QPK=q_per_kv, D=head_dim, WANT_LSE=want_lse, HAS_LOCAL=has_local,
            WIDE=(max(partial.shape[0] * q_per_kv * head_dim,
                      q_count * out.stride(0)) >= (1 << 31)),
            num_warps=int(os.environ.get("SEMBLO_COMBINE_WARPS", "4")))
        return out
    _combine_kernel[(q_count, q_per_kv)](
        out, lse_out if want_lse else out, partial, lse, order, start, count,
        lo, ll,
        kv_head, int(entry_base), out.stride(0), out.stride(1),
        lse_out.stride(0) if want_lse else 0,
        lse_out.stride(1) if want_lse else 0,
        lo.stride(0) if has_local else 0, lo.stride(1) if has_local else 0,
        ll.stride(0) if has_local else 0, ll.stride(1) if has_local else 0,
        QPK=q_per_kv, D=head_dim, WANT_LSE=want_lse, HAS_LOCAL=has_local,
        # int32 addressing unless the partials exceed 2^31 elements.
        WIDE=(max(partial.shape[0] * q_per_kv * head_dim,
                  q_count * out.stride(0)) >= (1 << 31)),
        # The merge is a serial walk over one query's partials, so more warps
        # only widen the reduction; 2 is fastest.
        num_warps=int(os.environ.get("SEMBLO_COMBINE_WARPS", "2")))
    return out


def kv_outer_attention(
    q,                  # [Q, Hq, D] bf16, post-rope
    packed_kv,          # [2, Lp, Hkv, D] bf16, Lp % page_size == 0
    qmask,              # [Q, Hkv, C/32] int32, per-query selected clusters
    causal_sizes,       # [Hkv, C] int32, causally visible tokens per cluster
    bucket_indptr,      # [Hkv, C+1] int32
    *,
    sm_scale,
    page_size,
    aligned=False,
    local_kv=None,      # [2, n_local, Hkv, D] bf16, ORIGINAL order
    query_split=0,      # cap on queries per sequence; 0 = no split
    fa_version=3,
    want_stats=False,
    table_cache=None,   # dict owned by the index: chunk-independent page tables
    gather_from=None,   # (k [L,Hkv,D], v [L,Hkv,D], posting_positions [Hkv,Lp]) in PROMPT order
    num_splits=None,    # FA3 split-KV count for the cluster call; None = SEMBLO_KV_OUTER_NUM_SPLITS
):
    """Attention with the cluster on the outside and its queries gathered.

    Returns (out [Q, Hq, D], stats dict).  The sparse part charges each query
    only what it selected.  `local_kv`, if given, carries the exactly causal
    window of the current chunk and is merged through the log-sum-exp.  By
    default the window is one GQA varlen call whose partial seeds the
    combine, and the page table is built once per index
    (SEMBLO_KV_OUTER_FUSED_LOCAL=0 merges the window separately afterwards).

    `gather_from` replaces the cluster-ordered copy: the visible rows of every
    cluster this chunk touches are gathered from the prompt-order K/V into a
    scratch buffer with the copy's layout, and FA3 reads the scratch.  The
    output is the same; the gather is paid on every chunk instead of once per
    prompt.

    `want_stats` reads the KV-read and charged-token totals, which costs a
    host sync per (KV head, chunk).
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func as varlen

    q_count, num_q_heads, head_dim = q.shape
    dump_dir = os.environ.get("SEMBLO_KV_OUTER_DUMP_DIR")
    if dump_dir:
        # Save one call's inputs for offline kernel benchmarking.  The Nth
        # call is saved (SEMBLO_KV_OUTER_DUMP_CALL, default 100).
        global _DUMP_CALLS
        _DUMP_CALLS = globals().get("_DUMP_CALLS", 0) + 1
        if _DUMP_CALLS == int(os.environ.get("SEMBLO_KV_OUTER_DUMP_CALL", "100")):
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(dict(q=q, packed_kv=packed_kv, qmask=qmask, causal_sizes=causal_sizes,
                            bucket_indptr=bucket_indptr, sm_scale=float(sm_scale),
                            page_size=int(page_size), aligned=bool(aligned),
                            local_kv=local_kv, call=_DUMP_CALLS),
                       os.path.join(dump_dir, "kv_outer_chunk.pt"))
    packed_len, num_kv_heads = packed_kv.shape[1], packed_kv.shape[2]
    n_words = int(qmask.shape[2])
    c_dim = n_words * 32
    q_per_kv = num_q_heads // num_kv_heads
    device = q.device
    page = page_size
    if packed_len % page:
        raise ValueError(
            f"packed KV length {packed_len} is not a whole number of "
            f"{page}-token pages")

    fused_local = (os.environ.get("SEMBLO_KV_OUTER_FUSED_LOCAL", "1") == "1"
                   and local_kv is not None and table_cache is not None)
    # The gathered queries and FA3's output are the body's two big buffers.
    # Releasing each as soon as it is consumed, rather than when the next
    # head rebinds the name, keeps one fewer of them alive at the peak.
    release_early = os.environ.get("SEMBLO_KV_OUTER_RELEASE", "1") == "1"
    if num_splits is None:
        # 0 leaves the count to FA3's heuristic, which at large query counts
        # may choose split-KV and allocate a large fp32 accumulator.
        num_splits = int(os.environ.get("SEMBLO_KV_OUTER_NUM_SPLITS", "0"))
    table_base = None
    if fused_local and (page, aligned) in table_cache:
        start, head_pad, first, table_base = table_cache[(page, aligned)]
    else:
        start = bucket_indptr[:, :-1].to(torch.int64)
        head_pad = torch.zeros_like(start) if aligned \
            else ((-start) & (page - 1))
        first = ((start + head_pad) // page).to(torch.int32)
        if fused_local:
            # The page ids a cluster spans do not depend on the chunk; only
            # how many of them are causally visible does, and FA3 reads that
            # from seqused_k.  So the table is built once per index, wide
            # enough for the largest cluster, and every chunk hands FA3 a
            # view of it.  Entries past a cluster's last page are clamped
            # into the buffer and never read.
            full = bucket_indptr[:, 1:].to(torch.int64) - start - head_pad
            pmax_all = max(int(((full.clamp_min(0) + page - 1) // page).max()), 1)
            ar = torch.arange(pmax_all, device=device, dtype=torch.int64)
            table_base = (first.to(torch.int64)[:, :, None] + ar[None, None, :]
                          ).clamp_max(packed_len // page - 1).to(torch.int32).contiguous()
            table_cache[(page, aligned)] = (start, head_pad, first, table_base)
    # Visible rows of each cluster, counted from the head-padded origin.
    nvis = (causal_sizes.to(torch.int64) - head_pad).clamp_min(0)
    # Pages are counted with ceil, and seqused_k carries the exact row count.
    # FA3 accepts any seqused_k, so the last page is passed partially filled.
    # Every bucket is padded to a whole number of pages, so the extra page
    # stays inside the bucket; it holds the bucket's own future rows or
    # padding rows, and seqused_k keeps both out of the softmax.
    if os.environ.get("SEMBLO_PAGE_CEIL", "1") == "1":
        npage = (nvis + page - 1) // page
    else:
        # Floor instead (SEMBLO_PAGE_CEIL=0): drops the last nvis % page
        # visible rows of every cluster.
        npage = nvis // page
        nvis = npage * page
    live = (npage > 0).to(torch.int32).contiguous()

    # Size the reverse index by its upper bound (every query selects every
    # cluster) instead of reading the entry count, when that bound is small
    # enough.  This saves one host sync per (layer, chunk).
    # SEMBLO_KV_OUTER_INDEX_CAP_MB bounds the two buffers (default 128 MB
    # each); above it the exact count is read.
    cap_entries = q_count * num_kv_heads * c_dim
    cap_mb = float(os.environ.get("SEMBLO_KV_OUTER_INDEX_CAP_MB", "128"))
    capacity = cap_entries if cap_entries * 4 <= cap_mb * (1 << 20) else None
    qvals, counts, indptr_rev, order_all, start_q, per_q_all, pref = \
        build_reverse_index(qmask, live, num_queries=q_count,
                            capacity=capacity)

    # One host sync for the whole layer-chunk: FA3 needs max_seqlen_q,
    # max_seqlen_k and the per-head slice bounds as Python ints.
    head_lo = indptr_rev[:, 0]
    head_hi = torch.cat([head_lo[1:], counts.sum(dtype=torch.int64)
                         .reshape(1).to(head_lo.dtype)])
    pmax_h = npage.max(dim=1).values
    rows_max_h = (counts.to(torch.int64) * q_per_kv).max(dim=1).values
    scalars = torch.stack([head_lo.to(torch.int64), head_hi.to(torch.int64),
                           pmax_h, rows_max_h]).cpu()
    lo_l, hi_l, pmax_l, rowsmax_l = scalars.tolist()
    # Cluster batching (SEMBLO_KV_OUTER_CBATCH) needs each batch's first entry
    # id on the host.
    cbatch = int(os.environ.get("SEMBLO_KV_OUTER_CBATCH", "0"))
    if cbatch:
        cbatch = max(32, (cbatch // 32) * 32)     # a batch edge is a
    whole_head = cbatch <= 0 or cbatch >= c_dim   # 32-cluster word edge
    # The host copy is a sync, so it is taken only when batching needs it.
    indptr_l = None if whole_head else indptr_rev.cpu().tolist()

    n_pages = packed_len // page
    if gather_from is not None:
        k_orig, v_orig, posting = gather_from
        key = ("gather_scratch", tuple(packed_kv.shape))
        scratch = None if table_cache is None else table_cache.get(key)
        if scratch is None:
            scratch = torch.zeros_like(packed_kv)
            if table_cache is not None:
                table_cache[key] = scratch
        with _attend_part("semblo::gather_kv"):
            for h in range(num_kv_heads):
                # The clusters this chunk reads: some query picked them and
                # they have a visible row.  Their visible rows, from the
                # head-padded origin, are what FA3's seqused_k covers.
                touched = (counts[h] > 0) & (nvis[h] > 0)
                starts = (start[h] + head_pad[h])[touched]
                lens = nvis[h][touched]
                if lens.numel() == 0:
                    continue
                offs = torch.cumsum(lens, 0) - lens
                slots = (torch.repeat_interleave(starts, lens)
                         + torch.arange(int(lens.sum()), device=device, dtype=torch.int64)
                         - torch.repeat_interleave(offs, lens))
                pos = posting[h].to(torch.int64)[slots]
                scratch[0, :, h].index_copy_(0, slots, k_orig[:, h].index_select(0, pos))
                scratch[1, :, h].index_copy_(0, slots, v_orig[:, h].index_select(0, pos))
        kpages = scratch[0].view(n_pages, page, num_kv_heads, head_dim)
        vpages = scratch[1].view(n_pages, page, num_kv_heads, head_dim)
    else:
        kpages = packed_kv[0].view(n_pages, page, num_kv_heads, head_dim)
        vpages = packed_kv[1].view(n_pages, page, num_kv_heads, head_dim)
    out = torch.zeros_like(q)
    # One arange for the whole chunk; each head slices the prefix it needs.
    ar_max = torch.arange(max(max(pmax_l), 1), device=device)
    lse_s = torch.full((q_count, num_q_heads), float("-inf"),
                       dtype=torch.float32, device=device)
    fused_local = fused_local and whole_head and not want_stats
    o_local = ll_local = None
    if fused_local:
        # The chunk's own causal window, all query heads in one GQA call,
        # ahead of the clusters so its partial seeds the combine.
        n_local = local_kv.shape[1]
        cu_qk = torch.tensor([0, q_count, 0, n_local], device=device,
                             dtype=torch.int32)
        with _attend_part("semblo::attend_current_chunk"):
            o_local, l_local = varlen(
                q, local_kv[0], local_kv[1],
                max_seqlen_q=q_count, cu_seqlens_q=cu_qk[:2],
                max_seqlen_k=n_local, cu_seqlens_k=cu_qk[2:],
                softmax_scale=sm_scale, causal=True, return_softmax_lse=True,
                fa_version=fa_version)
        ll_local = (l_local.transpose(0, 1)
                    if l_local.shape == (num_q_heads, q_count) else l_local)
        ll_local = ll_local.float().contiguous()
    # `charged` counts, per query, the tokens of the clusters it selected.
    # `kv_read` is what the kernel loads, shared across every query that
    # picked a cluster.
    stats = {"partials": 0, "kv_read": 0, "charged": 0}

    for h in range(num_kv_heads):
        lo, hi, pmax = int(lo_l[h]), int(hi_l[h]), int(pmax_l[h])
        if hi <= lo or pmax == 0:
            if fused_local:
                # No cluster contributes: the window is the whole output.
                sl = slice(h * q_per_kv, (h + 1) * q_per_kv)
                out[:, sl, :] = o_local[:, sl, :]
            continue
        # Clusters in batches when asked (SEMBLO_KV_OUTER_CBATCH).  The gathered
        # queries are the body's peak allocation (nnz * q_per_kv * D elements),
        # and early in a long prompt, when clusters are causally short, a
        # query selects many of them.  A batch edge is a 32-cluster word edge,
        # so `pref` is exactly the per-query offset into the batch, and the
        # extra cost is one log-sum-exp merge per batch.
        for bstart, bend in _cluster_batches(c_dim, cbatch):
            blo = lo if bstart == 0 else int(indptr_l[h][bstart])
            bhi = hi if bend >= c_dim else int(indptr_l[h][bend])
            if bhi <= blo:
                continue
            qv = qvals[blo:bhi].to(torch.int64)
            cnt_h = counts[h][bstart:bend]
            # Every cluster is a sequence, including the ones no query picked.
            # Those get zero query rows and contribute nothing.  Compacting
            # them out with torch.nonzero would cost a host sync.
            seq_n = cnt_h
            pages = npage[h][bstart:bend]
            pmax_b = pmax
            if table_base is not None:
                table = table_base[h][bstart:bend]
            else:
                ar = ar_max[:pmax_b]
                table = (first[h][bstart:bend, None] + ar[None, :]).to(torch.int32)
                table.masked_fill_(ar[None, :] >= pages[:, None], 0)
            seqused = nvis[h][bstart:bend].to(torch.int32)
            if query_split:
                # A cluster's query count ranges from a few to thousands.
                # Cut long sequences along the query axis to balance the work.
                # A query still lands in exactly one piece, so this adds no
                # partials.
                nsplit = ((seq_n.to(torch.int64) + query_split - 1)
                          // query_split)
                rep = torch.repeat_interleave(
                    torch.arange(seq_n.numel(), device=device), nsplit)
                offs = torch.cumsum(nsplit, 0) - nsplit
                # repeat_interleave already resolved the total on the host, so
                # rep.numel() costs no further sync.
                within = (torch.arange(rep.numel(), device=device)
                          - torch.repeat_interleave(offs, nsplit))
                sub_n = (seq_n[rep].to(torch.int64)
                         - within * query_split).clamp_max(query_split)
                seqused = seqused[rep]
                table = table[rep]
                seq_n = sub_n
            rows = (seq_n * q_per_kv).to(torch.int32)
            cu = torch.empty(rows.numel() + 1, dtype=torch.int32, device=device)
            # cu[0] = 0 would be a pageable host-to-device copy that blocks
            # the queue; zero_() on the view is a device-side fill.
            cu[:1].zero_()
            torch.cumsum(rows, 0, out=cu[1:])
            # Slice the head group before gathering, so that only this KV
            # group's query heads are materialised.
            qg = torch.index_select(
                q[:, h * q_per_kv:(h + 1) * q_per_kv, :], 0, qv
            ).reshape(-1, 1, head_dim).contiguous()
            with _attend_part("semblo::attend_selected_clusters"):
                o, l = varlen(
                    qg, kpages[:, :, h:h + 1, :], vpages[:, :, h:h + 1, :],
                    max_seqlen_q=max(int(rowsmax_l[h]), 1), cu_seqlens_q=cu,
                    max_seqlen_k=pmax_b * page, seqused_k=seqused,
                    softmax_scale=sm_scale, causal=False, block_table=table,
                    return_softmax_lse=True, fa_version=fa_version,
                    num_splits=num_splits)
            if release_early:
                del qg
            nnz = qv.numel()
            # The slots in start_q are global, so `order` is passed whole and
            # the combine kernel subtracts this batch's entry base.  A batch's
            # entries for one query are a contiguous slice of that query's run,
            # because the entries are cluster-major, so the slice starts at
            # start + pref[word] and its length is a difference of prefixes.
            w0, w1 = bstart // 32, bend // 32
            b_start = start_q[h] + pref[:, h, w0]
            b_count = (per_q_all[:, h] if w1 >= n_words
                       else pref[:, h, w1]) - pref[:, h, w0]
            if whole_head:
                dst_out, dst_lse = out, lse_s
            else:
                dst_out = torch.zeros_like(out)
                dst_lse = torch.full_like(lse_s, float("-inf"))
            combine_partials(
                dst_out, o.view(nnz, q_per_kv, head_dim),
                l.reshape(-1).float().contiguous(), order_all,
                b_start.contiguous(), b_count.contiguous(),
                kv_head=h, q_per_kv=q_per_kv, lse_out=dst_lse, entry_base=blo,
                local_o=o_local if fused_local else None,
                local_lse=ll_local if fused_local else None)
            if release_early:
                del o, l
            if not whole_head:
                _merge_into(out, lse_s, dst_out, dst_lse,
                            slice(h * q_per_kv, (h + 1) * q_per_kv))
            stats["partials"] += nnz
            if want_stats:
                stats["kv_read"] += int(seqused.sum())
                stats["charged"] += int((seq_n.to(torch.int64)
                                         * seqused.to(torch.int64)).sum())

    if local_kv is None or fused_local:
        return out, stats

    n_local = local_kv.shape[1]
    # One H2D copy for the chunk, not one per KV head.
    cu_qk = torch.tensor([0, q_count, 0, n_local], device=device,
                         dtype=torch.int32)
    cu_q, cu_k = cu_qk[:2], cu_qk[2:]
    for h in range(num_kv_heads):
        sl = slice(h * q_per_kv, (h + 1) * q_per_kv)
        with _attend_part("semblo::attend_current_chunk"):
            o_l, l_l = varlen(
                q[:, sl, :], local_kv[0][:, h:h + 1, :], local_kv[1][:, h:h + 1, :],
                max_seqlen_q=q_count, cu_seqlens_q=cu_q,
                max_seqlen_k=n_local, cu_seqlens_k=cu_k,
                softmax_scale=sm_scale, causal=True, return_softmax_lse=True,
                fa_version=fa_version)
        ll = l_l.transpose(0, 1) if l_l.shape == (q_per_kv, q_count) else l_l
        ls = lse_s[:, sl]
        live_s = torch.isfinite(ls)
        m = torch.where(live_s, torch.maximum(ls, ll), ll)
        ws = torch.where(live_s, (ls - m).exp(), torch.zeros_like(ls))
        wl = (ll - m).exp()
        merged = (out[:, sl, :].float() * ws[..., None]
                  + o_l.float() * wl[..., None]) \
            / (ws + wl).clamp_min(1e-30)[..., None]
        out[:, sl, :] = merged.to(out.dtype)
    return out, stats
