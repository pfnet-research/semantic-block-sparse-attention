"""Prefill and decode fast paths: theta selection, then an attention body.

For one query chunk of a prompt forward, this runs the fused kernels of
fused_select (selection), fused_union (union of a query tile's selections)
and fused_body, or the FlashAttention-3 bodies of fa3_body and kv_outer.

Semantics:
- Per-query selection takes buckets in descending penalized score until
  the token budget is reached, via a bisected score threshold (theta).
- The budget is counted against a causal snapshot of the bucket sizes
  (per union block, or at a fixed position for the FA3 bodies) rather
  than per query.
- The union body reads the union of a query tile's selections, so a query
  attends a superset of its own selection.  Causality stays exact per row
  via the key-position mask.  The KV-outer body charges each query only its
  own selection.

Overflow of the sized buffers (union bucket count > rmax, union tokens >
umax) raises via torch._assert_async, with no host sync on the normal path.
"""

from __future__ import annotations

import os

import torch

from semblo.vllm_plugin import oracle_select
from semblo.vllm_plugin.kv_outer import _attend_part

from semblo.vllm_plugin.fused_body import union_block_attention
from semblo.vllm_plugin.fused_select import (
    theta_select_bitmask_flat,
    theta_select_bitmask_precomputed,
    theta_select_ranges,
)
from semblo.vllm_plugin.fused_union import build_union_slots
from semblo.vllm_plugin.fused_probe import FUSED_STATS
from semblo.vllm_plugin.paged_attention import (
    PagedSparseAttentionResult,
)


class CausalSnapshot:
    """Incremental per-bucket causal token counts for one prefill forward.

    Keys enter buckets in position order inside each bucket, so the causal
    prefix count of bucket b at position p is |{keys in b, position <= p}|.
    One bincount over each chunk's key interval keeps the running counts,
    O(L) across the whole forward.
    """

    def __init__(self, index) -> None:
        num_kv_heads = index.num_kv_heads
        b_dim = index.num_buckets
        device = index.posting_positions.device
        # Two different lengths: the packed axis carries page-alignment
        # padding (state._page_align_order) while the position axis does not.
        packed_len = index.posting_positions.shape[1]
        length = int(index.prompt_length)
        # Bucket id of each packed KV slot ([Hkv, packed_len]).  Not the same
        # as index.slot_bucket, which maps compact non-empty-bucket slots to
        # bucket ids; the strict body indexes by packed slot.  Derived from
        # indptr rather than from a repeat of the sizes, because with padding
        # the buckets do not abut.
        rows = torch.arange(packed_len, device=device, dtype=torch.int32)
        slot_bucket = (torch.searchsorted(
            index.bucket_indptr.contiguous(),
            rows.expand(num_kv_heads, -1).contiguous(),
            right=True) - 1).clamp_(0, b_dim - 1).to(torch.int32)
        self.packed_slot_bucket = slot_bucket.contiguous()
        # Bucket of each original position, via the position -> packed row
        # map (a scatter of the posting list would be wrong because padding
        # rows repeat a real row's position).
        self.orig_bucket = torch.gather(
            slot_bucket, 1, index.inverse_positions().long()
        ).contiguous()
        # Offset per head so one bincount covers all heads at once.
        self.orig_bucket_flat = (
            self.orig_bucket.long()
            + (torch.arange(num_kv_heads, device=device) * b_dim)[:, None]
        )
        self.sizes = torch.zeros(
            num_kv_heads, b_dim, dtype=torch.int32, device=device
        )
        self.next_pos = 0
        self.length = length
        self.b_dim = b_dim
        # Per-chunk scratch, allocated lazily and reused across chunks.
        self.bitmask = None
        self.union_idx = None
        self.slot_rowmask = None
        self.qmask = None
        self.b2u = None          # semantic-unit table (fused_units)
        self.num_units = 0
        # Optional fp8 mirror of packed_kv (built on first use).
        self.kv8 = None
        self.kv_scales = None

    def block_sizes(self, chunk_start: int, chunk_end: int,
                    g_union: int) -> torch.Tensor:
        """Per-union-block causal snapshot [NB, H, B] for one chunk.

        Block nb's counts are taken at its last query position, so the
        budget granularity is g_union tokens.  base = counts before the
        chunk; the in-chunk keys are histogrammed by (relative block, bucket)
        in one bincount and prefix-summed over the block axis.
        """
        base = self.advance(chunk_start - 1) if chunk_start > 0 else \
            self.sizes
        n = chunk_end - chunk_start + 1
        nb = (n + g_union - 1) // g_union
        num_kv_heads, b_dim = self.sizes.shape
        device = self.sizes.device
        ob = self.orig_bucket[:, chunk_start:chunk_end + 1].long()
        rel = (torch.arange(n, device=device) // g_union)[None, :]
        head_off = (torch.arange(num_kv_heads, device=device)
                    * (nb * b_dim))[:, None]
        flat = head_off + rel * b_dim + ob
        hist = torch.bincount(
            flat.reshape(-1), minlength=num_kv_heads * nb * b_dim,
        ).view(num_kv_heads, nb, b_dim).cumsum(dim=1)
        table = (base[:, None, :].long() + hist).permute(1, 0, 2)
        return table.contiguous().int()

    def advance(self, last_pos: int) -> torch.Tensor:
        """Counts at position <= last_pos (must be non-decreasing calls)."""
        hi = min(last_pos + 1, self.length)
        if hi > self.next_pos:
            self.sizes += torch.bincount(
                self.orig_bucket_flat[:, self.next_pos:hi].reshape(-1),
                minlength=self.sizes.numel(),
            ).view_as(self.sizes).int()
            self.next_pos = hi
        return self.sizes


def _resolve_umax(length: int, target_tokens: int) -> int:
    """Union scratch bound (tokens per (block, head) slot list).

    The default of 16x the budget fits natural text, but repetitive content
    can produce larger unions and trip the overflow assert.
    SEMBLO_FUSED_UMAX=L makes overflow impossible at the cost of a larger
    scratch buffer (about 1 GB per 256K chunk); a number sets the bound
    directly.  Writes are bounded by umax in all cases; the assert only
    prevents silent truncation."""
    env = os.environ.get("SEMBLO_FUSED_UMAX")
    if env == "L":
        return length
    if env:
        return min(length, int(env))
    return min(length, max(16 * target_tokens, 16384))


def _dense_chunk_attention(q_chunk, index, chunk_end, *, sm_scale,
                           fa_version=3):
    """Exact attention over the whole causal prefix, in original order.

    Used for the chunks whose prefix is short enough that the selection would
    cover almost all of it anyway (SEMBLO_FA3_DENSE_PREFIX_X).  Routing and
    selection are skipped for them.

    FA3's causal mask is bottom-right aligned, so with len_k - len_q equal to
    the chunk's start the query at position p sees exactly up to p.
    """
    from vllm.vllm_flash_attn import flash_attn_varlen_func as varlen

    q_count, num_q_heads, head_dim = q_chunk.shape
    num_kv_heads = index.packed_kv.shape[2]
    q_per_kv = num_q_heads // num_kv_heads
    device = q_chunk.device
    inv = index.inverse_positions()[:, :chunk_end + 1]
    rows = inv.transpose(0, 1).long()
    heads = torch.arange(num_kv_heads, device=device)[None, :]
    kv = torch.stack((index.packed_kv[0][rows, heads],
                      index.packed_kv[1][rows, heads]), dim=0)
    n_local = kv.shape[1]
    cu_q = torch.tensor([0, q_count], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, n_local], device=device, dtype=torch.int32)
    out = torch.empty_like(q_chunk)
    for h in range(num_kv_heads):
        sl = slice(h * q_per_kv, (h + 1) * q_per_kv)
        with _attend_part("semblo::attend_dense_prefix"):
            out[:, sl, :] = varlen(
                q_chunk[:, sl, :], kv[0][:, h:h + 1, :], kv[1][:, h:h + 1, :],
                max_seqlen_q=q_count, cu_seqlens_q=cu_q,
                max_seqlen_k=n_local, cu_seqlens_k=cu_k,
                softmax_scale=sm_scale, causal=True, fa_version=fa_version)
    return out


def _fitted_sink_tokens() -> int:
    """SEMBLO_SINK_TOKENS for the fitted partition, whose cluster 0 is the sink (state._build_flat).

    0 when unset or when the fixed-block partition handles it in torch_select."""
    sink = int(os.environ.get("SEMBLO_SINK_TOKENS", "0") or "0")
    if sink <= 0 or int(os.environ.get("SEMBLO_BLOCK_TOKENS", "0") or "0") > 0:
        return 0
    return sink


def _hold_out_sink(sizes: torch.Tensor, target_tokens: int, sink: int) -> tuple[torch.Tensor, int]:
    """Sizes for the selector with the sink cluster 0 hidden, and the budget the other clusters share.

    Returns a copy: the caller's sizes may alias the snapshot's live counts, and the attention
    still needs cluster 0's real size to read it."""
    held = sizes.clone()
    held[..., 0] = 0
    return held, max(0, int(target_tokens) - sink)


def _force_sink_bit(*masks: torch.Tensor | None) -> None:
    """Select cluster 0 (bit 0 of word 0) for every row of every mask, in place."""
    for m in masks:
        if m is not None:
            m[..., 0].bitwise_or_(1)


def fused_chunk_attention(
    q_chunk: torch.Tensor,        # [Q, Hq, D] bf16 (post-rope)
    query_state,                  # SembloQueryState (vectors 5-d)
    index,                        # ProductPostingIndex
    q_positions: torch.Tensor,    # [Q] int32
    snapshot_state: CausalSnapshot,
    chunk_start: int,
    chunk_end: int,
    *,
    temperature: float,
    beta: float,
    target_tokens: int,
    queries_per_block: int,
    sm_scale: float,
    iters: int = 16,
    umax_factor: int = 8,
    block_n: int = 64,
    num_warps_body: int = 8,
    body_splits: int = 4,
    factor_precompute: bool = True,
    layer_idx: int = -1,
    fa3_page_size: int = 0,
    kv_outer_default: bool = False,
    dense_prefix_x: int = 4,
) -> torch.Tensor:
    q_count = q_chunk.shape[0]
    num_kv_heads = index.num_kv_heads
    b_dim = index.num_buckets
    device = q_chunk.device
    g_union = min(queries_per_block, q_count)
    nb = (q_count + g_union - 1) // g_union
    # FA3 body.  FA3 has no positional causal test over a permuted KV, so the
    # work is split at a position `wstart`: everything before it is read
    # sparsely (truncated there, so every packed token is causal for every
    # query in the chunk) and everything from it up to the chunk's end is read
    # by a dense, exactly causal FA3 call over the original order.
    #
    # Early in the prompt, buckets are causally short (often under a page),
    # and the selection would cover most of the prefix anyway.  While the
    # prefix is below a few budgets, it is read entirely densely instead.
    fa3_page = int(os.environ.get("SEMBLO_FA3_BODY", fa3_page_size))
    if fa3_page:
        # The caller sets the threshold in budgets (dense_prefix_x);
        # SEMBLO_FA3_DENSE_PREFIX_X overrides it.
        cap = int(os.environ.get("SEMBLO_FA3_DENSE_PREFIX_X", str(dense_prefix_x))) \
            * target_tokens
        local = int(os.environ.get("SEMBLO_FA3_LOCAL", "0"))
        wstart = 0 if chunk_start <= cap else max(0, chunk_start - local)
        # advance() only moves forward, so take the wstart snapshot before
        # anything advances past it; wstart is non-decreasing across chunks.
        base = snapshot_state.advance(wstart - 1) if wstart > 0 \
            else torch.zeros_like(snapshot_state.sizes)
        # advance() returns the live accumulator, which later chunks mutate in
        # place, so materialise it rather than hold a view.
        causal_sizes = base[None].expand(nb, -1, -1).contiguous()
        if wstart == 0 and os.environ.get("SEMBLO_FA3_DENSE_SKIP", "1") == "1":
            # Nothing is read sparsely in this chunk, so routing and
            # selection are skipped.
            return _dense_chunk_attention(
                q_chunk, index, chunk_end, sm_scale=sm_scale,
                fa_version=int(os.environ.get("SEMBLO_FA_VERSION", "3")))
    else:
        causal_sizes = snapshot_state.block_sizes(
            chunk_start, chunk_end, g_union
        )
    # Strict per-query semantics: keep the union read layout but mask each
    # row to its own selection in the body.
    strict_mask = os.environ.get("SEMBLO_FUSED_STRICT_MASK") == "1"
    # KV-outer indexes by query, so it needs the per-query mask.
    kv_outer = bool(fa3_page) and (
        os.environ["SEMBLO_KV_OUTER"] == "1" if "SEMBLO_KV_OUTER" in os.environ
        else kv_outer_default)
    want_qmask = strict_mask or kv_outer

    vectors = query_state.vectors[0]          # [Q or NB, Hq, R, I]
    logmix = (
        query_state.mixture_weights[0].float().clamp_min(1e-12).log()
        .contiguous()
    )
    # Consensus mode: the caller pooled the queries per union block before
    # the router, so the selection sees one query per block (block-shared
    # selection); the body still runs on the full chunk.
    consensus = vectors.shape[0] == nb and nb != q_count
    # SEMBLO_CONSENSUS_SCORE=sumexp or max: the caller passes one routing
    # vector per query (padded to whole g-query blocks) and the selector forms
    # one score row per block from them (torch_select.tile_sumexp_log_scores)
    # instead of routing a pooled mean.
    tile_score = os.environ.get("SEMBLO_CONSENSUS_SCORE", "mean")
    if tile_score not in ("mean", "sumexp", "max"):
        raise RuntimeError(f"SEMBLO_CONSENSUS_SCORE={tile_score!r}: expected mean, sumexp or max")
    tile_sumexp = tile_score in ("sumexp", "max")
    if tile_sumexp:
        if os.environ.get("SEMBLO_FUSED_CONSENSUS") != "1" or os.environ.get("SEMBLO_TORCH_SELECT") != "1":
            raise RuntimeError("SEMBLO_CONSENSUS_SCORE=sumexp or max needs SEMBLO_FUSED_CONSENSUS=1 and SEMBLO_TORCH_SELECT=1")
        # The caller pads to whole blocks of the configured size, which can
        # exceed g_union when the chunk is shorter than one block.
        if vectors.shape[0] % nb or vectors.shape[0] < q_count:
            raise RuntimeError(
                f"sumexp tiles: {vectors.shape[0]} routing rows for {nb} blocks of {q_count} queries")
        consensus = True
    sel_q = nb if consensus else q_count
    sel_g = 1 if consensus else g_union

    # The union is not used by KV-outer.  The token-level union list
    # ([NB, Hkv, umax] int32, up to gigabytes per layer) is read only by the
    # Triton union body: the FA3 path compacts the bitmask into bucket ranges
    # and builds a page table from those.
    needs_union = not kv_outer
    needs_union_tokens = needs_union and not fa3_page
    umax = _resolve_umax(snapshot_state.length, target_tokens) \
        if needs_union_tokens else 0
    snapshot = snapshot_state
    if snapshot.bitmask is None or snapshot.bitmask.shape[0] < nb \
            or (needs_union_tokens
                and (snapshot.union_idx is None
                     or snapshot.union_idx.shape[2] < umax)):
        snapshot.bitmask = torch.zeros(
            nb, num_kv_heads, b_dim // 32, dtype=torch.int32, device=device
        )
        if needs_union_tokens:
            snapshot.union_idx = torch.empty(
                nb, num_kv_heads, umax, dtype=torch.int32, device=device,
            )
    bitmask = snapshot.bitmask[:nb]
    bitmask.zero_()
    qmask = None
    if want_qmask:
        if consensus:
            raise RuntimeError(
                "SEMBLO_FUSED_STRICT_MASK needs per-query "
                "selection; incompatible with SEMBLO_FUSED_CONSENSUS"
            )
        if snapshot.qmask is None or snapshot.qmask.shape[0] < q_count:
            snapshot.qmask = torch.empty(
                q_count, num_kv_heads, b_dim // 32, dtype=torch.int32,
                device=device,
            )
        qmask = snapshot.qmask[:q_count]
    if oracle_select.enabled() and getattr(index, "is_flat", False):
        # Diagnostic: clusters scored by this layer's dense attention mass
        # (oracle_select.py); the budget rule and everything after unchanged.
        _, sel_tokens = oracle_select.prefill_select(
            q_chunk, index, causal_sizes, bitmask, chunk_start=chunk_start,
            target_tokens=target_tokens, sm_scale=sm_scale, sel_q=sel_q,
            sel_g=sel_g, iters=iters, qmask=qmask)
    elif os.environ.get("SEMBLO_TORCH_SELECT") == "1":
        # Selection as a GEMM and a sort (torch_select.py).  The fused kernel
        # is sized for a few hundred clusters; this path scales to the
        # thousands of blocks of a fixed-block partition.  Same contract, so
        # nothing downstream changes.
        from semblo.vllm_plugin.torch_select import budget_select_bitmask
        _, sel_tokens = budget_select_bitmask(
            vectors.reshape(vectors.shape[0] if tile_sumexp else sel_q,
                            vectors.shape[1], vectors.shape[2], -1)
            .contiguous(),
            logmix, index.codebooks, causal_sizes, bitmask,
            temperature=temperature, beta=beta,
            target_tokens=target_tokens, g_union=sel_g, qmask=qmask,
            tile_queries=vectors.shape[0] // nb if tile_sumexp else 0,
            tile_reduce=tile_score if tile_sumexp else "sumexp",
        )
    elif getattr(index, "is_flat", False):
        # Flat partition: one centroid per cluster, no product lattice.
        sink = _fitted_sink_tokens()
        select_sizes, select_target = (_hold_out_sink(causal_sizes, target_tokens, sink)
                                       if sink else (causal_sizes, target_tokens))
        _, sel_tokens = theta_select_bitmask_flat(
            vectors.reshape(sel_q, vectors.shape[1], vectors.shape[2], -1)
            .contiguous(),
            logmix, index.codebooks, select_sizes, bitmask,
            group_desc=getattr(index, "group_descriptors", None),
            # The cluster descriptor, when the checkpoint has one.  The query
            # side is sliced the same way the routing vectors are, so its
            # component axis lines up with theirs.
            cluster_descriptors=getattr(index, "cluster_descriptors", None),
            query_descriptors=(
                None if query_state.descriptor_vectors is None
                else query_state.descriptor_vectors[0]
            ),
            descriptor_gamma=query_state.descriptor_gamma,
            temperature=temperature, beta=beta,
            target_tokens=select_target, g_union=sel_g, iters=iters,
            qmask=qmask,
        )
        if sink:
            # The sink cluster is read by every query of the chunk, inside the budget.
            _force_sink_bit(bitmask, qmask)
    else:
        raise RuntimeError("the selection needs the flat partition (SEMBLO_FLAT_UNIT_TOKENS > 0)")

    if fa3_page:
        # FA3 reads pages, so the token-level union list is not built; only
        # the compacted bucket ranges are needed.
        from semblo.vllm_plugin.fa3_body import fa3_union_attention
        if not kv_outer:
            union_tokens, ranges = build_union_slots(
                bitmask, causal_sizes, index.bucket_indptr,
                rmax=b_dim, umax=0, ranges_only=True,
            )
            rstart, rlen, _, rcount, _ = ranges
        # The window [wstart, chunk_end] in original order.  FA3's causal
        # test with unequal lengths lets query i see keys up to
        # i + (len_k - len_q), and len_k - len_q is chunk_start - wstart here,
        # so the query at position chunk_start + i sees positions up to
        # chunk_start + i.
        original_kv = getattr(index, "original_kv", None)
        if original_kv is not None \
                and os.environ.get("SEMBLO_KV_OUTER_FUSED_LOCAL", "1") == "1" \
                and os.environ.get("SEMBLO_KV_OUTER_GATHER", "0") != "1":
            # The layer's K/V is still held in prompt order, so the window is
            # a slice of it rather than a gather through the inverse
            # permutation of the cluster-ordered copy.
            local_kv = torch.stack((
                original_kv[0][wstart:chunk_end + 1],
                original_kv[1][wstart:chunk_end + 1],
            ), dim=0)
        else:
            inv = index.inverse_positions()[:, wstart:chunk_end + 1]
            rows = inv.transpose(0, 1).long()
            heads = torch.arange(num_kv_heads, device=device)[None, :]
            local_kv = torch.stack((
                index.packed_kv[0][rows, heads],
                index.packed_kv[1][rows, heads],
            ), dim=0)
        # Keeping a bucket's leading page is only safe when its start is a
        # page multiple, i.e. when the layout's alignment is a multiple of the
        # page size.  Otherwise that page is skipped.
        align = int(getattr(index, "page_align", 0) or 0)
        if kv_outer:
            # Cluster on the outside, its queries gathered: each query is
            # charged its own selection instead of the union of its tile's.
            from semblo.vllm_plugin.kv_outer import kv_outer_attention
            gather_from = None
            if os.environ.get("SEMBLO_KV_OUTER_GATHER", "0") == "1":
                # Without the cluster-ordered copy: gather the chunk's clusters
                # out of the prompt-order K/V on every chunk.
                if original_kv is None:
                    raise RuntimeError(
                        "SEMBLO_KV_OUTER_GATHER=1 needs the prompt-order K/V, which this "
                        "index dropped; the streamed path keeps it only under the flag")
                gather_from = (original_kv[0], original_kv[1], index.posting_positions)
            out, kv_stats = kv_outer_attention(
                q_chunk, index.packed_kv, qmask, causal_sizes[0].contiguous(),
                index.bucket_indptr,
                sm_scale=sm_scale, page_size=fa3_page,
                aligned=align > 0 and align % fa3_page == 0,
                local_kv=local_kv,
                gather_from=gather_from,
                query_split=int(os.environ.get("SEMBLO_KV_OUTER_SPLIT", "0")),
                want_stats=os.environ.get("SEMBLO_FUSED_STATS") == "1",
                table_cache=index.__dict__.setdefault("_kv_outer_tables", {}),
            )
            if os.environ.get("SEMBLO_FUSED_STATS") == "1":
                # Report the tokens each query is charged for.  The kernel
                # loads less, because a cluster is read once for all the
                # queries that picked it.
                charged = kv_stats["charged"] // max(
                    q_count * num_kv_heads, 1)
                per_query = torch.full((1, 1), charged, dtype=torch.int32,
                                       device=device)
                FUSED_STATS.record(
                    sel_tokens, per_query,
                    torch.zeros_like(per_query, dtype=torch.bool))
            return out
        out, dropped = fa3_union_attention(
            q_chunk, index.packed_kv, rstart, rlen, rcount,
            queries_per_block=g_union, sm_scale=sm_scale,
            page_size=fa3_page, local_kv=local_kv,
            aligned=align > 0 and align % fa3_page == 0,
        )
        if os.environ.get("SEMBLO_FUSED_STATS") == "1":
            # union_tokens here is the union before page-boundary trimming;
            # `dropped` is what the trimming gave up.
            FUSED_STATS.record(sel_tokens, union_tokens - dropped,
                               torch.zeros_like(rcount, dtype=torch.bool))
        return out

    # With per-query strict semantics, the union build records which rows of
    # each block selected each slot's bucket, so the body does not re-derive
    # it per (row, key) with a scattered gather in its inner loop.
    # SEMBLO_FUSED_ROWMASK=0 uses the gather instead.
    want_rowmask = strict_mask and g_union <= 32 \
        and os.environ.get("SEMBLO_FUSED_ROWMASK", "1") == "1"
    slot_rowmask = None
    if want_rowmask:
        if snapshot.slot_rowmask is None \
                or snapshot.slot_rowmask.shape != snapshot.union_idx.shape:
            snapshot.slot_rowmask = torch.empty_like(snapshot.union_idx)
        union_idx, union_tokens, rfull, slot_rowmask = build_union_slots(
            bitmask, causal_sizes, index.bucket_indptr,
            rmax=b_dim, umax=umax,
            out_union_idx=snapshot.union_idx[:nb],
            qmask=qmask, queries_per_block=g_union,
            out_slot_rowmask=snapshot.slot_rowmask[:nb],
        )
    else:
        union_idx, union_tokens, rfull = build_union_slots(
            bitmask, causal_sizes, index.bucket_indptr,
            rmax=b_dim, umax=umax,
            out_union_idx=snapshot.union_idx[:nb],
        )
    torch._assert_async(
        (union_tokens <= umax).all(),
        "fused prefill: union tokens exceeded umax",
    )
    # The expand kernel and the body are bounded by umax regardless of the
    # assert (which fires asynchronously); clamp the count the body sees.
    overflowed = union_tokens > umax
    union_tokens = union_tokens.clamp_max(umax)
    if os.environ.get("SEMBLO_FUSED_STATS") == "1":
        FUSED_STATS.record(sel_tokens, union_tokens, overflowed)
    del rfull

    kv = index.packed_kv
    kv_scales = None
    out = union_block_attention(
        q_chunk, kv, index.posting_positions,
        union_idx, union_tokens, q_positions,
        queries_per_block=g_union, sm_scale=sm_scale,
        block_n=int(os.environ.get("SEMBLO_FUSED_BODY_BN", block_n)),
        num_warps=int(os.environ.get("SEMBLO_FUSED_BODY_WARPS",
                                     num_warps_body)),
        num_splits=int(os.environ.get("SEMBLO_FUSED_BODY_SPLITS",
                                      body_splits)),
        kv_scales=kv_scales,
        slot_bucket=None if slot_rowmask is not None
        else (snapshot.packed_slot_bucket if strict_mask else None),
        query_sel_mask=None if slot_rowmask is not None
        else (qmask if strict_mask else None),
        slot_rowmask=slot_rowmask,
    )
    return out


# The decode body's launch shape    return out


# The decode body's launch shape, tuned together with the default split
# count of 32 (the layer doubles it).
_DECODE_BODY_LAUNCH = {"block_n": 128, "num_warps": 4, "num_stages": 2}


def _dense_rows_are_exact(index) -> bool:
    """True when rows [0, prompt_length) of the packed K/V are the prompt's
    tokens, each once.

    The packed buffer pads every bucket to a whole page with copies of one
    real row (state._page_align_order), so a plain read over it would count
    those rows twice.  A fixed-block partition whose block is a multiple of
    the page has no padding; the fitted partition always has some.  One host
    sync per request, cached on the index.  Inside a graph capture the answer
    is not computed and the caller keeps the selective path.
    """
    cached = getattr(index, "_dense_rows_exact", None)
    if cached is None:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return False
        totals = index.bucket_indptr[:, -1].to(torch.int64)
        length = int(index.prompt_length)
        cached = bool(int(totals.max()) == length and int(totals.min()) == length)
        index._dense_rows_exact = cached
    return cached


def dense_prompt_decode(
    q_chunk: torch.Tensor,        # [1, Hq, D] bf16 (post-rope)
    index,
    *,
    sm_scale: float,
    fa_version: int = 3,
) -> "PagedSparseAttentionResult | None":
    """One FA3 call over the whole packed prompt, for a decode step whose
    budget is the whole prompt.

    A budget of the whole prompt selects every block, and the selective path
    would pay its per-block bookkeeping for each of them.  For one query with
    no causal cut the order of the keys does not matter, so the packed rows
    are read as one sequence per KV head.  Returns None when the packed rows
    are not exactly the prompt (padding rows), or when
    SEMBLO_DENSE_DECODE_FAST=0; the caller then keeps the selective path.
    """
    if os.environ.get("SEMBLO_DENSE_DECODE_FAST", "1") != "1":
        return None
    if not _dense_rows_are_exact(index):
        return None
    from vllm.vllm_flash_attn import flash_attn_varlen_func as varlen
    length = int(index.prompt_length)
    q_count, num_q_heads, _ = q_chunk.shape
    cu = getattr(index, "_dense_decode_cu", None)
    if cu is None or cu[0].numel() != 2 or int(cu[2]) != q_count:
        device = q_chunk.device
        cu = (torch.tensor([0, q_count], device=device, dtype=torch.int32),
              torch.tensor([0, length], device=device, dtype=torch.int32),
              q_count)
        index._dense_decode_cu = cu
    out, lse = varlen(
        q_chunk, index.packed_kv[0, :length], index.packed_kv[1, :length],
        max_seqlen_q=q_count, cu_seqlens_q=cu[0],
        max_seqlen_k=length, cu_seqlens_k=cu[1],
        softmax_scale=sm_scale, causal=False, return_softmax_lse=True,
        fa_version=fa_version)
    if lse.shape == (num_q_heads, q_count):
        lse = lse.transpose(0, 1)
    return PagedSparseAttentionResult(output=out, logsumexp=lse.float().contiguous())


def fused_decode_attention(
    q_chunk: torch.Tensor,        # [1, Hq, D] bf16 (post-rope)
    query_state,                  # SembloQueryState
    index,                        # ProductPostingIndex
    q_positions: torch.Tensor,    # [1] int32
    *,
    temperature: float,
    beta: float,
    target_tokens: int,
    sm_scale: float,
    iters: int = 16,
    num_splits: int = 16,
) -> PagedSparseAttentionResult:
    """Decode step over the packed prompt: theta-select the single query's
    buckets, expand to slots, one split attention pass.  The generated
    suffix is merged by the caller (needs the logsumexp this returns).
    At decode every prompt key is causal, so the budget snapshot is simply
    the full bucket sizes."""
    if target_tokens >= int(index.prompt_length):
        dense = dense_prompt_decode(
            q_chunk, index, sm_scale=sm_scale,
            fa_version=int(os.environ.get("SEMBLO_FA_VERSION", "3")))
        if dense is not None:
            return dense
    num_kv_heads = index.num_kv_heads
    b_dim = index.num_buckets
    device = q_chunk.device
    vectors = query_state.vectors[0]
    logmix = (
        query_state.mixture_weights[0].float().clamp_min(1e-12).log()
        .contiguous()
    )
    sizes3 = index.bucket_sizes.unsqueeze(0).contiguous()
    bitmask = torch.zeros(1, num_kv_heads, b_dim // 32, dtype=torch.int32,
                          device=device)
    vec4 = vectors.reshape(1, vectors.shape[1], vectors.shape[2], -1) \
        .contiguous()
    if (getattr(index, "cluster_descriptors", None) is not None
            and not getattr(index, "is_flat", False)):
        raise NotImplementedError(
            "the cluster descriptor exists for the flat partition only")
    if oracle_select.enabled() and getattr(index, "is_flat", False):
        oracle_select.decode_select(
            q_chunk, index, sizes3, bitmask, target_tokens=target_tokens,
            sm_scale=sm_scale, iters=iters)
    elif (os.environ.get("SEMBLO_TORCH_SELECT") == "1"
            and getattr(index, "is_flat", False)
            and getattr(index, "cluster_descriptors", None) is None):
        # The fused flat selector scales num_warps with the cluster count and
        # exceeds Triton's 1,024-thread limit at tens of thousands of
        # clusters, so the torch path is used when it was used at prefill.
        from semblo.vllm_plugin.torch_select import budget_select_bitmask
        budget_select_bitmask(
            vec4, logmix, index.codebooks, sizes3, bitmask,
            temperature=temperature, beta=beta,
            target_tokens=target_tokens, g_union=1,
        )
    elif getattr(index, "is_flat", False):
        sink = _fitted_sink_tokens()
        select_sizes, select_target = (_hold_out_sink(sizes3, target_tokens, sink)
                                       if sink else (sizes3, target_tokens))
        theta_select_bitmask_flat(
            vec4, logmix, index.codebooks, select_sizes, bitmask,
            cluster_descriptors=getattr(index, "cluster_descriptors", None),
            query_descriptors=(
                None if query_state.descriptor_vectors is None
                else query_state.descriptor_vectors[0]
            ),
            descriptor_gamma=query_state.descriptor_gamma,
            temperature=temperature, beta=beta,
            target_tokens=select_target, g_union=1, iters=iters,
        )
        if sink:
            _force_sink_bit(bitmask)
    else:
        theta_select_bitmask_precomputed(
            vec4, logmix, index.codebooks, sizes3, bitmask,
            temperature=temperature, beta=beta,
            target_tokens=target_tokens, g_union=1, iters=iters,
        )
    umax = index.posting_positions.shape[1]
    union_idx, union_tokens, _ = build_union_slots(
        bitmask, sizes3, index.bucket_indptr,
        rmax=b_dim, umax=umax,
    )
    union_tokens = union_tokens.clamp_max(umax)
    out, lse = union_block_attention(
        q_chunk, index.packed_kv, index.posting_positions,
        union_idx, union_tokens, q_positions,
        queries_per_block=1, sm_scale=sm_scale,
        **_DECODE_BODY_LAUNCH, num_splits=num_splits,
        return_logsumexp=True,
    )
    return PagedSparseAttentionResult(output=out, logsumexp=lse)
