"""Block selection as a GEMM and a sort, for many small blocks.

The fused selector (fused_select.py) scores, penalises, walks the budget and
writes the bitmask in one Triton kernel. It suits the few hundred blocks of the
k-means partition, but its cost grows linearly with the block count with a
large constant, which makes it slow for the thousands of blocks of the
positional baseline (SEMBLO_BLOCK_TOKENS). This module is the selector for
that case:

    scores   the fused scorer (flat_mixture_scores_qtiled) on the GPU, or one
             fp32 GEMM per KV head otherwise
    combine  softmax over the blocks, then the sum over a KV head's query
             heads, or the max of the raw scores over those heads
             (SEMBLO_SELECT_COMBINE=max)
    budget   the kernel's rule: blocks in score order until the causally
             visible tokens reach the budget
    output   the same packed bitmask the kernel writes

It has the same contract as ``theta_select_bitmask_flat``, so the two are
interchangeable. Everything downstream reads only the bitmask and the qmask.
"""
from __future__ import annotations

import os

import torch

from semblo.vllm_plugin.fused_units import block_bitmask_from_qmask
from semblo.vllm_plugin.query_tiles import pack_bits

try:  # unavailable without Triton
    from semblo.vllm_plugin.fused_select import (
        flat_mixture_scores_qtiled as _qtiled,
    )
except Exception:  # pragma: no cover - exercised only without triton
    _qtiled = None


def _budget_walk(score, visible, target_tokens, k_bound, num_clusters):
    """[.., C] bool: blocks by decreasing score until the budget is reached.

    ``k_bound`` is the largest number of blocks the budget can select. It is
    exact for equal-sized blocks and C otherwise. A bound below C replaces
    the full sort with a top-k.
    """
    if k_bound < num_clusters:
        top_score, top_idx = score.topk(k_bound, dim=-1)
        vis_top = visible.gather(-1, top_idx)
        run = vis_top.cumsum(dim=-1)
        take = (run - vis_top < float(target_tokens)) & (top_score > float("-inf"))
        short = (run[..., -1] < float(target_tokens)) & (
            top_score[..., -1] > float("-inf"))
        if bool(short.any()):
            raise RuntimeError(
                f"top-{k_bound} did not reach the budget for "
                f"{int(short.sum())} rows: the blocks are not all the same "
                "size, so the exact bound does not hold")
        sel = torch.zeros_like(score, dtype=torch.bool)
        sel.scatter_(-1, top_idx, take)
        return sel
    order = score.argsort(dim=-1, descending=True)
    vis_all = visible.gather(-1, order)
    run = vis_all.cumsum(dim=-1)
    take = (run - vis_all < float(target_tokens)) & (
        score.gather(-1, order) > float("-inf"))
    sel = torch.zeros_like(score, dtype=torch.bool)
    sel.scatter_(-1, order, take)
    return sel


# Queries are scored in slices of this size to bound the [q, Hkv, T, C] score
# tensor. At C=16,384, T=4 and fp32, a slice of 512 queries is 128 MB.
_SLICE = int(os.environ.get("SEMBLO_TORCH_SELECT_SLICE", "512"))


def mixture_log_scores(
    vectors: torch.Tensor,        # [sel_q, Hq, R, 2*DF] bf16
    logmix: torch.Tensor | None,  # [sel_q, Hq, R] fp32, or None for unweighted
    codebooks: torch.Tensor,      # [Hkv, 2, C, DF]
    *,
    temperature: float,
    fused: bool | None = None,
) -> torch.Tensor:
    """[sel_q, Hkv, C] fp32: log sum_h exp(logmix_h) softmax_c(q_h . mu_c / tau).

    Equation 2 without the size penalty. ``fused=None`` uses the kernel
    (flat_mixture_scores_qtiled) when it is available and the tensors are on
    the GPU, and the torch GEMM otherwise. Both form bf16 products with fp32
    accumulation, so they agree up to summation order.
    """
    sel_q, num_q_heads, rank, twice_df = vectors.shape
    num_kv_heads, two, num_clusters, df = codebooks.shape
    if two != 2 or twice_df != 2 * df:
        raise ValueError(
            f"codebooks {tuple(codebooks.shape)} and vectors "
            f"{tuple(vectors.shape)} disagree on the descriptor width")
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be a multiple of the KV heads")
    group = num_q_heads // num_kv_heads
    per_kv = group * rank
    index_dim = 2 * df
    if fused is None:
        fused = vectors.is_cuda and _qtiled is not None
    if fused:
        if _qtiled is None:
            raise RuntimeError("the fused scorer is not available here")
        if logmix is None:
            logmix = torch.zeros(sel_q, num_q_heads, rank,
                                 dtype=torch.float32, device=vectors.device)
        prob = _qtiled(vectors, logmix, codebooks, temperature=temperature)
        return prob.clamp_min_(1e-30).log_()
    # The GEMM runs in fp32 on bf16 values, matching the kernel (bf16 operands
    # in tl.dot, fp32 accumulation). A bf16 @ bf16 matmul in torch would round
    # the logits to bf16 and move the log-scores by up to 0.2. Every bf16
    # product is exact in fp32 (and in TF32), so only the summation order
    # differs from the kernel.
    cents = torch.cat([codebooks[:, 0], codebooks[:, 1]], dim=-1).float()
    out = torch.empty(sel_q, num_kv_heads, num_clusters, dtype=torch.float32,
                      device=vectors.device)
    for lo in range(0, sel_q, _SLICE):
        hi = min(lo + _SLICE, sel_q)
        rows = hi - lo
        # Folding the query heads into the M axis makes this one large matrix
        # product (M = rows * per_kv, N = C, K = 256).
        q = (vectors[lo:hi].reshape(rows, num_kv_heads, per_kv, index_dim)
             .float())
        for h in range(num_kv_heads):
            flat = q[:, h].reshape(rows * per_kv, index_dim)
            sc = flat @ cents[h].transpose(0, 1)           # [rows*T, C] fp32
            sc.div_(temperature)
            sc = torch.softmax(sc, dim=-1).view(rows, per_kv, num_clusters)
            if logmix is not None:
                sc = sc * (logmix[lo:hi, h * group:(h + 1) * group]
                           .reshape(rows, per_kv, 1).exp())
            out[lo:hi, h] = sc.sum(dim=1).clamp_min_(1e-30).log_()
            del sc
    return out


def tile_sumexp_log_scores(
    vectors: torch.Tensor,        # [NB * tile, Hq, 1, 2*DF] bf16, one row per query
    codebooks: torch.Tensor,      # [Hkv, 2, C, DF]
    *,
    tile: int,
    temperature: float,
    rows_per_slice: int = 1024,
    reduce: str = "sumexp",
    heads: str = "perhead",
) -> torch.Tensor:
    """[NB, Hkv, C] fp32: one score row per tile of ``tile`` consecutive queries.

    As in FlashPrefill (arXiv 2608.19758, Eq. 1-2), the tile's scores sum
    exp(q_i . k_J) over its queries and normalise over the blocks, so a block
    that any one query strongly matches keeps a high score. Per query head,
    the tile's logit for block c is logsumexp_i (q_i . mu_c / tau). A softmax
    over c turns it into a distribution, and the heads of one KV head are
    summed and logged as in Equation 2. With ``tile == 1`` this equals
    ``mixture_log_scores`` without mixture weights.

    The logits are fp32 products of the bf16 values, formed per KV head in
    slices of ``rows_per_slice`` queries to bound memory.

    ``reduce="max"`` takes the tile's largest logit instead of the logsumexp,
    as QuoKA does. ``heads="packed"`` merges the query heads of a KV head
    before the softmax, as in FlashPrefill V2's PackGQA.
    """
    if reduce not in ("sumexp", "max"):
        raise ValueError(f"reduce={reduce!r}: expected sumexp or max")
    if heads not in ("perhead", "packed"):
        raise ValueError(f"heads={heads!r}: expected perhead or packed")
    rows_total, num_q_heads, rank, twice_df = vectors.shape
    num_kv_heads, two, num_clusters, df = codebooks.shape
    if rank != 1:
        raise ValueError("the tile score is defined for one routing vector per head (the flat index)")
    if two != 2 or twice_df != 2 * df:
        raise ValueError(f"codebooks {tuple(codebooks.shape)} and vectors {tuple(vectors.shape)} disagree")
    if tile < 1 or rows_total % tile:
        raise ValueError(f"{rows_total} query rows are not a whole number of {tile}-query tiles")
    group = num_q_heads // num_kv_heads
    nb = rows_total // tile
    step = max(tile, (rows_per_slice // tile) * tile)
    cents = torch.cat([codebooks[:, 0], codebooks[:, 1]], dim=-1).float()     # [Hkv, C, I]
    out = torch.empty(nb, num_kv_heads, num_clusters, dtype=torch.float32, device=vectors.device)
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        for h in range(num_kv_heads):
            cent_t = cents[h].transpose(0, 1)                                  # [I, C]
            for lo in range(0, rows_total, step):
                hi = min(lo + step, rows_total)
                q = vectors[lo:hi, h * group:(h + 1) * group, 0].float()     # [rows, group, I]
                logits = (q.reshape(-1, q.shape[-1]) @ cent_t).div_(temperature)
                logits = logits.view((hi - lo) // tile, tile, group, num_clusters)
                tile_logit = (torch.logsumexp(logits, dim=1) if reduce == "sumexp"
                              else logits.amax(dim=1))                        # [tiles, group, C]
                del logits
                if heads == "packed":
                    merged = (torch.logsumexp(tile_logit, dim=1) if reduce == "sumexp"
                              else tile_logit.amax(dim=1))
                    out[lo // tile:hi // tile, h] = torch.log_softmax(merged, dim=-1)
                else:
                    prob = torch.softmax(tile_logit, dim=-1).sum(dim=1)        # [tiles, C]
                    out[lo // tile:hi // tile, h] = prob.clamp_min_(1e-30).log_()
                del tile_logit
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return out


def budget_select_bitmask(
    vectors: torch.Tensor,        # [sel_q, Hq, R, 2*DF] bf16
    logmix: torch.Tensor,         # [sel_q, Hq, R] fp32
    codebooks: torch.Tensor,      # [Hkv, 2, C, DF]
    causal_sizes: torch.Tensor,   # [NB, Hkv, C] int32
    bitmask: torch.Tensor,        # [NB, Hkv, C/32] int32, written in place
    *,
    temperature: float,
    beta: float,
    target_tokens: int,
    g_union: int,
    qmask: torch.Tensor | None = None,   # [sel_q, Hkv, C/32] int32, in place
    tile_queries: int = 0,
    tile_reduce: str = "sumexp",
) -> tuple[torch.Tensor, torch.Tensor]:
    """(counts, tokens), both [sel_q, Hkv] int32; bitmask and qmask in place.

    ``tile_queries > 0``: ``vectors`` holds one row per query and the score is one row per tile of
    that many queries (``tile_sumexp_log_scores``); sel_q is then the number of tiles.

    The score is Equation 2 of the paper, as in the fused kernel: per query
    head a softmax over the blocks, the log of the sum over the heads of one
    KV head, minus beta*log(1+n_c). Blocks with no causally visible token are
    excluded. Blocks are taken in score order until the visible tokens reach
    ``target_tokens``, including the block that crosses it.
    """
    sel_q, num_q_heads, rank, twice_df = vectors.shape
    if tile_queries:
        sel_q = sel_q // tile_queries
    num_kv_heads, two, num_clusters, df = codebooks.shape
    if two != 2 or twice_df != 2 * df:
        raise ValueError(
            f"codebooks {tuple(codebooks.shape)} and vectors "
            f"{tuple(vectors.shape)} disagree on the descriptor width")
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be a multiple of the KV heads")
    group = num_q_heads // num_kv_heads
    per_kv = group * rank            # the T axis: the heads a KV head serves
    combine = os.environ.get("SEMBLO_SELECT_COMBINE", "sum")
    if combine not in ("sum", "max"):
        raise RuntimeError(
            f"SEMBLO_SELECT_COMBINE={combine!r}: expected sum or max")
    nb = bitmask.shape[0]
    device = vectors.device

    if True:
        cb0 = codebooks[:, 0].float()                  # [Hkv, C, DF]
        cb1 = codebooks[:, 1].float()
        index_dim = 2 * df
        block_of = torch.div(torch.arange(sel_q, device=device), g_union,
                             rounding_mode="floor").clamp_max_(nb - 1)
        sizes_f = causal_sizes.float()
        penalty = beta * torch.log1p(sizes_f)          # [NB, Hkv, C]
        # The number of blocks the budget can select. In general any block may
        # cross the budget, so the whole order is needed. With positional
        # blocks every visible block holds exactly `block` tokens except the
        # one containing the causal frontier, so at most ceil(K / block) + 1
        # blocks are selected and top-k is exact.
        block = int(os.environ.get("SEMBLO_BLOCK_TOKENS", "0"))
        k_bound = num_clusters
        if block > 0:
            k_bound = min(num_clusters,
                          (target_tokens + block - 1) // block + 1)
        # Initial block (SEMBLO_SINK_TOKENS=S). With positional blocks, block i
        # holds positions [iB, (i+1)B), so the first S tokens are blocks
        # 0 .. ceil(S/B)-1. They are selected first, inside the budget, when
        # visible. With k-means blocks the initial block is block 0, which is
        # likewise taken first and counts against the budget (see
        # fused_prefill._hold_out_sink).
        sink = int(os.environ.get("SEMBLO_SINK_TOKENS", "0"))
        sink_blocks = 0
        if sink > 0:
            if block <= 0:
                sink_blocks = 1
            else:
                sink_blocks = min(num_clusters, (sink + block - 1) // block)
            if sink_blocks >= k_bound:
                raise RuntimeError(f"a {sink}-token sink fills the whole budget of {target_tokens}")
        words = torch.empty(sel_q, num_kv_heads, num_clusters // 32,
                            dtype=torch.int32, device=device)
        counts = torch.empty(sel_q, num_kv_heads, dtype=torch.int32,
                             device=device)
        tokens = torch.empty(sel_q, num_kv_heads, dtype=torch.int32,
                             device=device)
        # The centroids, one matrix per KV head, fp32 on bf16 values (see
        # mixture_log_scores).
        cents = torch.cat([cb0, cb1], dim=-1)                  # [Hkv, C, I]

        # The sum combine goes through mixture_log_scores, which uses the
        # fused scorer on the GPU. The max combine needs the raw per-head
        # scores, which that kernel sums away, so it uses the loop below.
        if tile_queries:
            if combine != "sum":
                raise RuntimeError("the tile score is defined for the sum combine only")
            if vectors.is_cuda and tile_queries % 16 == 0 and os.environ.get("SEMBLO_TILE_SCORE_KERNEL", "1") == "1":
                from semblo.vllm_plugin.tile_scores import tile_log_scores_triton
                score_all = tile_log_scores_triton(vectors, codebooks, tile=tile_queries,
                                                   temperature=temperature, reduce=tile_reduce,
                                                   heads=os.environ.get("SEMBLO_TILE_HEADS", "perhead"))
            else:
                score_all = tile_sumexp_log_scores(vectors, codebooks, tile=tile_queries,
                                                   temperature=temperature, reduce=tile_reduce,
                                                   heads=os.environ.get("SEMBLO_TILE_HEADS", "perhead"))
            step = sel_q
        elif combine == "sum":
            score_all = mixture_log_scores(vectors, logmix, codebooks,
                                           temperature=temperature)
            step = sel_q
        else:
            score_all = None
            step = _SLICE
        for lo in range(0, sel_q, step):
            hi = min(lo + step, sel_q)
            rows = hi - lo
            blocks = block_of[lo:hi]
            visible = sizes_f[blocks]                           # [q, Hkv, C]
            if score_all is not None:
                score = score_all[lo:hi] - penalty[blocks]
            else:
                q = (vectors[lo:hi].reshape(rows, num_kv_heads, per_kv,
                                            index_dim).float())
                combined = None
                for h in range(num_kv_heads):
                    flat = q[:, h].reshape(rows * per_kv, index_dim)
                    sc = flat @ cents[h].transpose(0, 1)
                    sc.div_(temperature)
                    # The best head rather than the sum of the heads'
                    # distributions. Without the size penalty the ranking
                    # does not depend on the temperature.
                    head = sc.view(rows, per_kv, num_clusters).amax(dim=1)
                    if combined is None:
                        combined = head.new_empty(rows, num_kv_heads,
                                                  num_clusters)
                    combined[:, h] = head
                    del sc, head
                score = combined - penalty[blocks]
            if sink_blocks:
                score = score.clone()
                score[..., :sink_blocks] = float("inf")
            score = score.masked_fill(visible <= 0, float("-inf"))

            sel = _budget_walk(score, visible, target_tokens,
                               k_bound, num_clusters)

            words[lo:hi] = pack_bits(sel)
            counts[lo:hi] = sel.sum(-1).to(torch.int32)
            tokens[lo:hi] = (sel * visible).sum(-1).to(torch.int32)

        if qmask is not None:
            qmask.copy_(words)
        bitmask.copy_(block_bitmask_from_qmask(words, g_union)
                      if g_union > 1 else words)
        return counts, tokens
