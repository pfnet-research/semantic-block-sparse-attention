"""The attention body over a union of bucket runs, run by FlashAttention-3.

`flash_attn_varlen_func` accepts a `block_table`, a page-granular gather from
an arbitrary page list, which is what a union of bucket runs in the packed KV
is.  One FA3 call is made per KV head, because each head packs its KV under
its own permutation while a varlen block table is per sequence and shared
across heads.  Within a call there is one sequence per query block.

FA3 has neither an arbitrary mask nor a positional causal mask over a
permuted KV, which is handled as follows:

  * Every query in a block reads the block's union (union semantics), since
    per-query row masking is not available.
  * FA3's causal test is in sequence-index space, so the sparse part is
    truncated at the chunk's start, where every packed token is causal for
    every query in the chunk.  The chunk's own tokens are attended by a
    second, dense, exactly-causal FA3 call over the contiguous original-order
    slice, and the two results are merged through their log-sum-exps.

A page is all-or-nothing and packed order is bucket-id order, so a page
straddling a bucket boundary would pull in tokens at arbitrary positions,
including future ones.  fa3_pages.py emits only pages lying wholly inside a
bucket, which drops each bucket's partial end page and never over-includes.
"""

from __future__ import annotations

import torch

from semblo.vllm_plugin.fa3_pages import build_union_pages


def _varlen():
    from vllm.vllm_flash_attn import flash_attn_varlen_func
    return flash_attn_varlen_func


def page_table_bound(packed_length: int, page_size: int, rmax: int) -> int:
    """Worst-case pages one (block, head) can list.

    A union can be the whole causal prefix, so the bound is the buffer in
    pages plus one partial page per bucket.  This page table is far smaller
    than a token-level union (68 MB of int32 across [NB, Hkv] at 256K with
    page 16).
    """
    return packed_length // page_size + rmax + 8


def _as_lse_rows(lse: torch.Tensor, total_q: int, heads: int) -> torch.Tensor:
    """FA3 returns the log-sum-exp as [heads, total_q]; want [total_q, heads]."""
    if lse.shape == (heads, total_q):
        return lse.transpose(0, 1)
    if lse.shape == (total_q, heads):
        return lse
    raise RuntimeError(
        f"unexpected logsumexp shape {tuple(lse.shape)} for total_q={total_q}, "
        f"heads={heads}")


def fa3_union_attention(
    q: torch.Tensor,            # [Q, Hq, D] bf16, post-rope
    packed_kv: torch.Tensor,    # [2, Lp, Hkv, D] bf16, Lp % page_size == 0
    rstart: torch.Tensor,       # [NB, Hkv, RMAX] int32, from build_union_slots
    rlen: torch.Tensor,         # [NB, Hkv, RMAX] int32
    rcount: torch.Tensor,       # [NB, Hkv] int32
    *,
    queries_per_block: int,
    sm_scale: float,
    page_size: int,
    local_kv: torch.Tensor | None = None,   # [2, n_local, Hkv, D] orig order
    pmax: int | None = None,
    out_table: torch.Tensor | None = None,
    aligned: bool = False,
    fa_version: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (out [Q, Hq, D], dropped [NB, Hkv] int32).

    `local_kv` is the chunk's own K/V in original order, whose last token is
    the chunk's last query, and carries the exactly-causal part.  Without it
    the result is only block-causal: a query may see up to
    queries_per_block-1 tokens ahead of itself.
    """
    varlen = _varlen()
    nb, num_kv_heads, rmax = rstart.shape
    q_count, num_q_heads, head_dim = q.shape
    q_per_kv = num_q_heads // num_kv_heads
    g = queries_per_block
    if nb * g < q_count:
        raise ValueError(f"{nb} blocks of {g} cannot cover {q_count} queries")
    packed_length = packed_kv.shape[1]
    if packed_length % page_size:
        raise ValueError(
            f"packed KV length {packed_length} is not a multiple of the page "
            f"size {page_size}; the paged view would run past the buffer")
    if pmax is None:
        pmax = page_table_bound(packed_length, page_size, rmax)

    table, seqused, dropped = build_union_pages(
        rstart, rlen, rcount, page_size=page_size, pmax=pmax,
        aligned=aligned, out_table=out_table)

    n_pages = packed_length // page_size
    kpages = packed_kv[0].view(n_pages, page_size, num_kv_heads, head_dim)
    vpages = packed_kv[1].view(n_pages, page_size, num_kv_heads, head_dim)

    device = q.device
    cu_q = torch.arange(0, (nb + 1) * g, g, device=device,
                        dtype=torch.int32).clamp_max_(q_count)
    max_k = int(seqused.max())
    out = torch.empty_like(q)
    # The first chunk has no prefix, so every sequence lists zero pages.
    # Skip the sparse call then instead of relying on FA3 to return a clean
    # zero with lse = -inf for an empty sequence.
    sparse_empty = max_k == 0

    if local_kv is not None:
        n_local = local_kv.shape[1]
        cu_local = torch.tensor([0, q_count], device=device,
                                dtype=torch.int32)
        cu_local_k = torch.tensor([0, n_local], device=device,
                                  dtype=torch.int32)

    for h in range(num_kv_heads):
        qh = q[:, h * q_per_kv:(h + 1) * q_per_kv, :]
        rows = slice(h * nb, (h + 1) * nb)
        if not sparse_empty:
            o_sparse, l_sparse = varlen(
                qh, kpages[:, :, h:h + 1, :], vpages[:, :, h:h + 1, :],
                max_seqlen_q=g, cu_seqlens_q=cu_q,
                max_seqlen_k=max_k, seqused_k=seqused[rows].contiguous(),
                softmax_scale=sm_scale, causal=False,
                block_table=table[rows], return_softmax_lse=True,
                fa_version=fa_version)
        if local_kv is None:
            if sparse_empty:
                raise RuntimeError(
                    "fa3_union_attention: no pages and no local KV, so there "
                    "is nothing for this chunk to attend")
            out[:, h * q_per_kv:(h + 1) * q_per_kv, :] = o_sparse
            continue
        if sparse_empty:
            # Without return_softmax_lse, varlen returns the output tensor alone.
            out[:, h * q_per_kv:(h + 1) * q_per_kv, :] = varlen(
                qh, local_kv[0][:, h:h + 1, :], local_kv[1][:, h:h + 1, :],
                max_seqlen_q=q_count, cu_seqlens_q=cu_local,
                max_seqlen_k=n_local, cu_seqlens_k=cu_local_k,
                softmax_scale=sm_scale, causal=True,
                fa_version=fa_version)
            continue
        # Contiguous and in original order, so FA3's own causal test is
        # exact: with equal q and k lengths, query i attends keys <= i.
        o_local, l_local = varlen(
            qh, local_kv[0][:, h:h + 1, :], local_kv[1][:, h:h + 1, :],
            max_seqlen_q=q_count, cu_seqlens_q=cu_local,
            max_seqlen_k=n_local, cu_seqlens_k=cu_local_k,
            softmax_scale=sm_scale, causal=True,
            return_softmax_lse=True, fa_version=fa_version)
        ls = _as_lse_rows(l_sparse, q_count, q_per_kv)
        ll = _as_lse_rows(l_local, q_count, q_per_kv)
        # A single block can still list zero pages.  FA3's output for such a
        # sequence is unspecified, and NaN times a zero weight stays NaN, so
        # both the weight and the value are zeroed.
        live = torch.isfinite(ls)
        m = torch.where(live, torch.maximum(ls, ll), ll)
        ws = torch.where(live, (ls - m).exp(), torch.zeros_like(ls))
        wl = (ll - m).exp()
        merged = (torch.nan_to_num(o_sparse.float()) * ws[..., None]
                  + o_local.float() * wl[..., None]) \
            / (ws + wl).clamp_min(1e-30)[..., None]
        out[:, h * q_per_kv:(h + 1) * q_per_kv, :] = merged.to(out.dtype)
    return out, dropped
