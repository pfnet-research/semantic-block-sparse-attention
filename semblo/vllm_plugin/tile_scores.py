"""FlashPrefill-style shared-selection tile score as a Triton kernel (SEMBLO_CONSENSUS_SCORE=sumexp | max).

``torch_select.tile_sumexp_log_scores`` materialises every query's logit against every block descriptor
as an fp32 tensor and reduces it over the tile afterwards, which is slow at long context.  This kernel
keeps the logits in registers.  For one tile of queries, one KV head and one slab of block descriptors,
it takes bf16 tensor-core dots of 16 queries at a time and folds them into a running logsumexp (or max)
over the tile's queries, so only the tile logit [tiles, Hkv, T, C] is written.  The softmax over the
blocks and the sum over a KV head's query heads then run in torch on that smaller table.

Heads: FlashPrefill V2 packs a KV head's query heads into the tile's rows (PackGQA), so its sum of
exponentials also runs over heads.  By default each query head is instead normalised over the blocks and
the heads' distributions are summed, the same rule as per-query and mean-query selection, so only the way
a tile's queries are merged differs.

The kernel does the same arithmetic as per-query scoring (every query against every descriptor), so this
selection costs about as much as per-query scoring of the same blocks, and never less than mean-query
scoring.
"""
from __future__ import annotations

import torch

try:  # pragma: no cover - exercised only without triton
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None


if triton is not None:
    @triton.jit
    def _tile_logit_kernel(
        q_ptr, c_ptr, out_ptr,
        C, inv_tau,
        stride_q0, stride_q1, stride_q2, stride_q3,     # q [rows, Hkv, T, I]
        stride_c0, stride_c1, stride_c2,                # c [Hkv, C, I]
        stride_o0, stride_o1, stride_o2, stride_o3,     # out [tiles, Hkv, T, C]
        TILE: tl.constexpr, T: tl.constexpr, I: tl.constexpr,
        QB: tl.constexpr, TC: tl.constexpr, USE_MAX: tl.constexpr,
    ):
        # Grid (tiles, C blocks, KV heads): consecutive programs walk the tiles of one C block, so a C block
        # is read from memory once per head and reused from L2 by every tile.  The order affects speed only.
        b = tl.program_id(0)
        cblk = tl.program_id(1)
        h = tl.program_id(2)
        cols = cblk * TC + tl.arange(0, TC)
        cmask = cols < C
        offs_i = tl.arange(0, I)
        cent = tl.load(c_ptr + h * stride_c0 + cols[:, None] * stride_c1 + offs_i[None, :] * stride_c2,
                       mask=cmask[:, None], other=0.0)                         # [TC, I] bf16
        for t in tl.static_range(T):
            m = tl.full([TC], -float("inf"), tl.float32)
            l = tl.zeros([TC], tl.float32)
            for q0 in tl.static_range(0, TILE, QB):
                rows = b * TILE + q0 + tl.arange(0, QB)
                q = tl.load(q_ptr + rows[:, None] * stride_q0 + h * stride_q1 + t * stride_q2
                            + offs_i[None, :] * stride_q3)                    # [QB, I] bf16
                s = tl.dot(q, tl.trans(cent)) * inv_tau                        # [QB, TC] fp32
                smax = tl.max(s, axis=0)
                m_new = tl.maximum(m, smax)
                if USE_MAX:
                    m = m_new
                else:
                    l = l * tl.exp(m - m_new) + tl.sum(tl.exp(s - m_new[None, :]), axis=0)
                    m = m_new
            if USE_MAX:
                res = m
            else:
                res = m + tl.log(l)
            tl.store(out_ptr + b * stride_o0 + h * stride_o1 + t * stride_o2 + cols * stride_o3,
                     res, mask=cmask)


def tile_log_scores_triton(vectors: torch.Tensor, codebooks: torch.Tensor, *, tile: int,
                           temperature: float, reduce: str = "sumexp", heads: str = "perhead",
                           tc: int | None = None) -> torch.Tensor:
    """[tiles, Hkv, C] fp32, the same score as ``torch_select.tile_sumexp_log_scores``.

    vectors [tiles * tile, Hq, 1, I] bf16 (I = 2 * DF), codebooks [Hkv, 2, C, DF].  The tile must be a
    multiple of 16 (the dot's smallest M).

    ``heads="packed"`` (SEMBLO_TILE_HEADS=packed) merges a KV head's query heads into the same logsumexp (or
    max) before the one softmax over the blocks, as FlashPrefill V2's PackGQA tile does.  The default
    normalises each head and sums the distributions.  The launch shape (SEMBLO_TILE_TC, SEMBLO_TILE_WARPS,
    SEMBLO_TILE_STAGES) affects speed only: each output column is reduced in the same order."""
    import os
    tc = int(tc or os.environ.get("SEMBLO_TILE_TC", "64"))
    warps = int(os.environ.get("SEMBLO_TILE_WARPS", "4"))
    stages = int(os.environ.get("SEMBLO_TILE_STAGES", "2"))
    if heads not in ("perhead", "packed"):
        raise ValueError(f"heads={heads!r}")
    if triton is None or not vectors.is_cuda:
        raise RuntimeError("the tile-score kernel needs triton and a GPU")
    if reduce not in ("sumexp", "max"):
        raise ValueError(f"reduce={reduce!r}")
    rows_total, num_q_heads, rank, index_dim = vectors.shape
    num_kv_heads, two, num_clusters, df = codebooks.shape
    if rank != 1 or two != 2 or index_dim != 2 * df:
        raise ValueError("expected one routing vector per head and a two-half codebook")
    qb = 16
    if tile % qb or rows_total % tile:
        raise ValueError(f"tile {tile} must be a multiple of {qb} dividing {rows_total} rows")
    group = num_q_heads // num_kv_heads
    tiles = rows_total // tile
    q = vectors.reshape(rows_total, num_kv_heads, group, index_dim).to(torch.bfloat16)
    cents = torch.cat([codebooks[:, 0], codebooks[:, 1]], dim=-1).to(torch.bfloat16).contiguous()
    out = torch.empty(tiles, num_kv_heads, group, num_clusters, dtype=torch.float32, device=vectors.device)
    grid = (tiles, triton.cdiv(num_clusters, tc), num_kv_heads)
    _tile_logit_kernel[grid](
        q, cents, out, num_clusters, 1.0 / float(temperature),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cents.stride(0), cents.stride(1), cents.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        TILE=tile, T=group, I=index_dim, QB=qb, TC=tc, USE_MAX=(reduce == "max"),
        num_warps=warps, num_stages=stages,
    )
    if heads == "packed":
        merged = torch.logsumexp(out, dim=2) if reduce == "sumexp" else out.amax(dim=2)   # [tiles, Hkv, C]
        return torch.log_softmax(merged, dim=-1)
    prob = torch.softmax(out, dim=-1).sum(dim=2)                   # [tiles, Hkv, C]
    return prob.clamp_min_(1e-30).log_()
