"""Oracle cluster selection: ``SEMBLO_ORACLE_SELECT=clusters``.

A diagnostic, not a serving path.  Each query scores the clusters by the dense attention mass it
puts on them, computed from this layer's post-RoPE query and the packed prompt keys.  The usual
budget rule is then applied unchanged: clusters in score order until ``target_tokens`` visible
tokens, the crossing cluster included, with size penalty ``SEMBLO_ORACLE_BETA`` (default 0).  The
partition, the exact current chunk, the dense prefix and the generated suffix are unchanged.  The
task score under this switch therefore shows what the method would reach if the centroid scorer were
replaced by the dense attention it approximates.

Prefill: the queries of a chunk see the keys before the chunk (the chunk itself is read exactly), so
the mass is over the packed rows whose position is below ``chunk_start``.  Decode: over the whole
prompt.  The dense logits are formed in fp32, in tiles of ``SEMBLO_ORACLE_TILE`` queries against
every packed row.  The per-cluster sums are cumsum differences over the bucket-major rows, and the
padding rows of the page-aligned layout are masked out.  Not graph-capturable (host-side table build
on first use, per-tile loops), so run with SEMBLO_DECODE_GRAPH=0.
"""
from __future__ import annotations

import os

import torch


def enabled() -> bool:
    return os.environ.get("SEMBLO_ORACLE_SELECT", "") == "clusters"


def beta() -> float:
    return float(os.environ.get("SEMBLO_ORACLE_BETA", "0"))


def _tables(index) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per KV head: (rows, cluster of each packed row [rows] int64, real-row mask [rows] bool,
    position of each row [rows] int32).  Built once per index (the padded layout is fixed)."""
    cached = getattr(index, "_oracle_tables", None)
    if cached is not None:
        return cached
    indptr = index.bucket_indptr.to(torch.int64)          # [Hkv, C+1], padded offsets
    sizes = index.bucket_sizes.to(torch.int64)            # [Hkv, C], true counts
    out = []
    for h in range(index.num_kv_heads):
        rows = int(indptr[h, -1])
        row = torch.arange(rows, device=indptr.device)
        cluster = torch.searchsorted(indptr[h, 1:], row, right=True)   # row < indptr[c+1]
        real = (row - indptr[h, cluster]) < sizes[h, cluster]
        pos = index.posting_positions[h, :rows]
        # The rows are bucket-major, so a cluster's mass is a contiguous segment sum: the
        # inclusive cumsum at each cluster's last real row minus the one before its first row.
        last = (indptr[h, :-1] + sizes[h] - 1).clamp_min(0)            # [C]; empty clusters read a zero-width segment
        first_minus_one = indptr[h, :-1] - 1                           # -1 for the first cluster
        out.append((rows, cluster, real, pos, last, first_minus_one, sizes[h] > 0))
    index._oracle_tables = out
    return out


def oracle_cluster_mass(q_chunk: torch.Tensor, index, boundary: int, *, sm_scale: float) -> torch.Tensor:
    """[Q, Hkv, C] fp32: the dense attention mass of each query (its heads of one KV head summed)
    on each cluster, over the packed rows with position < ``boundary``."""
    q_count, num_q_heads, _ = q_chunk.shape
    num_kv_heads, num_clusters = index.num_kv_heads, index.num_buckets
    group = num_q_heads // num_kv_heads
    keys = index.packed_kv[0]                              # [rows_max, Hkv, D]
    tile = int(os.environ.get("SEMBLO_ORACLE_TILE", "256"))
    out = torch.zeros(q_count, num_kv_heads, num_clusters, dtype=torch.float32, device=q_chunk.device)
    # TF32 for the [tile*group, D] x [D, rows] products (SEMBLO_ORACLE_TF32=0 for full fp32).
    # Ten mantissa bits are ample for ranking clusters.
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = os.environ.get("SEMBLO_ORACLE_TF32", "1") == "1"
    try:
        _mass_by_head(q_chunk, index, boundary, sm_scale, keys, tile, out, group, num_clusters)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    return out


def _mass_by_head(q_chunk, index, boundary, sm_scale, keys, tile, out, group, num_clusters) -> None:
    q_count = q_chunk.shape[0]
    for h, (rows, cluster, real, pos, last, first_minus_one, nonempty) in enumerate(_tables(index)):
        valid = real & (pos < boundary)                    # [rows]
        if not bool(valid.any()):
            continue
        k_h = keys[:rows, h].float()                       # [rows, D] post-RoPE; fp32 so the logits are not bf16-rounded
        neg = torch.zeros(rows, dtype=torch.float32, device=q_chunk.device).masked_fill_(~valid, float("-inf"))
        for lo in range(0, q_count, tile):
            hi = min(lo + tile, q_count)
            q_t = q_chunk[lo:hi, h * group:(h + 1) * group].reshape(-1, q_chunk.shape[-1])   # [t*group, D]
            logits = (q_t.float() @ k_h.T).mul_(sm_scale).add_(neg)                        # [t*group, rows] fp32
            p = torch.softmax(logits, dim=-1)
            cs = p.cumsum(dim=-1)                                                          # masked rows contribute 0
            upper = cs[:, last]                                                            # [t*group, C]
            lower = torch.where(first_minus_one[None, :] >= 0, cs[:, first_minus_one.clamp_min(0)], torch.zeros_like(upper))
            mass = (upper - lower).clamp_min_(0.0) * nonempty[None, :].float()
            out[lo:hi, h] = mass.view(hi - lo, group, num_clusters).sum(dim=1)
            del logits, p, cs, upper, lower, mass


def select(prob: torch.Tensor, causal_sizes: torch.Tensor, bitmask: torch.Tensor, *, target_tokens: int,
           g_union: int, iters: int = 16, qmask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The usual budget rule over the oracle scores, with the flat selector's kernel.

    Returns (counts, tokens) [Q, Hkv] int32 and writes the bitmask (and qmask) in place."""
    from semblo.vllm_plugin.fused_select import _run_streaming_select
    q_count, num_kv_heads, num_clusters = prob.shape
    counts = torch.empty(q_count, num_kv_heads, dtype=torch.int32, device=prob.device)
    tokens = torch.empty_like(counts)
    _run_streaming_select(prob.contiguous(), causal_sizes, bitmask, counts, tokens,
                          beta=beta(), target_tokens=target_tokens, b_dim=num_clusters,
                          tile=512, iters=iters, g_union=g_union, qmask=qmask)
    return counts, tokens


def prefill_select(q_chunk, index, causal_sizes, bitmask, *, chunk_start: int, target_tokens: int,
                   sm_scale: float, sel_q: int, sel_g: int, iters: int, qmask=None):
    """The prefill hook: per-query oracle scores, or one score per block when the caller runs in
    the block-consensus form (sel_q blocks of q_count/sel_q queries)."""
    prob = oracle_cluster_mass(q_chunk, index, chunk_start, sm_scale=sm_scale)
    q_count = prob.shape[0]
    if sel_q != q_count:                                   # consensus: one row per block
        per = (q_count + sel_q - 1) // sel_q
        padded = torch.zeros(sel_q * per, *prob.shape[1:], dtype=prob.dtype, device=prob.device)
        padded[:q_count] = prob
        prob = padded.view(sel_q, per, *prob.shape[1:]).sum(dim=1)
    return select(prob, causal_sizes, bitmask, target_tokens=target_tokens, g_union=sel_g, iters=iters, qmask=qmask)


def decode_select(q_chunk, index, sizes3, bitmask, *, target_tokens: int, sm_scale: float, iters: int):
    prob = oracle_cluster_mass(q_chunk, index, int(index.prompt_length), sm_scale=sm_scale)
    return select(prob, sizes3, bitmask, target_tokens=target_tokens, g_union=1, iters=iters)
