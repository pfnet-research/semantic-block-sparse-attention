"""Quest-style min/max descriptors for the contiguous-block baseline.

The positional baseline (SEMBLO_BLOCK_TOKENS=B) represents a block by the mean
of its keys.  With SEMBLO_BLOCK_DESCRIPTOR=minmax it keeps the per-coordinate
minimum and maximum instead, and a query scores a block as

    s_c(q) = sum_j max(q_j k^min_{c,j}, q_j k^max_{c,j}),

the query-aware upper bound that Quest (Tang et al., 2024) ranks KV pages with.
The partition, the budget, the query sharing and the attention body are
unchanged.

The bound is a dot product in a doubled space.  For q_j >= 0 the max picks
q_j k^max_j, for q_j < 0 it picks q_j k^min_j, so with

    descriptor  d_c = [k^max_c ; k^min_c]          (width 2 d)
    query       t(q) = [max(q, 0) ; min(q, 0)]     (width 2 d)

s_c(q) = t(q) . d_c exactly, and the existing centroid scorer computes it
unchanged.  The routing map must stack the identity twice, so that projected
keys and queries arrive as [k ; k] and [q ; q].  The descriptor takes the
block maximum over the first half and the minimum over the second, and the
query transform below rectifies each half.
"""
from __future__ import annotations

import os

import torch


def enabled() -> bool:
    return os.environ.get("SEMBLO_BLOCK_DESCRIPTOR", "mean") == "minmax"


def block_minmax_descriptors(space: torch.Tensor, assign: torch.Tensor,
                             num_clusters: int) -> torch.Tensor:
    """[L, 2d] projected keys ([k ; k]) and [L] block ids -> [C, 2d] descriptors.

    Row c is [max over the block of the first half ; min over the block of the
    second half].  Blocks with no token keep a zero row; they are never
    selected (n_c = 0).
    """
    length, width = space.shape
    if width % 2:
        raise ValueError(f"the doubled routing width must be even, got {width}")
    half = width // 2
    idx = assign.long().unsqueeze(1)
    # Half-precision keys are reduced in fp32. float64 input keeps its precision.
    src = space if space.dtype == torch.float64 else space.float()
    kmax = torch.zeros(num_clusters, half, dtype=src.dtype, device=space.device)
    kmin = torch.zeros(num_clusters, half, dtype=src.dtype, device=space.device)
    kmax.scatter_reduce_(0, idx.expand(length, half), src[:, :half], reduce="amax", include_self=False)
    kmin.scatter_reduce_(0, idx.expand(length, half), src[:, half:], reduce="amin", include_self=False)
    return torch.cat([kmax, kmin], dim=1).to(space.dtype)


def quest_query_vectors(projected: torch.Tensor) -> torch.Tensor:
    """[.., 2d] projected queries ([q ; q]) -> [.., 2d] = [max(q, 0) ; min(q, 0)]."""
    width = projected.shape[-1]
    if width % 2:
        raise ValueError(f"the doubled routing width must be even, got {width}")
    half = width // 2
    pos = torch.relu(projected[..., :half])
    neg = -torch.relu(-projected[..., half:])
    return torch.cat([pos, neg], dim=-1)


def quest_bound_reference(q: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> torch.Tensor:
    """sum_j max(q_j kmin_j, q_j kmax_j) for q [.., d], kmin/kmax [C, d] -> [.., C]."""
    a = q.unsqueeze(-2) * kmin.unsqueeze(0)
    b = q.unsqueeze(-2) * kmax.unsqueeze(0)
    return torch.maximum(a, b).sum(-1)
