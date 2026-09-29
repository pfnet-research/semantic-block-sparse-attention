"""The routing map: pre-RoPE keys and queries into the space the blocks are built in.

The method routes on the pre-RoPE keys themselves, so the map is the identity
by default.  A linear map ``W`` per layer and KV head ([Hkv, I, D]) can be
loaded instead, for the routing-key comparisons of the paper (the non-rotary
dimensions, PCA, random projections, the min/max descriptor); see
``scripts/make_routing_map.py``.  Keys and queries go through the same map, and
the result is normalized as ``SEMBLO_ROUTING_NORMALIZE`` says.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SembloQueryState:
    """What the selection kernels read for a chunk of queries."""

    vectors: torch.Tensor                 # [B, Q, Hq, R, I]
    mixture_weights: torch.Tensor         # [B, Q, Hq, R], uniform
    logit_scales: torch.Tensor | None = None
    descriptor_vectors: torch.Tensor | None = None
    descriptor_gamma: torch.Tensor | None = None


_NORMALIZE_ROWS_PER_STEP = 65536


def routing_normalize(x: torch.Tensor, num_factors: int = 2,
                      rows_per_step: int = _NORMALIZE_ROWS_PER_STEP) -> torch.Tensor:
    """Map projected vectors into the routing space (``SEMBLO_ROUTING_NORMALIZE``).

    ``whole`` (default) makes the whole vector unit-norm, ``half`` each of
    ``num_factors`` equal parts, ``none`` leaves it as it is, and ``split:<n>``
    is handled by :func:`semblo.inference.flat_index.routing_space`.  Rows are
    normalized in slices of ``rows_per_step``, which gives the same result and
    bounds the fp32 copies at long prompts.
    """

    mode = os.environ.get("SEMBLO_ROUTING_NORMALIZE", "whole") or "whole"
    if mode == "none":
        return x
    if mode not in ("whole", "half"):
        from semblo.inference.flat_index import routing_space
        return routing_space(x).to(x.dtype)
    if mode == "whole":
        rows = x.reshape(-1, x.shape[-1])
    else:
        if x.shape[-1] % num_factors != 0:
            raise ValueError("index dimension must be divisible by num_factors")
        rows = x.reshape(-1, num_factors, x.shape[-1] // num_factors)
    if torch.is_grad_enabled() or rows.shape[0] <= rows_per_step:
        return F.normalize(rows.float(), dim=-1).reshape_as(x).to(x.dtype)
    out = torch.empty_like(rows)
    for start in range(0, rows.shape[0], rows_per_step):
        stop = min(rows.shape[0], start + rows_per_step)
        out[start:stop] = F.normalize(rows[start:stop].float(), dim=-1).to(x.dtype)
    return out.reshape_as(x)


class SembloKeyRouter(nn.Module):
    """The routing map of one layer: ``weight`` [Hkv, I, D], the identity by default."""

    def __init__(self, num_kv_heads: int, head_dim: int, index_dim: int,
                 num_factors: int = 2) -> None:
        super().__init__()
        if index_dim <= 0 or index_dim % num_factors != 0:
            raise ValueError("index_dim must be positive and divisible by num_factors")
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.index_dim = int(index_dim)
        self.num_factors = int(num_factors)
        self.weight = nn.Parameter(torch.zeros(num_kv_heads, index_dim, head_dim))
        if index_dim == head_dim:
            with torch.no_grad():
                self.weight.copy_(torch.eye(head_dim).expand(num_kv_heads, head_dim, head_dim))

    def project(self, keys: torch.Tensor) -> torch.Tensor:
        """[..., Hkv, D] -> [..., Hkv, I], without normalization."""

        return torch.einsum("...hd,hid->...hi", keys, self.weight.to(dtype=keys.dtype))

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim != 4:
            raise ValueError("keys must have shape [B,L,Hkv,D]")
        if keys.shape[2:] != (self.num_kv_heads, self.head_dim):
            raise ValueError("key tensor does not match router dimensions")
        return routing_normalize(self.project(keys), self.num_factors)
