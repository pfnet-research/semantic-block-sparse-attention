"""The trivial router as one kernel.

With zero residual and zero mixture weights, the router's query state is the
query itself, normalised per factor.  In torch that is six passes over a
[Q, Hq, D] tensor (cast, norm, divide, cast, norm, divide). This kernel does
the same chain in one pass with the same rounding points: normalise in fp32,
round to bf16, normalise the rounded vector, round again.
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
    def _trivial_router_kernel(
        x_ptr, out_ptr, n_rows,
        stride_x, stride_o,
        D: tl.constexpr,           # head dim
        F: tl.constexpr,           # factors: 2 halves, or 1 for the whole vector
        BLOCK: tl.constexpr,       # rows per program
    ):
        DF: tl.constexpr = D // F
        pid = tl.program_id(0)
        rows = pid * BLOCK + tl.arange(0, BLOCK)
        live = rows < n_rows
        d = tl.arange(0, DF)
        for f in tl.static_range(F):
            x = tl.load(x_ptr + rows[:, None] * stride_x + (f * DF + d)[None, :],
                        mask=live[:, None], other=0.0).to(tl.float32)
            norm = tl.sqrt(tl.sum(x * x, axis=1))
            unit = x / tl.maximum(norm, 1e-12)[:, None]
            unit = unit.to(tl.bfloat16).to(tl.float32)
            norm2 = tl.sqrt(tl.sum(unit * unit, axis=1))
            out = unit / tl.maximum(norm2, 1e-12)[:, None]
            tl.store(out_ptr + rows[:, None] * stride_o + (f * DF + d)[None, :],
                     out.to(tl.bfloat16), mask=live[:, None])


def trivial_router_vectors(x: torch.Tensor, factors: int,
                           out: torch.Tensor | None = None) -> torch.Tensor:
    """[Q, Hq, D] bf16 -> [Q, Hq, D] bf16, normalised per factor.

    Falls back to torch off the GPU.  If given, ``out`` (same shape, bf16,
    contiguous) receives the result, so a captured decode step does not
    allocate."""
    if triton is None or not x.is_cuda:
        q, h, d = x.shape
        halves = x.float().view(q, h, factors, d // factors)
        unit = halves / halves.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit = unit.to(x.dtype).float()
        unit = unit / unit.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return unit.to(x.dtype).view(q, h, d)
    q, h, d = x.shape
    flat = x.reshape(q * h, d)
    if flat.stride(1) != 1:
        flat = flat.contiguous()
    if out is None:
        out = torch.empty_like(flat)
    else:
        assert out.shape == x.shape and out.dtype == x.dtype and out.is_contiguous(), (out.shape, x.shape)
        out = out.view(q * h, d)
    block = 16
    _trivial_router_kernel[((flat.shape[0] + block - 1) // block,)](
        flat, out, flat.shape[0], flat.stride(0), out.stride(0),
        D=d, F=factors, BLOCK=block, num_warps=4)
    return out.view(q, h, d)
