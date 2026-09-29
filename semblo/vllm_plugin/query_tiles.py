"""Pack and unpack the selector's per-query cluster masks."""

from __future__ import annotations

import torch


def unpack_qmask(qmask: torch.Tensor, num_buckets: int) -> torch.Tensor:
    """[Q, H, W] int32 packed bits -> [Q, H, B] bool."""
    shifts = torch.arange(32, device=qmask.device, dtype=torch.int32)
    bits = (qmask.unsqueeze(-1) >> shifts) & 1
    return bits.flatten(-2)[..., :num_buckets] > 0


def pack_bits(bits: torch.Tensor) -> torch.Tensor:
    """[..., B] bool -> [..., B/32] int32, matching the selector's layout."""
    if bits.shape[-1] % 32:
        raise ValueError("the bucket axis must be a multiple of 32")
    shifts = torch.arange(32, device=bits.device, dtype=torch.int64)
    words = (bits.reshape(*bits.shape[:-1], -1, 32).long() << shifts).sum(-1)
    # Bit 31 makes the word negative in int32.  The truncating cast gives the
    # two's-complement value the kernel's arithmetic shift expects.
    return words.to(torch.int32)
