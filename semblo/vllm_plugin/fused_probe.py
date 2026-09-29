"""Effective-token accounting for fused prefill (SEMBLO_FUSED_STATS=1)."""

from __future__ import annotations

import os

import torch


class FusedSelectionStats:
    """Effective-token accounting for the fused path (SEMBLO_FUSED_STATS=1).

    Records what the selection took (per-query causal tokens), what the body
    read (block union tokens), and overflow events.  Accumulates on the
    device; call summary() once after the request."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.calls = 0
        self.sel_sum = None      # sum over (query, head) of selected tokens
        self.sel_max = None
        self.sel_n = 0
        self.union_sum = None    # sum over (block, head) of union tokens
        self.union_max = None
        self.union_n = 0
        self.overflow = None     # unions clamped by umax

    def record(self, sel_tokens, union_tokens, overflowed) -> None:
        s_sum = sel_tokens.sum(dtype=torch.int64)
        s_max = sel_tokens.max()
        u_sum = union_tokens.sum(dtype=torch.int64)
        u_max = union_tokens.max()
        o = overflowed.sum(dtype=torch.int64)
        if self.sel_sum is None:
            self.sel_sum, self.sel_max = s_sum, s_max
            self.union_sum, self.union_max = u_sum, u_max
            self.overflow = o
        else:
            self.sel_sum += s_sum
            self.sel_max = torch.maximum(self.sel_max, s_max)
            self.union_sum += u_sum
            self.union_max = torch.maximum(self.union_max, u_max)
            self.overflow += o
        self.sel_n += sel_tokens.numel()
        self.union_n += union_tokens.numel()
        self.calls += 1

    def print_summary(self) -> None:
        """Print the accumulated accounting (registered at process exit)."""
        s = self.summary()
        if s.get("calls"):
            print("FUSEDSTATS " + " ".join(
                f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in s.items()), flush=True)

    def summary(self) -> dict:
        if self.sel_sum is None:
            return {"calls": 0}
        return {
            "calls": self.calls,
            "selected_tokens_mean": self.sel_sum.item() / max(self.sel_n, 1),
            "selected_tokens_max": int(self.sel_max.item()),
            "union_tokens_mean": self.union_sum.item() / max(self.union_n, 1),
            "union_tokens_max": int(self.union_max.item()),
            "union_overflow_events": int(self.overflow.item()),
        }


FUSED_STATS = FusedSelectionStats()

if os.environ.get("SEMBLO_FUSED_STATS") == "1":  # pragma: no cover
    import atexit

    atexit.register(FUSED_STATS.print_summary)
