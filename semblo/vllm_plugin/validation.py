"""Safety checks on the search path that do not synchronise with the host.

Reading a device tensor on the host for every check would stall the pipeline
and rule out CUDA graphs. Instead, the checked values are accumulated into one
small device tensor and read once per forward pass. A failure is reported one
step after it happened, and the message still carries the offending value,
because the slots hold values rather than flags.

The checker has three states. Disabled (the default): each checked function
raises immediately on the host. Active: checks are deferred as described
above; the model enables this once the runtime is built. Off: no checks at
all, which removes their extra kernel launches from the decode step. Off is
chosen at model construction.
"""

from __future__ import annotations

import torch

try:  # Optional on CPU-only development machines.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


SLOT_MIN_SELECTED = 0
SLOT_MAX_SELECTED = 1
SLOT_MAX_RANGES = 2
SLOT_SHORTFALL = 3
SLOT_MAX_TILES = 4
SLOT_SUFFIX_BREAK = 5
SLOT_DECODE_ORDER = 6
SLOT_UNION_OVERFLOW = 7
SLOT_BUCKET_COLLAPSE = 8
NUM_SLOTS = 9

_UNSET = -1

MODE_MAX = 0
MODE_MIN = 1
MODE_COUNT = 2


if triton is not None:

    @triton.jit
    def _fold_check_kernel(
        values_ptr,   # [NUM_SLOTS] int64
        source_ptr,   # flattened tensor to reduce
        slot,
        count,
        MODE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fold one tensor into one slot: largest, smallest, or how many hold."""

        offs = tl.arange(0, BLOCK)
        if MODE == 0:
            acc = tl.full((), -(2**62), tl.int64)
        elif MODE == 1:
            acc = tl.full((), 2**62, tl.int64)
        else:
            acc = tl.zeros((), tl.int64)
        base = 0
        while base < count:
            index = base + offs
            live = index < count
            chunk = tl.load(source_ptr + index, mask=live, other=0).to(tl.int64)
            if MODE == 0:
                acc = tl.maximum(acc, tl.max(tl.where(live, chunk, -(2**62)), axis=0))
            elif MODE == 1:
                acc = tl.minimum(acc, tl.min(tl.where(live, chunk, 2**62), axis=0))
            else:
                acc += tl.sum(tl.where(live, chunk, 0), axis=0)
            base += BLOCK
        previous = tl.load(values_ptr + slot)
        if MODE == 0:
            result = tl.maximum(previous, acc)
        elif MODE == 1:
            result = tl.where(previous < 0, acc, tl.minimum(previous, acc))
        else:
            result = tl.maximum(previous, 0) + acc
        tl.store(values_ptr + slot, result)

# How a slot's value is judged when it is read: the recorded number must stay
# at or below the limit ("at_most"), at or above it ("at_least"), or the slot
# counts occurrences of something that must never happen ("never").
AT_MOST = "at_most"
AT_LEAST = "at_least"
NEVER = "never"


class DeferredSearchChecks:
    """Accumulate the search path's checks on the device, read them once."""

    def __init__(self) -> None:
        self._values: torch.Tensor | None = None
        self._rules: dict[int, tuple[str, str, int]] = {}
        self._off = False
        self._staging: torch.Tensor | None = None
        self._pending: torch.cuda.Event | None = None

    def enable(self, device: torch.device) -> None:
        self._values = torch.full((NUM_SLOTS,), _UNSET, dtype=torch.int64, device=device)
        self._rules.clear()
        if device.type == "cuda":
            self._staging = torch.full(
                (NUM_SLOTS,), _UNSET, dtype=torch.int64, pin_memory=True
            )
        self._pending = None

    def disable(self) -> None:
        self._values = None
        self._rules.clear()
        self._staging = None
        self._pending = None

    def turn_off(self) -> None:
        """Skip every check, deferred and immediate, until ``allow`` is called.

        Call sites test ``off`` before both their deferred and synchronous
        branches, so no check kernels or host reads remain.
        """

        self._off = True
        self.disable()

    def allow(self) -> None:
        self._off = False

    @property
    def off(self) -> bool:
        return self._off

    @property
    def active(self) -> bool:
        return self._values is not None

    def _rule(self, slot: int, message: str, comparison: str, limit: int) -> None:
        self._rules[slot] = (message, comparison, int(limit))

    def record_max(
        self, slot: int, value: torch.Tensor, *, message: str, limit: int
    ) -> None:
        """Fail later if the largest value recorded here exceeds ``limit``."""

        self._rule(slot, message, AT_MOST, limit)
        self._fold(slot, value, MODE_MAX)

    def record_min(
        self, slot: int, value: torch.Tensor, *, message: str, limit: int
    ) -> None:
        """Fail later if the smallest value recorded here falls below ``limit``."""

        self._rule(slot, message, AT_LEAST, limit)
        self._fold(slot, value, MODE_MIN)

    def record_violations(
        self, slot: int, condition: torch.Tensor, *, message: str
    ) -> None:
        """Fail later if ``condition`` was ever true."""

        self._rule(slot, message, NEVER, 0)
        self._fold(slot, condition, MODE_COUNT)

    def _fold(self, slot: int, value: torch.Tensor, mode: int) -> None:
        """Reduce ``value`` into one slot.

        On CUDA this is a single Triton kernel, since decode is bound by the
        number of launched operations. The torch fallback takes several.
        """

        flat = value.reshape(-1)
        if triton is not None and flat.is_cuda:
            _fold_check_kernel[(1,)](
                self._values, flat, slot, int(flat.numel()),
                MODE=mode, BLOCK=1024,
                num_warps=4, num_stages=1,
            )
            return
        if mode == MODE_MAX:
            self._values[slot] = torch.maximum(
                self._values[slot], flat.max().to(torch.int64)
            )
        elif mode == MODE_MIN:
            smallest = flat.min().to(torch.int64)
            unset = self._values[slot] < 0
            self._values[slot] = torch.where(
                unset, smallest, torch.minimum(self._values[slot], smallest)
            )
        else:
            self._values[slot] = torch.clamp_min(self._values[slot], 0) + flat.sum().to(
                torch.int64
            )

    def check_and_reset(self) -> None:
        """Read every slot in one transfer and raise on the first failure."""

        if not self.active or not self._rules:
            return
        values = self._values.tolist()
        self._values.fill_(_UNSET)
        self._pending = None
        self._judge(values)

    def check_async(self) -> None:
        """Deferred read without a host synchronisation.

        ``check_and_reset`` waits for all queued kernels. Here the values are
        copied to pinned host memory asynchronously and judged on a later call,
        once the copy's event has completed, so a violation surfaces a few
        steps late but the host can run ahead of the device. Values keep
        accumulating while a copy is in flight, so nothing is lost between
        reads.
        """

        if not self.active or not self._rules:
            return
        if self._staging is None:
            self.check_and_reset()
            return
        if self._pending is not None:
            if not self._pending.query():
                return
            self._pending = None
            self._judge([int(v) for v in self._staging.tolist()])
        self._staging.copy_(self._values, non_blocking=True)
        # Stream-ordered after the copy, so the next accumulation starts clean.
        self._values.fill_(_UNSET)
        event = torch.cuda.Event()
        event.record()
        self._pending = event

    def _judge(self, values: list) -> None:
        for slot, (message, comparison, limit) in sorted(self._rules.items()):
            value = int(values[slot])
            if value < 0:
                continue  # nothing recorded since the last read
            if comparison == AT_MOST and value > limit:
                raise RuntimeError(f"{message} (got {value}, limit {limit})")
            if comparison == AT_LEAST and value < limit:
                raise RuntimeError(f"{message} (got {value}, minimum {limit})")
            if comparison == NEVER and value > 0:
                raise RuntimeError(f"{message} ({value} occurrences)")


DEFERRED_CHECKS = DeferredSearchChecks()
