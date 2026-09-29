"""Capture one sparse layer's decode step in a CUDA graph and replay it.

Decode is bound by host dispatch rather than by kernel time. The sparse
attention core cannot go into vLLM's CUDA graphs because it reads an index
built per request. Within one request, however, the index is immutable, every
buffer can be preallocated and the kernel sequence does not vary, so the core
is captured once per request and replayed on later steps.

A captured graph replays the addresses it was captured with, so every tensor
the body reads or writes must stay at a fixed address for the request:

  * ``q``, ``q_pre`` and the output buffer come from vLLM's piecewise CUDA
    graph pool and are fixed for a batch shape (decode of one request is
    always batch shape 1).
  * the packed prompt KV, posting positions, codebooks and block sizes belong
    to the request's index.
  * the paged KV cache and block table belong to the engine.
  * everything the body allocates during capture comes from the graph's
    private pool and is reused on replay.

``signature`` holds the addresses and shapes the body depends on. The graph is
replayed only while they are unchanged, and any change triggers a new capture.
If capture fails, the layer runs eagerly for the rest of the process.

The body must not read a device value on the host or branch on anything that
varies per step. The generated suffix is therefore laid out once over its
whole reservation, and the kernel's causal mask excludes the part not yet
generated.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch


def graphs_enabled() -> bool:
    """Whether decode-step capture is on. ``SEMBLO_DECODE_GRAPH=0`` turns it off."""

    if os.environ.get("SEMBLO_FULL_GRAPH_DECODE") == "1":
        return False          # vLLM captures the whole step
    return os.environ.get("SEMBLO_DECODE_GRAPH", "1") == "1"


def _warmup_steps() -> int:
    """Steps to run eagerly before capturing.

    Two by default: the first decode step allocates the workspaces later steps
    reuse, and Triton compiles on the first launch of a shape, which must not
    happen inside a capture.
    """

    return max(1, int(os.environ.get("SEMBLO_DECODE_GRAPH_WARMUP", "2")))


def _capture(body: Callable[[], None]) -> torch.cuda.CUDAGraph:
    """Record ``body`` into a graph, without emptying the allocator first.

    ``torch.cuda.graph`` is not used because its ``__enter__`` calls
    ``torch.cuda.empty_cache()``. With a KV pool of tens of gigabytes this
    costs about 170 ms per request.

    Each graph gets its own memory pool. A pool shared across layers breaks
    when a request's graphs are all dropped: the next ``capture_begin`` fails
    an allocator assertion (``use_count > 0``). The decode intermediates are
    a few megabytes per layer, so separate pools cost little.

    The ``torch.cuda.synchronize()`` that ``torch.cuda.graph`` also performs
    is required. Without it, memory allocated during capture can still be in
    use by in-flight work, and the graph is committed to addresses that are
    later reused, which surfaces as ``CUDA error: invalid argument`` at a
    later synchronisation. ``stream.wait_stream`` only orders the capture
    after the current stream, not after the engine's other streams.
    """

    graph = torch.cuda.CUDAGraph()
    # SEMBLO_CAPTURE_SYNC=0 skips the synchronisation, for measuring its cost.
    if os.environ.get("SEMBLO_CAPTURE_SYNC", "1") == "1":
        torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.capture_begin(capture_error_mode="thread_local")
        try:
            body()
        finally:
            graph.capture_end()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


def _report_device_time(name: str, graph: torch.cuda.CUDAGraph) -> None:
    """Print the device time of one layer's sparse decode step, if asked.

    ``SEMBLO_DECODE_GRAPH_TIME=<n>`` times n replays with CUDA events right
    after the capture. The graph contains exactly the sparse core of one
    layer and a replay issues no host work, so this measures its device time.

    A replay recomputes the same step, which leaves the output unchanged but
    not the deferred-check accumulators, so this is for diagnostics only.
    """

    iterations = int(os.environ.get("SEMBLO_DECODE_GRAPH_TIME", "0"))
    if iterations <= 0:
        return
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    print(
        f"[semblo] {name}: sparse decode core "
        f"{start.elapsed_time(end) / iterations:.4f} ms/step on the device",
        flush=True,
    )


class DecodeStepGraph:
    """The captured decode step of one layer of one request."""

    __slots__ = (
        "_graph", "_pool", "_signature", "_steps", "_failed", "_name",
        "_wasted", "_replays",
    )

    # If the addresses keep moving, capture is abandoned in favour of the
    # eager path. Only wasted captures count: those discarded before any
    # replay. Each request needs a fresh capture because its index is new
    # memory, so counting all captures would be wrong.
    _CAPTURE_LIMIT = 8

    def __init__(self, name: str) -> None:
        self._graph: torch.cuda.CUDAGraph | None = None
        self._pool: Any = None
        self._signature: tuple | None = None
        self._steps = 0
        self._wasted = 0
        self._replays = 0
        self._failed = False
        self._name = name

    @property
    def captured(self) -> bool:
        return self._graph is not None

    def run(self, signature: tuple, body: Callable[[], None]) -> None:
        """Produce this step's result, replaying the capture when it applies.

        ``body`` must write its result into buffers named by ``signature`` and
        must not read a device value on the host. It is called eagerly while
        warming up, once more to capture, and not again while the signature
        holds.
        """

        if self._failed or not graphs_enabled():
            body()
            return

        if self._graph is not None:
            if signature == self._signature:
                self._replay("reuse", signature)
                self._replays += 1
                return
            # An address moved, so the old capture would write to freed
            # memory. Discard it and warm up again. Only a capture that was
            # never replayed counts as wasted (see _CAPTURE_LIMIT).
            if os.environ.get("SEMBLO_DECODE_GRAPH_DIAG") == "1":
                print(
                    f"[semblo] {self._name}: re-capturing, moved: "
                    f"{_describe_signature_change(self._signature, signature)}",
                    flush=True,
                )
            if self._replays == 0:
                self._wasted += 1
            self.reset()

        self._signature = signature
        self._steps += 1
        if self._steps <= _warmup_steps():
            body()
            return

        if self._wasted >= self._CAPTURE_LIMIT:
            self._failed = True
            print(
                f"[semblo] decode-step capture given up for "
                f"{self._name}: the addresses it depends on kept moving",
                flush=True,
            )
            body()
            return

        try:
            graph = _capture(body)
        except Exception as error:  # capture is only an optimisation
            self._graph = None
            self._failed = True
            print(
                f"[semblo] decode-step capture disabled for "
                f"{self._name}: {type(error).__name__}: {error}",
                flush=True,
            )
            body()
            return
        self._graph = graph
        self._pool = graph.pool()
        # Counts reuses only, not the replay that follows the capture.
        self._replays = 0
        _report_device_time(self._name, graph)
        # Capture records the work without running it, so replay once to
        # produce this step's result.
        self._replay("after capture", signature)

    def _replay(self, when: str, signature: tuple) -> None:
        """Replay the graph, printing the layer and signature if it fails.

        A replay failure means the signature does not describe everything the
        graph depends on, so it is re-raised after the details are printed.

        ``replay`` is asynchronous, so a fault raised by the graph surfaces at
        the next synchronisation. For debugging, ``SEMBLO_DECODE_GRAPH_SYNC``
        synchronises after a replay: ``1`` or ``both`` after either site,
        ``capture`` after the replay that follows a capture, ``reuse`` after
        the per-step replay. This adds a device round trip per layer per step.
        """

        try:
            self._graph.replay()
            want = os.environ.get("SEMBLO_DECODE_GRAPH_SYNC", "")
            if want in ("1", "both") or want == when.split()[-1]:
                torch.cuda.synchronize()
        except Exception as error:
            print(
                f"[semblo] {self._name}: replay ({when}) failed after "
                f"{self._steps} steps and {self._replays} reuses: "
                f"{type(error).__name__}: {error}",
                flush=True,
            )
            for name, part in signature:
                print(f"[semblo]     {name}: {part}", flush=True)
            raise

    def reset(self) -> None:
        self._graph = None
        self._pool = None
        self._signature = None
        self._steps = 0
        self._replays = 0


def tensor_signature(**tensors: torch.Tensor | None) -> tuple:
    """Address, shape, stride and dtype of each named tensor.

    A graph replays the addresses, and its kernels were compiled for the
    rest. Entries are named so that a change can report which tensor moved.
    """

    parts: list[Any] = []
    for name, tensor in tensors.items():
        if tensor is None:
            parts.append((name, None))
            continue
        parts.append(
            (
                name,
                (
                    tensor.data_ptr(),
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    tensor.dtype,
                ),
            )
        )
    return tuple(parts)


def _describe_signature_change(before: tuple, after: tuple) -> str:
    """Name the signature entries that changed."""

    if not isinstance(before, tuple) or not isinstance(after, tuple):
        return "the whole signature"
    names = []
    for old, new in zip(before, after):
        if old == new:
            continue
        name = new[0] if isinstance(new, tuple) and new else "?"
        names.append(str(name))
    extra = abs(len(before) - len(after))
    if extra:
        names.append(f"(+{extra} entries)")
    return ", ".join(names) or "nothing named"
