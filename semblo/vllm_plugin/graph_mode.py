"""The CUDA-graph mode a run asked for against the one the engine resolved.

vLLM resolves the requested ``cudagraph_mode`` against the model and its
attention backends and may silently pick a weaker one (for example PIECEWISE
instead of FULL_DECODE_ONLY). Benchmarks record the resolved mode next to the
requested one, and ``--require-cudagraph-mode`` makes a run fail when the
resolved mode differs.
"""
from __future__ import annotations

from typing import Any


def resolved_cudagraph_mode(llm: Any) -> str:
    """The name of the mode the engine ended up with, or "unknown" if it cannot be read."""

    engine = getattr(llm, "llm_engine", llm)
    for path in (("vllm_config",), ("engine_core", "vllm_config"), ("model_config", "vllm_config")):
        obj = engine
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            cc = getattr(obj, "compilation_config", None)
            mode = getattr(cc, "cudagraph_mode", None)
            if mode is not None:
                return str(getattr(mode, "name", mode))
    return "unknown"


def check_cudagraph_mode(requested: str | None, resolved: str, require: str | None) -> str:
    """A log line comparing the requested and resolved modes.

    ``requested`` is the command-line mode (None: vLLM's default), ``resolved``
    the mode the engine reports, and ``require`` the mode the run must use
    (None: no requirement). Raises RuntimeError if ``require`` is not met.
    """

    line = f"[bench] cudagraph mode: requested {requested or 'default'}, resolved {resolved}"
    if requested and resolved not in ("unknown", requested):
        line += "  <-- NOT what was asked for"
    if require and resolved != require:
        raise RuntimeError(
            f"this measurement requires cudagraph mode {require} but the engine resolved "
            f"{resolved} (requested {requested or 'default'}); refusing to produce a number "
            "that would be compared against the wrong kind of run"
        )
    return line
