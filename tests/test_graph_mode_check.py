"""A clock records the CUDA-graph mode the engine resolved, and can refuse the wrong one.

vLLM may resolve a requested mode such as FULL_DECODE_ONLY to PIECEWISE
without an error, so timings taken under different modes could be compared
by mistake.  The resolved mode is reported, and can be required.
"""
from __future__ import annotations

import types

import pytest

from semblo.vllm_plugin.graph_mode import check_cudagraph_mode, resolved_cudagraph_mode


def _llm(mode_name):
    mode = types.SimpleNamespace(name=mode_name)
    cc = types.SimpleNamespace(cudagraph_mode=mode)
    cfg = types.SimpleNamespace(compilation_config=cc)
    return types.SimpleNamespace(llm_engine=types.SimpleNamespace(vllm_config=cfg))


def test_the_resolved_mode_is_read_from_the_engine():
    assert resolved_cudagraph_mode(_llm("PIECEWISE")) == "PIECEWISE"
    assert resolved_cudagraph_mode(types.SimpleNamespace()) == "unknown"


def test_a_weakened_mode_is_named_in_the_line():
    line = check_cudagraph_mode("FULL_DECODE_ONLY", "PIECEWISE", None)
    assert "requested FULL_DECODE_ONLY" in line and "resolved PIECEWISE" in line
    assert "NOT what was asked for" in line
    assert "NOT" not in check_cudagraph_mode("FULL_DECODE_ONLY", "FULL_DECODE_ONLY", None)
    assert "NOT" not in check_cudagraph_mode(None, "PIECEWISE", None)


def test_a_required_mode_refuses_anything_else():
    with pytest.raises(RuntimeError, match="requires cudagraph mode FULL_DECODE_ONLY"):
        check_cudagraph_mode("FULL_DECODE_ONLY", "PIECEWISE", "FULL_DECODE_ONLY")
    check_cudagraph_mode("FULL_DECODE_ONLY", "FULL_DECODE_ONLY", "FULL_DECODE_ONLY")
