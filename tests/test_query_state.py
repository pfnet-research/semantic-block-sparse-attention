"""The routing vectors of the queries (SembloAttention._query_state).

A query goes through its KV head's routing map, the identity by default, and is
normalized like the keys (SEMBLO_ROUTING_NORMALIZE).
"""
from __future__ import annotations

import types

import pytest
import torch


def _query_state(layer, q, weight):
    from semblo.vllm_plugin.model import SembloAttention
    source = types.SimpleNamespace(semblo_key_router=types.SimpleNamespace(weight=weight))
    return SembloAttention._query_state(layer, q, source)


def _layer(heads=4, kv_heads=2):
    return types.SimpleNamespace(layer_idx=3, num_heads=heads, num_kv_heads=kv_heads,
                                 runtime_config=types.SimpleNamespace(num_factors=2))


def _identity(kv_heads=2, dim=256):
    return torch.eye(dim).expand(kv_heads, dim, dim).contiguous()


def test_the_default_state_is_the_normalised_query(monkeypatch):
    """The default routing space: one unit vector over the whole vector."""
    monkeypatch.delenv("SEMBLO_ROUTING_NORMALIZE", raising=False)
    torch.manual_seed(0)
    q = torch.randn(7, 4, 256).to(torch.bfloat16)
    state = _query_state(_layer(), q, _identity())
    assert tuple(state.vectors.shape) == (1, 7, 4, 1, 256)
    assert tuple(state.mixture_weights.shape) == (1, 7, 4, 1)
    assert torch.all(state.mixture_weights == 1.0)
    vec = state.vectors[0, :, :, 0].float()
    torch.testing.assert_close(vec, torch.nn.functional.normalize(q.float(), dim=-1),
                               atol=1e-2, rtol=1e-2)


def test_the_half_split_space(monkeypatch):
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "half")
    torch.manual_seed(0)
    q = torch.randn(7, 4, 256).to(torch.bfloat16)
    halves = _query_state(_layer(), q, _identity()).vectors[0, :, :, 0].float().view(7, 4, 2, 128)
    want = torch.nn.functional.normalize(q.float().view(7, 4, 2, 128), dim=-1)
    torch.testing.assert_close(halves, want, atol=1e-2, rtol=1e-2)


def test_the_unnormalised_space_passes_the_query_through(monkeypatch):
    """Positional blocks score raw queries against pooled keys."""
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "none")
    torch.manual_seed(0)
    q = torch.randn(7, 4, 256).to(torch.bfloat16)
    state = _query_state(_layer(), q, _identity())
    torch.testing.assert_close(state.vectors[0, :, :, 0].float(), q.float())


def test_a_linear_map_projects_with_the_kv_head_of_each_query(monkeypatch):
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "none")
    torch.manual_seed(1)
    weight = torch.randn(2, 32, 256)                   # [Hkv, I, D]
    q = torch.randn(5, 4, 256).to(torch.bfloat16)
    got = _query_state(_layer(), q, weight).vectors[0, :, :, 0].float()
    q_to_kv = torch.tensor([0, 0, 1, 1])
    want = torch.einsum("qhd,hid->qhi", q.float(), weight[q_to_kv])
    torch.testing.assert_close(got, want, atol=0.5, rtol=2e-2)


def test_the_routing_normalisation_modes(monkeypatch):
    from semblo.vllm_plugin.routing_map import routing_normalize
    torch.manual_seed(3)
    x = torch.randn(9, 2, 256).to(torch.bfloat16)
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "half")
    halves = routing_normalize(x, 2).float()
    torch.testing.assert_close(halves.view(9, 2, 2, 128).norm(dim=-1), torch.ones(9, 2, 2), atol=1e-2, rtol=0)
    monkeypatch.delenv("SEMBLO_ROUTING_NORMALIZE", raising=False)
    torch.testing.assert_close(routing_normalize(x, 2).float().norm(dim=-1), torch.ones(9, 2), atol=1e-2, rtol=0)


def test_the_kernel_matches_the_torch_chain():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    try:
        import triton  # noqa: F401
    except ImportError:
        pytest.skip("needs triton")
    from semblo.vllm_plugin import trivial_router as tr
    torch.manual_seed(4)
    for factors in (2, 1):
        x = torch.randn(1000, 16, 256, device="cuda").to(torch.bfloat16)
        want = tr.trivial_router_vectors(x.cpu(), factors).cuda()      # the torch chain
        got = tr.trivial_router_vectors(x, factors)                     # the kernel
        torch.testing.assert_close(got.float(), want.float(), atol=1e-2, rtol=1e-2)
        differing = (got != want).float().mean().item()
        assert differing < 0.02, f"{differing:.3%} of the elements differ at the last bit"


def test_the_identity_test_runs_once_per_layer_not_once_per_call(monkeypatch):
    """torch.equal synchronises the device, which would break CUDA graph capture
    of the decode step, so the identity check is cached per layer."""
    monkeypatch.delenv("SEMBLO_ROUTING_NORMALIZE", raising=False)
    calls = []
    real_equal = torch.equal
    monkeypatch.setattr(torch, "equal", lambda a, b: (calls.append(1), real_equal(a, b))[1])
    layer, weight = _layer(), _identity()
    q = torch.randn(3, 4, 256, dtype=torch.bfloat16)
    states = [_query_state(layer, q, weight) for _ in range(3)]
    assert len(calls) == 1, f"torch.equal ran {len(calls)} times for three calls"
    assert all(real_equal(s.vectors, states[0].vectors) for s in states[1:])
