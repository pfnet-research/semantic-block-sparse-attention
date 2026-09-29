"""The sliced routing normalisation is the whole one, bit for bit.

The slicing tests set the routing space to the half-split, the only mode with a
sliced loop (the whole-vector and split modes take one pass).
"""

from __future__ import annotations

import torch

from semblo.vllm_plugin.routing_map import routing_normalize as _factor_normalize


def _whole(x, num_factors):
    factors = x.reshape(*x.shape[:-1], num_factors, x.shape[-1] // num_factors)
    return torch.nn.functional.normalize(factors.float(), dim=-1).reshape_as(x).to(x.dtype)


def test_sliced_normalisation_matches_the_whole_bit_for_bit(monkeypatch):
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "half")
    torch.manual_seed(0)
    x = (torch.randn(1, 1000, 4, 16) * 3).to(torch.bfloat16)
    with torch.no_grad():
        sliced = _factor_normalize(x, 2, rows_per_step=300)   # 4,000 rows in 14 slices
        whole = _factor_normalize(x, 2)                        # 4,000 rows, one step
    assert sliced.dtype == x.dtype and sliced.shape == x.shape
    assert torch.equal(sliced, _whole(x, 2))
    assert torch.equal(whole, _whole(x, 2))


def test_the_key_side_follows_the_routing_space(monkeypatch):
    """The key-side normaliser follows SEMBLO_ROUTING_NORMALIZE, including its default.

    It is called by the key router, so it must agree with the query side's
    routing_space in every mode.
    """
    from semblo.inference.flat_index import routing_space

    torch.manual_seed(0)
    x = (torch.randn(1, 40, 4, 16) * 3).to(torch.bfloat16)
    for mode in (None, "whole", "half", "split:4", "none"):
        if mode is None:
            monkeypatch.delenv("SEMBLO_ROUTING_NORMALIZE", raising=False)
        else:
            monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", mode)
        with torch.no_grad():
            got = _factor_normalize(x, 2)
        if mode == "none":
            assert torch.equal(got, x)
            continue
        want = routing_space(x).to(x.dtype)
        torch.testing.assert_close(got.float(), want.float(), atol=2e-2, rtol=2e-2)
