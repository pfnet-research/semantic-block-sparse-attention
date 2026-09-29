"""The cluster-ordered copy against gathering from prompt order every chunk.

`gather_from` rebuilds, for each chunk, the visible rows of the
clusters the chunk touches out of the prompt-order K/V, in the copy's own
layout, and FA3 reads that scratch.  The values FA3 sees are the same, so the
output must equal the packed path's bit for bit.  `num_splits` is FA3's
split-KV count, which only rearranges the same sums.
"""
from __future__ import annotations

import pytest
import torch

import importlib.util
import pathlib

_sibling = pathlib.Path(__file__).with_name("test_kv_outer_fused_local.py")
_spec = importlib.util.spec_from_file_location("_kv_outer_fused_local_helpers", _sibling)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_needs_hopper, _synthetic = _helpers._needs_hopper, _helpers._synthetic


def _prompt_order(case, seed):
    """Prompt-order K/V and a per-head posting order that reproduce the packed copy."""
    packed = case["packed_kv"]
    _, length, num_kv_heads, head_dim = packed.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    posting = torch.stack([torch.randperm(length, generator=g) for _ in range(num_kv_heads)])
    posting = posting.to(torch.int32).to(packed.device).contiguous()
    k = torch.empty(length, num_kv_heads, head_dim, dtype=packed.dtype, device=packed.device)
    v = torch.empty_like(k)
    for h in range(num_kv_heads):
        k[posting[h].long(), h] = packed[0, :, h]
        v[posting[h].long(), h] = packed[1, :, h]
    return k, v, posting


def test_gathering_from_prompt_order_is_the_packed_copy_to_the_bit(monkeypatch):
    _needs_hopper()
    from semblo.vllm_plugin.kv_outer import kv_outer_attention
    monkeypatch.setenv("SEMBLO_KV_OUTER_FUSED_LOCAL", "1")
    monkeypatch.delenv("SEMBLO_KV_OUTER_CBATCH", raising=False)
    case = _synthetic(3)
    k, v, posting = _prompt_order(case, 30)
    sm_scale = case["q"].shape[-1] ** -0.5
    common = dict(sm_scale=sm_scale, page_size=case["page"], aligned=True,
                  local_kv=case["local_kv"])
    packed = kv_outer_attention(
        case["q"], case["packed_kv"], case["qmask"], case["causal_sizes"],
        case["bucket_indptr"], table_cache={}, **common)[0]
    cache = {}
    gathered = kv_outer_attention(
        case["q"], case["packed_kv"], case["qmask"], case["causal_sizes"],
        case["bucket_indptr"], table_cache=cache, gather_from=(k, v, posting), **common)[0]
    assert torch.isfinite(gathered.float()).all()
    assert torch.equal(gathered, packed), "the gathered scratch changed bits"
    # the scratch is reused across chunks of the same index, not reallocated
    assert any(isinstance(key, tuple) and key[0] == "gather_scratch" for key in cache)
    later = case["causal_sizes"] // 2
    packed_later = kv_outer_attention(
        case["q"], case["packed_kv"], case["qmask"], later,
        case["bucket_indptr"], table_cache={}, **common)[0]
    gathered_later = kv_outer_attention(
        case["q"], case["packed_kv"], case["qmask"], later,
        case["bucket_indptr"], table_cache=cache, gather_from=(k, v, posting), **common)[0]
    assert torch.equal(gathered_later, packed_later)


@pytest.mark.parametrize("splits", [1, 2])
def test_a_fixed_split_count_agrees_with_the_heuristic(monkeypatch, splits):
    _needs_hopper()
    from semblo.vllm_plugin.kv_outer import kv_outer_attention
    monkeypatch.setenv("SEMBLO_KV_OUTER_FUSED_LOCAL", "1")
    monkeypatch.delenv("SEMBLO_KV_OUTER_CBATCH", raising=False)
    case = _synthetic(4)
    sm_scale = case["q"].shape[-1] ** -0.5
    run = lambda n: kv_outer_attention(
        case["q"], case["packed_kv"], case["qmask"], case["causal_sizes"],
        case["bucket_indptr"], sm_scale=sm_scale, page_size=case["page"],
        aligned=True, local_kv=case["local_kv"], table_cache={}, num_splits=n)[0].float()
    torch.testing.assert_close(run(splits), run(0), atol=3e-2, rtol=3e-2)
