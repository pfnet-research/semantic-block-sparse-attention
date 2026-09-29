"""An attention sink in the fitted partition (SEMBLO_SINK_TOKENS).

The first S positions form cluster 0, the other positions are fitted into clusters 1..C-1, and
cluster 0's descriptor is zero, so no score and no nearest-cluster assignment ever lands there.
The selectors then take cluster 0 first, inside the budget (fused_prefill._hold_out_sink,
_force_sink_bit).  These tests check that:
  * with S unset the partition is bit-identical to the plain fit;
  * positions < S are exactly cluster 0, in order, and nothing else is;
  * cluster 0's descriptor is zero and the other clusters' are not;
  * the settings a reserved cluster does not compose with are refused;
  * the selector's view hides cluster 0, the budget shrinks by S, and the forced bit is bit 0.
"""
from __future__ import annotations

import os

import pytest
import torch

from semblo.vllm_plugin.fused_prefill import _force_sink_bit, _hold_out_sink
from semblo.vllm_plugin.query_tiles import unpack_qmask
from semblo.vllm_plugin.state import ProductPostingIndex


def _keys(length, hkv=2, dim=16, seed=0):
    return torch.randn(length, hkv, dim, generator=torch.Generator().manual_seed(seed))


def _build(keys, *, sink=0, unit_tokens=16, min_log2=3):
    env = {"SEMBLO_FLAT_UNIT_TOKENS": str(unit_tokens)}
    if sink:
        env["SEMBLO_SINK_TOKENS"] = str(sink)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return ProductPostingIndex.build(
            keys, torch.arange(keys.shape[0]), num_centroids=4, kmeans_iters=2,
            kmeans_sample_size=256, flat_min_clusters_log2=min_log2, codebook_seed=0)
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _members(index, head, cluster):
    start = int(index.bucket_indptr[head][cluster])
    size = int(index.bucket_sizes[head][cluster])
    return index.posting_positions[head][start:start + size]


def test_without_the_sink_the_partition_is_unchanged(monkeypatch):
    monkeypatch.delenv("SEMBLO_SINK_TOKENS", raising=False)
    keys = _keys(512)
    a = _build(keys)
    b = _build(keys, sink=0)
    assert torch.equal(a.bucket_sizes, b.bucket_sizes)
    assert torch.equal(a.posting_positions, b.posting_positions)
    assert torch.equal(a.codebooks, b.codebooks)


def test_the_first_positions_are_cluster_zero_and_nothing_else_is(monkeypatch):
    monkeypatch.delenv("SEMBLO_SINK_TOKENS", raising=False)
    sink = 32
    index = _build(_keys(512), sink=sink)
    for head in range(index.bucket_sizes.shape[0]):
        assert torch.equal(_members(index, head, 0), torch.arange(sink))
        for c in range(1, index.bucket_sizes.shape[1]):
            assert (_members(index, head, c) >= sink).all()
        assert int(index.bucket_sizes[head].sum()) == 512


def test_the_sink_descriptor_is_zero_and_the_others_are_not(monkeypatch):
    monkeypatch.delenv("SEMBLO_SINK_TOKENS", raising=False)
    index = _build(_keys(512), sink=32)
    cb = index.codebooks.float()                       # [Hkv, 2, C, DF]
    assert torch.count_nonzero(cb[:, :, 0]) == 0
    assert torch.isfinite(cb).all()
    used = index.bucket_sizes[:, 1:] > 0
    norms = cb[:, :, 1:].norm(dim=-1).sum(1)          # [Hkv, C-1]
    assert (norms[used] > 0).all()


@pytest.mark.parametrize("env, match", [
    ({"SEMBLO_SINK_TOKENS": "24"}, "multiple of 16"),
    ({"SEMBLO_SINK_TOKENS": "32", "SEMBLO_FLAT_MIN_TOKENS": "4"}, "split/merge"),
    ({"SEMBLO_SINK_TOKENS": "32", "SEMBLO_FIT_PREFIX_TOKENS": "16"}, "FIT_PREFIX"),
])
def test_what_the_sink_does_not_compose_with_is_refused(monkeypatch, env, match):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("SEMBLO_FLAT_UNIT_TOKENS", "16")
    with pytest.raises(RuntimeError, match=match):
        ProductPostingIndex.build(_keys(512), torch.arange(512), num_centroids=4, kmeans_iters=2,
                                  kmeans_sample_size=256, flat_min_clusters_log2=3)


def test_the_selector_sees_no_sink_and_a_smaller_budget():
    sizes = torch.tensor([[[32, 10, 0, 7]], [[32, 3, 5, 0]]], dtype=torch.int32)
    held, target = _hold_out_sink(sizes, 100, 32)
    assert target == 68
    assert (held[..., 0] == 0).all() and torch.equal(held[..., 1:], sizes[..., 1:])
    assert int(sizes[0, 0, 0]) == 32, "the caller's sizes must not change"


def test_the_forced_bit_is_cluster_zero():
    qmask = torch.zeros(3, 2, 2, dtype=torch.int32)
    qmask[1, 0, 1] = 5
    _force_sink_bit(qmask, None)
    got = unpack_qmask(qmask, 64)
    assert got[..., 0].all()
    assert int(got[..., 1:].sum()) == 2            # cluster 32 and 34 of row 1, head 0, untouched
