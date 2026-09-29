"""The adjacent-block baseline: contiguous blocks, mean-pooled representatives.

The baseline uses the method's own scoring, selection, gather and attention.
Only the partition differs.

These tests check that:
  * the routing space modes are selected by SEMBLO_ROUTING_NORMALIZE;
  * token i is in block i // B, and a block's representative is the plain mean
    of its keys, not a fitted or normalised centroid;
  * the block count is ceil(L / B), rounded up for the kernels, independent of
    the cluster floor and the mean-cluster-size target.
"""
from __future__ import annotations

import os

import pytest
import torch

from semblo.inference.flat_index import (
    half_normalised_space, routing_space,
)
from semblo.vllm_plugin.state import ProductPostingIndex


def _keys(*shape, seed):
    """Seeded random keys, so that norm assertions are deterministic."""
    return torch.Generator().manual_seed(seed) and torch.randn(
        *shape, generator=torch.Generator().manual_seed(seed))


def _build(keys, positions, *, unit_tokens=8, min_log2=2):
    os.environ["SEMBLO_FLAT_UNIT_TOKENS"] = str(unit_tokens)
    try:
        return ProductPostingIndex.build(
            keys, positions, num_centroids=4, kmeans_iters=2,
            kmeans_sample_size=64, flat_min_clusters_log2=min_log2,
        )
    finally:
        del os.environ["SEMBLO_FLAT_UNIT_TOKENS"]


def _members(index, head, cluster):
    """The positions of one bucket, without the page padding.

    Under SEMBLO_FA3_ALIGN the slice between two indptr entries is padded to a
    page boundary with repeats of row 0. The true size is bucket_sizes.
    """
    start = int(index.bucket_indptr[head][cluster])
    size = int(index.bucket_sizes[head][cluster])
    return index.posting_positions[head][start:start + size]


def test_the_routing_space_has_three_modes(monkeypatch):
    x = _keys(37, 16, seed=107)
    # the default is the whole vector
    monkeypatch.delenv("SEMBLO_ROUTING_NORMALIZE", raising=False)
    assert torch.allclose(routing_space(x).norm(dim=-1), torch.ones(37), atol=1e-5)
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "whole")
    got = routing_space(x)
    assert torch.allclose(got.norm(dim=-1), torch.ones(37), atol=1e-5)
    # the half-split mode
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "half")
    assert torch.equal(routing_space(x), half_normalised_space(x))
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "none")
    assert torch.equal(routing_space(x), x.float())
    # split:n normalises the first n components and the rest separately
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "split:4")
    got = routing_space(x)
    assert torch.allclose(got[..., :4].norm(dim=-1), torch.ones(37), atol=1e-5)
    assert torch.allclose(got[..., 4:].norm(dim=-1), torch.ones(37), atol=1e-5)
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "split:8")
    assert torch.equal(routing_space(x), half_normalised_space(x))
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "split:16")
    with pytest.raises(RuntimeError, match="inside"):
        routing_space(x)
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "sideways")
    with pytest.raises(RuntimeError, match="half, whole, none or"):
        routing_space(x)


def test_a_block_partition_is_arithmetic(monkeypatch):
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "none")
    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", "8")
    heads, dim, length, block = 2, 16, 64, 8
    keys = _keys(length, heads, dim, seed=101)
    index = _build(keys, torch.arange(length))

    assert index.bucket_sizes.shape[-1] == length // block == 8
    for h in range(heads):
        for c in range(length // block):
            rows = sorted(int(p) for p in _members(index, h, c))
            assert rows == list(range(c * block, (c + 1) * block))


def test_the_representative_is_the_plain_mean(monkeypatch):
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "none")
    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", "8")
    heads, dim, length, block = 2, 16, 64, 8
    keys = _keys(length, heads, dim, seed=102)
    index = _build(keys, torch.arange(length))

    for h in range(heads):
        for c in range(length // block):
            want = keys[c * block:(c + 1) * block, h].float().mean(dim=0)
            # the kernel scores q0.c0 + q1.c1, so a representative is stored
            # as its two halves
            got = torch.cat([index.codebooks[h][0][c].float(),
                             index.codebooks[h][1][c].float()])
            assert torch.allclose(got, want, atol=2e-2, rtol=2e-2), (h, c)
    # not projected onto the unit sphere, so the norms vary
    for h in range(heads):
        norms = index.codebooks[h][0].float().norm(dim=-1)[:length // block]
        assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
        assert float(norms.std()) > 1e-3


def test_a_short_last_block_and_the_power_of_two_padding(monkeypatch):
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "none")
    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", "8")
    heads, dim, length, block = 1, 16, 40, 8      # 5 blocks -> 8 buckets
    keys = _keys(length, heads, dim, seed=103)
    index = _build(keys, torch.arange(length))

    sizes = [int(s) for s in index.bucket_sizes[0]]
    assert len(sizes) == 8
    assert sizes[:5] == [block] * 5
    assert sizes[5:] == [0, 0, 0]


def test_the_block_size_overrides_the_cluster_floor(monkeypatch):
    """The floor and the mean-cluster-size target do not apply to blocks."""
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "none")
    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", "16")
    keys = _keys(64, 1, 16, seed=104)
    # min_log2=5 would floor a fitted partition at 32 clusters and
    # unit_tokens=8 would ask for 8. The block size gives four blocks. The
    # bucket axis has a minimum of eight (the kernels pack it into 32-bit
    # words), so the last four buckets are empty.
    index = _build(keys, torch.arange(64), unit_tokens=8, min_log2=5)
    assert index.bucket_sizes.shape[-1] == 8
    assert [int(s) for s in index.bucket_sizes[0]] == [16, 16, 16, 16, 0, 0, 0, 0]


def test_blocks_and_a_fixed_codebook_are_two_partitions(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", "8")
    path = tmp_path / "book.pt"
    torch.save({"codebooks": {(0, 8): _keys(1, 8, 16, seed=105)}}, path)
    monkeypatch.setenv("SEMBLO_FIXED_CODEBOOK", str(path))
    from semblo.vllm_plugin import state as state_mod
    state_mod._FIXED_CODEBOOK_CACHE.clear()
    os.environ["SEMBLO_FLAT_UNIT_TOKENS"] = "8"
    try:
        with pytest.raises(RuntimeError, match="different partitions"):
            ProductPostingIndex.build(
                _keys(64, 1, 16, seed=106), torch.arange(64),
                num_centroids=4, kmeans_iters=2, kmeans_sample_size=64,
                flat_min_clusters_log2=2, layer_idx=0,
            )
    finally:
        del os.environ["SEMBLO_FLAT_UNIT_TOKENS"]
        state_mod._FIXED_CODEBOOK_CACHE.clear()


def test_the_shipped_path_is_unchanged(monkeypatch):
    """Without SEMBLO_BLOCK_TOKENS the partition is the fitted one, not the blocks.

    Checked as "the two partitions differ", since on 64 tokens k-means may put
    exactly the first eight positions in bucket 0.
    """
    monkeypatch.delenv("SEMBLO_ROUTING_NORMALIZE", raising=False)
    keys = _keys(64, 1, 16, seed=3)

    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", "8")
    blocks = _build(keys, torch.arange(64))
    monkeypatch.delenv("SEMBLO_BLOCK_TOKENS")
    fitted = _build(keys, torch.arange(64))

    assert int(sum(int(s) for s in fitted.bucket_sizes[0])) == 64
    assert [int(s) for s in blocks.bucket_sizes[0]] == [8] * 8
    same = all(
        sorted(int(x) for x in _members(fitted, 0, c))
        == sorted(int(x) for x in _members(blocks, 0, c))
        for c in range(blocks.bucket_sizes.shape[-1]))
    assert not same, "the fitted partition came out as the block partition"
