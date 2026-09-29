"""The fixed-codebook baseline (SEMBLO_FIXED_CODEBOOK) replaces the per-prompt partition.

A missing or mis-shaped codebook entry raises instead of falling back to the
per-prompt fit. The cluster count follows the prompt length, so a silent
fallback would mix the two partitions within one run.
"""
from __future__ import annotations

import os

import pytest
import torch

from semblo.inference.flat_index import (
    half_normalised_space, routing_space,
)
from semblo.vllm_plugin import state as state_mod
from semblo.vllm_plugin.state import ProductPostingIndex


def _build(keys, positions, unit_tokens=8, min_log2=2):
    previous = os.environ.get("SEMBLO_FLAT_UNIT_TOKENS")
    os.environ["SEMBLO_FLAT_UNIT_TOKENS"] = str(unit_tokens)
    try:
        return ProductPostingIndex.build(
            keys, positions, num_centroids=4, kmeans_iters=2,
            kmeans_sample_size=64, flat_min_clusters_log2=min_log2,
            layer_idx=0,
        )
    finally:
        if previous is None:
            os.environ.pop("SEMBLO_FLAT_UNIT_TOKENS", None)
        else:
            os.environ["SEMBLO_FLAT_UNIT_TOKENS"] = previous


def test_codebook_loader_rejects_arbitrary_pickle_objects(tmp_path, monkeypatch):
    import pickle

    path = tmp_path / "book.pt"
    torch.save({"codebooks": {(0, 8): torch.zeros(1, 8, 16)}, "meta": object()}, path)
    monkeypatch.setenv("SEMBLO_FIXED_CODEBOOK", str(path))
    with pytest.raises(pickle.UnpicklingError):
        state_mod._load_fixed_codebook(0, 8, 1, 16)


def test_half_normalised_space_matches_the_transform_it_replaced():
    x = torch.randn(37, 16)
    a, b = x.chunk(2, dim=-1)
    want = torch.cat([torch.nn.functional.normalize(a.float(), dim=-1),
                      torch.nn.functional.normalize(b.float(), dim=-1)], dim=-1)
    assert torch.equal(half_normalised_space(x), want)


def test_no_env_var_means_the_shipped_path(monkeypatch):
    monkeypatch.delenv("SEMBLO_FIXED_CODEBOOK", raising=False)
    assert state_mod._load_fixed_codebook(0, 8, 2, 16) is None


def test_a_missing_entry_raises_rather_than_falling_back(tmp_path, monkeypatch):
    path = tmp_path / "book.pt"
    torch.save({"codebooks": {(0, 8): torch.randn(2, 8, 16)}}, path)
    monkeypatch.setenv("SEMBLO_FIXED_CODEBOOK", str(path))
    state_mod._FIXED_CODEBOOK_CACHE.clear()
    with pytest.raises(RuntimeError, match="no codebook for layer 0 with 16 clusters"):
        state_mod._load_fixed_codebook(0, 16, 2, 16)
    with pytest.raises(RuntimeError, match="no codebook for layer 3"):
        state_mod._load_fixed_codebook(3, 8, 2, 16)


def test_a_wrong_shape_raises(tmp_path, monkeypatch):
    path = tmp_path / "book.pt"
    torch.save({"codebooks": {(0, 8): torch.randn(2, 8, 32)}}, path)
    monkeypatch.setenv("SEMBLO_FIXED_CODEBOOK", str(path))
    state_mod._FIXED_CODEBOOK_CACHE.clear()
    with pytest.raises(RuntimeError, match="expected"):
        state_mod._load_fixed_codebook(0, 8, 2, 16)


def test_two_different_prompts_get_the_same_codebook(tmp_path, monkeypatch):
    """With a fixed codebook, the codebook does not depend on the prompt."""
    torch.manual_seed(11)
    heads, dim, length = 2, 16, 512
    # Take the cluster count from the runtime instead of re-deriving its
    # floor, power-of-two rounding and length cap here.
    monkeypatch.delenv("SEMBLO_FIXED_CODEBOOK", raising=False)
    state_mod._FIXED_CODEBOOK_CACHE.clear()
    probe = _build(torch.randn(length, heads, dim), torch.arange(length))
    clusters = int(probe.bucket_sizes.shape[-1])

    book = torch.nn.functional.normalize(torch.randn(heads, clusters, dim), dim=-1)
    path = tmp_path / "book.pt"
    torch.save({"codebooks": {(0, clusters): book}}, path)
    monkeypatch.setenv("SEMBLO_FIXED_CODEBOOK", str(path))
    state_mod._FIXED_CODEBOOK_CACHE.clear()

    built = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        keys = torch.randn(length, heads, dim)
        index = _build(keys, torch.arange(length))
        assert index.bucket_sizes.shape[-1] == clusters
        built.append((keys, index))

    # Two unrelated prompts, one codebook (see the control below).
    assert torch.equal(built[0][1].codebooks, built[1][1].codebooks)

    # The tokens are assigned by that codebook. The runtime takes the argmax
    # in bf16, because the GEMM dominates the build, so the chosen centroid is
    # only required to be within bf16 resolution of the best one.
    tol = 2 * torch.finfo(torch.bfloat16).eps
    for keys, index in built:
        for h in range(heads):
            space = torch.nn.functional.normalize(
                routing_space(keys[:, h]), dim=-1)
            sim = space @ book[h].T                       # [L, C]
            chosen = torch.empty(length, dtype=torch.long)
            start = index.bucket_indptr[h]
            size = index.bucket_sizes[h]
            for c in range(clusters):
                # Slice by the true size, not by the next indptr. Under
                # SEMBLO_FA3_ALIGN each bucket's slice is padded to a page
                # boundary with repeats of row 0, which are not members (see
                # _page_align_order).
                rows = index.posting_positions[h][int(start[c]):int(start[c]) + int(size[c])]
                chosen[rows.long()] = c
            got = sim.gather(1, chosen[:, None]).squeeze(1)
            assert torch.all(got >= sim.max(dim=1).values - tol)


def test_the_shipped_path_still_depends_on_the_prompt(monkeypatch):
    """Control: without the env var, two prompts get two different partitions."""
    monkeypatch.delenv("SEMBLO_FIXED_CODEBOOK", raising=False)
    state_mod._FIXED_CODEBOOK_CACHE.clear()
    torch.manual_seed(5)
    length = 32
    a = _build(torch.randn(length, 2, 16), torch.arange(length))
    b = _build(torch.randn(length, 2, 16), torch.arange(length))
    assert not torch.equal(a.codebooks, b.codebooks)


def test_run_environment_is_recorded(tmp_path, monkeypatch):
    """The SEMBLO_* and VLLM_* variables and the git commit are written beside the results.

    These variables set the cluster size, the execution order and whether a
    compiled graph is reused, so they are part of a run's settings.
    """
    from semblo.lm_eval_runner import _write_run_environment
    import json as _json

    monkeypatch.setenv("SEMBLO_FLAT_UNIT_TOKENS", "512")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("SOMETHING_ELSE", "not recorded")
    _write_run_environment(tmp_path, "a_model")
    got = _json.loads((tmp_path / "run_environment.json").read_text())
    assert got["environment"]["SEMBLO_FLAT_UNIT_TOKENS"] == "512"
    assert got["environment"]["VLLM_DISABLE_COMPILE_CACHE"] == "1"
    assert "SOMETHING_ELSE" not in got["environment"]
    assert got["model_key"] == "a_model"
    assert "hostname" not in got
    # the commit may be None outside a checkout, but the key has to be there
    assert "git_commit" in got and "git_dirty" in got
