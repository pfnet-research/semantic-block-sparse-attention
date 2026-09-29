from __future__ import annotations

import json
import os
import sys
import types

import pytest
import torch

from semblo.vllm_plugin.routing_map import SembloQueryState
from semblo.vllm_plugin.config import SembloVllmConfig
from semblo.vllm_plugin.paged_attention import (
    PagedSparseAttentionResult,
    contiguous_positions_to_posting_tiles,
    merge_paged_attention_results,
    ranked_buckets_to_posting_tiles,
    reference_paged_posting_attention,
)
from semblo.vllm_plugin.state import (
    SembloRuntimeState,
    ProductPostingIndex,
)


def _runtime_dict() -> dict:
    return {
        "layer_types": ["sliding_attention", "full_attention"],
        "semblo_enabled": True,
        "semblo_vllm_mode": "sparse",
        "semblo_layer_indices": [1],
        "semblo_index_dim": 8,
        "semblo_num_factors": 2,
        "semblo_num_centroids": 2,
        "semblo_router_parameter_dtype": "float32",
        "semblo_selection_temperature": 0.1,
        # The runtime requires exhaustive factor proposals.
        "semblo_top_centroids_per_factor": 2,
        "semblo_target_tokens": 4,
        "semblo_max_probe_buckets": 4,
        "semblo_bucket_size_penalty": 0.0,
        "semblo_vllm_query_chunk_size": 2,
        "semblo_vllm_token_capacity": 8,
        "semblo_vllm_decode_num_splits": 2,
        "semblo_vllm_token_tile": 8,
        "semblo_runtime_weights_file": "semblo_runtime.safetensors",
        "semblo_runtime_manifest_file": "semblo_runtime.json",
    }


def test_runtime_config_accepts_nested_dict():
    cfg = SembloVllmConfig.from_hf_config({"text_config": _runtime_dict()})
    assert cfg.layer_indices == (1,)
    assert cfg.index_dim == 8


def test_runtime_checks_flag_reads_config_and_env_override(
    monkeypatch: pytest.MonkeyPatch,
):
    raw = _runtime_dict()
    monkeypatch.delenv("SEMBLO_RUNTIME_CHECKS", raising=False)
    assert SembloVllmConfig.from_hf_config(raw).runtime_checks is True
    raw["semblo_vllm_runtime_checks"] = False
    assert SembloVllmConfig.from_hf_config(raw).runtime_checks is False
    # The environment override wins in both directions, so a benchmark can
    # flip the checks without editing a prepared checkpoint.
    monkeypatch.setenv("SEMBLO_RUNTIME_CHECKS", "1")
    assert SembloVllmConfig.from_hf_config(raw).runtime_checks is True
    monkeypatch.setenv("SEMBLO_RUNTIME_CHECKS", "0")
    del raw["semblo_vllm_runtime_checks"]
    assert SembloVllmConfig.from_hf_config(raw).runtime_checks is False


def test_the_prefill_slice_comes_from_the_environment_and_stale_knobs_refuse(
    monkeypatch: pytest.MonkeyPatch,
):
    raw = _runtime_dict()
    for name in ("SEMBLO_PREFILL_CHUNK", "SEMBLO_PREFILL_PROJECTION_CHUNK",
                 "SEMBLO_MLP_CHUNK", "SEMBLO_GDN_CHUNK", "SEMBLO_MLP_SPLIT_GRAPH"):
        monkeypatch.delenv(name, raising=False)
    assert SembloVllmConfig.from_hf_config(raw).prefill_chunk == 0
    raw["semblo_vllm_prefill_chunk"] = 4096
    assert SembloVllmConfig.from_hf_config(raw).prefill_chunk == 4096
    monkeypatch.setenv("SEMBLO_PREFILL_CHUNK", "2048")
    assert SembloVllmConfig.from_hf_config(raw).prefill_chunk == 2048
    # Unsupported per-block chunk variables raise instead of being ignored.
    monkeypatch.setenv("SEMBLO_MLP_CHUNK", "8192")
    with pytest.raises(ValueError, match="SEMBLO_MLP_CHUNK is gone"):
        SembloVllmConfig.from_hf_config(raw)


def test_checks_off_advances_the_suffix_counter_without_checking():
    from semblo.vllm_plugin.validation import DEFERRED_CHECKS

    torch.manual_seed(0)
    index = ProductPostingIndex.build(
        torch.randn(12, 1, 16), torch.arange(12), num_centroids=4, kmeans_iters=2,
        kmeans_sample_size=64, flat_unit_tokens=4, flat_min_clusters_log2=2)
    DEFERRED_CHECKS.turn_off()
    try:
        index.record_generated_positions(torch.tensor([12, 13]))
        assert index.max_position == 13
        # With checks off a non-contiguous append is not caught. The counter
        # advances by the token count regardless of the values.
        index.record_generated_positions(torch.tensor([99]))
        assert index.max_position == 14
    finally:
        DEFERRED_CHECKS.allow()
    # With checks allowed again the same violation raises immediately on CPU.
    with pytest.raises(ValueError, match="append after the indexed prefix"):
        index.record_generated_positions(torch.tensor([3]))


def test_runtime_state_resets_only_at_first_layer():
    state = SembloRuntimeState(reset_layer_idx=4)
    state.source_indexes[4] = object()  # type: ignore[assignment]
    state.maybe_begin_prompt(9, torch.arange(4))
    assert 4 in state.source_indexes
    state.maybe_begin_prompt(4, torch.arange(4))
    assert state.source_indexes == {}
    assert state.prompt_epoch == 1


def test_runtime_state_tracks_decode_position_without_device_read():
    """Batch-1 decode takes its position from a host-side counter.

    Reading positions[0] from the device would synchronise once per generated
    token.
    """

    state = SembloRuntimeState(reset_layer_idx=0)
    state.maybe_begin_prompt(0, torch.arange(10))
    assert state.step_first_position == 0
    assert state.expected_next_position == 10
    # Decode steps advance from the host counter. The tensor content is wrong
    # on purpose, so the test fails if anything reads it. Contiguity is
    # verified on the device by the SLOT_DECODE_ORDER check.
    state.maybe_begin_prompt(0, torch.tensor([999]))
    assert state.step_first_position == 10
    state.maybe_begin_prompt(0, torch.tensor([999]))
    assert state.step_first_position == 11
    # A new prompt re-anchors through the synchronous read and resets.
    state.maybe_begin_prompt(0, torch.arange(6))
    assert state.step_first_position == 0
    assert state.expected_next_position == 6
    # Before any prefill has anchored the counter, a single-token call falls
    # back to reading the tensor (engine warmup shapes).
    fresh = SembloRuntimeState(reset_layer_idx=0)
    fresh.maybe_begin_prompt(0, torch.tensor([7]))
    assert fresh.step_first_position == 7


def test_runtime_state_serves_short_prompts_densely():
    """A prompt at or below the engage threshold marks the whole request dense.

    Decode steps keep the mark, and the next prompt decides again.
    """

    state = SembloRuntimeState(reset_layer_idx=0)
    state.maybe_begin_prompt(0, torch.arange(10), engage_above_tokens=16)
    assert state.dense_request
    state.maybe_begin_prompt(0, torch.tensor([999]), engage_above_tokens=16)
    assert state.dense_request          # a decode step of the same request
    state.maybe_begin_prompt(0, torch.arange(20), engage_above_tokens=16)
    assert not state.dense_request      # longer than the threshold
    state.maybe_begin_prompt(0, torch.arange(16), engage_above_tokens=16)
    assert state.dense_request          # the threshold itself is still dense
    state.maybe_begin_prompt(0, torch.arange(4), engage_above_tokens=0)
    assert not state.dense_request      # 0 engages everywhere
    state.maybe_begin_prompt(9, torch.arange(4), engage_above_tokens=16)
    assert not state.dense_request      # a non-reset layer changes nothing


def test_posting_tiles_obey_causality_and_whole_bucket_budget():
    ranked = torch.tensor([[[0, 1]]], dtype=torch.int32)
    counts = torch.tensor([[2]], dtype=torch.int32)
    indptr = torch.tensor([[0, 3, 6]], dtype=torch.int32)
    postings = torch.tensor([[0, 2, 4, 1, 3, 5]], dtype=torch.int32)
    tiles = ranked_buckets_to_posting_tiles(
        ranked,
        counts,
        indptr,
        postings,
        torch.tensor([3], dtype=torch.int32),
        target_tokens=3,
        tile_size=2,
        max_tiles=4,
    )
    # Bucket 0 contributes positions 0 and 2 at query position 3; bucket 1
    # is then opened whole and contributes positions 1 and 3, overshooting K.
    assert int(tiles.selected_tokens[0, 0]) == 4
    assert int(tiles.counts[0, 0]) == 2


def test_contiguous_suffix_tiles_and_lse_merge():
    posting, tiles = contiguous_positions_to_posting_tiles(
        torch.arange(10, 15), num_queries=1, num_kv_heads=2, tile_size=2
    )
    assert posting.shape == (2, 5)
    assert tiles.starts.shape == (1, 2, 3)
    assert int(tiles.lengths[0, 0, -1]) == 1

    left = PagedSparseAttentionResult(
        output=torch.tensor([[[1.0, 0.0]]]),
        logsumexp=torch.tensor([[0.0]]),
    )
    right = PagedSparseAttentionResult(
        output=torch.tensor([[[0.0, 1.0]]]),
        logsumexp=torch.tensor([[0.0]]),
    )
    merged = merge_paged_attention_results(left, right)
    torch.testing.assert_close(merged.output, torch.tensor([[[0.5, 0.5]]]))
    torch.testing.assert_close(merged.logsumexp, torch.tensor([[torch.log(torch.tensor(2.0))]]))


def test_reference_paged_attention_reads_vllm_block_table():
    torch.manual_seed(1)
    block_size = 2
    num_blocks = 3
    hkv = 1
    hq = 2
    dim = 4
    k = torch.randn(num_blocks, block_size, hkv, dim)
    v = torch.randn_like(k)
    q = torch.randn(1, hq, dim)
    block_table = torch.tensor([[2, 0, 1]], dtype=torch.int32)
    posting = torch.tensor([[0, 1, 4]], dtype=torch.int32)
    from semblo.vllm_plugin.paged_attention import PostingTiles
    tiles = PostingTiles(
        starts=torch.tensor([[[0, 2]]], dtype=torch.int32),
        lengths=torch.tensor([[[2, 1]]], dtype=torch.int32),
        counts=torch.tensor([[2]], dtype=torch.int32),
        selected_tokens=torch.tensor([[3]], dtype=torch.int32),
    )
    result = reference_paged_posting_attention(
        q,
        k,
        v,
        posting,
        tiles,
        torch.tensor([4]),
        torch.tensor([0]),
        block_table,
        sm_scale=1.0,
        softcap=None,
    )
    assert result.shape == q.shape
    assert torch.isfinite(result).all()


def _install_fake_vllm_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    models_config_map: dict[str, object],
) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    # register() fills in the paper's environment (paper_defaults.py); keep
    # it out of the rest of the session.
    monkeypatch.setattr(os, "environ", os.environ.copy())

    class Registry:
        @staticmethod
        def register_model(name: str, target: str) -> None:
            calls.append((name, target))

    config_module = types.ModuleType("vllm.model_executor.models.config")
    config_module.MODELS_CONFIG_MAP = models_config_map
    models_module = types.ModuleType("vllm.model_executor.models")
    models_module.config = config_module
    executor_module = types.ModuleType("vllm.model_executor")
    executor_module.models = models_module
    vllm_module = types.ModuleType("vllm")
    vllm_module.ModelRegistry = Registry
    vllm_module.model_executor = executor_module

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", executor_module)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models_module)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.config", config_module
    )
    return calls


def _backend_vllm_config(backend: object, *, head_dim: int, global_head_dim: int):
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_config={
                "text_config": {
                    "head_dim": head_dim,
                    "global_head_dim": global_head_dim,
                }
            }
        ),
        attention_config=types.SimpleNamespace(backend=backend),
    )


def _has_sm90_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
@pytest.mark.parametrize("num_splits", [1, 4])
def test_h200_paged_posting_kernel_matches_reference(num_splits: int):
    from semblo.vllm_plugin.paged_attention import paged_posting_attention

    torch.manual_seed(7)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    block_size = 16
    num_blocks = 8
    hkv = 1
    hq = 8
    dim = 512
    logical_length = 64
    permutation = torch.randperm(num_blocks, device=device, dtype=torch.int32)
    block_table = permutation.view(1, -1)
    kv_cache = torch.randn(
        2, num_blocks, block_size, hkv, dim, device=device, dtype=dtype
    )
    q = torch.randn(2, hq, dim, device=device, dtype=dtype)
    posting = torch.arange(logical_length, device=device, dtype=torch.int32).view(1, -1)
    _, tiles = contiguous_positions_to_posting_tiles(
        posting[0], num_queries=2, num_kv_heads=1, tile_size=16
    )
    query_positions = torch.tensor([31, 63], device=device, dtype=torch.int32)
    request_ids = torch.zeros(2, device=device, dtype=torch.int32)
    result = paged_posting_attention(
        q,
        kv_cache,
        posting,
        tiles,
        query_positions,
        request_ids,
        block_table,
        block_size=block_size,
        sm_scale=1.0,
        softcap=30.0,
        token_tile=16,
        num_splits=num_splits,
        return_logsumexp=True,
    )
    reference = reference_paged_posting_attention(
        q,
        kv_cache[0],
        kv_cache[1],
        posting,
        tiles,
        query_positions,
        request_ids,
        block_table,
        sm_scale=1.0,
        softcap=30.0,
    )
    torch.testing.assert_close(result.output, reference, atol=3e-2, rtol=3e-2)
    assert result.logsumexp is not None
    assert torch.isfinite(result.logsumexp).all()


def _import_model_module_with_fake_vllm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    attention_context: tuple,
):
    """Import the lazy model module against a minimal fake vLLM.

    ``attention_context`` is what ``get_attention_context`` returns.  vLLM hands
    back ``None`` metadata while it profiles memory and real metadata while it
    serves, and the wrapper must take a different path for each.
    """

    import importlib

    # Provide only the vLLM modules imported by the lazy model module.
    vllm_root = types.ModuleType("vllm")
    config_module = types.ModuleType("vllm.config")
    attention_module = types.ModuleType(
        "vllm.model_executor.layers.attention.attention"
    )

    class FakeVllmConfig:
        pass

    config_module.VllmConfig = FakeVllmConfig
    attention_module.get_attention_context = lambda name: attention_context
    attention_module.unified_kv_cache_update = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vllm", vllm_root)
    monkeypatch.setitem(sys.modules, "vllm.config", config_module)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.layers.attention.attention", attention_module
    )
    # Restore the real module afterwards so a later test never sees a model
    # module that was built against these fakes.
    monkeypatch.delitem(sys.modules, "semblo.vllm_plugin.model", raising=False)
    model_module = importlib.import_module("semblo.vllm_plugin.model")
    return model_module, None


@pytest.mark.parametrize("tokens", [1, 31, 64, 127])
def test_sparse_prefill_rejects_too_few_tokens_before_fitting(monkeypatch, tokens):
    module, _ = _import_model_module_with_fake_vllm(monkeypatch, attention_context=(None, None, None))
    with pytest.raises(ValueError, match="at least 128"):
        module.SembloAttention._fit_index(types.SimpleNamespace(), torch.empty(tokens, 1, 8), torch.arange(tokens))


def test_full_graph_rejects_shorter_prompt_before_adopting(monkeypatch):
    module, _ = _import_model_module_with_fake_vllm(monkeypatch, attention_context=(None, None, None))
    monkeypatch.delenv("SEMBLO_SUFFIX_FLUSH_EVERY", raising=False)
    adopted = []
    index = types.SimpleNamespace(prompt_length=128, packed_kv=torch.empty(0))
    layer = types.SimpleNamespace(
        source_layer_idx=0,
        runtime_config=types.SimpleNamespace(own_prompt_cache=True, token_capacity=256),
        runtime_state=types.SimpleNamespace(source_indexes={0: index}, prompt_epoch=1),
        _arena=lambda kv: types.SimpleNamespace(adopt=lambda *args: adopted.append(args)),
    )
    with pytest.raises(ValueError, match="prompt length"):
        module.SembloAttention._adopt_into_arena(layer)
    assert not adopted
    index.prompt_length = 256
    module.SembloAttention._adopt_into_arena(layer)
    assert adopted == [(index, 1)]


def test_posting_tile_capacity_accounts_for_bucket_tails():
    # Eight one-token buckets need eight work tiles although only eight tokens
    # are selected, so ceil(selected_tokens / tile_size) is not a valid bound.
    ranked = torch.arange(8, dtype=torch.int32).view(1, 1, 8)
    counts = torch.tensor([[8]], dtype=torch.int32)
    indptr = torch.arange(9, dtype=torch.int32).view(1, 9)
    postings = torch.arange(8, dtype=torch.int32).view(1, 8)
    tiles = ranked_buckets_to_posting_tiles(
        ranked,
        counts,
        indptr,
        postings,
        torch.tensor([7], dtype=torch.int32),
        target_tokens=8,
        token_capacity=8,
        tile_size=16,
        max_tiles=8,
    )
    assert int(tiles.counts[0, 0]) == 8
    with pytest.raises(RuntimeError, match="capacity=7"):
        ranked_buckets_to_posting_tiles(
            ranked,
            counts,
            indptr,
            postings,
            torch.tensor([7], dtype=torch.int32),
            target_tokens=8,
            token_capacity=8,
            tile_size=16,
            max_tiles=7,
        )


def test_posting_selection_fails_when_last_bucket_exceeds_token_capacity():
    ranked = torch.tensor([[[0, 1]]], dtype=torch.int32)
    counts = torch.tensor([[2]], dtype=torch.int32)
    indptr = torch.tensor([[0, 3, 10]], dtype=torch.int32)
    postings = torch.arange(10, dtype=torch.int32).view(1, 10)
    with pytest.raises(RuntimeError, match="causal tokens"):
        ranked_buckets_to_posting_tiles(
            ranked,
            counts,
            indptr,
            postings,
            torch.tensor([9], dtype=torch.int32),
            target_tokens=4,
            token_capacity=8,
            tile_size=4,
            max_tiles=8,
        )


def test_posting_positions_to_tiles_accepts_sparse_logical_positions():
    from semblo.vllm_plugin.paged_attention import posting_positions_to_tiles

    positions = torch.tensor([0, 3, 7, 12, 19], dtype=torch.int32)
    posting, tiles = posting_positions_to_tiles(
        positions,
        num_queries=2,
        num_kv_heads=1,
        tile_size=2,
    )
    assert posting.tolist() == [[0, 3, 7, 12, 19]]
    assert tiles.starts.tolist() == [[[0, 2, 4]], [[0, 2, 4]]]
    assert tiles.lengths.tolist() == [[[2, 2, 1]], [[2, 2, 1]]]
    assert tiles.counts.tolist() == [[3], [3]]
    assert tiles.selected_tokens.tolist() == [[5], [5]]


def test_contiguous_positions_to_posting_tiles_rejects_sparse_positions():
    from semblo.vllm_plugin.paged_attention import (
        contiguous_positions_to_posting_tiles,
    )

    with pytest.raises(ValueError, match="contiguous"):
        contiguous_positions_to_posting_tiles(
            torch.tensor([0, 2, 3], dtype=torch.int32),
            num_queries=1,
            num_kv_heads=1,
            tile_size=2,
        )


@pytest.mark.parametrize(
    ("backend_shape", "cache_shape"),
    [
        # FlashAttention leads with the key/value pair.
        ((2, 3, 16, 5, 7), (2, 4, 16, 1, 8)),
        # The Triton backend puts it second.
        ((3, 2, 16, 5, 7), (4, 2, 16, 1, 8)),
    ],
)
def test_paged_cache_view_follows_the_backend_axis_order(
    monkeypatch: pytest.MonkeyPatch,
    backend_shape: tuple[int, ...],
    cache_shape: tuple[int, ...],
):
    """The kernel reads a canonical ``[2,blocks,block,Hkv,D]`` view of the cache.

    The key/value axis is located from the backend's reported shape.
    """

    model_module, _ = _import_model_module_with_fake_vllm(
        monkeypatch, attention_context=(object(), None, None, None)
    )

    class FakeBackend:
        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
            sizes = {
                2: 2,
                3: num_blocks,
                16: block_size,
                5: num_kv_heads,
                7: head_size,
            }
            return tuple(sizes[value] for value in backend_shape)

    attn_layer = types.SimpleNamespace(attn_backend=FakeBackend)
    cache = torch.arange(int(torch.tensor(cache_shape).prod())).reshape(cache_shape)

    view = model_module._canonical_kv_cache_view(attn_layer, cache)

    assert tuple(view.shape) == (2, 4, 16, 1, 8)
    # A view, not a copy: the kernel addresses the cache through strides.
    assert view.data_ptr() == cache.data_ptr()
    keys = cache[0] if backend_shape[0] == 2 else cache[:, 0]
    torch.testing.assert_close(view[0], keys)


def test_paged_cache_view_rejects_an_unknown_backend_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    model_module, _ = _import_model_module_with_fake_vllm(
        monkeypatch, attention_context=(object(), None, None, None)
    )

    class FusedBackend:
        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
            return (num_blocks, block_size, num_kv_heads, 2 * head_size)

    attn_layer = types.SimpleNamespace(attn_backend=FusedBackend)
    with pytest.raises(RuntimeError, match="unsupported vLLM KV-cache layout"):
        model_module._canonical_kv_cache_view(attn_layer, torch.zeros(4, 16, 1, 16))


def test_sparse_runtime_requires_exhaustive_factor_proposals():
    """A narrower factor search can leave an early query with no causal token.

    Empty hard retrieval is an error rather than a dense fallback, so the
    configuration is rejected before a request reaches the kernel.
    """

    raw = _runtime_dict()
    raw["semblo_num_centroids"] = 64
    raw["semblo_top_centroids_per_factor"] = 16
    with pytest.raises(ValueError, match="exhaustive factor proposals"):
        SembloVllmConfig.from_hf_config(raw)

    raw["semblo_top_centroids_per_factor"] = 64
    cfg = SembloVllmConfig.from_hf_config(raw)
    assert cfg.top_centroids_per_factor == cfg.num_centroids


def _ranking_inputs(num_centroids: int, seed: int = 3):
    torch.manual_seed(seed)
    q, hq, r, factor_dim = 5, 4, 2, 3
    index_dim = 2 * factor_dim
    state = SembloQueryState(
        vectors=torch.randn(1, q, hq, r, index_dim),
        mixture_weights=torch.rand(1, q, hq, r) + 0.1,
        logit_scales=torch.ones(1, q, hq, r),
    )
    codebooks = torch.nn.functional.normalize(
        torch.randn(1, 2, num_centroids, factor_dim), dim=-1
    )
    sizes = torch.randint(0, 8, (1, num_centroids**2), dtype=torch.int32)
    return state, codebooks, sizes


def test_selection_stats_report_slots_against_valid_tokens():
    """Work tiles are padded, so selected tokens alone understate kernel work."""

    from semblo.vllm_plugin.state import SelectionStats

    stats = SelectionStats()
    tiles = types.SimpleNamespace(
        selected_tokens=torch.tensor([[10], [6]], dtype=torch.int32),
        counts=torch.tensor([[1], [1]], dtype=torch.int32),
    )
    stats.record(
        tiles,
        tile_size=8,
        ranked_candidates=torch.tensor([[3], [2]], dtype=torch.int32),
        touched_blocks=torch.tensor([[4], [3]], dtype=torch.int64),
    )
    summary = stats.summary()
    assert summary["selected_valid_tokens"] == 16
    assert summary["processed_tile_slots"] == 16  # 2 tiles x 8 slots
    assert summary["slots_per_valid_token"] == pytest.approx(1.0)
    assert summary["ranked_candidate_buckets"] == 5
    assert summary["touched_kv_blocks"] == 7
    assert summary["sampled_tokens_per_touched_kv_block"] == pytest.approx(16 / 7)

    # Later calls accumulate the cheap counters and leave the sampled chunk
    # alone, so the block ratio keeps describing one representative chunk.
    stats.record(
        tiles,
        tile_size=8,
        ranked_candidates=torch.tensor([[3], [2]], dtype=torch.int32),
        touched_blocks=None,
    )
    second = stats.summary()
    assert second["calls"] == 2
    assert second["selected_valid_tokens"] == 32
    assert second["touched_kv_blocks"] == 7


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
@pytest.mark.parametrize("block_n", [16, 32])
@pytest.mark.parametrize("softcap", [None, 30.0])
def test_packed_range_attention_matches_reference(block_n: int, softcap):
    """Contiguous runs of a bucket-packed K/V array, the layout packing creates."""

    from semblo.vllm_plugin.paged_attention import (
        packed_range_attention,
        reference_packed_range_attention,
    )

    torch.manual_seed(5)
    device = torch.device("cuda")
    q_count, num_q_heads, num_kv_heads, head_dim = 6, 8, 1, 512
    packed_tokens = 512

    q = torch.randn(q_count, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    packed = torch.randn(
        2, packed_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    # Ragged runs, including one query that reads a single token and one that
    # uses fewer ranges than the metadata allows.
    starts = torch.tensor(
        [[[0, 100, 300]], [[7, 250, 0]], [[64, 0, 0]], [[0, 0, 0]],
         [[128, 384, 0]], [[500, 0, 0]]],
        device=device, dtype=torch.int32,
    )
    lengths = torch.tensor(
        [[[40, 33, 17]], [[5, 60, 0]], [[1, 0, 0]], [[96, 0, 0]],
         [[64, 12, 0]], [[12, 0, 0]]],
        device=device, dtype=torch.int32,
    )
    counts = torch.tensor([[3], [2], [1], [1], [2], [1]], device=device, dtype=torch.int32)

    actual = packed_range_attention(
        q, packed, starts, lengths, counts,
        sm_scale=1.0 / (head_dim ** 0.5),
        softcap=softcap,
        block_n=block_n,
        return_logsumexp=True,
    )
    expected = reference_packed_range_attention(
        q, packed, starts, lengths, counts,
        sm_scale=1.0 / (head_dim ** 0.5),
        softcap=softcap,
    )
    torch.testing.assert_close(actual.output, expected, atol=3e-2, rtol=3e-2)
    assert actual.logsumexp is not None
    assert torch.isfinite(actual.logsumexp).all()


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
@pytest.mark.parametrize("num_splits", [2, 8, 16])
def test_packed_range_attention_splits_agree_with_one_program(num_splits: int):
    """Splitting the selection across programs does not change the result.

    In decode, one query and one KV head would otherwise make the grid a single
    program. Each split computes a partial softmax and a second pass rescales
    them against the maximum across splits. The case includes a query whose
    slice is empty, which happens when the splits outnumber the column tiles.
    """

    from semblo.vllm_plugin.paged_attention import packed_range_attention

    torch.manual_seed(7)
    device = torch.device("cuda")
    q_count, num_q_heads, num_kv_heads, head_dim = 3, 8, 1, 512
    packed_tokens = 1024

    q = torch.randn(q_count, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    packed = torch.randn(
        2, packed_tokens, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    # The last query reads a single tile, which is fewer tiles than there are
    # splits at every parameter above.
    starts = torch.tensor(
        [[[0, 300, 700]], [[16, 512, 0]], [[64, 0, 0]]],
        device=device, dtype=torch.int32,
    )
    lengths = torch.tensor(
        [[[200, 150, 90]], [[64, 128, 0]], [[8, 0, 0]]],
        device=device, dtype=torch.int32,
    )
    counts = torch.tensor([[3], [2], [1]], device=device, dtype=torch.int32)

    common = dict(
        sm_scale=1.0 / (head_dim**0.5),
        softcap=None,
        block_n=32,
        return_logsumexp=True,
    )
    single = packed_range_attention(q, packed, starts, lengths, counts, **common)
    split = packed_range_attention(
        q, packed, starts, lengths, counts, num_splits=num_splits, **common
    )
    torch.testing.assert_close(split.output, single.output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(split.logsumexp, single.logsumexp, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
def test_packed_range_attention_rejects_shapes_it_cannot_serve():
    from semblo.vllm_plugin.paged_attention import packed_range_attention

    device = torch.device("cuda")
    q = torch.randn(2, 8, 512, device=device, dtype=torch.bfloat16)
    packed = torch.randn(2, 64, 1, 512, device=device, dtype=torch.bfloat16)
    starts = torch.zeros(2, 1, 1, device=device, dtype=torch.int32)
    lengths = torch.full((2, 1, 1), 8, device=device, dtype=torch.int32)
    counts = torch.ones(2, 1, device=device, dtype=torch.int32)

    with pytest.raises(ValueError, match="block_n"):
        packed_range_attention(
            q, packed, starts, lengths, counts,
            sm_scale=1.0, softcap=None, block_n=128,
        )
    with pytest.raises(ValueError, match=r"\[tiles,Hkv\]"):
        packed_range_attention(
            q, packed, starts, lengths, counts[:1],
            sm_scale=1.0, softcap=None,
        )


def test_packed_ranges_take_buckets_whole_until_the_budget():
    """One bucket is one interval of the packed array, so it needs no tiling."""

    from semblo.vllm_plugin.paged_attention import ranked_buckets_to_packed_ranges

    # Three buckets of three positions each, interleaved in original position.
    indptr = torch.tensor([[0, 3, 6, 9]], dtype=torch.int32)
    posting = torch.tensor([[0, 3, 6, 1, 4, 7, 2, 5, 8]], dtype=torch.int32)
    ranked = torch.tensor([[[2, 0, 1]]], dtype=torch.int32)
    counts = torch.tensor([[3]], dtype=torch.int32)

    ranges = ranked_buckets_to_packed_ranges(
        ranked, counts, indptr, posting,
        torch.tensor([5], dtype=torch.int32),
        target_tokens=3,
    )
    # At position 5, bucket 2 holds {2,5} and bucket 0 holds {0,3}. The budget
    # of three is reached after the second, and bucket 1 is dropped.
    assert ranges.counts.tolist() == [[2]]
    assert ranges.starts[0, 0, :2].tolist() == [6, 0]
    assert ranges.lengths[0, 0, :2].tolist() == [2, 2]
    assert ranges.selected_tokens.tolist() == [[4]]


def test_packed_ranges_skip_dropped_buckets_without_overwriting():
    """A dropped bucket must not land on the previous kept bucket's slot."""

    from semblo.vllm_plugin.paged_attention import ranked_buckets_to_packed_ranges

    indptr = torch.tensor([[0, 2, 4, 6]], dtype=torch.int32)
    # Bucket 1 holds only positions after the query, so it contributes nothing
    # and sits between two buckets that do.
    posting = torch.tensor([[0, 1, 90, 91, 2, 3]], dtype=torch.int32)
    ranked = torch.tensor([[[0, 1, 2]]], dtype=torch.int32)
    counts = torch.tensor([[3]], dtype=torch.int32)

    ranges = ranked_buckets_to_packed_ranges(
        ranked, counts, indptr, posting,
        torch.tensor([10], dtype=torch.int32),
        target_tokens=8,
        ranked_list_is_complete=True,
    )
    assert ranges.counts.tolist() == [[2]]
    assert ranges.starts[0, 0, :2].tolist() == [0, 4]
    assert ranges.lengths[0, 0, :2].tolist() == [2, 2]


def test_packed_ranges_reject_an_empty_or_oversized_selection():
    from semblo.vllm_plugin.paged_attention import ranked_buckets_to_packed_ranges

    indptr = torch.tensor([[0, 2]], dtype=torch.int32)
    posting = torch.tensor([[50, 51]], dtype=torch.int32)
    ranked = torch.tensor([[[0]]], dtype=torch.int32)
    counts = torch.tensor([[1]], dtype=torch.int32)

    with pytest.raises(RuntimeError, match="no causal token"):
        ranked_buckets_to_packed_ranges(
            ranked, counts, indptr, posting,
            torch.tensor([10], dtype=torch.int32),
            target_tokens=4,
        )
    with pytest.raises(RuntimeError, match="capacity"):
        ranked_buckets_to_packed_ranges(
            ranked, counts, indptr, posting,
            torch.tensor([100], dtype=torch.int32),
            target_tokens=1,
            token_capacity=1,
        )


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
def test_packed_range_kernel_matches_the_cpu_reference():
    """The kernel walks the ranked list; the reference scores all of it."""

    from semblo.vllm_plugin.paged_attention import (
        ranked_buckets_to_packed_ranges,
    )

    torch.manual_seed(0)
    tokens, num_buckets, queries = 4096, 256, 96
    bucket_of_token = torch.randint(0, num_buckets, (tokens,))
    sizes = torch.bincount(bucket_of_token, minlength=num_buckets)
    indptr = torch.zeros(num_buckets + 1, dtype=torch.int64)
    indptr[1:] = sizes.cumsum(0)
    indptr = indptr.unsqueeze(0)
    posting = torch.argsort(bucket_of_token, stable=True).to(torch.int32).unsqueeze(0)

    live = torch.nonzero(sizes > 0).flatten()
    ranked = live[torch.argsort(torch.rand(queries, live.numel()), dim=-1)]
    ranked = ranked.to(torch.int32).unsqueeze(1).contiguous()
    counts = torch.full((queries, 1), live.numel(), dtype=torch.int32)
    # Positions spread over the prompt, so some queries exhaust the ranked list
    # and some fill the budget after a handful of buckets.
    positions = torch.linspace(64, tokens - 1, queries).to(torch.int32)

    args = (ranked, counts, indptr, posting, positions)
    expected = ranked_buckets_to_packed_ranges(
        *args, target_tokens=256, ranked_list_is_complete=True
    )
    actual = ranked_buckets_to_packed_ranges(
        *(tensor.cuda() for tensor in args),
        target_tokens=256,
        ranked_list_is_complete=True,
    )

    width = int(expected.starts.shape[-1])
    assert torch.equal(actual.counts.cpu(), expected.counts)
    assert torch.equal(actual.selected_tokens.cpu(), expected.selected_tokens)
    assert torch.equal(actual.max_kept_rank.cpu(), expected.max_kept_rank)
    # Only the entries a query uses are defined. The kernel leaves the rest of
    # the row unwritten, since nothing reads past the count.
    starts, lengths = actual.starts.cpu(), actual.lengths.cpu()
    for query in range(queries):
        used = int(expected.counts[query, 0])
        assert torch.equal(starts[query, 0, :used], expected.starts[query, 0, :used])
        assert torch.equal(lengths[query, 0, :used], expected.lengths[query, 0, :used])
    assert int(expected.counts.max()) <= width


def test_packed_ranges_refuse_a_budget_cut_short_by_the_probe_cap():
    """A budget shortfall caused by capping the candidate list raises."""

    from semblo.vllm_plugin.paged_attention import ranked_buckets_to_packed_ranges

    # Two buckets of two causal tokens each. The ranking was capped at two, so
    # the four selected tokens may not be all that were reachable.
    indptr = torch.tensor([[0, 2, 4]], dtype=torch.int32)
    posting = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)
    ranked = torch.tensor([[[0, 1]]], dtype=torch.int32)
    counts = torch.tensor([[2]], dtype=torch.int32)
    args = (ranked, counts, indptr, posting, torch.tensor([10], dtype=torch.int32))

    with pytest.raises(RuntimeError, match="max_probe_buckets"):
        ranked_buckets_to_packed_ranges(*args, target_tokens=64)

    # The same shortfall is fine when the ranking covered every bucket.
    ranges = ranked_buckets_to_packed_ranges(
        *args, target_tokens=64, ranked_list_is_complete=True
    )
    assert ranges.selected_tokens.tolist() == [[4]]
    assert ranges.max_kept_rank.tolist() == [[1]]


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
def test_query_tile_attention_matches_one_query_per_tile():
    """Two queries sharing a K/V read each attend to exactly what they selected.

    The tile reads the union of the pair's buckets and the per-row lengths
    separate the two queries, so an error there shows up as one query seeing
    the other's tokens.
    """

    from semblo.vllm_plugin.paged_attention import (
        pair_packed_ranges,
        packed_range_attention,
        ranked_buckets_to_packed_ranges,
    )

    torch.manual_seed(0)
    tokens, num_buckets, queries, head_dim, heads = 2048, 128, 64, 512, 8
    bucket_of_token = torch.randint(0, num_buckets, (tokens,))
    sizes = torch.bincount(bucket_of_token, minlength=num_buckets)
    indptr = torch.zeros(num_buckets + 1, dtype=torch.int64)
    indptr[1:] = sizes.cumsum(0)
    posting = torch.argsort(bucket_of_token, stable=True).to(torch.int32)

    live = torch.nonzero(sizes > 0).flatten()
    ranked = live[torch.argsort(torch.rand(queries, live.numel()), dim=-1)]
    ranked = ranked.to(torch.int32).unsqueeze(1).contiguous().cuda()
    counts = torch.full((queries, 1), live.numel(), dtype=torch.int32).cuda()
    positions = torch.linspace(32, tokens - 1, queries).to(torch.int32).cuda()
    indptr, posting = indptr.unsqueeze(0).cuda(), posting.unsqueeze(0).cuda()

    from semblo.vllm_plugin.state import _bucket_slot_table

    sizes_cuda = (indptr[:, 1:] - indptr[:, :-1]).to(torch.int32)
    slot_count = int((sizes_cuda > 0).sum(dim=-1).max().item())
    bucket_slot, slot_start, _slot_bucket = _bucket_slot_table(
        sizes_cuda, indptr, slot_count
    )
    ranges = ranked_buckets_to_packed_ranges(
        ranked, counts, indptr, posting, positions,
        target_tokens=192, ranked_list_is_complete=True,
    )
    tiled_ranges = ranked_buckets_to_packed_ranges(
        ranked, counts, indptr, posting, positions,
        target_tokens=192, ranked_list_is_complete=True,
        queries_per_tile=2, bucket_slot=bucket_slot, num_slots=slot_count,
    )
    q = torch.randn(queries, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    packed = torch.randn(2, tokens, 1, head_dim, device="cuda", dtype=torch.bfloat16)

    single = packed_range_attention(
        q, packed, ranges.starts, ranges.lengths, ranges.counts,
        sm_scale=0.05, softcap=None,
    ).output
    paired_ranges = pair_packed_ranges(
        tiled_ranges, slot_start, queries_per_tile=2
    )
    paired = packed_range_attention(
        q, packed, paired_ranges.starts, paired_ranges.lengths, paired_ranges.counts,
        sm_scale=0.05, softcap=None,
        range_row_lengths=paired_ranges.row_lengths, queries_per_tile=2,
    ).output

    # The union changes only the order tokens are accumulated in, so the
    # difference must stay at the level of bfloat16 rounding.
    difference = (paired.float() - single.float()).abs().max().item()
    assert difference < 5e-2, difference
    # Every token each query selected must still be covered exactly once.
    assert torch.equal(
        paired_ranges.row_lengths[:, :, :, 0].sum(-1),
        ranges.selected_tokens[0::2],
    )
    assert torch.equal(
        paired_ranges.row_lengths[:, :, :, 1].sum(-1),
        ranges.selected_tokens[1::2],
    )


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
def test_attention_merge_kernel_matches_the_tensor_path():
    """One kernel must combine the two states exactly as the tensor path does."""

    from semblo.vllm_plugin.paged_attention import (
        _merge_attention_results_on_device,
    )

    torch.manual_seed(0)
    queries, heads, head_dim = 37, 8, 512
    left = PagedSparseAttentionResult(
        output=torch.randn(queries, heads, head_dim, device="cuda", dtype=torch.bfloat16),
        logsumexp=torch.randn(queries, heads, device="cuda") * 3,
    )
    right = PagedSparseAttentionResult(
        output=torch.randn(queries, heads, head_dim, device="cuda", dtype=torch.bfloat16),
        logsumexp=torch.randn(queries, heads, device="cuda") * 3,
    )
    fused = _merge_attention_results_on_device(left, right)
    expected_lse = torch.logaddexp(left.logsumexp, right.logsumexp)
    weight_left = torch.exp(left.logsumexp - expected_lse).unsqueeze(-1)
    weight_right = torch.exp(right.logsumexp - expected_lse).unsqueeze(-1)
    expected = (
        left.output.float() * weight_left + right.output.float() * weight_right
    ).to(left.output.dtype)

    torch.testing.assert_close(fused.logsumexp, expected_lse, atol=1e-5, rtol=1e-5)
    assert (fused.output.float() - expected.float()).abs().max().item() < 5e-3


def _vllm_config_like(cudagraph_mode: str, *, splitting_ops=None):
    """The parts of a vLLM configuration the sparse guard actually reads."""

    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=1,
            enable_chunked_prefill=False,
            max_num_batched_tokens=131072,
        ),
        cache_config=types.SimpleNamespace(
            enable_prefix_caching=False, kv_sharing_fast_prefill=False
        ),
        compilation_config=types.SimpleNamespace(
            cudagraph_mode=cudagraph_mode, splitting_ops=splitting_ops
        ),
        model_config=types.SimpleNamespace(
            enforce_eager=False, max_model_len=131072
        ),
        speculative_config=None,
    )


def test_capture_of_the_sparse_layer_is_refused_but_piecewise_is_not():
    """Piecewise capture is allowed; capture that takes the layer in is not.

    The sparse layer sits in ``splitting_ops`` and stays outside the captured
    pieces, and the operator writes through a caller-allocated buffer so the
    pieces around it replay valid addresses.  Full capture would freeze the
    request-local index and the Python-side suffix state, so every FULL
    variant is refused with the reason.
    """

    runtime = SembloVllmConfig.from_hf_config(_runtime_dict())
    runtime.validate_vllm_config(_vllm_config_like("NONE"))
    runtime.validate_vllm_config(_vllm_config_like("PIECEWISE"))

    for refused in ("FULL", "FULL_AND_PIECEWISE", "FULL_DECODE_ONLY"):
        with pytest.raises(ValueError, match="cannot use cudagraph_mode"):
            runtime.validate_vllm_config(_vllm_config_like(refused))


def test_full_graph_decode_adds_a_two_token_warm_up_size(monkeypatch: pytest.MonkeyPatch):
    """With SEMBLO_FULL_GRAPH_DECODE=1 a FULL decode-only capture is allowed.

    compile_sizes gains a 2 so that vLLM's first trace of the model is symbolic
    in the token dimension. A first trace at the capture's single token would
    specialise it, and the piecewise backend would then fail. An existing size
    of two or more is enough. PIECEWISE is left alone.
    """

    monkeypatch.setenv("SEMBLO_FULL_GRAPH_DECODE", "1")
    runtime = SembloVllmConfig.from_hf_config(_runtime_dict())
    config = _vllm_config_like("FULL_DECODE_ONLY")
    runtime.validate_vllm_config(config)
    assert config.compilation_config.compile_sizes == [2]
    runtime.validate_vllm_config(config)                       # idempotent
    assert config.compilation_config.compile_sizes == [2]
    # vLLM skips the warm-up of a compile size that is also a capture size, so
    # the added size steps past the capture sizes.
    config = _vllm_config_like("FULL_DECODE_ONLY")
    config.compilation_config.cudagraph_capture_sizes = [1, 2]
    runtime.validate_vllm_config(config)
    assert config.compilation_config.compile_sizes == [3]
    config = _vllm_config_like("FULL_DECODE_ONLY")
    config.compilation_config.compile_sizes = [16]
    runtime.validate_vllm_config(config)
    assert config.compilation_config.compile_sizes == [16]
    config = _vllm_config_like("PIECEWISE")
    runtime.validate_vllm_config(config)
    assert getattr(config.compilation_config, "compile_sizes", None) is None


def test_the_sparse_layer_is_added_to_the_compiler_splitting_ops(
    monkeypatch: pytest.MonkeyPatch,
):
    """The sparse layer is listed in ``splitting_ops`` so piecewise capture excludes it.

    vLLM fills ``splitting_ops`` with its own attention operators only when it
    finds the field unset, so the plugin appends rather than assigns.
    Assigning would drop vLLM's entries and capture across its attention too.
    """

    model_module, _ = _import_model_module_with_fake_vllm(
        monkeypatch, attention_context=(object(), None, None, None)
    )

    existing = ["vllm::unified_attention"]
    config = _vllm_config_like("PIECEWISE", splitting_ops=existing)
    model_module._split_the_graph_at_sparse_attention(config)
    assert config.compilation_config.splitting_ops == [
        "vllm::unified_attention",
        "semblo::attention",
    ]
    # The plugin adds exactly one split point. A long prefill is streamed
    # outside the compiled model (streamed_prefill.py), so no other operator
    # needs a split, and every extra split point would slow every decode step.

    # Running twice must not list it twice.
    model_module._split_the_graph_at_sparse_attention(config)
    assert config.compilation_config.splitting_ops.count(
        "semblo::attention") == 1

    # With compilation off there is no list and nothing to do.
    off = _vllm_config_like("NONE", splitting_ops=None)
    model_module._split_the_graph_at_sparse_attention(off)
    assert off.compilation_config.splitting_ops is None


def test_the_benchmark_notices_a_configuration_that_changes_the_answer(tmp_path):
    """The benchmark compares its output tokens against a reference run.

    Repeating a run shows determinism, not correctness. A configuration can
    keep the token count and the number of sparse calls and still generate
    different tokens.
    """

    from semblo.vllm_plugin.benchmark import report_token_mismatch

    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps({"runs": [{"output_token_ids": [5, 6, 7, 8]}]}), encoding="utf-8"
    )

    same = {"runs": [{"output_token_ids": [5, 6, 7, 8]}]}
    report_token_mismatch(same, reference_path)  # no exception

    late = {"runs": [{"output_token_ids": [5, 6, 7, 9]}]}
    with pytest.raises(SystemExit, match="1 of 4 tokens, first at position 3"):
        report_token_mismatch(late, reference_path)

    broken = {"runs": [{"output_token_ids": [1, 2, 3, 4]}]}
    with pytest.raises(SystemExit, match="4 of 4 tokens, first at position 0"):
        report_token_mismatch(broken, reference_path)


@pytest.mark.parametrize(
    ("actual_ids", "expected_ids"),
    [
        ([1, 2], [1, 2, 3]),
        ([1, 2, 3], [1, 2]),
        ([], [1]),
        ([1], []),
        ([9, 2], [1, 2, 3]),
    ],
)
def test_the_benchmark_reports_output_length_mismatches(
    tmp_path, actual_ids, expected_ids,
):
    from semblo.vllm_plugin.benchmark import report_token_mismatch

    reference_path = tmp_path / "reference.json"
    matching_run = {"output_token_ids": [7]}
    reference_path.write_text(
        json.dumps({"runs": [matching_run, {"output_token_ids": expected_ids}]}),
        encoding="utf-8",
    )
    result = {"runs": [matching_run, {"output_token_ids": actual_ids}]}

    with pytest.raises(SystemExit) as exc:
        report_token_mismatch(result, reference_path)
    assert str(exc.value) == (
        f"generated token count differs from {reference_path} in run 1: "
        f"got {len(actual_ids)}, expected {len(expected_ids)}."
    )


@pytest.mark.parametrize(
    ("actual_runs", "expected_runs"),
    [([], [[1]]), ([[1]], []), ([[1]], [[1], [2]]), ([[1], [2]], [[1]])],
)
def test_the_benchmark_reports_run_count_mismatches(
    tmp_path, actual_runs, expected_runs,
):
    from semblo.vllm_plugin.benchmark import report_token_mismatch

    reference_path = tmp_path / "reference.json"
    reference_path.write_text(
        json.dumps({"runs": [{"output_token_ids": ids} for ids in expected_runs]}),
        encoding="utf-8",
    )
    result = {"runs": [{"output_token_ids": ids} for ids in actual_runs]}

    with pytest.raises(SystemExit) as exc:
        report_token_mismatch(result, reference_path)
    assert str(exc.value) == (
        f"run count differs from {reference_path}: "
        f"got {len(actual_runs)}, expected {len(expected_runs)}."
    )


@pytest.mark.parametrize("token_runs", [[], [[]], [[1], [2, 3]]])
def test_the_benchmark_accepts_identical_runs(tmp_path, token_runs):
    from semblo.vllm_plugin.benchmark import report_token_mismatch

    reference_path = tmp_path / "reference.json"
    result = {"runs": [{"output_token_ids": ids} for ids in token_runs]}
    reference_path.write_text(json.dumps(result), encoding="utf-8")

    report_token_mismatch(result, reference_path)


def test_the_benchmark_checks_token_values_in_later_runs(tmp_path):
    from semblo.vllm_plugin.benchmark import report_token_mismatch

    reference_path = tmp_path / "reference.json"
    matching_run = {"output_token_ids": [1, 2]}
    reference_path.write_text(
        json.dumps({"runs": [matching_run, {"output_token_ids": [3, 4]}]}),
        encoding="utf-8",
    )
    result = {"runs": [matching_run, {"output_token_ids": [3, 9]}]}

    with pytest.raises(SystemExit, match="1 of 2 tokens, first at position 1"):
        report_token_mismatch(result, reference_path)


def test_packed_ranges_accept_per_query_targets():
    """Each query stops at its own token target."""

    from semblo.vllm_plugin.paged_attention import ranked_buckets_to_packed_ranges

    # Three buckets of three positions each, and two queries that differ only
    # in their budget.
    indptr = torch.tensor([[0, 3, 6, 9]], dtype=torch.int32)
    posting = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.int32)
    ranked = torch.tensor([[[0, 1, 2]], [[0, 1, 2]]], dtype=torch.int32)
    counts = torch.tensor([[3], [3]], dtype=torch.int32)
    positions = torch.tensor([8, 8], dtype=torch.int32)

    ranges = ranked_buckets_to_packed_ranges(
        ranked, counts, indptr, posting, positions,
        target_tokens=torch.tensor([2, 8], dtype=torch.int32),
        ranked_list_is_complete=True,
    )
    # Query 0 (budget 2) keeps the first bucket whole, crossing the budget.
    # Query 1 (budget 8) crosses it with the third bucket, at nine tokens.
    assert ranges.selected_tokens.tolist() == [[3], [9]]
    assert ranges.counts.tolist() == [[1], [3]]

    # A uniform per-query vector matches a scalar budget.
    uniform_vec = ranked_buckets_to_packed_ranges(
        ranked, counts, indptr, posting, positions,
        target_tokens=torch.tensor([4, 4], dtype=torch.int32),
        ranked_list_is_complete=True,
    )
    uniform_int = ranked_buckets_to_packed_ranges(
        ranked, counts, indptr, posting, positions,
        target_tokens=4,
        ranked_list_is_complete=True,
    )
    assert uniform_vec.selected_tokens.tolist() == uniform_int.selected_tokens.tolist()
    assert uniform_vec.counts.tolist() == uniform_int.counts.tolist()


def test_topk_index_attention_reference_handles_padding_and_softcap():
    from semblo.vllm_plugin.paged_attention import (
        reference_topk_index_attention,
    )

    torch.manual_seed(3)
    length, hkv, hpk, dim = 32, 1, 2, 8
    q = torch.randn(2, hkv * hpk, dim)
    packed = torch.randn(2, length, hkv, dim)
    idx = torch.full((2, hkv, 6), -1, dtype=torch.int32)
    idx[0, 0, :3] = torch.tensor([4, 9, 17])
    # Query 1 stays fully padded: its output must be exactly zero.
    result = reference_topk_index_attention(
        q, packed, idx, sm_scale=0.7, softcap=30.0, return_logsumexp=True,
    )
    assert torch.all(result.output[1] == 0)
    rows = torch.tensor([4, 9, 17])
    for h in range(hkv * hpk):
        scores = (q[0, h].float() @ packed[0, rows, 0].float().T) * 0.7
        scores = 30.0 * torch.tanh(scores / 30.0)
        ref = torch.softmax(scores, dim=-1) @ packed[1, rows, 0].float()
        torch.testing.assert_close(
            result.output[0, h].float(), ref, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            result.logsumexp[0, h], torch.logsumexp(scores, dim=-1)
        )



def test_the_paged_context_is_a_dataclass():
    from semblo.vllm_plugin.model import _PagedContext

    context = _PagedContext(kv_cache=1, block_table=2, block_size=3)
    assert context.block_size == 3
    with pytest.raises(Exception):
        context.block_size = 4


def test_the_partition_depends_on_the_prompt_and_the_seed_alone():
    """With codebook_seed, the same prompt gets the same partition whatever the global RNG did before."""

    torch.manual_seed(0)
    keys = torch.randn(64, 2, 16)

    def build():
        return ProductPostingIndex.build(
            keys, torch.arange(64), num_centroids=4, kmeans_iters=3, kmeans_sample_size=64,
            codebook_seed=7, flat_unit_tokens=8, flat_min_clusters_log2=2)

    first = build()
    torch.rand(1000)            # move the global RNG
    second = build()
    assert torch.equal(first.posting_positions, second.posting_positions)
    assert torch.equal(first.bucket_sizes, second.bucket_sizes)
