from __future__ import annotations

import pytest

from semblo.parallel import (
    parse_vllm_json_arg_overrides,
    example_in_shard,
    parse_human_int,
    parse_parameter_count,
    torch_cuda_device_count,
    validate_vllm_parallel_config,
)


def test_dcp_must_divide_tp() -> None:
    with pytest.raises(ValueError, match="decode_context_parallel_size"):
        validate_vllm_parallel_config(
            {"tensor_parallel_size": 3, "decode_context_parallel_size": 2},
            available_cuda_devices=4,
        )


def test_tp_pp_must_fit_visible_gpus() -> None:
    with pytest.raises(ValueError, match="exceed visible CUDA devices"):
        validate_vllm_parallel_config(
            {"tensor_parallel_size": 4, "pipeline_parallel_size": 2},
            available_cuda_devices=4,
        )


def test_dcp_summary_reuses_tp_world_size() -> None:
    summary = validate_vllm_parallel_config(
        {"tensor_parallel_size": 4, "decode_context_parallel_size": 2},
        available_cuda_devices=4,
    )
    assert summary.local_worker_gpus == 4
    assert summary.total_logical_gpus == 4


def test_stable_hash_sharding_partitions_examples() -> None:
    ids = [f"ex-{i}" for i in range(100)]
    seen = []
    for shard_id in range(4):
        seen.extend([x for i, x in enumerate(ids) if example_in_shard(x, i, num_shards=4, shard_id=shard_id, shard_by="hash")])
    assert sorted(seen) == sorted(ids)


def test_index_sharding_partitions_examples() -> None:
    ids = [f"ex-{i}" for i in range(10)]
    assert [x for i, x in enumerate(ids) if example_in_shard(x, i, num_shards=3, shard_id=1, shard_by="index")] == ["ex-1", "ex-4", "ex-7"]


def test_parse_parameter_count_suffixes() -> None:
    assert parse_parameter_count("4B") == 4_000_000_000
    assert parse_parameter_count("0.6B") == 600_000_000
    assert parse_parameter_count("125M") == 125_000_000


def test_parse_human_int_suffixes() -> None:
    assert parse_human_int("128000") == 128000
    assert parse_human_int("128k") == 128000
    assert parse_human_int("128K") == 131072
    assert parse_human_int("auto") == "auto"


def test_cuda_device_count_uses_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    assert torch_cuda_device_count() == 2


def test_cuda_device_count_does_not_import_torch_without_env(monkeypatch) -> None:
    import builtins

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError("torch should not be imported by default")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert torch_cuda_device_count() is None


def test_parse_vllm_json_arg_overrides() -> None:
    parsed = parse_vllm_json_arg_overrides([
        'limit_mm_per_prompt={"image":0,"audio":0}',
        'speculative_config={"method":"mtp","num_speculative_tokens":1}',
    ])
    assert parsed == {
        "limit_mm_per_prompt": {"image": 0, "audio": 0},
        "speculative_config": {"method": "mtp", "num_speculative_tokens": 1},
    }
