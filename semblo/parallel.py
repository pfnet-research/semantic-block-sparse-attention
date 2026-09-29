from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from semblo.config import ModelSpec, parse_scalar


@dataclass(frozen=True)
class ParallelConfigSummary:
    tensor_parallel_size: int
    pipeline_parallel_size: int
    decode_context_parallel_size: int
    prefill_context_parallel_size: int
    data_parallel_size: int
    local_worker_gpus: int
    data_parallel_replicas: int
    total_logical_gpus: int
    available_cuda_devices: int | None
    warnings: list[str]

    def to_record(self) -> dict[str, Any]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "decode_context_parallel_size": self.decode_context_parallel_size,
            "prefill_context_parallel_size": self.prefill_context_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "local_worker_gpus": self.local_worker_gpus,
            "data_parallel_replicas": self.data_parallel_replicas,
            "total_logical_gpus": self.total_logical_gpus,
            "available_cuda_devices": self.available_cuda_devices,
            "warnings": list(self.warnings),
        }



def parse_human_int(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Expected an integer-like value, got {value!r}")
        return int(value)
    text = str(value).strip()
    if text.lower() == "auto":
        return "auto"
    match = re.fullmatch(r"([+-]?[0-9]+(?:\.[0-9]+)?)([kKmMgG]?)", text)
    if not match:
        raise ValueError(f"Expected an integer or k/m/g-suffixed integer, got {value!r}")
    number = float(match.group(1))
    suffix = match.group(2)
    if suffix == "k":
        number *= 1_000
    elif suffix == "K":
        number *= 1_024
    elif suffix == "m":
        number *= 1_000_000
    elif suffix == "M":
        number *= 1_048_576
    elif suffix == "g":
        number *= 1_000_000_000
    elif suffix == "G":
        number *= 1_073_741_824
    if not number.is_integer():
        raise ValueError(f"Expected integer token count after suffix expansion, got {value!r}")
    return int(number)


def _split_key_value(item: str, flag_name: str) -> tuple[str, str]:
    key, sep, value = item.partition("=")
    if not sep:
        raise ValueError(f"{flag_name} must be KEY=VALUE, got {item!r}")
    if not key:
        raise ValueError(f"{flag_name} key must not be empty: {item!r}")
    return key, value


def parse_vllm_arg_overrides(items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        key, value = _split_key_value(item, "--vllm-arg")
        overrides[key] = parse_scalar(value)
    return overrides


def parse_vllm_json_arg_overrides(items: list[str]) -> dict[str, Any]:
    """Parse JSON-valued vLLM EngineArgs overrides.

    For EngineArgs whose value is a dict or list, e.g.
    ``--vllm-json-arg 'limit_mm_per_prompt={"image":0,"audio":0}'``. Scalars use ``--vllm-arg``.
    """
    overrides: dict[str, Any] = {}
    for item in items:
        key, value = _split_key_value(item, "--vllm-json-arg")
        overrides[key] = json.loads(value)
    return overrides


def apply_vllm_cli_overrides(model_spec: ModelSpec, args: Any) -> dict[str, Any]:
    vllm = dict(model_spec.vllm)
    direct_map = {
        "tensor_parallel_size": "tensor_parallel_size",
        "pipeline_parallel_size": "pipeline_parallel_size",
        "decode_context_parallel_size": "decode_context_parallel_size",
        "prefill_context_parallel_size": "prefill_context_parallel_size",
        "data_parallel_size": "data_parallel_size",
        "distributed_executor_backend": "distributed_executor_backend",
        "gpu_memory_utilization": "gpu_memory_utilization",
        "max_num_seqs": "max_num_seqs",
        "max_num_batched_tokens": "max_num_batched_tokens",
    }
    for arg_name, key in direct_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            if key == "max_num_batched_tokens":
                value = parse_human_int(value)
            vllm[key] = value
    if getattr(args, "enable_chunked_prefill", False):
        vllm["enable_chunked_prefill"] = True
    if getattr(args, "disable_chunked_prefill", False):
        vllm["enable_chunked_prefill"] = False
    vllm.update(parse_vllm_arg_overrides(getattr(args, "vllm_arg", []) or []))
    vllm.update(parse_vllm_json_arg_overrides(getattr(args, "vllm_json_arg", []) or []))
    return vllm


def visible_cuda_device_count_from_env() -> int | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        return None
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if any(value in {"-1", "none", "None", "void"} for value in values):
        return 0
    return len(values)


def torch_cuda_device_count(*, allow_torch_import: bool = False) -> int | None:
    """Return the visible CUDA device count, or None if it is unknown.

    The count comes from CUDA_VISIBLE_DEVICES when it is set. Otherwise torch is
    imported only with ``allow_torch_import=True``, and None leaves the check to vLLM.
    """
    env_count = visible_cuda_device_count_from_env()
    if env_count is not None:
        return env_count
    if not allow_torch_import:
        return None

    import torch

    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def _int_kw(kwargs: dict[str, Any], key: str, default: int) -> int:
    value = kwargs.get(key, default)
    if value is None:
        return default
    return int(value)


def summarize_vllm_parallel_config(vllm_kwargs: dict[str, Any], available_cuda_devices: int | None = None) -> ParallelConfigSummary:
    tp = _int_kw(vllm_kwargs, "tensor_parallel_size", 1)
    pp = _int_kw(vllm_kwargs, "pipeline_parallel_size", 1)
    dcp = _int_kw(vllm_kwargs, "decode_context_parallel_size", 1)
    pcp = _int_kw(vllm_kwargs, "prefill_context_parallel_size", 1)
    dp = _int_kw(vllm_kwargs, "data_parallel_size", 1)
    local_worker_gpus = tp * pp
    total_logical_gpus = local_worker_gpus * dp
    warnings: list[str] = []
    if dp > 1:
        warnings.append(
            "vLLM data_parallel_size creates replicas for serving throughput; native semblo-run submits one request at a time, so use semblo-run --num-shards for evaluation throughput unless you know the vLLM DP path you want."
        )
    if pp > 1 and tp > 1:
        warnings.append("TP and PP can be combined, but single-node small-model evaluation usually starts with TP only.")
    if dcp > 1:
        warnings.append("DCP reuses the TP group and reduces KV-cache duplication; it is only valid when tensor_parallel_size is divisible by decode_context_parallel_size.")
    if pcp > 1:
        warnings.append("PCP targets long-prefill latency; validate support/performance with a pilot run on the exact vLLM build.")
    return ParallelConfigSummary(
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        decode_context_parallel_size=dcp,
        prefill_context_parallel_size=pcp,
        data_parallel_size=dp,
        local_worker_gpus=local_worker_gpus,
        data_parallel_replicas=dp,
        total_logical_gpus=total_logical_gpus,
        available_cuda_devices=available_cuda_devices,
        warnings=warnings,
    )


def validate_vllm_parallel_config(
    vllm_kwargs: dict[str, Any],
    *,
    available_cuda_devices: int | None = None,
    enforce_available_gpus: bool = True,
) -> ParallelConfigSummary:
    summary = summarize_vllm_parallel_config(vllm_kwargs, available_cuda_devices=available_cuda_devices)
    positive_fields = {
        "tensor_parallel_size": summary.tensor_parallel_size,
        "pipeline_parallel_size": summary.pipeline_parallel_size,
        "decode_context_parallel_size": summary.decode_context_parallel_size,
        "prefill_context_parallel_size": summary.prefill_context_parallel_size,
        "data_parallel_size": summary.data_parallel_size,
    }
    bad = {key: value for key, value in positive_fields.items() if value < 1}
    if bad:
        raise ValueError(f"vLLM parallel sizes must be >= 1, got {bad}")
    if summary.tensor_parallel_size % summary.decode_context_parallel_size != 0:
        raise ValueError(
            "decode_context_parallel_size must divide tensor_parallel_size "
            f"(tp={summary.tensor_parallel_size}, dcp={summary.decode_context_parallel_size})."
        )
    if enforce_available_gpus and available_cuda_devices is not None and available_cuda_devices > 0:
        if summary.local_worker_gpus > available_cuda_devices:
            raise ValueError(
                "Requested vLLM TP*PP worker GPUs exceed visible CUDA devices: "
                f"tp={summary.tensor_parallel_size}, pp={summary.pipeline_parallel_size}, "
                f"tp*pp={summary.local_worker_gpus}, CUDA devices={available_cuda_devices}. "
                "Set CUDA_VISIBLE_DEVICES or reduce --tensor-parallel-size/--pipeline-parallel-size."
            )
        if summary.total_logical_gpus > available_cuda_devices and summary.data_parallel_size > 1:
            raise ValueError(
                "Requested vLLM TP*PP*DP logical GPUs exceed visible CUDA devices: "
                f"tp={summary.tensor_parallel_size}, pp={summary.pipeline_parallel_size}, dp={summary.data_parallel_size}, "
                f"total={summary.total_logical_gpus}, CUDA devices={available_cuda_devices}."
            )
    return summary


def stable_shard_id(value: str, num_shards: int) -> int:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % num_shards


def example_in_shard(example_id: str, index: int, *, num_shards: int, shard_id: int, shard_by: str) -> bool:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_id < num_shards:
        raise ValueError(f"shard_id must satisfy 0 <= shard_id < num_shards, got shard_id={shard_id}, num_shards={num_shards}")
    if shard_by == "hash":
        return stable_shard_id(example_id, num_shards) == shard_id
    if shard_by == "index":
        return index % num_shards == shard_id
    raise ValueError(f"Unknown shard_by={shard_by!r}; expected 'hash' or 'index'.")


def parse_parameter_count(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([bBmMkK]?)", text)
    if not match:
        return None
    number = float(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "b":
        return number * 1_000_000_000
    if suffix == "m":
        return number * 1_000_000
    if suffix == "k":
        return number * 1_000
    return number
