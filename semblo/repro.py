from __future__ import annotations

import hashlib
import os
import platform
import random
import sys
from importlib import metadata, util
from typing import Any

from semblo.config import ModelSpec

_VERSION_PACKAGES = [
    "torch",
    "transformers",
    "vllm",
    "datasets",
    "huggingface_hub",
    "lm_eval",
    "numpy",
    "pandas",
]


def apply_runtime_env(model_spec: ModelSpec) -> None:
    env = dict(getattr(model_spec, "env", {}) or {})
    runtime_env = getattr(model_spec, "runtime_env", None)
    if runtime_env:
        env.update(runtime_env)
    for key, value in env.items():
        os.environ[str(key)] = str(value)


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if util.find_spec("numpy") is not None:
        import numpy as np
        np.random.seed(seed % (2**32 - 1))

    if util.find_spec("torch") is not None:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: package_version(name) for name in _VERSION_PACKAGES},
        "env": {
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "PYTHONHASHSEED",
                "CUBLAS_WORKSPACE_CONFIG",
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "TOKENIZERS_PARALLELISM",
                "NCCL_DEBUG",
                "NCCL_IB_DISABLE",
                "NCCL_P2P_DISABLE",
                "NCCL_SOCKET_IFNAME",
                "HF_HOME",
                "HF_DATASETS_CACHE",
                "TRANSFORMERS_CACHE",
            ]
            if os.environ.get(key) is not None
        },
    }
    if util.find_spec("torch") is not None:
        import torch
        info["torch"] = {
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            info["torch"]["devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


def seed_for_example(base_seed: int, example_id: str) -> int:
    payload = f"{int(base_seed)}::{example_id}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16)
