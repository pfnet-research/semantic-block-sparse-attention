from __future__ import annotations

from semblo.backends.base import Backend
from semblo.config import ModelSpec


def build_backend(
    model_spec: ModelSpec,
    backend_name: str | None = None,
    engine_seed: int | None = None,
) -> Backend:
    name = backend_name or model_spec.backend
    if name == "vllm":
        from semblo.backends.vllm import VLLMBackend

        return VLLMBackend(model_spec, engine_seed=engine_seed)
    raise ValueError(f"Unknown backend: {name!r}")
