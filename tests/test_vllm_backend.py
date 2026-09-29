from __future__ import annotations

from semblo.backends.vllm import VLLMBackend
from semblo.config import GenerationConfig


def test_vllm_sampling_params_preserve_special_token_setting():
    backend = object.__new__(VLLMBackend)

    params = backend._sampling_params(GenerationConfig(skip_special_tokens=False))

    assert params["skip_special_tokens"] is False
