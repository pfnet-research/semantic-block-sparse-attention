from __future__ import annotations

from dataclasses import asdict, is_dataclass
import math
import time
from typing import Any

from semblo.backends.base import GenerationOutput
from semblo.config import GenerationConfig, ModelSpec
from semblo.hf_loading import load_auto_tokenizer
from semblo.parallel import (
    validate_vllm_parallel_config,
    visible_cuda_device_count_from_env,
)
from semblo.prompting import (
    Message,
    PromptRender,
    render_prompt,
    tokenizer_prompt_metadata,
)
from semblo.repro import apply_runtime_env


def _jsonable_metric_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value):
        return _jsonable_metric_value(asdict(value))
    if isinstance(value, dict):
        return {
            str(key): _jsonable_metric_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable_metric_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _jsonable_metric_value(value.model_dump())
    if hasattr(value, "__dict__"):
        return _jsonable_metric_value(vars(value))
    return repr(value)


class VLLMBackend:
    """Offline ``vllm.LLM`` backend, for dense models and for models served with the plugin.

    The plugin is selected by the model's architecture name, so both run through this class.
    """

    def __init__(self, model_spec: ModelSpec, engine_seed: int | None = None):
        apply_runtime_env(model_spec)

        from vllm import LLM

        self.model_spec = model_spec
        tokenizer_id = model_spec.tokenizer_id or model_spec.hf_id
        self.tokenizer = load_auto_tokenizer(
            tokenizer_id,
            revision=model_spec.revision,
            trust_remote_code=model_spec.trust_remote_code,
            **model_spec.tokenizer_kwargs,
        )

        vllm_kwargs: dict[str, Any] = dict(model_spec.vllm)
        lora_path = vllm_kwargs.pop("lora_path", None)
        lora_name = str(vllm_kwargs.pop("lora_name", model_spec.key))
        lora_int_id = int(vllm_kwargs.pop("lora_int_id", 1))
        if model_spec.hf_overrides:
            vllm_kwargs.setdefault("hf_overrides", model_spec.hf_overrides)

        defaults = {
            "model": model_spec.hf_id,
            "tokenizer": tokenizer_id,
            "revision": model_spec.revision,
            "trust_remote_code": model_spec.trust_remote_code,
            "dtype": model_spec.dtype,
            "max_model_len": model_spec.max_model_len,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.92,
            "max_num_seqs": 1,
            "seed": (
                model_spec.generation.seed
                if engine_seed is None
                else int(engine_seed)
            ),
        }
        for key, value in defaults.items():
            if value is not None:
                vllm_kwargs.setdefault(key, value)

        self.lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest

            vllm_kwargs.setdefault("enable_lora", True)
            self.lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=str(lora_path),
            )

        available_gpus = visible_cuda_device_count_from_env()
        self.parallel_summary = validate_vllm_parallel_config(
            vllm_kwargs,
            available_cuda_devices=available_gpus,
        )
        self.vllm_kwargs = vllm_kwargs
        self.llm = LLM(**vllm_kwargs)

    def render_prompt(self, messages: list[Message]) -> PromptRender:
        return render_prompt(self.tokenizer, messages, self.model_spec)

    def count_tokens(self, messages: list[Message]) -> int | None:
        return self.render_prompt(messages).token_count

    @staticmethod
    def _sampling_params(generation: GenerationConfig) -> dict[str, Any]:
        if generation.max_new_tokens < 1:
            raise ValueError(
                f"vLLM max_tokens must be >= 1, got {generation.max_new_tokens}"
            )
        params: dict[str, Any] = {
            "max_tokens": generation.max_new_tokens,
            "temperature": generation.temperature,
            "top_p": generation.top_p,
            "seed": generation.seed,
            "ignore_eos": generation.ignore_eos,
            "skip_special_tokens": generation.skip_special_tokens,
        }
        optional = {
            "top_k": generation.top_k,
            "min_p": generation.min_p,
            "repetition_penalty": generation.repetition_penalty,
            "presence_penalty": generation.presence_penalty,
            "frequency_penalty": generation.frequency_penalty,
            "include_stop_str_in_output": generation.include_stop_str_in_output,
        }
        for key, value in optional.items():
            if value is not None:
                params[key] = value
        if generation.stop:
            params["stop"] = generation.stop
        if generation.stop_token_ids:
            params["stop_token_ids"] = generation.stop_token_ids
        return params

    def generate(
        self,
        messages: list[Message],
        generation: GenerationConfig,
        prompt_render: PromptRender | None = None,
    ) -> GenerationOutput:
        from vllm import SamplingParams

        bundle = prompt_render or self.render_prompt(messages)
        if not bundle.token_ids:
            raise ValueError(
                "vLLM backend requires prompt token ids from tokenizer rendering"
            )

        params = self._sampling_params(generation)
        sampling_params = SamplingParams(**params)
        token_prompt = {
            "prompt_token_ids": [int(token_id) for token_id in bundle.token_ids]
        }
        started = time.perf_counter()
        outputs = self.llm.generate(
            [token_prompt],
            sampling_params,
            lora_request=self.lora_request,
        )
        latency = time.perf_counter() - started
        output = outputs[0]
        candidate = output.outputs[0]
        completion_token_ids = [int(token_id) for token_id in candidate.token_ids]
        prompt_tokens = (
            len(output.prompt_token_ids)
            if output.prompt_token_ids is not None
            else bundle.token_count
        )
        return GenerationOutput(
            text=candidate.text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(completion_token_ids),
            latency_s=latency,
            prompt_render=bundle,
            completion_token_ids=completion_token_ids,
            sampling_params=params,
            raw={
                "backend": "vllm",
                "finish_reason": candidate.finish_reason,
                "stop_reason": getattr(candidate, "stop_reason", None),
                "prompt_input_mode": bundle.renderer,
                "lora_request": _jsonable_metric_value(self.lora_request),
                "vllm_metrics": _jsonable_metric_value(
                    getattr(output, "metrics", None)
                ),
            },
        )

    def get_run_manifest(self) -> dict[str, Any]:
        return {
            "backend": "vllm",
            "vllm_kwargs": self.vllm_kwargs,
            "lora_request": _jsonable_metric_value(self.lora_request),
            "parallel": self.parallel_summary.to_record(),
            "tokenizer": tokenizer_prompt_metadata(
                self.tokenizer,
                self.model_spec,
            ),
            "prompt_input_mode": "token_ids_from_prompt_renderer",
        }

    def describe(self) -> dict[str, Any]:
        return self.get_run_manifest()
