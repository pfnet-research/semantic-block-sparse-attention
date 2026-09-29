from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int = 0
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    include_stop_str_in_output: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GenerationConfig":
        data = dict(data or {})
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown generation config keys: {unknown}")
        if isinstance(data.get("stop"), str):
            data["stop"] = [data["stop"]]
        if data.get("stop") is None:
            data["stop"] = []
        if data.get("stop_token_ids") is None:
            data["stop_token_ids"] = []
        return cls(**data)

    def sampling_record(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
            "stop": list(self.stop),
            "stop_token_ids": list(self.stop_token_ids),
            "ignore_eos": self.ignore_eos,
            "skip_special_tokens": self.skip_special_tokens,
            "spaces_between_special_tokens": self.spaces_between_special_tokens,
            "include_stop_str_in_output": self.include_stop_str_in_output,
        }


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_id: str
    backend: str = "vllm"
    tokenizer_id: str | None = None
    revision: str | None = None
    dtype: str = "auto"
    max_model_len: int = 32768
    trust_remote_code: bool = True
    family: str = ""
    display_name: str = ""
    mode: str = ""
    system_prompt: str | None = None
    system_prompt_path: str | None = None
    system_prompt_file: str | None = None
    system_prompt_policy: str = "prepend_if_missing"
    system_prompt_role: str = "system"
    prompt_format: str = "chat_template"
    add_generation_prompt: bool = True
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: dict[str, Any] = field(default_factory=dict)
    hf_overrides: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    vllm: dict[str, Any] = field(default_factory=dict)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    generation_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_family: str | None = None
    architecture: dict[str, Any] = field(default_factory=dict)
    official_context: dict[str, Any] = field(default_factory=dict)
    source_urls: list[str] = field(default_factory=list)
    known_caveats: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    prompt_notes: str = ""

    @classmethod
    def from_dict(cls, key: str, data: dict[str, Any]) -> "ModelSpec":
        data = dict(data)
        gen = GenerationConfig.from_dict(data.pop("generation", {}))
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        allowed.discard("key")
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown model config keys for {key}: {unknown}")
        return cls(key=key, generation=gen, **data)


    def with_profile(self, profile_name: str) -> "ModelSpec":
        """Apply a named profile to the whole model spec.

        Profiles may either be generation-only::

            official_non_thinking:
              temperature: 0.7
              top_p: 0.8

        or may override model-side behavior as well::

            official_thinking:
              chat_template_kwargs:
                enable_thinking: true
              generation:
                temperature: 0.6
        """
        if profile_name not in self.generation_profiles:
            raise KeyError(f"Unknown generation profile {profile_name!r}; available: {sorted(self.generation_profiles)}")

        profile = dict(self.generation_profiles[profile_name] or {})
        generation_keys = set(GenerationConfig.__dataclass_fields__)
        generation_data: dict[str, Any] = {}

        if "generation" in profile:
            generation_data = dict(profile.pop("generation") or {})

        # Top-level generation fields are generation overrides. The other keys
        # override the model spec.
        for key in list(profile.keys()):
            if key in generation_keys:
                generation_data[key] = profile.pop(key)

        allowed_model_keys = set(self.__dataclass_fields__) - {"key", "generation", "generation_profiles"}
        unknown = sorted(set(profile) - allowed_model_keys)
        if unknown:
            raise ValueError(f"Unknown model profile keys for {self.key}/{profile_name}: {unknown}")

        updates: dict[str, Any] = {"mode": profile_name}
        for key, value in profile.items():
            current = getattr(self, key)
            if isinstance(current, dict) and isinstance(value, dict):
                updates[key] = deep_merge(current, value)
            else:
                updates[key] = value

        updates["generation"] = GenerationConfig.from_dict(deep_merge(self.generation.sampling_record(), generation_data))
        return replace(self, **updates)

    def resolved_system_prompt(self) -> str | None:
        if self.system_prompt is not None:
            return self.system_prompt
        path = self.system_prompt_path or self.system_prompt_file
        if path:
            return Path(path).read_text(encoding="utf-8")
        return None


def generation_overrides_from_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    profile = dict(profile or {})
    generation_keys = set(GenerationConfig.__dataclass_fields__)
    generation_data: dict[str, Any] = dict(profile.pop("generation", {}) or {})
    for key in list(profile.keys()):
        if key in generation_keys:
            generation_data[key] = profile.pop(key)
    return generation_data
