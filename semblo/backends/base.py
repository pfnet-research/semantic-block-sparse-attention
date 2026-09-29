from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from semblo.config import GenerationConfig, ModelSpec
from semblo.prompting import Message, PromptRender


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    prompt_render: PromptRender | None = None
    completion_token_ids: list[int] | None = None
    sampling_params: dict[str, Any] = field(default_factory=dict)
    context_budget: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    model_spec: ModelSpec

    def render_prompt(self, messages: list[Message]) -> PromptRender:
        ...

    def count_tokens(self, messages: list[Message]) -> int | None:
        ...

    def generate(self, messages: list[Message], generation: GenerationConfig, prompt_render: PromptRender | None = None) -> GenerationOutput:
        ...

    def get_run_manifest(self) -> dict[str, Any]:
        ...
