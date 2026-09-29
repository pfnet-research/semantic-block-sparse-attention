from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal

from semblo.config import GenerationConfig, ModelSpec
from semblo.prompting import PromptRender

BudgetPolicy = Literal["skip", "error", "shrink_generation", "none"]


@dataclass(frozen=True)
class ContextBudget:
    prompt_tokens: int | None
    requested_max_new_tokens: int
    effective_max_new_tokens: int
    max_model_len: int
    margin_tokens: int = 0
    min_new_tokens: int = 1
    available_new_tokens: int | None = None
    requested_total_tokens: int | None = None
    effective_total_tokens: int | None = None
    overflow_tokens: int | None = None
    prompt_overflow_tokens: int | None = None
    fits_prompt: bool | None = None
    fits_requested: bool | None = None
    was_shrunk: bool = False
    policy: str = "skip"
    decision: str = "run"  # run | skip | error | unchecked
    reason: str | None = None

    def to_record(self) -> dict[str, int | bool | str | None]:
        return asdict(self)


class ContextBudgetError(ValueError):
    def __init__(self, budget: ContextBudget):
        super().__init__(format_budget_error(budget))
        self.budget = budget


def format_budget_error(budget: ContextBudget) -> str:
    r = budget.reason or "context_budget_error"
    if budget.prompt_tokens is None:
        return f"{r}: prompt token count is unavailable"
    return (
        f"{r}: prompt_tokens={budget.prompt_tokens}, "
        f"max_new_tokens={budget.requested_max_new_tokens}, "
        f"margin_tokens={budget.margin_tokens}, "
        f"requested_total={budget.requested_total_tokens}, "
        f"max_model_len={budget.max_model_len}"
    )


def _unchecked_budget(
    *,
    generation: GenerationConfig,
    model_spec: ModelSpec,
    margin_tokens: int,
    min_new_tokens: int,
    policy: str,
    reason: str,
) -> ContextBudget:
    return ContextBudget(
        prompt_tokens=None,
        requested_max_new_tokens=int(generation.max_new_tokens),
        effective_max_new_tokens=int(generation.max_new_tokens),
        max_model_len=int(model_spec.max_model_len),
        margin_tokens=int(margin_tokens),
        min_new_tokens=int(min_new_tokens),
        fits_prompt=None,
        fits_requested=None,
        policy=policy,
        decision="unchecked",
        reason=reason,
    )


def resolve_context_budget(
    *,
    prompt_render: PromptRender,
    generation: GenerationConfig,
    model_spec: ModelSpec,
    policy: BudgetPolicy = "skip",
    margin_tokens: int = 0,
    min_new_tokens: int = 1,
) -> tuple[GenerationConfig, ContextBudget]:
    """Check that the prompt plus the generated tokens fit in ``max_model_len``.

    The check uses the token count of the rendered chat template, so the template,
    the system prompt and the generation reserve are all included.
    """
    if policy not in {"skip", "error", "shrink_generation", "none"}:
        raise ValueError(f"Unknown context budget policy: {policy}")
    if generation.max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {generation.max_new_tokens}")
    if margin_tokens < 0:
        raise ValueError(f"margin_tokens must be >= 0, got {margin_tokens}")
    if min_new_tokens < 1:
        raise ValueError(f"min_new_tokens must be >= 1, got {min_new_tokens}")

    prompt_tokens = prompt_render.token_count
    if policy == "none":
        return generation, _unchecked_budget(
            generation=generation,
            model_spec=model_spec,
            margin_tokens=margin_tokens,
            min_new_tokens=min_new_tokens,
            policy=policy,
            reason="budget_check_disabled",
        )
    if prompt_tokens is None:
        budget = _unchecked_budget(
            generation=generation,
            model_spec=model_spec,
            margin_tokens=margin_tokens,
            min_new_tokens=min_new_tokens,
            policy=policy,
            reason="prompt_token_count_unavailable",
        )
        if policy == "error":
            raise ContextBudgetError(replace(budget, decision="error"))
        return generation, budget

    prompt_tokens = int(prompt_tokens)
    requested_max_new = int(generation.max_new_tokens)
    max_model_len = int(model_spec.max_model_len)
    usable_context = max_model_len - int(margin_tokens)
    available_new = usable_context - prompt_tokens
    requested_total = prompt_tokens + requested_max_new + int(margin_tokens)
    fits_prompt = prompt_tokens + int(margin_tokens) <= max_model_len
    fits_requested = requested_total <= max_model_len

    base = ContextBudget(
        prompt_tokens=prompt_tokens,
        requested_max_new_tokens=requested_max_new,
        effective_max_new_tokens=requested_max_new,
        max_model_len=max_model_len,
        margin_tokens=int(margin_tokens),
        min_new_tokens=int(min_new_tokens),
        available_new_tokens=available_new,
        requested_total_tokens=requested_total,
        effective_total_tokens=requested_total,
        overflow_tokens=max(0, requested_total - max_model_len),
        prompt_overflow_tokens=max(0, prompt_tokens + int(margin_tokens) - max_model_len),
        fits_prompt=fits_prompt,
        fits_requested=fits_requested,
        policy=policy,
    )

    if fits_requested:
        return generation, base

    reason = "prompt_over_model_len" if not fits_prompt else "context_over_model_len"
    if policy == "error":
        raise ContextBudgetError(replace(base, decision="error", reason=reason))
    if policy == "skip":
        return generation, replace(base, decision="skip", reason=reason)

    # shrink_generation
    if available_new < int(min_new_tokens):
        return generation, replace(base, decision="skip", reason="insufficient_generation_budget")
    effective_max_new = min(requested_max_new, available_new)
    shrunk_generation = replace(generation, max_new_tokens=effective_max_new)
    return shrunk_generation, replace(
        base,
        effective_max_new_tokens=effective_max_new,
        effective_total_tokens=prompt_tokens + effective_max_new + int(margin_tokens),
        was_shrunk=True,
        decision="run",
        reason="max_new_tokens_shrunk",
    )
