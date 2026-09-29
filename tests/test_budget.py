
from semblo.budget import ContextBudgetError, resolve_context_budget
from semblo.config import GenerationConfig, ModelSpec
from semblo.prompting import make_prompt_render


def _prompt(n_tokens: int):
    return make_prompt_render(
        messages=[{"role": "user", "content": "x"}],
        rendered_prompt="x",
        prompt_tokens_estimate=n_tokens,
        renderer="unit_test",
        chat_template_kwargs={},
        system_prompt_applied=False,
        token_ids=list(range(n_tokens)),
    )


def test_context_budget_allows_fit():
    gen = GenerationConfig(max_new_tokens=10)
    spec = ModelSpec(key="m", hf_id="m", max_model_len=100)
    effective, budget = resolve_context_budget(prompt_render=_prompt(90), generation=gen, model_spec=spec)
    assert effective.max_new_tokens == 10
    assert budget.decision == "run"
    assert budget.fits_requested is True


def test_context_budget_skips_overflow_by_default():
    gen = GenerationConfig(max_new_tokens=11)
    spec = ModelSpec(key="m", hf_id="m", max_model_len=100)
    _, budget = resolve_context_budget(prompt_render=_prompt(90), generation=gen, model_spec=spec)
    assert budget.decision == "skip"
    assert budget.reason == "context_over_model_len"
    assert budget.overflow_tokens == 1


def test_context_budget_can_shrink_generation():
    gen = GenerationConfig(max_new_tokens=20)
    spec = ModelSpec(key="m", hf_id="m", max_model_len=100)
    effective, budget = resolve_context_budget(
        prompt_render=_prompt(90),
        generation=gen,
        model_spec=spec,
        policy="shrink_generation",
        min_new_tokens=4,
    )
    assert effective.max_new_tokens == 10
    assert budget.was_shrunk is True
    assert budget.effective_total_tokens == 100


def test_context_budget_error_policy_raises():
    gen = GenerationConfig(max_new_tokens=20)
    spec = ModelSpec(key="m", hf_id="m", max_model_len=100)
    try:
        resolve_context_budget(prompt_render=_prompt(90), generation=gen, model_spec=spec, policy="error")
    except ContextBudgetError as exc:
        assert exc.budget.decision == "error"
    else:
        raise AssertionError("error policy should raise")


def test_context_budget_margin_counts_against_limit():
    gen = GenerationConfig(max_new_tokens=10)
    spec = ModelSpec(key="m", hf_id="m", max_model_len=100)
    _, budget = resolve_context_budget(prompt_render=_prompt(90), generation=gen, model_spec=spec, margin_tokens=1)
    assert budget.decision == "skip"
    assert budget.overflow_tokens == 1
