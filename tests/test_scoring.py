from __future__ import annotations

import pytest

from semblo.scoring import (
    extract_longbench_v2_evalscope_answer,
    extract_longbench_v2_official_answer,
    score_multiple_choice,
    strip_visible_thinking,
)


def test_strip_visible_thinking_supports_prompt_opened_block():
    final, stripped = strip_visible_thinking(
        "reasoning emitted after a prompt marker</think>\nfinal"
    )

    assert final == "final"
    assert stripped is True


def test_longbench_v2_official_extraction_is_intentionally_strict():
    assert (
        extract_longbench_v2_official_answer(
            "Reasoning. The correct answer is (C)"
        )
        == "C"
    )
    assert extract_longbench_v2_official_answer("ANSWER: C") is None


def test_longbench_v2_evalscope_extraction_uses_last_explicit_answer():
    assert (
        extract_longbench_v2_evalscope_answer(
            "ANSWER: A\nreconsidered\nANSWER: [D]"
        )
        == "D"
    )


def test_multiple_choice_scores_after_visible_reasoning():
    metrics = score_multiple_choice(
        "<think>analysis mentioning A</think>\nThe correct answer is (B)",
        "B",
        extraction="official_longbench_v2",
    )

    assert metrics["score"] == 1.0
    assert metrics["prediction_letter"] == "B"
    assert metrics["thinking_stripped"] is True


def test_unknown_multiple_choice_extractor_is_rejected():
    with pytest.raises(ValueError, match="Unknown multiple-choice"):
        score_multiple_choice("A", "A", extraction="unsupported")

