from __future__ import annotations

import re
from typing import Any


def strip_visible_thinking(text: str) -> tuple[str, bool]:
    """Remove a leading visible reasoning block before scoring.

    Handles a ``<think>...</think>`` block. Only a leading block is removed,
    and the text after its closing marker is kept.
    """

    stripped = str(text).strip()
    think_end = stripped.find("</think>")
    if think_end >= 0 and (
        stripped.startswith("<think>") or not stripped.startswith("</think>")
    ):
        return stripped[think_end + len("</think>") :].lstrip(), True
    return str(text), False


def extract_mc_letter(text: str) -> str | None:
    """Lenient A-D extraction, for prompts other than the official one."""

    upper = str(text).upper()
    patterns = [
        r"FINAL\s+ANSWER\s*[:：]?\s*\[?\[?\s*([ABCD])\s*\]?\]?",
        r"ANSWER\s*[:：]?\s*\[?\[?\s*([ABCD])\s*\]?\]?",
        r"\[\[\s*([ABCD])\s*\]\]",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, upper)
        if matches:
            return matches[-1]
    standalone = re.findall(r"\b([ABCD])\b", upper)
    return standalone[-1] if standalone else None


def extract_longbench_v2_official_answer(text: str) -> str | None:
    """Mirror LongBench-v2's official answer extraction."""

    response = str(text).replace("*", "")
    matches = re.findall(r"The correct answer is \(([A-D])\)", response)
    if matches:
        return matches[-1]
    matches = re.findall(r"The correct answer is ([A-D])", response)
    return matches[-1] if matches else None


def extract_longbench_v2_evalscope_answer(text: str) -> str | None:
    response = str(text).replace("*", "")
    matches = re.findall(
        r"ANSWER\s*[:：]\s*\[?\s*([A-D])\s*\]?",
        response,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()
    return extract_longbench_v2_official_answer(response)


def score_multiple_choice(
    prediction: str,
    answer: str,
    extraction: str = "lenient",
) -> dict[str, Any]:
    prediction, thinking_stripped = strip_visible_thinking(prediction)
    if extraction == "official_longbench_v2":
        pred_letter = extract_longbench_v2_official_answer(prediction)
    elif extraction == "evalscope_longbench_v2":
        pred_letter = extract_longbench_v2_evalscope_answer(prediction)
    elif extraction == "lenient":
        pred_letter = extract_mc_letter(prediction)
    else:
        raise ValueError(f"Unknown multiple-choice extraction mode: {extraction}")

    gold = str(answer).strip().upper()[:1]
    return {
        "score": 1.0 if pred_letter == gold else 0.0,
        "prediction_letter": pred_letter,
        "gold_letter": gold,
        "extraction": extraction,
        "thinking_stripped": thinking_stripped,
    }

