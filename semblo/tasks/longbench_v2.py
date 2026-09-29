from __future__ import annotations

import functools
from typing import Any, Iterable

from semblo.scoring import score_multiple_choice
from semblo.tasks.base import Example


@functools.lru_cache(maxsize=4)
def _tokenizer(tokenizer_id: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)


def load_dataset(*args: Any, **kwargs: Any) -> Any:
    """Import the optional datasets dependency only when evaluation runs."""

    from datasets import load_dataset as hf_load_dataset

    return hf_load_dataset(*args, **kwargs)


OFFICIAL_TEMPLATE_0SHOT = (
    "Please read the following text and answer the question below.\n\n"
    "<text>\n"
    "$DOC$\n"
    "</text>\n\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n"
    "\n"
    "Format your response as follows: \"The correct answer is (insert answer here)\"."
)

OFFICIAL_TEMPLATE_COT = (
    "Please read the following text and answer the questions below.\n\n"
    "<text>\n"
    "$DOC$\n"
    "</text>\n\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n"
    "\n"
    "Let’s think step by step:"
)

OFFICIAL_TEMPLATE_COT_ANSWER = (
    "Please read the following text and answer the questions below.\n\n"
    "The text is too long and omitted here.\n\n"
    "What is the correct answer to this question: $Q$\n"
    "Choices:\n"
    "(A) $C_A$\n"
    "(B) $C_B$\n"
    "(C) $C_C$\n"
    "(D) $C_D$\n"
    "\n"
    "Let’s think step by step: $COT$\n\n"
    'Based on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".'
)

EVALSCOPE_TEMPLATE = (
    "Please read the following text and answer the questions below.\n\n"
    "<text>\n"
    "$DOC$\n"
    "</text>\n\n"
    "Answer the following multiple choice question. The last line of your response should be of the following format: "
    "'ANSWER: [LETTER]' (without quotes) where [LETTER] is one of A, B, C, D. Think step by step before answering.\n\n"
    "$Q$\n\n"
    "A) $C_A$\n"
    "B) $C_B$\n"
    "C) $C_C$\n"
    "D) $C_D$"
)


def _fill_template(template: str, row: dict[str, Any]) -> str:
    return (
        template.replace("$DOC$", str(row["context"]).strip())
        .replace("$Q$", str(row["question"]).strip())
        .replace("$C_A$", str(row["choice_A"]).strip())
        .replace("$C_B$", str(row["choice_B"]).strip())
        .replace("$C_C$", str(row["choice_C"]).strip())
        .replace("$C_D$", str(row["choice_D"]).strip())
    )


def fill_official_cot_answer_template(row: dict[str, Any], cot: str) -> str:
    return _fill_template(OFFICIAL_TEMPLATE_COT_ANSWER, row).replace("$COT$", str(cot).strip())


class LongBenchV2Task:
    name = "longbench_v2"

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.dataset_name = config.get("dataset_name", "THUDM/LongBench-v2")
        self.split = config.get("split", "train")
        self.cot = bool(config.get("cot", False))
        self.max_words = config.get("max_words")
        self.domains = set(config.get("domains", []))
        self.difficulties = set(config.get("difficulties", []))
        self.lengths = {str(x).lower() for x in config.get("lengths", [])}
        self.prompt_style = config.get("prompt_style", "official")
        self.extraction_mode = config.get("extraction_mode", "official_longbench_v2")
        self.streaming = bool(config.get("streaming", False))
        # Length control in tokens. The dataset's `length` column is categorical
        # (short/medium/long) and does not give a context length.
        self.tokenizer_id = config.get("tokenizer_id")
        # Counting tokens means tokenizing 503 contexts, about 120M tokens, so the
        # counts can be precomputed once (scripts/count_longbench_v2_tokens.py).
        self.token_counts_path = config.get("token_counts_path")
        self._token_counts = None
        self.max_prompt_tokens = config.get("max_prompt_tokens")
        self.min_prompt_tokens = config.get("min_prompt_tokens")
        # ``truncation: middle`` cuts the raw prompt text here. The CLI's
        # --prompt-truncation-policy middle cuts the rendered prompt instead and
        # records what it removed.
        self.truncation = config.get("truncation", "none")
        if self.truncation not in {"middle", "none"}:
            raise ValueError(f"Unknown LongBench v2 truncation: {self.truncation!r}")
        if (self.max_prompt_tokens or self.min_prompt_tokens) and not self.tokenizer_id:
            raise ValueError(
                "LongBench v2 token-length control needs tokenizer_id: without a "
                "tokenizer the prompt length is unknown and the truncation the "
                "official protocol applies cannot be reproduced."
            )

    def _minimal_prompt(self, row: dict[str, Any]) -> str:
        choices = "\n".join(
            [
                f"A. {row['choice_A']}",
                f"B. {row['choice_B']}",
                f"C. {row['choice_C']}",
                f"D. {row['choice_D']}",
            ]
        )
        if self.cot:
            instruction = "Think carefully using the context, then put the final answer letter in the form [[A]], [[B]], [[C]], or [[D]]."
        else:
            instruction = "Return only one answer letter: A, B, C, or D."
        return (
            "You are answering a long-context multiple-choice question.\n\n"
            f"Context:\n{row['context']}\n\n"
            f"Question:\n{row['question']}\n\n"
            f"Choices:\n{choices}\n\n"
            f"{instruction}"
        )

    def _prompt(self, row: dict[str, Any]) -> str:
        if self.prompt_style == "official":
            return _fill_template(OFFICIAL_TEMPLATE_COT if self.cot else OFFICIAL_TEMPLATE_0SHOT, row)
        if self.prompt_style == "evalscope":
            return _fill_template(EVALSCOPE_TEMPLATE, row)
        if self.prompt_style == "minimal":
            return self._minimal_prompt(row)
        raise ValueError(f"Unknown LongBench v2 prompt_style: {self.prompt_style}")

    def _counts(self) -> dict[str, int] | None:
        """The precomputed prompt-token counts, checked against this config.

        The file's tokenizer, prompt style and dataset must match this task's,
        since they determine which items are selected.
        """
        if self._token_counts is not None:
            return self._token_counts or None
        if not self.token_counts_path:
            self._token_counts = {}
            return None
        import json
        import pathlib

        path = pathlib.Path(self.token_counts_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"token_counts_path {path} does not exist; build it with "
                "scripts/count_longbench_v2_tokens.py"
            )
        blob = json.loads(path.read_text())
        want = {"tokenizer_id": self.tokenizer_id, "prompt_style": self.prompt_style,
                "cot": self.cot, "dataset_name": self.dataset_name, "split": self.split}
        got = {k: blob.get(k) for k in want}
        if got != want:
            raise ValueError(
                f"the token-count cache at {path} was built for {got}, but this "
                f"task is {want}; the band it selects would not be the one asked for"
            )
        self._token_counts = {str(k): int(v) for k, v in blob["counts"].items()}
        return self._token_counts

    def prompt_token_count(self, row: dict[str, Any]) -> int:
        """The number of tokens in this item's prompt, from the precomputed counts if given."""
        counts = self._counts()
        if counts is not None:
            n = counts.get(str(row.get("_id")))
            if n is not None:
                return int(n)
        if not self.tokenizer_id:
            return -1
        return len(_tokenizer(self.tokenizer_id)(
            self._prompt(row), add_special_tokens=False).input_ids)

    def _fit_prompt(self, prompt: str) -> tuple[str, int, bool]:
        """Return the prompt, its token count, and whether it was truncated.

        Keeps the first and last ``max_prompt_tokens // 2`` tokens of the raw
        prompt when ``truncation`` is ``middle``.
        """
        if not self.tokenizer_id:
            return prompt, -1, False
        tok = _tokenizer(self.tokenizer_id)
        ids = tok(prompt, add_special_tokens=False).input_ids
        if self.max_prompt_tokens is None or len(ids) <= int(self.max_prompt_tokens):
            return prompt, len(ids), False
        if self.truncation == "none":
            return prompt, len(ids), False
        half = int(self.max_prompt_tokens) // 2
        cut = (tok.decode(ids[:half], skip_special_tokens=True)
               + tok.decode(ids[-half:], skip_special_tokens=True))
        return cut, len(tok(cut, add_special_tokens=False).input_ids), True

    @staticmethod
    def _token_bin(n_tokens: int) -> str:
        """Name the token-length bin (64k, 128k, ..., 1m, over_1m)."""
        if n_tokens < 0:
            return "unknown"
        for edge, name in ((65536, "64k"), (131072, "128k"), (262144, "256k"),
                           (524288, "512k"), (1048576, "1m")):
            if n_tokens <= edge:
                return name
        return "over_1m"

    @staticmethod
    def _length_bin(row: dict[str, Any]) -> str:
        length = row.get("length", None)
        if length is None:
            return "unknown"
        if isinstance(length, str):
            lowered = length.lower().strip()
            if lowered in {"short", "medium", "long"}:
                return lowered
            try:
                n = int(lowered)
            except ValueError:
                return lowered or "unknown"
        else:
            n = int(length)
        if n <= 32000:
            return "short_0_32k_words"
        if n <= 128000:
            return "medium_32k_128k_words"
        return "long_128k_2m_words"

    def _passes_length_filters(self, row: dict[str, Any]) -> bool:
        length_bin = self._length_bin(row)
        if self.lengths and length_bin.lower() not in self.lengths:
            return False
        if self.max_words is None:
            return True
        raw_length = row.get("length")
        if isinstance(raw_length, str) and not raw_length.isdigit():
            raise ValueError("max_words requires a numeric LongBench v2 length column, but the dataset stores categorical length labels.")
        return int(raw_length or 0) <= int(self.max_words)

    def iter_examples(self) -> Iterable[Example]:
        ds = load_dataset(self.dataset_name, split=self.split, streaming=self.streaming)
        for row in ds:
            if self.domains and row.get("domain") not in self.domains:
                continue
            if self.difficulties and row.get("difficulty") not in self.difficulties:
                continue
            if not self._passes_length_filters(row):
                continue
            # Filter on the token count before building the prompt.
            n_tokens = self.prompt_token_count(row)
            if self.min_prompt_tokens is not None and 0 <= n_tokens < int(self.min_prompt_tokens):
                continue
            prompt = self._prompt(row)
            truncated = False
            if self.truncation != "none":
                prompt, n_tokens, truncated = self._fit_prompt(prompt)
            metadata = {
                "domain": row.get("domain"),
                "sub_domain": row.get("sub_domain"),
                "difficulty": row.get("difficulty"),
                "length": row.get("length"),
                "bin": self._length_bin(row),
                "cot": self.cot,
                "prompt_style": self.prompt_style,
                "extraction_mode": self.extraction_mode,
                "raw_prompt_tokens": n_tokens,
                "token_bin": self._token_bin(n_tokens),
                "truncated": truncated,
                "max_prompt_tokens": self.max_prompt_tokens,
                "tokenizer_id": self.tokenizer_id,
            }
            yield Example(
                id=f"longbench_v2/{row.get('_id')}",
                task_name=self.name,
                messages=[{"role": "user", "content": prompt}],
                answer=str(row["answer"]),
                metadata=metadata,
            )

    def score(self, prediction: str, example: Example) -> dict[str, Any]:
        metrics = score_multiple_choice(prediction, str(example.answer), extraction=self.extraction_mode)
        metrics["official_prompt"] = self.prompt_style == "official"
        return metrics
