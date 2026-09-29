#!/usr/bin/env python3
"""Rebuild the latency prompt, assets/latency_prompt/dolma1m_qwen_ids.npy.

The latency measurements read 1,050,624 Qwen3.5 token ids of Dolma text from
that file. ``--prompt-tokens L --prompt-offset w`` takes tokens [w L, (w + 1) L).

The ids are the first 1,050,624 tokens of the full-length rows of a tokenized
Dolma corpus, concatenated in row order. The corpus is a ``save_to_disk`` dataset
built from allenai/dolma3_longmino_pool with the Qwen/Qwen3.5-4B tokenizer, in
rows of 32,768 tokens (seed 1091). The script that builds it is not part of this
repository. Rows that are not filled to 32,768 tokens are skipped. The first 33
rows are all full. Each row is one document span, so the prompt is 33
documents back to back without a separator.

    python scripts/build_latency_prompt.py \\
        --corpus <corpus directory> \\
        --out /tmp/dolma1m_qwen_ids.npy --compare assets/latency_prompt/dolma1m_qwen_ids.npy
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib

import numpy as np

TOKENS = 1_050_624


def build(corpus: pathlib.Path, tokens: int = TOKENS) -> np.ndarray:
    """The first ``tokens`` ids of the corpus's full-length rows, concatenated in row order."""
    from datasets import load_from_disk

    ds = load_from_disk(str(corpus))
    parts: list[np.ndarray] = []
    have = 0
    for row in range(len(ds)):
        item = ds[row]
        if int(item["effective_nonpad_length"]) < len(item["input_ids"]):
            continue                      # skip padded rows
        ids = np.asarray(item["input_ids"], dtype=np.int32)
        parts.append(ids)
        have += len(ids)
        if have >= tokens:
            return np.concatenate(parts)[:tokens]
    raise SystemExit(f"{corpus} holds only {have} tokens, fewer than {tokens}")


def digest(ids: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(ids, dtype=np.int32).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=pathlib.Path, required=True,
                    help="the save_to_disk directory of the Qwen3.5 Dolma corpus (router-longdoc-32k-qwen35-full)")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--tokens", type=int, default=TOKENS)
    ap.add_argument("--compare", type=pathlib.Path, default=None,
                    help="an existing ids file that the rebuilt one must equal")
    a = ap.parse_args()
    ids = build(a.corpus, a.tokens)
    np.save(a.out, ids)
    print(f"{a.out}: {len(ids)} ids, sha256 of the int32 ids {digest(ids)}")
    if a.compare is not None:
        old = np.load(a.compare)
        if old.shape != ids.shape or not np.array_equal(old.astype(np.int32), ids):
            raise SystemExit(f"differs from {a.compare}")
        print(f"identical to {a.compare}")


if __name__ == "__main__":
    main()
