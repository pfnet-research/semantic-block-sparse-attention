#!/usr/bin/env python3
"""Count the tokens of every LongBench v2 prompt and write the counts to a file.

Selecting items by prompt length means tokenizing all 503 contexts, about 120M
tokens. This script does it once, and the evaluation reads the file through
``token_counts_path``.

The file also records the tokenizer, prompt style and dataset, and the evaluation
checks that they match its own config.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

from semblo.tasks.longbench_v2 import LongBenchV2Task, load_dataset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="THUDM/LongBench-v2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer-id", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--prompt-style", default="official")
    ap.add_argument("--cot", action="store_true")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    task = LongBenchV2Task({
        "dataset_name": a.dataset_name, "split": a.split,
        "tokenizer_id": a.tokenizer_id, "prompt_style": a.prompt_style, "cot": a.cot,
    })
    rows = load_dataset(a.dataset_name, split=a.split)
    counts: dict[str, int] = {}
    t0 = time.time()
    for i, row in enumerate(rows):
        counts[str(row["_id"])] = task.prompt_token_count(row)
        if (i + 1) % 50 == 0:
            print(f"   {i + 1} items, {time.time() - t0:.0f}s", flush=True)

    out = pathlib.Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset_name": a.dataset_name, "split": a.split,
        "tokenizer_id": a.tokenizer_id, "prompt_style": a.prompt_style,
        "cot": a.cot, "counts": counts,
    }, indent=0))

    ns = sorted(counts.values())
    bands = [(65536, "<=64k"), (131072, "64-128k"), (262144, "128-256k"),
             (524288, "256-512k"), (1048576, "512k-1M")]
    print(f"{len(ns)} items in {time.time() - t0:.0f}s -> {out}")
    print(f"   median {ns[len(ns) // 2]:,}   p90 {ns[int(0.9 * len(ns))]:,}   max {ns[-1]:,}")
    lo = 0
    for edge, name in bands:
        print(f"   {name:10} {sum(1 for n in ns if lo < n <= edge)}")
        lo = edge
    print(f"   {'>1M':10} {sum(1 for n in ns if n > 1048576)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
