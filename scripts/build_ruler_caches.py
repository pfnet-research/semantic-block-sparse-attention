#!/usr/bin/env python3
"""Build (or find) the cached RULER prompts a suite will read, without a model.

    python scripts/build_ruler_caches.py --suite configs/lm_eval/ruler_512k_focus_vllm_cached_fill.yaml \\
        --tokenizer Qwen/Qwen3.5-4B --task semblo_ruler_qa_squad --task semblo_ruler_cwe

Reads the suite's ``metadata`` (max_seq_lengths, samples_per_length, fill, ...) and calls the
task's factory (lm_eval_tasks/ruler_cached/cached_utils.py) with the same kwargs as
``semblo-lm-eval`` does. The cache key is therefore the same, and a later evaluation run reads the
prompts instead of generating them. Runs on the CPU and prints the cache directory and the token
lengths of a few items. SEMBLO_RULER_CACHE_DIR selects the cache root, as for the evaluation.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lm_eval_tasks" / "ruler_cached"))

FACTORY = {"semblo_ruler_qa_squad": "ruler_qa_squad", "semblo_ruler_cwe": "ruler_cwe", "semblo_ruler_vt": "ruler_vt",
           "semblo_ruler_niah_multikey_3": "niah_multikey_3", "semblo_ruler_niah_multikey_2": "niah_multikey_2",
           "semblo_ruler_niah_multivalue": "niah_multivalue", "semblo_ruler_qa_hotpot": "ruler_qa_hotpot",
           "semblo_ruler_fwe": "ruler_fwe"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tokenizer", required=True, help="the model entry's tokenizer_id (the runner forwards it as the task's tokenizer kwarg)")
    ap.add_argument("--task", action="append", required=True)
    ap.add_argument("--check-items", type=int, default=5)
    ap.add_argument("--copy-from", default=None, help="another cache root: a cache with the same key there is copied instead of rebuilt (the 4B, 35B-A3B and 2B share the tokenizer, so their roots hold the same prompts)")
    a = ap.parse_args()
    import cached_utils as cu
    from transformers import AutoTokenizer
    suite = yaml.safe_load(open(a.suite))
    metadata = dict(suite.get("metadata") or {})
    kwargs = {"tokenizer": a.tokenizer, **metadata}
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    for task in a.task:
        name = FACTORY[task]
        payload = cu._cache_payload(name, dict(kwargs))
        path = cu._cache_root() / name / cu._cache_key(payload)
        existed = path.exists()
        if not existed and a.copy_from and (pathlib.Path(a.copy_from) / name / path.name).is_dir():
            import shutil
            shutil.copytree(pathlib.Path(a.copy_from) / name / path.name, path)
            print(f"{task}: copied from {a.copy_from}", flush=True)
        ds = getattr(cu, name)(**kwargs)
        ds = ds["test"] if "test" in ds else ds
        n = len(ds); idx = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})[: a.check_items]
        lengths = [len(tok(ds[i]["input"])["input_ids"]) for i in idx]
        print(f"{task}: {'found' if existed else 'built'} {path} ({n} items; items {idx}: min {min(lengths)} median {int(statistics.median(lengths))} max {max(lengths)} tokens)", flush=True)


if __name__ == "__main__":
    main()
