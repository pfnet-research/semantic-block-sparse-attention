#!/usr/bin/env python3
"""Run a sample of a tokenised corpus through the sparse model so that the
plugin's key dump (SEMBLO_DUMP_PROJECTED_KEYS, SEMBLO_DUMP_PROJECTED_KEYS_SAMPLE)
writes one file per prompt and layer: the input of scripts/fit_corpus_codebook.py
(the fixed codebook) and of the PCA map of scripts/make_routing_map.py.

The corpus is a `datasets` directory of tokenised rows (`input_ids`), in the
paper 32K-token Dolma 3 documents, one per row.  `--rows` rows are drawn with a fixed seed, each is prefilled
once with one generated token, and the plugin dumps the keys it would have
clustered.

    SEMBLO_DUMP_PROJECTED_KEYS=<dir> SEMBLO_DUMP_PROJECTED_KEYS_SAMPLE=1024 \
    SEMBLO_TARGET_TOKENS=2048 SEMBLO_FLAT_UNIT_TOKENS=2048 VLLM_PLUGINS=semblo \
    python scripts/dump_corpus_keys.py --dataset <dir> --rows 1000 --seed 0 \
        --model <raw-keys checkpoint> --tokenizer Qwen/Qwen3.5-4B \
        --max-model-len 33792 --hf-text-override max_position_embeddings=33792 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any


def parse_text_overrides(entries: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for entry in entries:
        key, _, raw = entry.partition("=")
        if not key or not raw:
            raise SystemExit(f"--hf-text-override wants key=value, got {entry!r}")
        try:
            out[key] = json.loads(raw)
        except json.JSONDecodeError:
            try:
                out[key] = int(raw)
            except ValueError:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="a `datasets` directory with an input_ids column")
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sequence-length", type=int, default=32768,
                    help="rows shorter than this are skipped, longer ones are cut")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-model-len", type=int, required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    ap.add_argument("--compilation-config", default="")
    ap.add_argument("--hf-text-override", action="append", default=[])
    ap.add_argument("--manifest", required=True, help="where to write the list of rows used")
    args = ap.parse_args()

    dump_dir = os.environ.get("SEMBLO_DUMP_PROJECTED_KEYS")
    if not dump_dir:
        raise SystemExit("SEMBLO_DUMP_PROJECTED_KEYS is not set; the run would dump nothing")
    # The keys are dumped from the index build, which a prompt at or below the
    # engage threshold skips.  A dump run always wants the sparse path.
    os.environ.setdefault("SEMBLO_ENGAGE_ABOVE_TOKENS", "0")

    import numpy as np
    from datasets import load_from_disk

    ds = load_from_disk(args.dataset)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ds))
    chosen: list[int] = []
    for i in order:
        if len(chosen) >= args.rows:
            break
        ids = ds[int(i)]["input_ids"]
        if len(ids) >= args.sequence_length:
            chosen.append(int(i))
    if len(chosen) < args.rows:
        raise SystemExit(f"only {len(chosen)} rows of at least {args.sequence_length} tokens in {args.dataset}")
    sources = {}
    if "source" in ds.column_names:
        for i in chosen:
            s = ds[i]["source"]
            sources[s] = sources.get(s, 0) + 1
    with open(args.manifest, "w") as fh:
        json.dump({"dataset": args.dataset, "seed": args.seed, "rows": chosen,
                   "sequence_length": args.sequence_length, "sources": sources}, fh, indent=1)
    print(f"{len(chosen)} rows of {len(ds)} chosen (seed {args.seed}); sources: {sources}", flush=True)

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    engine: dict[str, Any] = dict(
        model=args.model, tokenizer=args.tokenizer, dtype="bfloat16",
        max_model_len=args.max_model_len, max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1, enable_chunked_prefill=False, enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=False, seed=1234,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    if args.compilation_config:
        engine["compilation_config"] = json.loads(args.compilation_config)
    overrides = parse_text_overrides(args.hf_text_override)
    if overrides:
        engine["hf_overrides"] = {"text_config": overrides}
    llm = LLM(**engine)
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, detokenize=False)

    started = time.perf_counter()
    for n, i in enumerate(chosen, 1):
        ids = [int(t) for t in ds[i]["input_ids"][: args.sequence_length]]
        llm.generate([TokensPrompt(prompt_token_ids=ids)], params, use_tqdm=False)
        if n == 1 and not [f for f in os.listdir(dump_dir) if f.endswith(".npz")]:
            # A run that writes nothing has a configuration problem; stop on
            # the first prompt.
            raise SystemExit(
                f"the first prompt wrote no dump into {dump_dir}: the sparse "
                "path did not build an index (is SEMBLO_ENGAGE_ABOVE_TOKENS=0 and "
                "VLLM_PLUGINS=semblo set?)")
        if n % 100 == 0 or n == len(chosen):
            files = len([f for f in os.listdir(dump_dir) if f.endswith(".npz")])
            print(f"   {n}/{len(chosen)} prompts, {files} dump files, {time.perf_counter() - started:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
