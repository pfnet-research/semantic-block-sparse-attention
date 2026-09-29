"""OpenAI MRCR evaluation driver (direct vLLM, no lm-eval).

MRCR (multi-round co-reference resolution): a long synthetic conversation
contains n near-identical needles ("write a poem about tapirs"); the model
must reproduce the i-th one, prefixed by a per-sample random string.  The
official metric is  SequenceMatcher.ratio(response, answer)  after checking
and stripping that prefix (0 if the prefix is missing).

In OpenAI MRCR, unlike GDM MRCR-v2, contexts are not shared across samples,
so the samples are graded independently.

The driver loads the requested needle split from the HF hub, tokenizes each
conversation with the model's chat template, selects the samples of one length
bin, and runs one vLLM engine over them.
"""

from __future__ import annotations

import argparse
import json
from difflib import SequenceMatcher
from pathlib import Path


# The sampling parameters recommended on the model card,
# https://huggingface.co/Qwen/Qwen3.5-4B. MRCR is a general task run in
# thinking mode, so thinking_general is the default.
SAMPLING_PRESETS: dict[str, dict[str, float | int | bool]] = {
    "thinking_general": {
        "thinking": True, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
    },
    "thinking_coding": {
        "thinking": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
    },
    "instruct_general": {
        "thinking": False, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
    },
    "instruct_reasoning": {
        "thinking": False, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
    },
    "greedy": {
        "thinking": False, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
        "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0,
    },
}


def visible_answer(response: str) -> str:
    """Drop the thinking channel, if the model opened one.

    The official metric checks that the response starts with the required hash,
    which a response that begins with a thought never does. Grading the visible
    answer is the usual adaptation for reasoning models.
    """

    marker = "</think>"
    index = response.rfind(marker)
    return response[index + len(marker):].lstrip() if index >= 0 else response


def grade(response: str, answer: str, prefix: str) -> float:
    """The official metric: the prefix must be the first thing said."""
    response = visible_answer(response)
    if not response.startswith(prefix):
        return 0.0
    response = response.removeprefix(prefix)
    answer = answer.removeprefix(prefix)
    return SequenceMatcher(None, response, answer).ratio()


def grade_lenient(response: str, answer: str, prefix: str) -> float:
    """The same, but the prefix may appear anywhere.

    A diagnostic, not the official metric. It separates two failures that the
    official metric scores as zero: an answer that does not start with the
    prefix, and a wrong answer.
    """
    response = visible_answer(response)
    idx = response.rfind(prefix)
    if idx < 0:
        return 0.0
    return SequenceMatcher(None, response[idx + len(prefix):],
                           answer.removeprefix(prefix)).ratio()


def _run_environment() -> dict:
    """Runtime settings and source revision of this run."""
    from semblo.run_metadata import collect_run_metadata

    return collect_run_metadata()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="model dir for vLLM")
    ap.add_argument("--tokenizer-id", default="Qwen/Qwen3.5-4B",
                    help="chat template + token counting; must match the pack")
    ap.add_argument("--needles", type=int, default=2, choices=(2, 4, 8))
    # The official bin, named by its upper edge. The dataset card defines 100
    # samples per bin, with bins cut by the tiktoken (o200k_base) count of
    # prompt + answer.
    ap.add_argument("--official-bin", type=int, default=None,
                    choices=(8192, 16384, 32768, 65536, 131072, 262144,
                             524288, 1048576),
                    help="upper edge of the OpenAI MRCR length bin to run, "
                         "cut with tiktoken exactly as the dataset card does")
    # The same bin edges, but cut by the length the model reads: the prompt
    # under the served tokenizer and chat template. The 800 rows are the same
    # either way. Only the bin of a row near an edge changes.
    ap.add_argument("--qwen-bin", type=int, default=None,
                    choices=(8192, 16384, 32768, 65536, 131072, 262144,
                             524288, 1048576),
                    help="upper edge of the length bin, cut with the served "
                         "tokenizer instead of tiktoken")
    ap.add_argument("--min-tokens", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N samples of the bin, so a bin can be "
                         "split across processes: --offset 25 --limit 25 is the "
                         "second quarter of the 100 that --limit 100 would take")
    # The official runner samples at the API's default temperature of 1.0.
    # --repeats re-runs the whole set with successive seeds.
    ap.add_argument("--sampling-preset", default="thinking_general",
                    choices=sorted(SAMPLING_PRESETS),
                    help="from the Qwen3.5 model card; MRCR is a general task "
                         "in thinking mode")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the preset's temperature")
    # Room for the thinking channel, on top of the longest answer in the
    # selected set. Qwen recommends 32,768 output tokens for thinking mode. A
    # thought that runs out of room never emits </think> and scores zero.
    ap.add_argument("--thinking-budget", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=66560)
    ap.add_argument("--gen-tokens", type=int, default=1024)
    ap.add_argument("--hf-overrides-json", default="",
                    help='extra text_config overrides as JSON, e.g. '
                         '\'{"semblo_target_tokens_per_layer":{"4":16384}}\'')
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="two cards at 1M for the sparse path: the packed "
                         "prompt K/V doubles the long-context KV")
    ap.add_argument("--kv-cache-memory-bytes", type=int, default=0,
                    help="pin the pool; vLLM's own sizing pass takes the "
                         "dense path and never sees the packed copy")
    ap.add_argument("--chunked-prefill-tokens", type=int, default=0,
                    help="chunk the prefill at this width, as the RULER "
                         "dense baseline does; 0 keeps it unchunked")
    ap.add_argument("--thinking", action="store_true",
                    help="leave the chat template's thinking channel open "
                         "(off by default, to match the RULER configs)")
    ap.add_argument("--preview-chars", type=int, default=400,
                    help="how much of each response to keep for inspection")
    ap.add_argument("--compilation-config", default=None,
                    help="JSON merged into compilation_config.  Needed for "
                         "settings with no flag of their own, such as "
                         "inductor_compile_config.")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    # Dense runs load no plugin, so install the check of vLLM's multimodal-RoPE YaRN here, before vLLM
    # builds the model (semblo.lm_eval_runner and semblo.cli get it on import).
    from semblo import mrope_yarn_fix
    mrope_yarn_fix.install()
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    # Qwen3.5's template ends the generation prompt with a bare "<think>"
    # unless enable_thinking is false, in which case it writes an empty
    # thought and closes it.
    preset = dict(SAMPLING_PRESETS[args.sampling_preset])
    if args.temperature is not None:
        preset["temperature"] = args.temperature
    # --thinking forces thinking mode. Otherwise the preset decides.
    thinking = bool(preset["thinking"] or args.thinking)
    print(f"sampling preset {args.sampling_preset}: "
          + ", ".join(f"{k}={v}" for k, v in preset.items() if k != "thinking")
          + f", thinking={thinking}")
    template_kwargs = {} if thinking else {"enable_thinking": False}
    ds = load_dataset("openai/mrcr", data_files=f"{args.needles}needle/*.parquet",
                      split="train")

    if args.official_bin is not None and args.qwen_bin is not None:
        raise SystemExit("choose one of --official-bin and --qwen-bin")

    official_encoder = None
    bin_lo = bin_hi = None
    edges = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    if args.qwen_bin is not None:
        bin_hi = args.qwen_bin
        bin_lo = edges[edges.index(bin_hi) - 1]
    if args.official_bin is not None:
        import tiktoken

        official_encoder = tiktoken.get_encoding("o200k_base")
        bin_hi = args.official_bin
        bin_lo = edges[edges.index(bin_hi) - 1]

    rows = []
    skipped_for_offset = 0
    for row in ds:
        messages = json.loads(row["prompt"])
        if official_encoder is not None:
            # Reproduce the dataset's own binning. The model then reads the
            # prompt through its own tokenizer, whose count differs.
            total = sum(len(official_encoder.encode(m["content"]))
                        for m in messages)
            total += len(official_encoder.encode(row["answer"]))
            if not (bin_lo < total <= bin_hi):
                continue
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            **template_kwargs,
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if args.qwen_bin is not None:
            if not (bin_lo < len(ids) <= bin_hi):
                continue
        elif official_encoder is None and not (
            args.min_tokens <= len(ids) < args.max_tokens
        ):
            continue
        if skipped_for_offset < args.offset:
            skipped_for_offset += 1
            continue
        rows.append({"ids": list(ids), "answer": row["answer"],
                     "prefix": row["random_string_to_prepend"],
                     "tokens": len(ids)})
        if len(rows) >= args.limit:
            break
    if bin_hi is not None:
        served = sorted(r["tokens"] for r in rows)
        cut = "tiktoken" if official_encoder is not None else "served tokens"
        print(f"bin ({bin_lo}, {bin_hi}] by {cut}: {len(rows)} samples; "
              f"served token count min {served[0]:,} median "
              f"{served[len(served) // 2]:,} max {served[-1]:,}")
        if official_encoder is not None and len(rows) != 100:
            print(f"WARNING: the bin should hold 100 samples, not {len(rows)};"
                  " the binning was not reproduced")
    if not rows:
        raise SystemExit("no samples in the requested token bin")
    lo = bin_lo if bin_hi is not None else args.min_tokens
    hi = bin_hi if bin_hi is not None else args.max_tokens
    print(f"{len(rows)} samples in ({lo},{hi}] "
          f"({args.needles} needles); median tokens "
          f"{sorted(r['tokens'] for r in rows)[len(rows)//2]}")

    overrides = {
        "semblo_vllm_collect_selection_stats": True,
        "semblo_max_probe_buckets": 4096,
        "semblo_bucket_size_penalty": 0.25,
    }
    if args.hf_overrides_json:
        overrides.update(json.loads(args.hf_overrides_json))

    engine = dict(
        model=args.pack,
        tokenizer=args.tokenizer_id,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=1,
        enable_prefix_caching=False,
        tensor_parallel_size=args.tensor_parallel_size,
        hf_overrides={"text_config": overrides},
        # --compilation-config reaches settings with no flag of their own. For
        # example, deterministic Inductor needs combo-kernel benchmarking off,
        # which vLLM turns on by default.
        compilation_config={"cudagraph_mode": "PIECEWISE",
                            **json.loads(args.compilation_config or "{}")},
    )
    if args.chunked_prefill_tokens:
        engine["enable_chunked_prefill"] = True
        engine["max_num_batched_tokens"] = args.chunked_prefill_tokens
    else:
        engine["enable_chunked_prefill"] = False
        engine["max_num_batched_tokens"] = args.max_model_len
    if args.kv_cache_memory_bytes:
        engine["kv_cache_memory_bytes"] = args.kv_cache_memory_bytes
    if args.tensor_parallel_size > 1:
        # vLLM's custom all-reduce fails during CUDA graph capture here
        # (custom_all_reduce.cuh: 'invalid argument'). NCCL's all-reduce works.
        engine["disable_custom_all_reduce"] = True
    llm = LLM(**engine)
    # The answer has to be reproduced verbatim, so its length is a floor on the
    # generation limit. The thinking budget is added on top.
    answer_tokens = [
        len(tokenizer(r["answer"], add_special_tokens=False).input_ids)
        for r in rows
    ]
    floor = max(answer_tokens) + (args.thinking_budget if thinking else 256)
    gen_tokens = max(args.gen_tokens, floor)
    print(f"generation limit {gen_tokens:,} tokens "
          f"(longest answer {max(answer_tokens):,} + "
          f"{'thinking budget ' if thinking else 'margin '}"
          f"{args.thinking_budget if thinking else 256:,})")

    longest_prompt = max(r["tokens"] for r in rows)
    if longest_prompt + gen_tokens > args.max_model_len:
        raise SystemExit(
            f"max_model_len {args.max_model_len:,} cannot hold the longest "
            f"prompt ({longest_prompt:,}) plus {gen_tokens:,} generated "
            "tokens.  Raise it rather than letting the long samples be "
            "dropped or truncated: dropping them removes the hardest ones "
            "and biases the bin."
        )

    prompts = [{"prompt_token_ids": r["ids"]} for r in rows]
    scores: list[float] = []
    details: list[dict] = []
    per_seed: list[float] = []
    for repeat in range(max(1, args.repeats)):
        seed = args.seed + repeat
        params = SamplingParams(
            temperature=float(preset["temperature"]),
            top_p=float(preset["top_p"]),
            top_k=int(preset["top_k"]),
            min_p=float(preset["min_p"]),
            presence_penalty=float(preset["presence_penalty"]),
            repetition_penalty=float(preset["repetition_penalty"]),
            max_tokens=gen_tokens,
            seed=None if float(preset["temperature"]) == 0.0 else seed,
        )
        outputs = llm.generate(prompts, params)
        round_scores = []
        for row, out in zip(rows, outputs):
            text = out.outputs[0].text
            visible = visible_answer(text)
            s = grade(text, row["answer"], row["prefix"])
            round_scores.append(s)
            n = args.preview_chars
            details.append({
                "seed": seed,
                "tokens": row["tokens"], "score": s,
                "score_lenient": grade_lenient(text, row["answer"], row["prefix"]),
                "prefix_ok": visible.startswith(row["prefix"]),
                "prefix_anywhere": row["prefix"] in text,
                "thought_present": "</think>" in text,
                "response_len": len(text),
                "finish_reason": out.outputs[0].finish_reason,
                "prefix": row["prefix"],
                "response_head": visible[:n],
                "answer_head": row["answer"][:n],
            })
        scores.extend(round_scores)
        per_seed.append(sum(round_scores) / len(round_scores))
        print(f"  seed {seed}: mean ratio {per_seed[-1]:.4f} over "
              f"{len(round_scores)} samples")
    mean = sum(scores) / len(scores)
    lenient = sum(d["score_lenient"] for d in details) / len(details)
    prefix_rate = sum(d["prefix_ok"] for d in details) / len(details)
    anywhere = sum(d["prefix_anywhere"] for d in details) / len(details)
    truncated = sum(d["finish_reason"] == "length" for d in details) / len(details)
    thought = sum(d["thought_present"] for d in details) / len(details)
    if truncated > 0.02:
        print(f"WARNING: {truncated:.0%} of responses hit the {gen_tokens}"
              "-token generation limit; the answer may be cut off, which the "
              "metric cannot distinguish from a wrong answer")
    print(f"  responses with a thinking channel: {thought:.0%}; "
          f"truncated at the token limit: {truncated:.0%}")
    print(f"MRCR {args.needles}needle mean ratio: {mean:.4f} "
          f"(lenient {lenient:.4f}, prefix-first {prefix_rate:.2f}, "
          f"prefix-anywhere {anywhere:.2f}, n={len(scores)})")
    Path(args.output_json).write_text(json.dumps(
        # With --qwen-bin or --official-bin, the bin edges select the samples
        # and min_tokens/max_tokens are unused.
        {"needles": args.needles,
         "bin": [bin_lo, bin_hi] if bin_hi is not None
                else [args.min_tokens, args.max_tokens],
         "bin_cut_by": ("tiktoken" if official_encoder is not None
                        else "served tokens" if bin_hi is not None else "served tokens"),
         "selection": ({"qwen_bin": args.qwen_bin} if args.qwen_bin is not None
                       else {"official_bin": args.official_bin}
                       if args.official_bin is not None
                       else {"token_range": [args.min_tokens, args.max_tokens]}),
         "offset": args.offset, "limit": args.limit,
         "n": len(scores), "n_prompts": len(rows),
         "sampling_preset": args.sampling_preset, "sampling": preset,
         "thinking_resolved": thinking, "seed": args.seed,
         "repeats": args.repeats, "per_seed_mean_ratio": per_seed,
         "mean_ratio": mean, "mean_ratio_lenient": lenient,
         "prefix_rate": prefix_rate, "prefix_anywhere_rate": anywhere,
         "thinking": thinking,
         "truncated_rate": truncated, "thought_rate": thought,
         "gen_tokens_effective": gen_tokens,
         "thinking_budget": args.thinking_budget,
         "longest_answer_tokens": max(answer_tokens),
         "overrides": overrides, "model": args.pack,
         # The plugin takes its settings from environment variables.
         "run_environment": _run_environment(),
         "tensor_parallel_size": args.tensor_parallel_size,
         "chunked_prefill_tokens": args.chunked_prefill_tokens,
         "tokenizer": args.tokenizer_id,
         "details": details}, indent=2))


if __name__ == "__main__":
    main()
