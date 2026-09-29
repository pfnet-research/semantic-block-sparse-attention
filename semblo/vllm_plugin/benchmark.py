from __future__ import annotations

"""End-to-end vLLM latency measurement for official, custom-dense, and sparse models."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Iterable
from semblo.vllm_plugin.graph_mode import check_cudagraph_mode, resolved_cudagraph_mode
from semblo.run_metadata import redact_sensitive_fields, runtime_environment


# vLLM 0.19.1 attaches ``RequestStateStats`` to every ``RequestOutput``.  Its
# ``arrival_time`` is a frontend wall clock while the ``*_ts`` fields are engine
# core monotonic timestamps, so the two clocks must never be subtracted from
# each other.  vLLM measures TTFT itself and stores it as ``first_token_latency``.
def _metric_dict(metrics: Any) -> dict[str, float | int | None]:
    if metrics is None:
        return {}
    names = (
        "arrival_time",
        "queued_ts",
        "scheduled_ts",
        "first_token_ts",
        "last_token_ts",
        "first_token_latency",
        "num_generation_tokens",
    )
    return {name: getattr(metrics, name, None) for name in names}


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def request_timing(metrics: Any, output_tokens: int) -> dict[str, float | None]:
    """Compute standard latency terms from vLLM request timestamps.

    The interval definitions mirror what vLLM reports for a finished request:
    queue is first queued to first scheduled, prefill is first scheduled to the
    first new token, and decode is the first new token to the last one.
    ``tpot_ms`` excludes the first generated token, so it is defined only when at
    least two output tokens were produced.
    """

    raw = _metric_dict(metrics)
    queued = _finite_float(raw.get("queued_ts"))
    scheduled = _finite_float(raw.get("scheduled_ts"))
    first_token = _finite_float(raw.get("first_token_ts"))
    last_token = _finite_float(raw.get("last_token_ts"))
    first_token_latency = _finite_float(raw.get("first_token_latency"))

    def delta_ms(end: float | None, start: float | None) -> float | None:
        if end is None or start is None or end < start:
            return None
        return (end - start) * 1000.0

    decode_interval_ms = delta_ms(last_token, first_token)
    return {
        "ttft_ms": (
            first_token_latency * 1000.0 if first_token_latency is not None else None
        ),
        "queue_ms": delta_ms(scheduled, queued),
        "prefill_ms": delta_ms(first_token, scheduled),
        "inference_ms": delta_ms(last_token, scheduled),
        "decode_interval_ms": decode_interval_ms,
        "tpot_ms": (
            decode_interval_ms / (output_tokens - 1)
            if decode_interval_ms is not None and output_tokens > 1
            else None
        ),
    }


def _median_non_null(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def _build_prompt_tokens(tokenizer: Any, target_tokens: int, seed_text: str) -> list[int]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise ValueError("seed text tokenized to an empty sequence")
    repeats = (target_tokens + len(seed_ids) - 1) // len(seed_ids)
    ids = (seed_ids * repeats)[:target_tokens]
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        ids[0] = int(bos)
    return [int(token_id) for token_id in ids]


def _scoped_overrides(model: str, overrides: dict) -> dict:
    """Place ``--hf-text-override`` entries where the config reads them.

    Multimodal configs with a ``text_config`` (Qwen3.5) take them there, and
    text-only configs (Nemotron-H) at the top level. Adding a ``text_config``
    to the latter would make the runtime read its settings from a dict that
    holds only the overrides.
    """
    import os
    path = os.path.join(model, "config.json")
    try:
        with open(path) as fh:
            has_text = "text_config" in json.load(fh)
    except OSError:
        has_text = True          # a hub id: assume a multimodal config
    return {"text_config": overrides} if has_text else dict(overrides)


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    if not torch.cuda.is_available():
        raise RuntimeError("the vLLM benchmark requires CUDA")
    major, _minor = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError("the current production benchmark targets H100/H200 (SM90)")

    def _sparse_calls() -> int:
        try:
            from semblo.vllm_plugin.model import sparse_attention_calls
        except Exception:  # pragma: no cover - dense-only builds
            return -1
        return sparse_attention_calls()

    tokenizer_path = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False)
    if getattr(args, "prompt_ids_npy", None):
        # Real-content prompt. Block sizes, and therefore selection cost,
        # depend on the data, and a repeated synthetic prompt gives very
        # skewed blocks. The file holds one token-id sequence.
        import numpy as _np

        loaded = _np.load(args.prompt_ids_npy).tolist()
        # --prompt-offset selects another window of the same document, so a
        # timing can be repeated on different content.
        start = int(args.prompt_offset) * args.prompt_tokens
        if len(loaded) < start + args.prompt_tokens:
            raise ValueError(
                f"{args.prompt_ids_npy} has {len(loaded)} tokens; "
                f"--prompt-tokens {args.prompt_tokens} at offset "
                f"{args.prompt_offset} needs {start + args.prompt_tokens}"
            )
        prompt_ids = [int(t) for t in loaded[start: start + args.prompt_tokens]]
    else:
        prompt_ids = _build_prompt_tokens(tokenizer, args.prompt_tokens, args.seed_text)
    # Concurrent requests get distinct prompts that differ only in the first
    # token, so every sequence has the same length and the same work.
    prompts = [TokensPrompt(prompt_token_ids=prompt_ids)]
    for extra in range(1, args.num_sequences):
        varied = list(prompt_ids)
        varied[0] = prompt_ids[extra % len(prompt_ids)]
        prompts.append(TokensPrompt(prompt_token_ids=varied))
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.output_tokens,
        ignore_eos=True,
        detokenize=False,
        seed=args.seed,
    )

    text_overrides: dict[str, Any] = {}
    for entry in args.hf_text_override:
        key, _, raw = entry.partition("=")
        if not key or not raw:
            raise ValueError(f"--hf-text-override wants KEY=VALUE, got {entry!r}")
        if raw[:1] in "[{" or raw in {"true", "false", "null"}:
            # JSON values: lists, dicts, booleans, null.
            text_overrides[key] = json.loads(raw)
            continue
        try:
            text_overrides[key] = int(raw)
        except ValueError:
            try:
                text_overrides[key] = float(raw)
            except ValueError:
                text_overrides[key] = raw

    engine_kwargs = dict(
        model=args.model,
        tokenizer=tokenizer_path,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=(
            args.max_num_batched_tokens or args.max_model_len
        ),
        max_num_seqs=args.num_sequences,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_prefix_caching=False,
        enforce_eager=not args.cuda_graphs,
        **(
            {"limit_mm_per_prompt": json.loads(args.limit_mm_per_prompt)}
            if args.limit_mm_per_prompt
            else {}
        ),
        **(
            {
                "compilation_config": {
                    "cudagraph_mode": args.cudagraph_mode,
                    **(
                        {"cudagraph_copy_inputs": True}
                        if args.cudagraph_copy_inputs
                        else {}
                    ),
                    # Caller-supplied keys win. This is the only way to set
                    # inductor_compile_config options such as combo_kernels,
                    # which vLLM enables unless the key is already present.
                    **json.loads(args.compilation_config or "{}"),
                }
            }
            if args.cudagraph_mode or args.compilation_config
            else {}
        ),
        gpu_memory_utilization=args.gpu_memory_utilization,
        **(
            {"kv_cache_memory_bytes": args.kv_cache_memory_bytes}
            if args.kv_cache_memory_bytes
            else {}
        ),
        trust_remote_code=False,
        disable_log_stats=False,
        seed=args.seed,
        **(
            {"hf_overrides": _scoped_overrides(args.model, text_overrides)}
            if text_overrides
            else {}
        ),
    )
    if args.profile_dir:
        engine_kwargs["profiler_config"] = {
            "profiler": "torch", "torch_profiler_dir": os.path.abspath(args.profile_dir),
            "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False}
    llm = LLM(**engine_kwargs)
    graph_mode = resolved_cudagraph_mode(llm)
    print(check_cudagraph_mode(args.cudagraph_mode, graph_mode, args.require_cudagraph_mode), flush=True)
    # Memory held before any prompt: the weights and the KV pool.
    torch.cuda.synchronize()
    engine_resident = int(torch.cuda.memory_allocated())

    for _ in range(args.warmup):
        _ = llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    # Memory held after a completed request: the engine plus per-prompt state
    # (centroids, block index). The gap to the peak below is the transient
    # memory of the forward pass.
    prompt_resident = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    # Optional allocator snapshot. Recording costs a stack trace per allocation.
    snapshot_path = os.environ.get("SEMBLO_MEMORY_SNAPSHOT")
    if snapshot_path:
        torch.cuda.memory._record_memory_history(
            max_entries=int(os.environ.get("SEMBLO_MEMORY_SNAPSHOT_ENTRIES",
                                           "400000")))

    rows: list[dict[str, Any]] = []
    reference_tokens: dict[int, list[int]] = {}
    token_mismatches = 0
    for repetition in range(args.repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        if args.profile_dir:
            llm.start_profile(f"rep{repetition}")
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        if args.profile_dir:
            llm.stop_profile()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000.0
        if len(outputs) != len(prompts):
            raise RuntimeError("unexpected vLLM output cardinality")
        for sequence, output in enumerate(outputs):
            if len(output.outputs) != 1:
                raise RuntimeError("unexpected vLLM completion cardinality")
            token_ids = [int(value) for value in output.outputs[0].token_ids]
            if len(token_ids) != args.output_tokens:
                raise RuntimeError(
                    f"expected {args.output_tokens} output tokens, got {len(token_ids)}"
                )
            # Each sequence is compared against its own first run.
            if sequence not in reference_tokens:
                reference_tokens[sequence] = token_ids
            elif token_ids != reference_tokens[sequence]:
                if args.allow_token_mismatch:
                    token_mismatches += 1
                    diff_at = next(i for i, (a, b) in enumerate(
                        zip(token_ids, reference_tokens[sequence])) if a != b)
                    print(f"WARNING: repeat {repetition} seq {sequence} "
                          f"tokens diverge from repeat 0 at position "
                          f"{diff_at}", flush=True)
                else:
                    raise RuntimeError(
                        "deterministic benchmark produced different tokens "
                        "across repeats"
                    )
            timing = request_timing(output.metrics, len(token_ids))
            if timing["ttft_ms"] is None:
                raise RuntimeError(
                    "vLLM did not expose first_token_latency; TTFT cannot be measured"
                )
            if len(token_ids) > 1 and timing["tpot_ms"] is None:
                raise RuntimeError(
                    "vLLM did not expose first_token_ts/last_token_ts; "
                    "time per output token cannot be measured"
                )
            rows.append(
                {
                    "repetition": repetition,
                    "sequence": sequence,
                    # One wall time covers the whole batch.
                    "wall_e2e_ms": wall_ms,
                    **timing,
                    "output_token_ids": token_ids,
                    "output_token_ids_sha256": hashlib.sha256(
                        json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
                    ).hexdigest(),
                    "request_metrics_raw": _metric_dict(output.metrics),
                }
            )

    if snapshot_path:
        torch.cuda.memory._dump_snapshot(snapshot_path)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"memory snapshot written to {snapshot_path}")

    # The runtime is configured through environment variables, so record them.
    environment = runtime_environment()

    result = {
        "schema_version": 3,
        "environment": environment,
        "model": str(Path(args.model).expanduser().resolve()),
        "tokenizer": str(tokenizer_path),
        "prompt_tokens": len(prompt_ids),
        "requested_output_tokens": args.output_tokens,
        "actual_output_tokens": len(reference_tokens.get(0, ())),
        "num_sequences": args.num_sequences,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "token_mismatches": token_mismatches,
        "wall_e2e_median_ms": _median_non_null(rows, "wall_e2e_ms"),
        "wall_e2e_mean_ms": statistics.fmean(
            float(row["wall_e2e_ms"]) for row in rows
        ),
        "ttft_median_ms": _median_non_null(rows, "ttft_ms"),
        "tpot_median_ms": _median_non_null(rows, "tpot_ms"),
        "prefill_median_ms": _median_non_null(rows, "prefill_ms"),
        "inference_median_ms": _median_non_null(rows, "inference_ms"),
        "decode_interval_median_ms": _median_non_null(rows, "decode_interval_ms"),
        # Concurrent sequences step together, so the time per output token is
        # also the time per batch step.
        "decode_tokens_per_second": (
            args.num_sequences * 1000.0 / _median_non_null(rows, "tpot_ms")
            if _median_non_null(rows, "tpot_ms")
            else None
        ),
        # Output tokens do not distinguish sparse from dense, so the sparse
        # calls are counted. Zero means the run was dense.
        "sparse_attention_calls": _sparse_calls(),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        # Breakdown of the peak, taken at the points marked above.
        "engine_resident_bytes": engine_resident,
        "prompt_resident_bytes": prompt_resident,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        # Driver view, which includes memory outside PyTorch's allocator
        # (FA3 workspaces, CUDA graph pools).
        "device_free_bytes": int(torch.cuda.mem_get_info()[0]),
        "device_total_bytes": int(torch.cuda.mem_get_info()[1]),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "engine_kwargs": redact_sensitive_fields(engine_kwargs),
        "cudagraph_mode_requested": args.cudagraph_mode,
        "cudagraph_mode_resolved": graph_mode,
        "runs": rows,
        "measurement_notes": {
            "ttft_ms": "vLLM first_token_latency, measured from request arrival",
            "prefill_ms": "vLLM first new token minus first scheduled",
            "tpot_ms": (
                "vLLM last new token minus first new token, divided by output_tokens-1"
            ),
            "wall_e2e_ms": "wall time around one blocking LLM.generate call",
            "sparse_attention_calls": (
                "product-routed attention calls; zero means the run was dense"
            ),
        },
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--prompt-tokens", type=int, default=32768)
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=1,
        help=(
            "How many requests to run concurrently.  The sparse path refuses "
            "more than one; it is here to measure how the dense baseline's "
            "decode grows with the batch, which is what batching would be "
            "worth."
        ),
    )
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument(
        "--allow-token-mismatch", action="store_true",
        help="warn (and count in the output json) instead of raising when "
        "repeats produce different tokens; for timing-only diagnostics",
    )
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile-dir", default="",
                        help="build the engine with vLLM's torch profiler and wrap each "
                             "timed generate in start/stop_profile (to see whether the "
                             "profiler changes the wall clock)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--kv-cache-memory-bytes", type=int, default=0,
        help="pin the KV pool to this many bytes instead of letting vLLM size "
             "it from a profiling pass.  That estimate is dominated by a "
             "transient the served request never coexists with, and it cannot "
             "see the packed prompt K/V this plugin allocates per request, so "
             "at very long contexts there may be no utilisation fraction that "
             "leaves room for both.  0 keeps vLLM's estimate.")
    parser.add_argument(
        "--require-cudagraph-mode",
        default=None,
        help=("Fail unless the engine resolves exactly this cudagraph mode (e.g. "
              "FULL_DECODE_ONLY).  vLLM may quietly weaken the requested mode for a "
              "model; a clock taken under the wrong mode must not be recorded as the "
              "requested one."),
    )
    parser.add_argument(
        "--cudagraph-mode",
        default=None,
        help=(
            "Force a capture mode, e.g. PIECEWISE.  PIECEWISE leaves attention "
            "eager and captures the rest, which is what a custom-op sparse "
            "attention would get; FULL_AND_PIECEWISE is vLLM's default."
        ),
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help=(
            "Enable CUDA graphs. Sparse runs support PIECEWISE, or "
            "FULL_DECODE_ONLY with SEMBLO_FULL_GRAPH_DECODE=1."
        ),
    )
    parser.add_argument(
        "--compilation-config",
        default=None,
        help="JSON merged into compilation_config.  Needed for settings with no "
             "environment variable, such as inductor_compile_config's "
             "combo_kernels.",
    )
    parser.add_argument(
        "--hf-text-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override one entry of the checkpoint's text config, e.g. "
            "semblo_vllm_packed_decode_num_splits=1.  For sweeping a "
            "runtime knob without writing a new checkpoint."
        ),
    )
    parser.add_argument(
        "--cudagraph-copy-inputs",
        action="store_true",
        help=(
            "Copy inputs into stable buffers for piecewise CUDA graph replay."
        ),
    )
    parser.add_argument(
        "--expect-tokens-from",
        metavar="RESULT_JSON",
        help=(
            "Compare generated tokens against an earlier result and fail "
            "if run counts, token counts or token values differ."
        ),
    )
    parser.add_argument(
        "--limit-mm-per-prompt",
        default=None,
        help=(
            "JSON dict passed to vLLM's limit_mm_per_prompt.  Text-only "
            "benchmarks of multimodal architectures (Qwen3.5) need "
            '\'{"image":0,"video":0}\': the vision-encoder profiling pass '
            "at long max_model_len otherwise crashes inside FlashAttention."
        ),
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    # Chunked prefill is for the dense baseline only. The sparse path does its
    # own chunking and rejects vLLM's chunked prefill.
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=0,
                        help="0 keeps the default (= max_model_len)")
    parser.add_argument(
        "--disable-custom-all-reduce", action="store_true",
        help="vLLM's custom all-reduce rejects the message sizes a 1M-token "
        "forward produces at TP>=2 ('invalid argument'); NCCL handles them.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--prompt-ids-npy",
        default=None,
        help="npy file with one token-id sequence; the first --prompt-tokens "
        "ids become the prompt (real-content latency; the synthetic repeated "
        "seed measures a bucket-skew-heavy regime only)",
    )
    parser.add_argument(
        "--prompt-offset", type=int, default=0,
        help="take window number N of --prompt-tokens instead of the first, "
        "so one timing can be repeated on different content",
    )
    parser.add_argument(
        "--seed-text",
        default=(
            "This is a deterministic long-context systems benchmark. "
            "The repeated text is not used for quality claims. "
        ),
    )
    parser.add_argument("--output-json", required=True)
    return parser


def report_token_mismatch(result: dict[str, Any], reference_path: Path) -> None:
    """Fail if the run generated different tokens from a reference run.

    Run and token counts must match before token values are compared.
    For equal-length outputs, the message reports how many tokens differ
    and the first position.
    """

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    actual_runs = result["runs"]
    expected_runs = reference["runs"]
    if len(actual_runs) != len(expected_runs):
        raise SystemExit(
            f"run count differs from {reference_path}: "
            f"got {len(actual_runs)}, expected {len(expected_runs)}."
        )
    for run_index, (run, expected) in enumerate(
        zip(actual_runs, expected_runs, strict=True)
    ):
        actual_ids = list(run["output_token_ids"])
        expected_ids = list(expected["output_token_ids"])
        if len(actual_ids) != len(expected_ids):
            raise SystemExit(
                f"generated token count differs from {reference_path} in run {run_index}: "
                f"got {len(actual_ids)}, expected {len(expected_ids)}."
            )
        if actual_ids == expected_ids:
            continue
        differing = sum(1 for a, b in zip(actual_ids, expected_ids) if a != b)
        first = next(
            index
            for index, (a, b) in enumerate(zip(actual_ids, expected_ids))
            if a != b
        )
        raise SystemExit(
            f"generated tokens differ from {reference_path}: "
            f"{differing} of {len(expected_ids)} tokens, first at position "
            f"{first}."
        )


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.prompt_tokens > args.max_model_len:
        raise ValueError("prompt_tokens must not exceed max_model_len")
    if args.output_tokens < 1:
        raise ValueError("output_tokens must be positive")
    result = run_case(args)
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    if args.expect_tokens_from:
        report_token_mismatch(result, Path(args.expect_tokens_from))


if __name__ == "__main__":
    main()
