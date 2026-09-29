from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from semblo.backends import build_backend
from semblo.budget import resolve_context_budget
from semblo.config import GenerationConfig, ModelSpec, deep_merge, generation_overrides_from_profile, load_yaml, parse_scalar
from semblo.io import read_jsonl
from semblo.parallel import apply_vllm_cli_overrides, example_in_shard
from semblo.prompting import append_prompt_prefill, prepare_messages, truncate_prompt_render
from semblo.repro import collect_environment, seed_for_example, set_global_seed
from semblo.run_metadata import runtime_environment
from semblo.tasks import build_task
from semblo.tasks.base import Example
from semblo.writer import RunWriter, validate_output_directory


def _select_model(models_config: dict[str, Any], model_key: str) -> ModelSpec:
    models = models_config.get("models", models_config)
    if model_key not in models:
        raise KeyError(f"Unknown model key {model_key}. Available: {sorted(models)}")
    return ModelSpec.from_dict(model_key, models[model_key])


def _merge_generation(model_spec: ModelSpec, bench_cfg: dict[str, Any], args: argparse.Namespace) -> GenerationConfig:
    generation = asdict(model_spec.generation)
    generation = deep_merge(generation, bench_cfg.get("generation", {}))
    if model_spec.mode and model_spec.mode in model_spec.generation_profiles:
        generation = deep_merge(generation, generation_overrides_from_profile(model_spec.generation_profiles[model_spec.mode]))

    overrides: dict[str, Any] = {}
    for cli_name in [
        "max_new_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "seed",
    ]:
        value = getattr(args, cli_name)
        if value is not None:
            overrides[cli_name] = value
    if args.stop:
        overrides["stop"] = args.stop
    if args.stop_token_id:
        overrides["stop_token_ids"] = args.stop_token_id
    if args.ignore_eos:
        overrides["ignore_eos"] = True
    return GenerationConfig.from_dict(deep_merge(generation, overrides))


def _apply_model_overrides(model_spec: ModelSpec, args: argparse.Namespace) -> ModelSpec:
    updates: dict[str, Any] = {}

    if args.system_prompt is not None:
        updates["system_prompt"] = args.system_prompt
        updates["system_prompt_file"] = None
        updates["system_prompt_path"] = None
    if args.system_prompt_file is not None:
        updates["system_prompt_file"] = args.system_prompt_file
        updates["system_prompt"] = None
    if args.system_prompt_policy is not None:
        updates["system_prompt_policy"] = args.system_prompt_policy
    elif (args.system_prompt is not None or args.system_prompt_file is not None) and model_spec.system_prompt_policy in {"none", "never", "off"}:
        updates["system_prompt_policy"] = "prepend_if_absent"
    if args.no_system_prompt:
        updates["system_prompt"] = None
        updates["system_prompt_file"] = None
        updates["system_prompt_path"] = None
        updates["system_prompt_policy"] = "none"

    if args.max_model_len is not None:
        updates["max_model_len"] = args.max_model_len
    vllm_overrides = apply_vllm_cli_overrides(model_spec, args)
    if vllm_overrides != model_spec.vllm:
        updates["vllm"] = vllm_overrides

    chat_kwargs = dict(model_spec.chat_template_kwargs)
    if args.chat_template_kwargs_json:
        chat_kwargs = deep_merge(chat_kwargs, json.loads(args.chat_template_kwargs_json))
    for item in args.chat_template_kw:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--chat-template-kw must be KEY=VALUE, got {item!r}")
        chat_kwargs[key] = parse_scalar(value)
    if chat_kwargs != model_spec.chat_template_kwargs:
        updates["chat_template_kwargs"] = chat_kwargs

    if not updates:
        return model_spec
    return replace(model_spec, **updates)


def _record_example_id(row: dict[str, Any]) -> str | None:
    example_id = row.get("example_id")
    if example_id is not None:
        return str(example_id)
    record_id = row.get("record_id")
    if isinstance(record_id, str) and "::seed=" in record_id:
        return record_id.split("::seed=", 1)[0]
    return str(record_id) if record_id is not None else None


def _existing_run_state(out_dir: Path) -> tuple[set[str], dict[str, int], int]:
    examples_path = out_dir / "examples.jsonl"
    skipped_path = out_dir / "skipped.jsonl"
    out: set[str] = set()
    seen_examples: set[str] = set()
    bin_by_example: dict[str, str] = {}
    for path in [examples_path, skipped_path]:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            out.add(str(row.get("record_id") or row.get("example_id")))
            example_id = _record_example_id(row)
            if example_id is None:
                continue
            seen_examples.add(example_id)
            metadata = row.get("metadata")
            if isinstance(metadata, dict):
                bin_by_example.setdefault(example_id, str(metadata.get("bin", metadata.get("length_bin", "all"))))
            else:
                bin_by_example.setdefault(example_id, "all")
    counts_by_bin: dict[str, int] = {}
    for example_id in seen_examples:
        bin_key = bin_by_example.get(example_id, "all")
        counts_by_bin[bin_key] = counts_by_bin.get(bin_key, 0) + 1
    return out, counts_by_bin, len(seen_examples)


def _parse_metadata_filters(filters: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in filters:
        if "=" not in raw:
            raise ValueError(f"metadata filters must be KEY=VALUE, got {raw!r}")
        key, value = raw.split("=", 1)
        parsed[key] = [parse_scalar(part) for part in value.split("|")] if "|" in value else parse_scalar(value)
    return parsed


def _parse_bins(bin_values: list[str], bins_csv: list[str]) -> set[str]:
    bins: set[str] = set()
    for raw in [*bin_values, *bins_csv]:
        for part in str(raw).replace(",", " ").split():
            if part:
                bins.add(part)
    return bins


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (list, tuple, set)):
        return any(_value_matches(actual, item) for item in expected)
    return actual == expected or str(actual) == str(expected)


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any], bins: set[str]) -> bool:
    if bins and str(metadata.get("bin", metadata.get("length_bin", ""))) not in bins:
        return False
    for key, expected in filters.items():
        if not _value_matches(metadata.get(key), expected):
            return False
    return True


def _resolve_request_seed(base_seed: int, example_id: str, seed_policy: str) -> int:
    if seed_policy == "fixed":
        return int(base_seed)
    if seed_policy == "example_hash":
        return seed_for_example(int(base_seed), example_id)
    raise ValueError(f"Unknown seed policy: {seed_policy}")


def _example_done_for_all_seeds(example_id: str, base_seeds: list[int], seed_policy: str, existing: set[str]) -> bool:
    return all(
        f"{example_id}::seed={_resolve_request_seed(seed, example_id, seed_policy)}" in existing or example_id in existing
        for seed in base_seeds
    )


def _candidate_examples(
    task: Any,
    skip_existing: set[str],
    base_seeds: list[int],
    seed_policy: str,
    metadata_filters: dict[str, Any],
    bins: set[str],
    example_ids: set[str],
    start_index: int | None,
    end_index: int | None,
    num_shards: int,
    shard_id: int,
    shard_by: str,
) -> Iterable[Example]:
    for index, example in enumerate(task.iter_examples()):
        if start_index is not None and index < start_index:
            continue
        if end_index is not None and index >= end_index:
            break
        if example_ids and example.id not in example_ids:
            continue
        if not example_in_shard(example.id, index, num_shards=num_shards, shard_id=shard_id, shard_by=shard_by):
            continue
        if _example_done_for_all_seeds(example.id, base_seeds, seed_policy, skip_existing):
            continue
        if metadata_filters or bins:
            if not _metadata_matches(example.metadata, metadata_filters, bins):
                continue
        yield example


def _iter_limited(
    task: Any,
    limit: int | None,
    limit_per_bin: int | None,
    skip_existing: set[str],
    base_seeds: list[int],
    seed_policy: str,
    metadata_filters: dict[str, Any],
    bins: set[str],
    example_ids: set[str],
    start_index: int | None,
    end_index: int | None,
    num_shards: int,
    shard_id: int,
    shard_by: str,
    shuffle: bool,
    shuffle_seed: int,
    initial_counts_by_bin: dict[str, int] | None = None,
    initial_count: int = 0,
) -> Iterable[Example]:
    candidates: Iterable[Example] = _candidate_examples(
        task,
        skip_existing,
        base_seeds,
        seed_policy,
        metadata_filters,
        bins,
        example_ids,
        start_index,
        end_index,
        num_shards,
        shard_id,
        shard_by,
    )
    if shuffle:
        examples = list(candidates)
        random.Random(int(shuffle_seed)).shuffle(examples)
        candidates = examples

    counts_by_bin: dict[str, int] = dict(initial_counts_by_bin or {})
    yielded = int(initial_count)
    for example in candidates:
        if limit is not None and yielded >= limit:
            return
        bin_key = str(example.metadata.get("bin", example.metadata.get("length_bin", "all")))
        if limit_per_bin is not None:
            count = counts_by_bin.get(bin_key, 0)
            if count >= limit_per_bin:
                continue
            counts_by_bin[bin_key] = count + 1
        yield example
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def _run_base_seeds(generation: GenerationConfig, args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(seed) for seed in args.seeds]
    return [int(generation.seed)]


def _effective_config_payload(
    *,
    model_key: str,
    model_spec: ModelSpec,
    generation: GenerationConfig,
    benchmark_config_path: str,
    models_config_path: str,
    bench_cfg: dict[str, Any],
    args: argparse.Namespace,
    metadata_filters: dict[str, Any],
    bins: set[str],
    example_ids: set[str],
    base_seeds: list[int],
) -> dict[str, Any]:
    return {
        "model_key": model_key,
        "model_spec": asdict(model_spec),
        "generation": asdict(generation),
        "generation_profile": args.generation_profile,
        "benchmark_config_path": benchmark_config_path,
        "models_config_path": models_config_path,
        "benchmark_config": bench_cfg,
        "backend_override": args.backend,
        "metadata_filters": metadata_filters,
        "bins": sorted(bins),
        "example_ids": sorted(example_ids),
        "start_index": args.start_index,
        "end_index": args.end_index,
        "limit": args.limit,
        "limit_per_bin": args.limit_per_bin,
        "skip_if_prompt_tokens_over": args.skip_if_prompt_tokens_over,
        "skip_if_context_over_model_len": args.skip_if_context_over_model_len,
        "context_budget_policy": args.context_budget_policy,
        "context_budget_margin_tokens": args.context_budget_margin_tokens,
        "min_new_tokens_for_budget": args.min_new_tokens_for_budget,
        "prompt_truncation_policy": args.prompt_truncation_policy,
        "prompt_truncation_target_tokens": args.prompt_truncation_target_tokens,
        "assistant_prefill": args.assistant_prefill,
        "shuffle": args.shuffle,
        "shuffle_seed": args.shuffle_seed if args.shuffle_seed is not None else generation.seed,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "shard_by": args.shard_by,
        "base_seeds": base_seeds,
        "seed_policy": args.seed_policy,
        "per_example_max_new_tokens_metadata_key": bench_cfg.get("per_example_max_new_tokens_metadata_key"),
        "chat_template_kw": args.chat_template_kw,
        "chat_template_kwargs_json": args.chat_template_kwargs_json,
        "deterministic_torch": args.deterministic,
        "deterministic_vllm": args.deterministic_vllm,
        "vllm_effective_kwargs": model_spec.vllm,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run native long-context evaluations.")
    parser.add_argument("--benchmark-config", required=True)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--model", required=True, help="Model key in configs/models.yaml")
    parser.add_argument("--generation-profile", default=None, help="Named profile from model.generation_profiles")
    parser.add_argument("--backend", choices=["vllm"], default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--run-tag", default=None, help="Optional human-readable tag appended to the auto run directory.")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples before seed expansion.")
    parser.add_argument("--limit-per-bin", type=int, default=None, help="Limit examples per metadata bin before seed expansion.")
    parser.add_argument("--metadata-filter", action="append", default=[], help="Filter examples by metadata, e.g. bin=64000 or difficulty=hard|medium.")
    parser.add_argument("--bin", dest="bin_values", action="append", default=[], help="Shortcut for metadata bin filtering. Repeatable.")
    parser.add_argument("--bins", dest="bins_csv", action="append", default=[], help="Comma/space-separated bin list, e.g. 32768_65536,65536_131072.")
    parser.add_argument("--example-id", action="append", default=[], help="Run only specific example id. Repeatable.")
    parser.add_argument("--start-index", type=int, default=None, help="Skip examples before this zero-based dataset index.")
    parser.add_argument("--end-index", type=int, default=None, help="Stop before this zero-based dataset index.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle filtered examples before applying limit/limit-per-bin.")
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1, help="Split examples across independent evaluator processes. Use with --shard-id.")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards).")
    parser.add_argument("--shard-by", choices=["hash", "index"], default="hash", help="hash is stable under dataset order changes; index matches loader order modulo N.")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=None, help="Override vLLM tensor_parallel_size for model-parallel execution.")
    parser.add_argument("--pipeline-parallel-size", type=int, default=None, help="Override vLLM pipeline_parallel_size. Usually leave 1 on a single H200 node unless TP is unsuitable.")
    parser.add_argument("--decode-context-parallel-size", type=int, default=None, help="Override vLLM decode_context_parallel_size; must divide tensor_parallel_size.")
    parser.add_argument("--prefill-context-parallel-size", type=int, default=None, help="Override vLLM prefill_context_parallel_size for long-prefill experiments.")
    parser.add_argument("--data-parallel-size", type=int, default=None, help="Pass vLLM data_parallel_size. For offline evaluation throughput, prefer --num-shards with one process per GPU.")
    parser.add_argument("--distributed-executor-backend", choices=["mp", "ray"], default=None, help="Override vLLM distributed_executor_backend.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="Override vLLM gpu_memory_utilization.")
    parser.add_argument("--max-num-seqs", type=int, default=None, help="Override vLLM max_num_seqs.")
    parser.add_argument("--max-num-batched-tokens", default=None, help="Override vLLM max_num_batched_tokens. vLLM accepts ints or strings such as 128k.")
    parser.add_argument("--enable-chunked-prefill", action="store_true", help="Set vLLM enable_chunked_prefill=True.")
    parser.add_argument("--disable-chunked-prefill", action="store_true", help="Set vLLM enable_chunked_prefill=False.")
    parser.add_argument("--vllm-arg", action="append", default=[], help="Arbitrary vLLM EngineArgs override as KEY=VALUE. Values are parsed as bool/int/float/string.")
    parser.add_argument("--vllm-json-arg", action="append", default=[], help="Arbitrary vLLM EngineArgs override as KEY=JSON. Useful for dict/list values such as limit_mm_per_prompt.")
    parser.add_argument("--skip-if-prompt-tokens-over", type=int, default=None, help="Skip examples whose locally rendered prompt exceeds this token count.")
    parser.add_argument("--skip-if-context-over-model-len", action="store_true", help="Deprecated alias for the default --context-budget-policy skip behavior.")
    parser.add_argument("--context-budget-policy", choices=["skip", "error", "shrink_generation", "none"], default="skip", help="How to handle prompt_tokens + max_new_tokens + margin > max_model_len. Default: skip and record skipped.jsonl.")
    parser.add_argument("--context-budget-margin-tokens", type=int, default=0, help="Extra safety margin reserved inside model_spec.max_model_len before generation.")
    parser.add_argument("--min-new-tokens-for-budget", type=int, default=1, help="Minimum generation budget required when --context-budget-policy shrink_generation is used.")
    parser.add_argument(
        "--prompt-truncation-policy",
        choices=["none", "middle"],
        default="none",
        help="Optionally truncate rendered prompt token ids before generation. 'middle' keeps the beginning and end.",
    )
    parser.add_argument(
        "--prompt-truncation-target-tokens",
        type=int,
        default=None,
        help="Maximum prompt tokens after truncation. Default with middle truncation is max_model_len - max_new_tokens - margin.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--presence-penalty", type=float, default=None)
    parser.add_argument("--frequency-penalty", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", nargs="*", default=None, help="Run several base seeds per example, e.g. --seeds 0 1 2.")
    parser.add_argument("--seed-policy", choices=["fixed", "example_hash"], default="fixed", help="fixed passes the base seed as-is; example_hash derives a stable per-example seed.")
    parser.add_argument("--stop", action="append", default=[])
    parser.add_argument("--stop-token-id", action="append", type=int, default=[])
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--chat-template-kwargs-json", default=None, help="JSON object merged into model.chat_template_kwargs.")
    parser.add_argument("--chat-template-kw", action="append", default=[], help="Override chat template kwarg, e.g. enable_thinking=true.")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--system-prompt-file", default=None)
    parser.add_argument(
        "--system-prompt-policy",
        choices=[
            "none",
            "prepend_if_missing",
            "prepend_if_absent",
            "prepend",
            "replace",
            "replace_or_prepend",
            "append_to_existing",
            "append_to_first",
        ],
        default=None,
    )
    parser.add_argument("--no-system-prompt", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="Enable PyTorch deterministic algorithms where possible.")
    parser.add_argument("--deterministic-vllm", action="store_true", help="Set VLLM_ENABLE_V1_MULTIPROCESSING=0 before vLLM import.")
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--save-token-ids", action="store_true")
    parser.add_argument(
        "--assistant-prefill",
        default="",
        help=(
            "Append text to the rendered assistant generation prompt and prepend "
            "the same text to saved predictions. Diagnostic use only; changes "
            "the actual prompt tokens."
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-effective-config", action="store_true")
    return parser


def run_main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"--shard-id must satisfy 0 <= shard_id < num_shards, got shard_id={args.shard_id}, num_shards={args.num_shards}")
    if args.start_index is not None and args.start_index < 0:
        raise ValueError(f"--start-index must be >= 0, got {args.start_index}")
    if args.end_index is not None and args.end_index < 0:
        raise ValueError(f"--end-index must be >= 0, got {args.end_index}")
    if args.start_index is not None and args.end_index is not None and args.start_index > args.end_index:
        raise ValueError(f"--start-index must be <= --end-index, got {args.start_index} > {args.end_index}")
    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        raise ValueError("Use only one of --enable-chunked-prefill and --disable-chunked-prefill.")

    models_cfg = load_yaml(args.models_config)
    bench_cfg = load_yaml(args.benchmark_config)
    model_spec = _select_model(models_cfg, args.model)
    if args.generation_profile:
        model_spec = model_spec.with_profile(args.generation_profile)
    model_spec = _apply_model_overrides(model_spec, args)
    generation = _merge_generation(model_spec, bench_cfg, args)
    base_seeds = _run_base_seeds(generation, args)
    metadata_filters = _parse_metadata_filters(args.metadata_filter)
    bins = _parse_bins(args.bin_values, args.bins_csv)
    example_ids = {str(x) for x in args.example_id}
    shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else generation.seed

    if args.deterministic_vllm:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    set_global_seed(generation.seed, deterministic=args.deterministic)

    effective_config = _effective_config_payload(
        model_key=args.model,
        model_spec=model_spec,
        generation=generation,
        benchmark_config_path=args.benchmark_config,
        models_config_path=args.models_config,
        bench_cfg=bench_cfg,
        args=args,
        metadata_filters=metadata_filters,
        bins=bins,
        example_ids=example_ids,
        base_seeds=base_seeds,
    )
    if args.print_effective_config:
        print(json.dumps(effective_config, ensure_ascii=False, indent=2, sort_keys=True, default=str))

    task = build_task(bench_cfg)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{args.model}_{task.name}{tag}"
    out_dir = Path(args.out_dir or Path("runs") / run_name)
    existing: set[str] = set()
    existing_counts_by_bin: dict[str, int] = {}
    existing_count = 0
    if args.resume:
        existing, existing_counts_by_bin, existing_count = _existing_run_state(out_dir)
    examples = _iter_limited(
        task,
        args.limit,
        args.limit_per_bin,
        existing,
        base_seeds,
        args.seed_policy,
        metadata_filters,
        bins,
        example_ids,
        args.start_index,
        args.end_index,
        args.num_shards,
        args.shard_id,
        args.shard_by,
        args.shuffle,
        int(shuffle_seed),
        existing_counts_by_bin,
        existing_count,
    )

    if args.dry_run:
        for example in itertools.islice(examples, args.limit or 5):
            prepared_messages = prepare_messages(example.messages, model_spec)
            resolved_seeds = [_resolve_request_seed(seed, example.id, args.seed_policy) for seed in base_seeds]
            print(
                {
                    "id": example.id,
                    "task": example.task_name,
                    "metadata": example.metadata,
                    "answer_preview": str(example.answer)[:200],
                    "message_chars": len(str(prepared_messages)),
                    "first_message": prepared_messages[0] if prepared_messages else None,
                    "base_seeds": base_seeds,
                    "resolved_seeds": resolved_seeds,
                    "seed_policy": args.seed_policy,
                    "num_shards": args.num_shards,
                    "shard_id": args.shard_id,
                    "shard_by": args.shard_by,
                }
            )
        return

    effective_config["runtime_environment"] = {**runtime_environment(), **model_spec.env}
    effective_config["save_prompts"] = args.save_prompts
    effective_config["save_token_ids"] = args.save_token_ids
    validate_output_directory(out_dir, effective_config, resume=args.resume)
    backend = build_backend(model_spec, args.backend, engine_seed=generation.seed)
    writer = RunWriter(
        out_dir=out_dir,
        run_config={
            **effective_config,
            "resume": args.resume,
            "environment": collect_environment(),
        },
        model_spec=model_spec,
        generation=generation,
        save_prompts=args.save_prompts,
        save_token_ids=args.save_token_ids,
        backend_manifest=backend.get_run_manifest(),
        resume=args.resume,
    )

    for example in tqdm(examples, desc=f"{args.model}/{task.name}"):
        for base_seed in base_seeds:
            request_seed = _resolve_request_seed(base_seed, example.id, args.seed_policy)
            record_id = f"{example.id}::seed={request_seed}"
            if record_id in existing or example.id in existing:
                continue

            seeded_generation = replace(generation, seed=request_seed)
            per_example_max_key = bench_cfg.get("per_example_max_new_tokens_metadata_key")
            if per_example_max_key:
                per_example_max = example.metadata.get(str(per_example_max_key))
                if per_example_max is None:
                    raise KeyError(
                        f"Benchmark config requested per-example max_new_tokens from metadata key "
                        f"{per_example_max_key!r}, but example {example.id} has no such metadata."
                    )
                seeded_generation = replace(seeded_generation, max_new_tokens=int(per_example_max))
            set_global_seed(request_seed, deterministic=args.deterministic)

            prompt_render = backend.render_prompt(example.messages)
            if args.prompt_truncation_policy != "none":
                if args.prompt_truncation_target_tokens is None:
                    max_prompt_tokens = (
                        int(model_spec.max_model_len)
                        - int(seeded_generation.max_new_tokens)
                        - int(args.context_budget_margin_tokens)
                    )
                else:
                    max_prompt_tokens = int(args.prompt_truncation_target_tokens)
                tokenizer = getattr(backend, "tokenizer", None)
                if tokenizer is None:
                    raise ValueError("--prompt-truncation-policy requires a backend with a local tokenizer")
                prompt_render = truncate_prompt_render(
                    prompt_render,
                    tokenizer=tokenizer,
                    max_prompt_tokens=max_prompt_tokens,
                    policy=args.prompt_truncation_policy,
                )
            if args.assistant_prefill:
                tokenizer = getattr(backend, "tokenizer", None)
                if tokenizer is None:
                    raise ValueError("--assistant-prefill requires a backend with a local tokenizer")
                prompt_render = append_prompt_prefill(
                    prompt_render,
                    tokenizer=tokenizer,
                    prefill=args.assistant_prefill,
                )
            prompt_tokens = prompt_render.token_count
            if prompt_tokens is not None and args.skip_if_prompt_tokens_over is not None and prompt_tokens > args.skip_if_prompt_tokens_over:
                writer.write_skip(
                    example,
                    reason="prompt_tokens_over_limit",
                    model_key=args.model,
                    run_seed=request_seed,
                    prompt_render=prompt_render,
                    base_seed=base_seed,
                    seed_policy=args.seed_policy,
                    details={
                        "prompt_tokens": prompt_tokens,
                        "limit": args.skip_if_prompt_tokens_over,
                        "base_seed": base_seed,
                        "seed_policy": args.seed_policy,
                    },
                )
                existing.add(record_id)
                continue

            effective_generation, budget = resolve_context_budget(
                prompt_render=prompt_render,
                generation=seeded_generation,
                model_spec=model_spec,
                policy=args.context_budget_policy,
                margin_tokens=args.context_budget_margin_tokens,
                min_new_tokens=args.min_new_tokens_for_budget,
            )
            if budget.decision == "skip":
                writer.write_skip(
                    example,
                    reason=budget.reason or "context_budget_skip",
                    model_key=args.model,
                    run_seed=request_seed,
                    prompt_render=prompt_render,
                    base_seed=base_seed,
                    seed_policy=args.seed_policy,
                    details={
                        **budget.to_record(),
                        "base_seed": base_seed,
                        "seed_policy": args.seed_policy,
                    },
                )
                existing.add(record_id)
                continue

            generation_output = backend.generate(example.messages, effective_generation, prompt_render=prompt_render)
            if args.assistant_prefill:
                raw = dict(generation_output.raw or {})
                raw["assistant_prefill"] = args.assistant_prefill
                generation_output = replace(
                    generation_output,
                    text=args.assistant_prefill + generation_output.text,
                    raw=raw,
                )
            generation_output = replace(generation_output, context_budget=budget.to_record())
            metrics = task.score(generation_output.text, example)
            writer.write(
                example,
                generation_output,
                metrics,
                model_key=args.model,
                run_seed=request_seed,
                base_seed=base_seed,
                seed_policy=args.seed_policy,
            )
            existing.add(record_id)

    print(f"Wrote run to {out_dir}")


if __name__ == "__main__":
    run_main()
