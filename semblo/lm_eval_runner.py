from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

from semblo.config import ModelSpec, load_yaml
from semblo.parallel import apply_vllm_cli_overrides, validate_vllm_parallel_config, visible_cuda_device_count_from_env
from semblo.repro import apply_runtime_env


def _select_model(models_config: dict, model_key: str) -> ModelSpec:
    models = models_config.get("models", models_config)
    if model_key not in models:
        raise KeyError(f"Unknown model key {model_key}. Available: {sorted(models)}")
    return ModelSpec.from_dict(model_key, models[model_key])


def _format_arg_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _model_args_dict(model: ModelSpec, suite: dict) -> dict[str, Any]:
    args: dict[str, Any] = {
        "pretrained": model.hf_id,
        "tokenizer": model.tokenizer_id or model.hf_id,
        "trust_remote_code": model.trust_remote_code,
        "dtype": model.dtype,
        "max_model_len": model.max_model_len,
    }
    args.update(model.vllm)
    if model.hf_overrides:
        args.setdefault("hf_overrides", model.hf_overrides)
    args.setdefault("tensor_parallel_size", 1)
    args.setdefault("gpu_memory_utilization", 0.92)
    args.setdefault("max_num_seqs", 1)
    if suite.get("apply_chat_template") and model.chat_template_kwargs:
        args["chat_template_args"] = dict(model.chat_template_kwargs)
    suite_model_args = dict(suite.get("model_args", {}))
    if "chat_template_args" in suite_model_args and "chat_template_args" in args:
        merged_chat_args = dict(args["chat_template_args"])
        merged_chat_args.update(suite_model_args["chat_template_args"] or {})
        suite_model_args["chat_template_args"] = merged_chat_args
    args.update(suite_model_args)
    if "max_length" in suite_model_args and args.get("max_length") is not None:
        # lm-eval's vLLM wrapper asserts that max_length and max_model_len are not both set.
        # The RULER suites set max_length, so a suite-level max_length wins.
        args.pop("max_model_len", None)
    validate_vllm_parallel_config(args, available_cuda_devices=visible_cuda_device_count_from_env())
    return {k: v for k, v in args.items() if v is not None}


def _model_args_string(model_args: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in model_args.items():
        formatted = _format_arg_value(value)
        if "," in formatted:
            raise ValueError(
                "lm-eval CLI --model_args is comma-separated and cannot safely carry comma-containing values "
                f"such as {key}={formatted!r}. Use --runner python_api, which passes model_args as a dict."
            )
        parts.append(f"{key}={formatted}")
    return ",".join(parts)


def _tasks_arg(suite: dict) -> str:
    tasks = suite["tasks"]
    if isinstance(tasks, list):
        return ",".join(tasks)
    return str(tasks)


def _tasks_list(suite: dict) -> list[str]:
    tasks = suite["tasks"]
    if isinstance(tasks, list):
        return [str(task) for task in tasks]
    return [part for part in str(tasks).replace(",", " ").split() if part]


def _samples_dict(suite: dict) -> dict[str, list[int]] | None:
    """Turn suite["samples_range"] = "START:STOP" into lm-eval's samples argument.

    lm-eval's ``samples`` argument takes explicit document indices, and the documents
    are in a fixed order. So 0:100 gives the same items as ``--limit 100``, and
    disjoint ranges split one task across processes.
    """
    spec = suite.get("samples_range")
    if spec is None:
        return None
    start, _, stop = str(spec).partition(":")
    lo, hi = int(start), int(stop)
    if hi <= lo:
        raise ValueError(f"samples_range must be START:STOP with STOP > START, got {spec!r}")
    return {task: list(range(lo, hi)) for task in _tasks_list(suite)}


def _gen_kwargs(model: ModelSpec, suite: dict) -> dict[str, Any]:
    gen: dict[str, Any] = {
        "temperature": model.generation.temperature,
        "top_p": model.generation.top_p,
    }
    for key in ["top_k", "min_p", "repetition_penalty", "presence_penalty", "frequency_penalty"]:
        value = getattr(model.generation, key)
        if value is not None:
            gen[key] = value
    gen.update(suite.get("gen_kwargs", {}))
    return gen


def _template_kwargs(suite: dict) -> dict[str, Any]:
    return {
        "system_instruction": suite.get("system_instruction"),
        "apply_chat_template": suite.get("apply_chat_template", False),
        "fewshot_as_multiturn": suite.get("fewshot_as_multiturn", True),
    }


def _task_manager_metadata(model_args: dict[str, Any], suite: dict) -> dict[str, Any] | None:
    keys = suite.get("task_manager_model_arg_keys")
    if keys is None:
        return None
    metadata = {str(key): model_args[str(key)] for key in keys if str(key) in model_args}
    metadata.update(suite.get("metadata") or {})
    return metadata


def _include_path(suite: dict) -> str | list[str] | None:
    include_path = suite.get("include_path")
    if include_path is None:
        return None
    if isinstance(include_path, list):
        return [str(path) for path in include_path]
    return str(include_path)


def _include_defaults(suite: dict) -> bool:
    return bool(suite.get("include_defaults", True))


def _compact_logged_samples(results: dict[str, Any]) -> dict[str, Any]:
    """Drop repeated long prompts while retaining predictions, scores, and hashes."""
    samples = results.get("samples")
    if not isinstance(samples, dict):
        return results
    for task_samples in samples.values():
        if not isinstance(task_samples, list):
            continue
        for sample in task_samples:
            if not isinstance(sample, dict):
                continue
            doc = sample.get("doc")
            if isinstance(doc, dict) and "input" in doc:
                sample["doc"] = {key: value for key, value in doc.items() if key != "input"}
            arguments = sample.get("arguments")
            if isinstance(arguments, list):
                sample["arguments"] = [
                    ["<prompt omitted>", *argument[1:]]
                    if isinstance(argument, (list, tuple)) and argument
                    else argument
                    for argument in arguments
                ]
    return results



def _build_cli_command(model: ModelSpec, suite: dict, model_args: dict[str, Any], gen_kwargs: dict[str, Any], output_path: str | None) -> list[str]:
    cmd = [
        "lm_eval",
        "--model", suite.get("lm_eval_model", "vllm"),
        "--model_args", _model_args_string(model_args),
        "--tasks", _tasks_arg(suite),
        "--batch_size", str(suite.get("batch_size", 1)),
    ]
    include_path = _include_path(suite)
    if include_path is not None:
        if isinstance(include_path, list):
            cmd += ["--include_path", ",".join(include_path)]
        else:
            cmd += ["--include_path", include_path]
    if suite.get("num_fewshot") is not None:
        cmd += ["--num_fewshot", str(suite["num_fewshot"])]
    if suite.get("limit") is not None:
        cmd += ["--limit", str(suite["limit"])]
    if gen_kwargs:
        cmd += ["--gen_kwargs", ",".join(f"{k}={_format_arg_value(v)}" for k, v in gen_kwargs.items())]
    if suite.get("metadata"):
        cmd += ["--metadata", json.dumps(suite["metadata"], ensure_ascii=False)]
    seed = suite.get("seed", model.generation.seed)
    if seed is not None:
        cmd += ["--seed", str(seed)]
    if output_path:
        cmd += ["--output_path", output_path]
    if suite.get("log_samples", True):
        cmd += ["--log_samples"]
    return cmd


def _sparse_runtime_stats() -> dict[str, Any] | None:
    """Selection and index-build statistics of the sparse plugin for this run.

    The counters are process globals, so they are visible only when the engine runs
    in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0). Returns None for dense runs.
    """

    try:
        from semblo.vllm_plugin.model import sparse_attention_calls
        from semblo.vllm_plugin.state import INDEX_BUILD_STATS, SELECTION_STATS
    except Exception:
        return None
    calls = int(sparse_attention_calls())
    if calls == 0:
        return None
    payload: dict[str, Any] = {"sparse_attention_calls": calls}
    try:
        if getattr(SELECTION_STATS, "calls", 0):
            payload["selection_stats"] = SELECTION_STATS.summary()
        else:
            payload["selection_stats"] = None
    except Exception as error:  # stats are diagnostics; never fail the run
        payload["selection_stats_error"] = repr(error)
    payload["index_build_stats"] = list(INDEX_BUILD_STATS)
    return payload


def _write_run_environment(out: Path, model_key: str) -> None:
    """Write ``run_environment.json`` beside the results.

    ``results_python_api.json`` records model_args but not the environment, and the
    plugin takes its settings (``SEMBLO_*``) from environment variables. The file
    also records the git commit.
    """
    from semblo.run_metadata import collect_run_metadata

    record = {"model_key": model_key, **collect_run_metadata()}
    with (out / "run_environment.json").open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _run_python_api(model: ModelSpec, suite: dict, model_args: dict[str, Any], gen_kwargs: dict[str, Any], output_path: str | None) -> None:
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import handle_non_serializable, make_table

    seed = suite.get("seed", model.generation.seed)
    task_manager_metadata = _task_manager_metadata(model_args, suite)
    include_path = _include_path(suite)
    task_manager = (
        TaskManager(
            include_path=include_path,
            include_defaults=_include_defaults(suite),
            metadata=task_manager_metadata,
        )
        if task_manager_metadata is not None or include_path is not None
        else None
    )
    lm_eval_model = suite.get("lm_eval_model", "vllm")
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        model_args=model_args,
        tasks=_tasks_list(suite),
        num_fewshot=suite.get("num_fewshot"),
        batch_size=suite.get("batch_size", 1),
        limit=None if _samples_dict(suite) is not None else suite.get("limit"),
        samples=_samples_dict(suite),
        log_samples=suite.get("log_samples", True),
        gen_kwargs=gen_kwargs or None,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        task_manager=task_manager,
        metadata=None if task_manager is not None else suite.get("metadata"),
        **_template_kwargs(suite),
    )
    if results is None:
        return
    if suite.get("compact_log_samples", False):
        results = _compact_logged_samples(results)
    if output_path:
        out = Path(output_path)
        out.mkdir(parents=True, exist_ok=True)
        _write_run_environment(out, model.key)
        with (out / "results_python_api.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, sort_keys=True, default=handle_non_serializable)
            f.write("\n")
        sparse_stats = _sparse_runtime_stats()
        if sparse_stats is not None:
            sparse_stats["model_key"] = model.key
            with (out / "sparse_runtime_stats.json").open("w", encoding="utf-8") as f:
                json.dump(sparse_stats, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
    print(make_table(results))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run lm-evaluation-harness suites with configs/models.yaml settings.")
    parser.add_argument("--suite", required=True)
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--generation-profile", default=None)
    parser.add_argument("--runner", choices=["python_api", "cli"], default="python_api", help="python_api passes nested model_args dicts safely; cli prints/runs the classic lm_eval command.")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--decode-context-parallel-size", type=int, default=None)
    parser.add_argument("--prefill-context-parallel-size", type=int, default=None)
    parser.add_argument("--data-parallel-size", type=int, default=None)
    parser.add_argument("--distributed-executor-backend", choices=["mp", "ray"], default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", default=None)
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--vllm-arg", action="append", default=[])
    parser.add_argument("--vllm-json-arg", action="append", default=[])
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--task", action="append", default=[], help="Override suite tasks; repeat for multiple tasks.")
    parser.add_argument("--limit", type=float, default=None, help="Override suite sample limit for a preflight.")
    parser.add_argument("--samples-range", default=None, help="Evaluate document indices START:STOP instead of the first --limit of them, so one task can be split across GPUs. 0:100 is the same items as --limit 100.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    suite = load_yaml(args.suite)
    if args.task:
        suite["tasks"] = list(args.task)
    if args.limit is not None:
        suite["limit"] = args.limit
    if args.samples_range is not None:
        suite["samples_range"] = args.samples_range
        suite.pop("limit", None)
    model = _select_model(load_yaml(args.models_config), args.model)
    if args.generation_profile:
        model = model.with_profile(args.generation_profile)
    if args.enable_chunked_prefill and args.disable_chunked_prefill:
        raise ValueError("Use only one of --enable-chunked-prefill and --disable-chunked-prefill.")
    vllm_overrides = apply_vllm_cli_overrides(model, args)
    if vllm_overrides != model.vllm:
        model = replace(model, vllm=vllm_overrides)
    apply_runtime_env(model)

    model_args = _model_args_dict(model, suite)
    gen_kwargs = _gen_kwargs(model, suite)
    output_path = args.output_path or suite.get("output_path")

    if args.runner == "cli":
        cmd = _build_cli_command(model, suite, model_args, gen_kwargs, output_path)
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        return

    payload = {
        "runner": "python_api",
        "model": suite.get("lm_eval_model", "vllm"),
        "model_args": model_args,
        "tasks": _tasks_list(suite),
        "batch_size": suite.get("batch_size", 1),
        "num_fewshot": suite.get("num_fewshot"),
        "limit": suite.get("limit"),
        "samples_range": suite.get("samples_range"),
        "gen_kwargs": gen_kwargs,
        "seed": suite.get("seed", model.generation.seed),
        "metadata": suite.get("metadata"),
        "task_manager_metadata": _task_manager_metadata(model_args, suite),
        "include_path": _include_path(suite),
        "include_defaults": _include_defaults(suite),
        "template_kwargs": _template_kwargs(suite),
        "output_path": output_path,
        "log_samples": suite.get("log_samples", True),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    if not args.dry_run:
        _run_python_api(model, suite, model_args, gen_kwargs, output_path)


if __name__ == "__main__":
    main()
