from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from semblo.backends.base import GenerationOutput
from semblo.config import GenerationConfig, ModelSpec
from semblo.io import append_jsonl, sha256_json, write_json
from semblo.prompting import PromptRender
from semblo.run_metadata import redact_sensitive_fields
from semblo.tasks.base import Example


def _run_settings(config: dict[str, Any]) -> dict[str, Any]:
    incidental = {"created_unix", "resume", "environment", "backend_manifest"}
    return {key: value for key, value in config.items() if key not in incidental}


def validate_output_directory(
    out_dir: str | Path, config: dict[str, Any], *, resume: bool = False,
) -> None:
    """Reject accidental reuse before loading a model or writing any files."""
    out = Path(out_dir)
    manifest = out / "run_config.json"
    has_results = any((out / name).exists() for name in ("examples.jsonl", "skipped.jsonl", "artifacts"))
    if not manifest.exists() and not has_results:
        return
    if not resume:
        raise FileExistsError(f"{out} already contains a run; use a new directory or --resume")
    if not manifest.is_file():
        raise ValueError(f"cannot resume {out}: run_config.json is missing")
    previous = json.loads(manifest.read_text(encoding="utf-8"))
    expected = _run_settings(redact_sensitive_fields(config))
    recorded = _run_settings(previous)
    changed = sorted(key for key in expected.keys() | recorded.keys()
                     if key not in expected or key not in recorded or expected[key] != recorded[key])
    if changed:
        raise ValueError(f"cannot resume {out}: settings differ ({', '.join(changed)}); use a new directory")


class RunWriter:
    def __init__(
        self,
        out_dir: str | Path,
        run_config: dict[str, Any],
        model_spec: ModelSpec,
        generation: GenerationConfig,
        save_prompts: bool = False,
        save_token_ids: bool = False,
        backend_manifest: dict[str, Any] | None = None,
        resume: bool = False,
    ):
        self.out_dir = Path(out_dir)
        self.artifact_dir = self.out_dir / "artifacts"
        self.save_prompts = save_prompts
        self.save_token_ids = save_token_ids
        full_config = dict(run_config)
        full_config["model_spec"] = asdict(model_spec)
        full_config["generation"] = asdict(generation)
        full_config["backend_manifest"] = backend_manifest or {}
        full_config = redact_sensitive_fields(full_config)
        validate_output_directory(self.out_dir, full_config, resume=resume)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if save_prompts or save_token_ids:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        full_config["created_unix"] = time.time()
        if not (self.out_dir / "run_config.json").exists():
            write_json(self.out_dir / "run_config.json", full_config)
        self.examples_path = self.out_dir / "examples.jsonl"
        self.skipped_path = self.out_dir / "skipped.jsonl"

    def write(
        self,
        example: Example,
        generation_output: GenerationOutput,
        metrics: dict[str, Any],
        model_key: str,
        run_seed: int | None = None,
        base_seed: int | None = None,
        seed_policy: str | None = None,
    ) -> None:
        prompt_render = generation_output.prompt_render
        raw_messages_sha = sha256_json(example.messages)
        prepared_sha = prompt_render.request_sha256 if prompt_render else raw_messages_sha
        rendered_sha = prompt_render.rendered_prompt_sha256 if prompt_render else None
        record_id = f"{example.id}::seed={run_seed}" if run_seed is not None else example.id
        completion_token_ids = generation_output.completion_token_ids
        completion_tokens = generation_output.completion_tokens
        tokens_per_second = None
        if completion_tokens is not None and generation_output.latency_s and generation_output.latency_s > 0:
            tokens_per_second = float(completion_tokens) / float(generation_output.latency_s)

        record = {
            "record_id": record_id,
            "example_id": example.id,
            "task_name": example.task_name,
            "model_key": model_key,
            "base_seed": base_seed,
            "seed_policy": seed_policy,
            "generation_seed": run_seed,
            "score": float(metrics.get("score", 0.0)),
            "metrics": metrics,
            "prediction": generation_output.text,
            "answer": example.answer,
            "metadata": example.metadata,
            "prompt_raw_messages_sha256": raw_messages_sha,
            "prompt_messages_sha256": prepared_sha,
            "rendered_prompt_sha256": rendered_sha,
            "prompt_message_chars": len(str(example.messages)),
            "rendered_prompt_chars": prompt_render.rendered_prompt_chars if prompt_render else None,
            "prompt_roles": prompt_render.message_roles if prompt_render else None,
            "prompt_chat_template_kwargs": prompt_render.chat_template_kwargs if prompt_render else None,
            "system_prompt_applied": prompt_render.system_prompt_applied if prompt_render else None,
            "prompt_renderer": prompt_render.renderer if prompt_render else None,
            "prompt_tokens": generation_output.prompt_tokens,
            "prompt_tokens_estimate": prompt_render.prompt_tokens_estimate if prompt_render else None,
            "completion_tokens": completion_tokens,
            "latency_s": generation_output.latency_s,
            "tokens_per_second": tokens_per_second,
            "sampling_params": generation_output.sampling_params,
            "generation_config": generation_output.sampling_params,
            "context_budget": generation_output.context_budget,
            "backend_raw": generation_output.raw,
            "prompt_token_ids_sha256": sha256_json(prompt_render.token_ids) if prompt_render and prompt_render.token_ids is not None else None,
            "completion_token_ids_sha256": sha256_json(completion_token_ids) if completion_token_ids is not None else None,
        }
        if completion_token_ids is not None and self.save_token_ids:
            record["completion_token_ids"] = completion_token_ids
        if self.save_prompts or self.save_token_ids:
            artifact_name = (rendered_sha or prepared_sha) + f"__seed{run_seed if run_seed is not None else 'na'}.json"
            artifact_path = self.artifact_dir / artifact_name
            write_json(
                artifact_path,
                {
                    "record_id": record_id,
                    "example_id": example.id,
                    "raw_messages": example.messages,
                    "prepared_messages": prompt_render.messages if prompt_render else example.messages,
                    "rendered_prompt": prompt_render.rendered_prompt if (prompt_render and self.save_prompts) else None,
                    "prompt_render": prompt_render.to_record(include_text=False, include_token_ids=self.save_token_ids) if prompt_render else None,
                    "answer": example.answer,
                    "prediction": generation_output.text,
                    "completion_token_ids": completion_token_ids if self.save_token_ids else None,
                    "metadata": example.metadata,
                    "metrics": metrics,
                    "sampling_params": generation_output.sampling_params,
                    "context_budget": generation_output.context_budget,
                    "backend_raw": generation_output.raw,
                },
            )
            record["artifact_path"] = str(artifact_path)
        append_jsonl(self.examples_path, record)

    def write_skip(
        self,
        example: Example,
        reason: str,
        model_key: str,
        run_seed: int | None = None,
        base_seed: int | None = None,
        seed_policy: str | None = None,
        prompt_render: PromptRender | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        raw_messages_sha = sha256_json(example.messages)
        prepared_sha = prompt_render.request_sha256 if prompt_render else raw_messages_sha
        rendered_sha = prompt_render.rendered_prompt_sha256 if prompt_render else None
        record_id = f"{example.id}::seed={run_seed}" if run_seed is not None else example.id
        record = {
            "record_id": record_id,
            "example_id": example.id,
            "task_name": example.task_name,
            "model_key": model_key,
            "base_seed": base_seed,
            "seed_policy": seed_policy,
            "generation_seed": run_seed,
            "skipped": True,
            "skip_reason": reason,
            "skip_details": details or {},
            "answer": example.answer,
            "metadata": example.metadata,
            "prompt_raw_messages_sha256": raw_messages_sha,
            "prompt_messages_sha256": prepared_sha,
            "rendered_prompt_sha256": rendered_sha,
            "prompt_message_chars": len(str(example.messages)),
            "rendered_prompt_chars": prompt_render.rendered_prompt_chars if prompt_render else None,
            "prompt_roles": prompt_render.message_roles if prompt_render else None,
            "prompt_chat_template_kwargs": prompt_render.chat_template_kwargs if prompt_render else None,
            "system_prompt_applied": prompt_render.system_prompt_applied if prompt_render else None,
            "prompt_renderer": prompt_render.renderer if prompt_render else None,
            "prompt_tokens_estimate": prompt_render.prompt_tokens_estimate if prompt_render else None,
            "prompt_token_ids_sha256": sha256_json(prompt_render.token_ids) if prompt_render and prompt_render.token_ids is not None else None,
        }
        if (self.save_prompts or self.save_token_ids) and prompt_render is not None:
            artifact_name = (rendered_sha or prepared_sha) + f"__seed{run_seed if run_seed is not None else 'na'}__skipped.json"
            artifact_path = self.artifact_dir / artifact_name
            write_json(
                artifact_path,
                {
                    "record_id": record_id,
                    "example_id": example.id,
                    "skip_reason": reason,
                    "skip_details": details or {},
                    "raw_messages": example.messages,
                    "prepared_messages": prompt_render.messages,
                    "rendered_prompt": prompt_render.rendered_prompt if self.save_prompts else None,
                    "prompt_render": prompt_render.to_record(include_text=False, include_token_ids=self.save_token_ids),
                    "answer": example.answer,
                    "metadata": example.metadata,
                },
            )
            record["artifact_path"] = str(artifact_path)
        append_jsonl(self.skipped_path, record)
