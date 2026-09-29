"""Run metadata suitable for saving alongside evaluation results."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Any


_SENSITIVE_NAME = re.compile(
    r"(?:^|_)(?:API_KEY|PASSWORD|SECRET|CREDENTIALS?|PRIVATE_KEY|AUTHORIZATION)(?:$|_)"
    r"|(?:^|_)(?:ACCESS_|AUTH_)?TOKEN$",
    re.IGNORECASE,
)
_RUNTIME_VARIABLES = {
    "CUDA_VISIBLE_DEVICES", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
    "TOKENIZERS_PARALLELISM", "TORCHINDUCTOR_DETERMINISTIC",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN", "VLLM_WORKER_MULTIPROC_METHOD",
    "VLLM_ENABLE_V1_MULTIPROCESSING", "VLLM_DISABLE_COMPILE_CACHE", "VLLM_PLUGINS",
}


def redact_sensitive_fields(value: Any) -> Any:
    """Redact credential fields in nested model and engine settings."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if _SENSITIVE_NAME.search(str(key))
            else redact_sensitive_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_fields(item) for item in value]
    return value


def runtime_environment() -> dict[str, str]:
    return {
        key: value for key, value in sorted(os.environ.items())
        if (key.startswith("SEMBLO_") or key in _RUNTIME_VARIABLES)
        and not _SENSITIVE_NAME.search(key)
    }


def collect_run_metadata() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    return {
        "environment": runtime_environment(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
    }
