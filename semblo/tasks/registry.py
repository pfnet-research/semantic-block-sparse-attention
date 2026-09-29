from __future__ import annotations

from typing import Any

from semblo.tasks.base import Task


def build_task(config: dict[str, Any]) -> Task:
    name = config.get("benchmark") or config.get("name")
    if name == "longbench_v2":
        from semblo.tasks.longbench_v2 import LongBenchV2Task

        return LongBenchV2Task(config)
    raise ValueError(
        f"Unknown benchmark/task {name!r}; supported: "
        "longbench_v2"
    )
