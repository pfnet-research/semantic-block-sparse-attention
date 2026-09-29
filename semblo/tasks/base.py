from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from semblo.prompting import Message


@dataclass(frozen=True)
class Example:
    id: str
    task_name: str
    messages: list[Message]
    answer: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class Task(Protocol):
    name: str

    def iter_examples(self) -> Iterable[Example]:
        ...

    def score(self, prediction: str, example: Example) -> dict[str, Any]:
        ...
