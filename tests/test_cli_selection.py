from __future__ import annotations

from semblo.cli import _iter_limited
from semblo.tasks.base import Example


class DummyTask:
    def __init__(self, count: int):
        self.count = count

    def iter_examples(self):
        for index in range(self.count):
            yield Example(
                id=f"ex-{index}",
                task_name="dummy",
                messages=[{"role": "user", "content": str(index)}],
                answer=str(index),
                metadata={"bin": "all"},
            )


def _selected_ids(**kwargs) -> list[str]:
    defaults = {
        "task": DummyTask(10),
        "limit": None,
        "limit_per_bin": None,
        "skip_existing": set(),
        "base_seeds": [0],
        "seed_policy": "fixed",
        "metadata_filters": {},
        "bins": set(),
        "example_ids": set(),
        "start_index": None,
        "end_index": None,
        "num_shards": 1,
        "shard_id": 0,
        "shard_by": "index",
        "shuffle": False,
        "shuffle_seed": 0,
        "initial_counts_by_bin": None,
        "initial_count": 0,
    }
    defaults.update(kwargs)
    return [example.id for example in _iter_limited(**defaults)]


def test_start_end_index_filters_dataset_order():
    assert _selected_ids(start_index=3, end_index=7) == ["ex-3", "ex-4", "ex-5", "ex-6"]


def test_start_end_index_composes_with_index_shards():
    assert _selected_ids(start_index=3, end_index=8, num_shards=2, shard_id=1) == ["ex-3", "ex-5", "ex-7"]


def test_start_index_composes_with_resume_skip():
    assert _selected_ids(start_index=3, skip_existing={"ex-4::seed=0", "ex-5"}) == [
        "ex-3",
        "ex-6",
        "ex-7",
        "ex-8",
        "ex-9",
    ]


def test_limit_per_bin_counts_existing_resume_rows():
    assert _selected_ids(limit_per_bin=3, initial_counts_by_bin={"all": 2}, initial_count=2) == ["ex-0"]


def test_limit_counts_existing_resume_rows():
    assert _selected_ids(limit=3, initial_count=2) == ["ex-0"]


def test_output_is_checked_before_backend_loading(tmp_path, monkeypatch):
    import types
    import pytest
    from semblo import cli

    models = tmp_path / "models.yaml"
    benchmark = tmp_path / "benchmark.yaml"
    models.write_text("models:\n  m:\n    hf_id: dummy\n")
    benchmark.write_text("benchmark: longbench_v2\n")
    calls = []
    def backend(*args, **kwargs):
        calls.append(1)
        return types.SimpleNamespace(get_run_manifest=lambda: {})
    monkeypatch.setattr(cli, "build_backend", backend)
    monkeypatch.setattr(cli, "build_task", lambda config: types.SimpleNamespace(name="unit", iter_examples=lambda: iter(())))
    monkeypatch.setattr(cli, "set_global_seed", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "collect_environment", lambda: {})
    args = ["--benchmark-config", str(benchmark), "--models-config", str(models),
            "--model", "m", "--out-dir", str(tmp_path / "out")]
    cli.run_main(args)
    with pytest.raises(FileExistsError):
        cli.run_main(args)
    with pytest.raises(ValueError, match="settings differ"):
        cli.run_main([*args, "--resume", "--max-new-tokens", "8"])
    with pytest.raises(ValueError, match="settings differ"):
        cli.run_main([*args, "--resume", "--assistant-prefill", "Answer:"])
    assert len(calls) == 1
    cli.run_main([*args, "--resume"])
    assert len(calls) == 2
