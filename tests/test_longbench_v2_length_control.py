"""LongBench v2 at a stated context length.

The dataset's own `length` column is categorical (short/medium/long), and its
long bin runs past two million words.  The task therefore counts tokens,
truncates the way the official predictor does, and records what it sent.
"""
import pytest

from semblo.tasks import longbench_v2 as lb


class FakeTokenizer:
    """One token per character, and decode is the identity on that alphabet."""

    class _Out:
        def __init__(self, ids):
            self.input_ids = ids

    def __call__(self, text, add_special_tokens=False):
        return self._Out(list(text))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(ids)


@pytest.fixture(autouse=True)
def fake_tokenizer(monkeypatch):
    monkeypatch.setattr(lb, "_tokenizer", lambda _id: FakeTokenizer())


def _row(context: str) -> dict:
    return {"_id": "x", "context": context, "question": "q", "answer": "A",
            "choice_A": "a", "choice_B": "b", "choice_C": "c", "choice_D": "d",
            "domain": "d", "sub_domain": "s", "difficulty": "hard", "length": "long"}


def test_a_prompt_that_fits_is_untouched():
    task = lb.LongBenchV2Task({"tokenizer_id": "fake", "max_prompt_tokens": 10_000})
    prompt = "x" * 100
    out, n, truncated = task._fit_prompt(prompt)
    assert out == prompt and n == 100 and truncated is False


def test_a_long_prompt_is_cut_from_the_middle():
    task = lb.LongBenchV2Task({"tokenizer_id": "fake", "max_prompt_tokens": 10,
                               "truncation": "middle"})
    out, n, truncated = task._fit_prompt("abcdefghijklmnopqrst")
    assert truncated is True
    assert out == "abcdepqrst", "the head and the tail are kept, the middle goes"
    assert n == 10


def test_truncation_is_off_by_default_because_the_cli_does_the_cutting():
    task = lb.LongBenchV2Task({"tokenizer_id": "fake", "max_prompt_tokens": 10})
    out, n, truncated = task._fit_prompt("abcdefghijklmnopqrst")
    assert out == "abcdefghijklmnopqrst" and n == 20 and truncated is False


def test_without_a_tokenizer_the_length_is_unknown_rather_than_wrong():
    task = lb.LongBenchV2Task({})
    out, n, truncated = task._fit_prompt("abc")
    assert out == "abc" and n == -1 and truncated is False
    assert task._token_bin(n) == "unknown"


def test_length_control_without_a_tokenizer_is_refused():
    with pytest.raises(ValueError, match="tokenizer_id"):
        lb.LongBenchV2Task({"max_prompt_tokens": 1024})


def test_an_unknown_truncation_mode_is_refused():
    with pytest.raises(ValueError, match="truncation"):
        lb.LongBenchV2Task({"truncation": "tail"})


@pytest.mark.parametrize("n,expected", [
    (0, "64k"), (65536, "64k"), (65537, "128k"), (131072, "128k"),
    (262144, "256k"), (524288, "512k"), (1048576, "1m"), (1048577, "over_1m"),
    (-1, "unknown"),
])
def test_token_bins_are_named_the_way_every_other_table_names_them(n, expected):
    assert lb.LongBenchV2Task._token_bin(n) == expected


def test_short_items_can_be_excluded_so_a_band_means_one_thing(monkeypatch):
    rows = [_row("x" * 50), _row("y" * 5_000)]
    monkeypatch.setattr(lb, "load_dataset", lambda *a, **k: rows)
    task = lb.LongBenchV2Task({"tokenizer_id": "fake", "min_prompt_tokens": 1_000})
    got = list(task.iter_examples())
    assert len(got) == 1, "the 50-character context is below the floor"
    assert got[0].metadata["raw_prompt_tokens"] > 1_000


def test_what_was_sent_is_recorded_on_every_example(monkeypatch):
    monkeypatch.setattr(lb, "load_dataset", lambda *a, **k: [_row("z" * 4_000)])
    task = lb.LongBenchV2Task({"tokenizer_id": "fake", "max_prompt_tokens": 1_000,
                               "truncation": "middle"})
    ex = next(iter(task.iter_examples()))
    assert ex.metadata["raw_prompt_tokens"] == 1_000
    assert ex.metadata["truncated"] is True
    assert ex.metadata["token_bin"] == "64k"
    assert ex.metadata["tokenizer_id"] == "fake"
    assert len(ex.messages[0]["content"]) == 1_000, "the model sees the cut prompt"


def test_a_cache_built_for_another_tokenizer_is_refused(tmp_path):
    import json
    p = tmp_path / "counts.json"
    p.write_text(json.dumps({"tokenizer_id": "other", "prompt_style": "official",
                             "cot": False, "dataset_name": "THUDM/LongBench-v2",
                             "split": "train", "counts": {}}))
    task = lb.LongBenchV2Task({"tokenizer_id": "fake", "token_counts_path": str(p)})
    with pytest.raises(ValueError, match="would not be the one asked for"):
        task.prompt_token_count(_row("x"))


def test_a_missing_cache_is_an_error_not_a_silent_retokenize(tmp_path):
    task = lb.LongBenchV2Task({"tokenizer_id": "fake",
                               "token_counts_path": str(tmp_path / "nope.json")})
    with pytest.raises(FileNotFoundError, match="count_longbench_v2_tokens"):
        task.prompt_token_count(_row("x"))


def test_the_cache_is_used_instead_of_tokenizing(tmp_path, monkeypatch):
    import json
    p = tmp_path / "counts.json"
    p.write_text(json.dumps({"tokenizer_id": "fake", "prompt_style": "official",
                             "cot": False, "dataset_name": "THUDM/LongBench-v2",
                             "split": "train", "counts": {"x": 300_000}}))
    task = lb.LongBenchV2Task({"tokenizer_id": "fake", "token_counts_path": str(p)})
    monkeypatch.setattr(lb, "_tokenizer", lambda _id: pytest.fail("tokenized anyway"))
    assert task.prompt_token_count(_row("short")) == 300_000


def test_an_item_missing_from_the_cache_falls_back_to_counting(tmp_path):
    import json
    p = tmp_path / "counts.json"
    p.write_text(json.dumps({"tokenizer_id": "fake", "prompt_style": "official",
                             "cot": False, "dataset_name": "THUDM/LongBench-v2",
                             "split": "train", "counts": {}}))
    task = lb.LongBenchV2Task({"tokenizer_id": "fake", "token_counts_path": str(p)})
    assert task.prompt_token_count(_row("abc")) > 0
