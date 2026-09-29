"""The official-fill behaviour of the two RULER generators that run out of material
(lm_eval_tasks/ruler_cached/cached_utils.py): the SQuAD pool is cycled, the word list enlarged,
the prompt-size search is binary, and the cache key changes for those two tasks only."""
from __future__ import annotations

import pathlib
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lm_eval_tasks" / "ruler_cached"))

import cached_utils as cu  # noqa: E402


class _WordTokenizer:
    """Tokens = whitespace-separated words, so lengths are exact and fast."""
    def __call__(self, text):
        return types.SimpleNamespace(input_ids=text.split())


def test_qa_pool_is_cycled_beyond_its_size():
    from lm_eval.tasks.ruler import qa_utils
    docs = [f"doc{i} " + "w " * 20 for i in range(10)]
    qas = [{"query": "q?", "outputs": ["a"], "context": [0], "more_context": [1, 2]}]
    with cu.official_fill_qa(42):
        text, answer = qa_utils.generate_input_output(0, 25, qas=qas, docs=docs)
    assert answer == ["a"]
    assert text.count("Document ") == 25
    for i in range(10):
        assert text.count(f"doc{i} ") >= 2          # every paragraph at least twice
    with cu.official_fill_qa(42):                    # below the pool size the port's own rule applies
        text, _ = qa_utils.generate_input_output(0, 5, qas=qas, docs=docs)
    assert text.count("Document ") == 5


def test_qa_binary_search_fills_the_length():
    from lm_eval.tasks.ruler import qa_utils
    docs = [f"doc{i} " + "w " * 20 for i in range(50)]          # 21 tokens each + prompt
    qas = [{"query": "q?", "outputs": ["a"], "context": [0], "more_context": []} for _ in range(3)]
    max_len = 20_000                                            # needs about 900 docs, 18 cycles of the pool
    with cu.official_fill_qa(42):
        rows = qa_utils.generate_samples(_WordTokenizer(), docs, qas, max_len, num_samples=3, tokens_to_generate=32)
    assert len(rows) == 3
    for r in rows:
        assert r["length"] <= max_len
        assert r["length"] >= 0.98 * max_len                    # filled, not capped at the pool
        assert r["input"].count("Document ") > len(docs)


def test_cwe_uses_a_larger_dictionary_and_fills(tmp_path):
    from lm_eval.tasks.ruler import cwe_utils
    words = tmp_path / "words.txt"
    def word(i):                                                # letters only: the filter drops digits
        s = ""
        while True:
            s = chr(ord("a") + i % 26) + s; i //= 26
            if i == 0: return "zz" + s
    words.write_text("\n".join(word(i) for i in range(30000)) + "\nBad\nno-hyphen\nab\n")
    big = cu.big_word_list(cwe_utils.WORDS, words)
    assert len(big) >= len(cwe_utils.WORDS) + 30000
    assert "Bad" not in big and "no-hyphen" not in big and "ab" not in big
    n0 = len(cwe_utils.WORDS)
    import os
    # Restore the caller's value, which may be exported in the environment.
    previous = os.environ.get(cu.WORDLIST_ENV)
    os.environ[cu.WORDLIST_ENV] = str(words)
    try:
        with cu.official_fill_cwe(42):
            assert len(cwe_utils.WORDS) > n0
            rows = cwe_utils.sys_word_pair_random(2, 60_000, tokenizer=_WordTokenizer())
    finally:
        if previous is None:
            os.environ.pop(cu.WORDLIST_ENV, None)
        else:
            os.environ[cu.WORDLIST_ENV] = previous
    assert len(cwe_utils.WORDS) == n0                          # restored
    for r in rows:
        assert r["length"] <= 60_000 and r["length"] >= 0.98 * 60_000
        assert len(r["outputs"]) == 10


def test_fill_is_in_the_cache_key_of_the_two_tasks_only(tmp_path, monkeypatch):
    words = tmp_path / "words.txt"
    words.write_text("apple\nbanana\n")
    monkeypatch.setenv("SEMBLO_WORDLIST", str(words))
    base = {"tokenizer": "Qwen/Qwen3.5-4B", "max_seq_lengths": [524288], "samples_per_length": 100}
    for task in ("ruler_qa_squad", "ruler_cwe"):
        a = cu._cache_key(cu._cache_payload(task, dict(base)))
        b = cu._cache_key(cu._cache_payload(task, dict(base, fill="official")))
        assert a != b
    for task in ("niah_multikey_3", "niah_multivalue", "ruler_vt"):
        a = cu._cache_key(cu._cache_payload(task, dict(base)))
        b = cu._cache_key(cu._cache_payload(task, dict(base, fill="official")))
        assert a == b


def test_the_generation_stops_at_the_kept_sample_count():
    import pytest
    from lm_eval.tasks.ruler import qa_utils, cwe_utils
    if not cu._wordlist_path().is_file():
        pytest.skip(f"no dictionary at {cu._wordlist_path()}; set {cu.WORDLIST_ENV}")
    docs = [f"doc{i} " + "w " * 20 for i in range(50)]
    qas = [{"query": "q?", "outputs": ["a"], "context": [0], "more_context": []} for _ in range(20)]
    with cu.official_fill_qa(42, samples_per_length=3):
        rows = qa_utils.generate_samples(_WordTokenizer(), docs, qas, 5_000, num_samples=20, tokens_to_generate=32)
    assert len(rows) == 3
    with cu.official_fill_cwe(42, samples_per_length=2):
        rows = cwe_utils.sys_word_pair_random(20, 5_000, tokenizer=_WordTokenizer())
    assert len(rows) == 2
