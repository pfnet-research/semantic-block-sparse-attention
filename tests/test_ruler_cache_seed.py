"""A RULER cache built afresh has the same items in every process (lm_eval_tasks/ruler_cached/cached_utils.py).

lm_eval's generators draw keys, values and depths from the shared ``random`` and ``numpy.random``
states, which Python seeds from the OS at start-up.  A cache miss therefore seeds both states from
the cache key, except under the official fill, which seeds them itself.
"""
from __future__ import annotations

import json
import os
import pathlib
import random
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lm_eval_tasks" / "ruler_cached"))

import cached_utils as cu  # noqa: E402

BUILD = r"""
import json, random, sys
import numpy as np
import datasets
sys.path.insert(0, sys.argv[1])
import cached_utils as cu
def factory(**kwargs):
    return {"test": datasets.Dataset.from_dict({
        "key": [str(random.getrandbits(128)) for _ in range(4)],
        "value": [int(np.random.randint(10000, 99999)) for _ in range(4)],
        "max_length": [4096] * 4})}
d = cu._cached("niah_multikey_3", factory, pretrained="p", max_seq_lengths=[4096], samples_per_length=4)
print(json.dumps({"key": list(d["test"]["key"]), "value": list(d["test"]["value"])}))
"""


def _build_in_a_fresh_process(cache_dir: pathlib.Path) -> dict:
    env = {"SEMBLO_RULER_CACHE_DIR": str(cache_dir), "PATH": "/usr/bin:/bin",
           "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")}
    out = subprocess.run([sys.executable, "-c", BUILD, str(ROOT / "lm_eval_tasks" / "ruler_cached")],
                         capture_output=True, text=True, check=True, env=env).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_two_fresh_processes_build_the_same_items(tmp_path):
    first = _build_in_a_fresh_process(tmp_path / "a")
    second = _build_in_a_fresh_process(tmp_path / "b")
    assert first == second
    manifest = next((tmp_path / "a" / "niah_multikey_3").glob("*/semblo_ruler_cache_manifest.json"))
    assert "shared_random_seed" in json.loads(manifest.read_text())


def test_the_build_restores_the_shared_states(tmp_path, monkeypatch):
    import datasets
    monkeypatch.setenv("SEMBLO_RULER_CACHE_DIR", str(tmp_path))
    random.seed(123)
    np.random.seed(123)
    expected = (random.random(), float(np.random.rand()))
    random.seed(123)
    np.random.seed(123)
    cu._cached("ruler_vt", lambda **k: {"test": datasets.Dataset.from_dict(
        {"x": [random.random(), float(np.random.rand())], "max_length": [4096, 4096]})},
        pretrained="p", max_seq_lengths=[4096])
    assert (random.random(), float(np.random.rand())) == expected


def test_the_official_fill_keeps_its_own_seed(tmp_path, monkeypatch):
    """Under ``fill: official`` the build draws from the state that official_fill_* sets."""
    import datasets
    monkeypatch.setenv("SEMBLO_RULER_CACHE_DIR", str(tmp_path))
    words = tmp_path / "words.txt"
    words.write_text("apple\nbanana\n")
    monkeypatch.setenv("SEMBLO_WORDLIST", str(words))
    random.seed(42)
    expected = [random.random() for _ in range(3)]
    random.seed(42)
    d = cu._cached("ruler_cwe", lambda **k: {"test": datasets.Dataset.from_dict(
        {"x": [random.random() for _ in range(3)], "max_length": [4096] * 3})},
        pretrained="p", max_seq_lengths=[4096], fill="official")
    assert list(d["test"]["x"]) == expected
    manifest = next((tmp_path / "ruler_cwe").glob("*/semblo_ruler_cache_manifest.json"))
    assert "shared_random_seed" not in json.loads(manifest.read_text())


def test_an_existing_cache_is_read_not_rebuilt(tmp_path, monkeypatch):
    import datasets
    monkeypatch.setenv("SEMBLO_RULER_CACHE_DIR", str(tmp_path))
    calls = []
    def factory(**k):
        calls.append(1)
        return {"test": datasets.Dataset.from_dict({"x": [len(calls)], "max_length": [4096]})}
    first = cu._cached("niah_multikey_2", factory, pretrained="p", max_seq_lengths=[4096])
    second = cu._cached("niah_multikey_2", factory, pretrained="p", max_seq_lengths=[4096])
    assert len(calls) == 1 and list(first["test"]["x"]) == list(second["test"]["x"]) == [1]


def test_changed_dictionary_rebuilds_cache(tmp_path, monkeypatch):
    import datasets
    monkeypatch.setenv("SEMBLO_RULER_CACHE_DIR", str(tmp_path / "cache"))
    words = tmp_path / "words.txt"
    words.write_text("apple\nbanana\n")
    monkeypatch.setenv("SEMBLO_WORDLIST", str(words))
    calls = []
    def factory(**kwargs):
        calls.append(1)
        return {"test": datasets.Dataset.from_dict({"x": [len(calls)], "max_length": [4096]})}
    kwargs = {"pretrained": "p", "max_seq_lengths": [4096], "fill": "official"}
    first = cu._cached("ruler_cwe", factory, **kwargs)
    words.write_text("apple\nbanana\ncherry\n")
    second = cu._cached("ruler_cwe", factory, **kwargs)
    assert len(calls) == 2
    assert list(first["test"]["x"]) == [1] and list(second["test"]["x"]) == [2]


def test_hotpot_content_changes_key_but_relocating_same_file_does_not(tmp_path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"data": 1}')
    second.write_bytes(first.read_bytes())
    monkeypatch.setenv("SEMBLO_HOTPOT_JSON", str(first))
    key = cu._cache_key(cu._cache_payload("ruler_qa_hotpot", {}))
    monkeypatch.setenv("SEMBLO_HOTPOT_JSON", str(second))
    assert key == cu._cache_key(cu._cache_payload("ruler_qa_hotpot", {}))
    second.write_text('{"data": 2}')
    assert key != cu._cache_key(cu._cache_payload("ruler_qa_hotpot", {}))
