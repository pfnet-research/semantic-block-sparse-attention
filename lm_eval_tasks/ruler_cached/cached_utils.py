from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import datasets

from lm_eval.tasks.ruler.common_utils import DEFAULT_SEQ_LENGTHS


CACHE_VERSION = 1
DEFAULT_CACHE_ROOT = Path("data/ruler_cache")
# ``fill: official`` (a task kwarg, from the suite's metadata) makes the two generators that run
# out of material behave as NVIDIA's RULER does instead of stopping short: SQuAD QA cycles its
# paragraph pool once every paragraph is in the prompt (lm_eval 0.4.12 stops at 213K tokens),
# and common-word extraction draws from a large dictionary once wonderwords' 8,166 words are
# used up (lm_eval stops at 197K tokens).  Both searches for the prompt size are binary, as in
# the official generator.  The kwarg is part of the cache key for these two tasks only, so every
# other task keeps its cache.
FILL_TASKS = ("ruler_qa_squad", "ruler_cwe")
LOCK_WAIT_SECONDS = int(os.environ.get("SEMBLO_RULER_CACHE_LOCK_WAIT_SECONDS", "86400"))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _cache_root() -> Path:
    return Path(os.environ.get("SEMBLO_RULER_CACHE_DIR", str(DEFAULT_CACHE_ROOT)))


def _cache_payload(task_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    lengths = kwargs.get("max_seq_lengths", DEFAULT_SEQ_LENGTHS)
    payload_kwargs = {
        "pretrained": kwargs.get("pretrained"),
        "tokenizer": kwargs.get("tokenizer"),
        "max_seq_lengths": list(lengths),
        "trust_remote_code": kwargs.get("trust_remote_code"),
        "revision": kwargs.get("revision"),
        "tokenizer_revision": kwargs.get("tokenizer_revision"),
        "ruler_generation_seed": kwargs.get("ruler_generation_seed", 42),
    }
    if kwargs.get("samples_per_length") is not None:
        payload_kwargs["samples_per_length"] = kwargs.get("samples_per_length")
    if task_name in FILL_TASKS and kwargs.get("fill"):
        payload_kwargs["fill"] = kwargs["fill"]
    payload = {
        "cache_version": CACHE_VERSION,
        "task": task_name,
        "kwargs": _jsonable(payload_kwargs),
        "versions": {
            "lm_eval": _package_version("lm_eval"),
            "transformers": _package_version("transformers"),
            "datasets": _package_version("datasets"),
        },
    }
    if task_name == "ruler_cwe" and kwargs.get("fill") == "official":
        payload["inputs"] = {"wordlist_sha256": _file_sha256(_wordlist_path())}
    elif task_name == "ruler_qa_hotpot":
        local = _hotpot_path()
        payload["inputs"] = ({"hotpot_sha256": _file_sha256(local)} if local.is_file()
                             else {"hotpot_source": "hotpot_dev_distractor_v1.json"})
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hotpot_path() -> Path:
    return Path(os.environ.get("SEMBLO_HOTPOT_JSON", str(DEFAULT_CACHE_ROOT / "hotpot_dev_distractor_v1.json")))


def _cache_key(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _acquire_lock(lock_dir: Path) -> bool:
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            lock_dir.mkdir(parents=True)
            return True
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for RULER dataset cache lock: {lock_dir}")
            time.sleep(10)


def _build_seed(payload: dict[str, Any]) -> int:
    """The seed of the shared random for one cache build: a function of the cache key."""
    return int(_cache_key(payload), 16)


@contextlib.contextmanager
def _seeded_shared_random(seed: int):
    """Seed the shared ``random`` and ``numpy.random`` states for one cache build, then restore them.

    lm_eval's generators use their seeded generator for the needle order only; keys, values,
    insertion depths (NIAH), variable values (VT), word samples (CWE) and distractor paragraphs
    (QA) come from the shared states.  Without this, a cache built in a fresh process differs
    from run to run under the same key.  A cache that already exists is read, so seeding does not
    affect it.
    """
    import numpy as np

    python_state, numpy_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed % 2**32)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _save_dataset(cache_path: Path, payload: dict[str, Any], dataset: dict[str, datasets.Dataset],
                  shared_random_seed: int | None = None) -> None:
    tmp_path = cache_path.with_name(f".{cache_path.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp_path, ignore_errors=True)
    datasets.DatasetDict(dataset).save_to_disk(str(tmp_path))
    # The seed is recorded in the manifest. It is not part of the key.
    manifest = dict(payload) if shared_random_seed is None else {**payload, "shared_random_seed": str(shared_random_seed)}
    (tmp_path / "semblo_ruler_cache_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.rename(cache_path)


def _load_dataset(cache_path: Path) -> datasets.DatasetDict:
    return datasets.load_from_disk(str(cache_path))


def _limit_samples_per_length(
    dataset: dict[str, datasets.Dataset] | datasets.DatasetDict,
    samples_per_length: int | None,
) -> datasets.DatasetDict:
    if samples_per_length is None:
        return datasets.DatasetDict(dataset)
    if samples_per_length <= 0:
        raise ValueError("samples_per_length must be positive when set.")

    limited: dict[str, datasets.Dataset] = {}
    for split, split_dataset in dataset.items():
        if "max_length" not in split_dataset.column_names:
            raise ValueError(f"RULER cached split {split!r} has no max_length column.")

        selected_indices: list[int] = []
        counts: dict[int, int] = {}
        for idx, max_length in enumerate(split_dataset["max_length"]):
            key = int(max_length)
            if counts.get(key, 0) >= samples_per_length:
                continue
            counts[key] = counts.get(key, 0) + 1
            selected_indices.append(idx)

        limited[split] = split_dataset.select(selected_indices)
    return datasets.DatasetDict(limited)


def _cached(task_name: str, factory: Callable[..., dict[str, datasets.Dataset]], **kwargs: Any) -> datasets.DatasetDict:
    samples_per_length = kwargs.get("samples_per_length")
    payload = _cache_payload(task_name, dict(kwargs))
    cache_path = _cache_root() / task_name / _cache_key(payload)
    if cache_path.exists():
        return _load_dataset(cache_path)

    factory_kwargs = dict(kwargs)
    factory_kwargs.pop("samples_per_length", None)
    factory_kwargs.pop("fill", None)
    lock_dir = cache_path.with_suffix(".lock")
    _acquire_lock(lock_dir)
    try:
        if cache_path.exists():
            return _load_dataset(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if task_name in FILL_TASKS and kwargs.get("fill"):
            # official_fill_qa / official_fill_cwe have already seeded the shared random with the
            # generation seed, so it is not reseeded here.
            dataset = _limit_samples_per_length(factory(**factory_kwargs), samples_per_length)
            _save_dataset(cache_path, payload, dataset)
        else:
            seed = _build_seed(payload)
            with _seeded_shared_random(seed):
                dataset = _limit_samples_per_length(factory(**factory_kwargs), samples_per_length)
            _save_dataset(cache_path, payload, dataset, shared_random_seed=seed)
        return _load_dataset(cache_path)
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def niah_single_1(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import niah_utils

    return _cached("niah_single_1", niah_utils.niah_single_1, **kwargs)


def niah_single_2(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import niah_utils

    return _cached("niah_single_2", niah_utils.niah_single_2, **kwargs)


def niah_single_3(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import niah_utils

    return _cached("niah_single_3", niah_utils.niah_single_3, **kwargs)


def niah_multikey_1(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import niah_utils

    return _cached("niah_multikey_1", niah_utils.niah_multikey_1, **kwargs)


def niah_multikey_2(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import niah_utils

    return _cached("niah_multikey_2", niah_utils.niah_multikey_2, **kwargs)


def niah_multikey_3(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import niah_utils

    return _cached("niah_multikey_3", niah_utils.niah_multikey_3, **kwargs)


def niah_multiquery(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import niah_utils

    return _cached("niah_multiquery", niah_utils.niah_multiquery, **kwargs)


def niah_multivalue(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import niah_utils

    return _cached("niah_multivalue", niah_utils.niah_multivalue, **kwargs)


def ruler_vt(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import vt_utils

    return _cached("ruler_vt", vt_utils.get_vt_dataset, **kwargs)


def ruler_cwe(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import cwe_utils

    if kwargs.get("fill") == "official":
        with official_fill_cwe(kwargs.get("ruler_generation_seed", 42), kwargs.get("samples_per_length")):
            return _cached("ruler_cwe", cwe_utils.get_cw_dataset, **kwargs)
    return _cached("ruler_cwe", cwe_utils.get_cw_dataset, **kwargs)


def ruler_fwe(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import fwe_utils

    return _cached("ruler_fwe", fwe_utils.fwe_download, **kwargs)


def ruler_qa_squad(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import qa_utils

    if kwargs.get("fill") == "official":
        with official_fill_qa(kwargs.get("ruler_generation_seed", 42), kwargs.get("samples_per_length")):
            return _cached("ruler_qa_squad", qa_utils.get_squad, **kwargs)
    return _cached("ruler_qa_squad", qa_utils.get_squad, **kwargs)


def ruler_qa_hotpot(**kwargs: Any) -> datasets.DatasetDict:
    from lm_eval.tasks.ruler import qa_utils

    # lm_eval downloads the HotpotQA dev-distractor file when it builds the
    # cache. A local copy (SEMBLO_HOTPOT_JSON, by default
    # data/ruler_cache/hotpot_dev_distractor_v1.json) is read instead when
    # present, for machines without network access.
    local = _hotpot_path()
    if local.is_file():
        import json
        original = qa_utils.download_json

        def download_json(url):
            if url.endswith("hotpot_dev_distractor_v1.json"):
                return json.loads(local.read_text())
            return original(url)

        qa_utils.download_json = download_json
        try:
            return _cached("ruler_qa_hotpot", qa_utils.get_hotpotqa, **kwargs)
        finally:
            qa_utils.download_json = original
    return _cached("ruler_qa_hotpot", qa_utils.get_hotpotqa, **kwargs)



# ---------------------------------------------------------------------------
# The official generator's behaviour for the two tasks that run out of material.
# Same templates, same document / word formatting, same per-sample loop as lm_eval's
# port; what changes is (a) the pool is cycled (QA) or enlarged (CWE) instead of
# capping the prompt, and (b) the prompt-size search is binary, as NVIDIA's is,
# because a linear search in steps of ten cannot reach 1M tokens.

import contextlib
import random as _random

WORDLIST_ENV = "SEMBLO_WORDLIST"


def _wordlist_path() -> Path:
    return Path(os.environ.get(WORDLIST_ENV, str(_cache_root().parent / "wordlists" / "usr_share_dict_words.txt")))


def big_word_list(base: list[str], path: Path | None = None) -> list[str]:
    """wonderwords' list plus the supplied dictionary's lower-case words (3 to 12 letters),
    shuffled with the same seeded RNG the port shuffles its list with."""
    path = _wordlist_path() if path is None else path
    if not path.is_file():
        raise FileNotFoundError(f"the fallback dictionary {path} is missing (set {WORDLIST_ENV})")
    extra = {w.strip() for w in path.read_text(errors="ignore").splitlines()}
    extra = {w for w in extra if w.isascii() and w.isalpha() and w.islower() and 3 <= len(w) <= 12}
    words = sorted(set(base) | extra)
    _random.Random(42).shuffle(words)
    return words


def _binary_search_size(fits, lower: int, upper: int) -> int:
    """The largest n in [lower, upper] with fits(n) true, assuming fits is monotone; lower if none."""
    best = lower
    while lower <= upper:
        mid = (lower + upper) // 2
        if fits(mid):
            best = mid
            lower = mid + 1
        else:
            upper = mid - 1
    return best


def qa_generate_input_output_filled(index, num_docs, qas, docs, qa_utils):
    if num_docs <= len(docs):
        return qa_utils._semblo_original_generate_input_output(index, num_docs, qas, docs)
    repeats = (num_docs + len(docs) - 1) // len(docs)
    all_docs = (list(docs) * repeats)[:num_docs]
    _random.Random(qa_utils.SEED).shuffle(all_docs)
    context = "\n\n".join(qa_utils.DOCUMENT_PROMPT.format(i=i + 1, document=d) for i, d in enumerate(all_docs))
    return qa_utils.TEMPLATE.format(context=context, query=qas[index]["query"]), qas[index]["outputs"]


def qa_generate_samples_filled(tokenizer, docs, qas, max_seq_length, num_samples=500, tokens_to_generate=32,
                               pre_samples=0, incremental=10, remove_newline_tab=False, qa_utils=None, cap=None):
    from tqdm import tqdm
    gio = qa_utils.generate_input_output
    # lm_eval's factories ask for 500 prompts per length and the wrapper keeps samples_per_length of
    # them; 500 filled 1M prompts overflow a pyarrow string array (2 GB of offsets) and take five
    # times longer to make, so the generation stops at the number the cache will keep.
    if cap:
        num_samples = min(num_samples, int(cap))

    def tokens(num_docs):
        text, answer = gio(0, num_docs, qas=qas, docs=docs)
        return len(tokenizer(text + f" {answer}").input_ids)

    probe = min(len(docs), 200)
    per_doc = max(1.0, tokens(probe) / probe)
    upper = int(1.5 * max_seq_length / per_doc) + incremental
    num_docs = _binary_search_size(lambda n: tokens(n) + tokens_to_generate <= max_seq_length, incremental, upper)
    write_jsons = []
    for index in tqdm(range(num_samples), desc=f"Generating QA Samples (filled) | {max_seq_length}"):
        used_docs = num_docs
        while True:
            try:
                input_text, answer = gio(index + pre_samples, used_docs, qas=qas, docs=docs)
                length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except Exception:  # the port's rule: shrink by the increment until it fits
                if used_docs > incremental:
                    used_docs -= incremental
        if remove_newline_tab:
            input_text = " ".join(input_text.replace("\n", " ").replace("\t", " ").strip().split())
        write_jsons.append({"index": index, "input": input_text, "outputs": answer, "length": length,
                            "max_length": max_seq_length, "gen_prefix": "Answer:"})
    return write_jsons


def cwe_sys_word_pair_random_filled(num_samples, max_seq_length, tokenizer=None, incremental=10,
                                    remove_newline_tab=False, tokens_to_generate=120, cwe_utils=None, cap=None):
    from tqdm import tqdm
    assert tokenizer is not None, "Tokenizer is not provided."
    if cap:
        num_samples = min(num_samples, int(cap))
    CONFIG = cwe_utils.CONFIG

    def tokens(num_words):
        input_example, input_text, answer = cwe_utils.generate_input_output(num_words, max_seq_length)
        return len(tokenizer(input_example + "\n" + input_text + " "
                             + " ".join(f"{i + 1}. {w}" for i, w in enumerate(answer))).input_ids)

    probe = min(len(cwe_utils.WORDS), 1000)
    per_word = max(0.1, tokens(probe) / probe)
    upper = min(len(cwe_utils.WORDS), int(1.5 * max_seq_length / per_word) + incremental)
    num_words = _binary_search_size(lambda n: tokens(n) + tokens_to_generate <= max_seq_length, incremental, upper)
    write_jsons = []
    for index in tqdm(range(num_samples), desc=f"Generating CWE Samples (filled) | {max_seq_length}"):
        used_words = num_words
        while True:
            try:
                input_example, input_text, answer = cwe_utils.generate_input_output(used_words, max_seq_length)
                length = len(tokenizer(input_text).input_ids) + tokens_to_generate
                assert length <= max_seq_length, f"{length} exceeds max_seq_length."
                break
            except Exception:
                if used_words > incremental:
                    used_words -= incremental
        if remove_newline_tab:
            input_text = " ".join(input_text.replace("\n", " ").replace("\t", " ").strip().split())
            input_example = " ".join(input_example.replace("\n", " ").replace("\t", " ").strip().split())
        gen_prefix_index = input_text.rfind(CONFIG["answer_prefix"])
        input_text = input_text[:gen_prefix_index]
        write_jsons.append({"index": index, "input": input_text.strip(), "input_example": input_example,
                            "outputs": answer, "length": length, "max_length": max_seq_length,
                            "gen_prefix": CONFIG["answer_prefix"].strip()})
    return write_jsons


@contextlib.contextmanager
def official_fill_qa(seed: int = 42, samples_per_length: int | None = None):
    from lm_eval.tasks.ruler import qa_utils
    saved = (qa_utils.generate_input_output, qa_utils.generate_samples)
    qa_utils._semblo_original_generate_input_output = saved[0]
    qa_utils.generate_input_output = lambda index, num_docs, qas, docs: qa_generate_input_output_filled(index, num_docs, qas, docs, qa_utils)
    qa_utils.generate_samples = lambda *a, **k: qa_generate_samples_filled(*a, qa_utils=qa_utils, cap=samples_per_length, **k)
    state = _random.getstate()
    _random.seed(seed)
    try:
        yield
    finally:
        qa_utils.generate_input_output, qa_utils.generate_samples = saved
        _random.setstate(state)


@contextlib.contextmanager
def official_fill_cwe(seed: int = 42, samples_per_length: int | None = None):
    from lm_eval.tasks.ruler import cwe_utils
    saved = (cwe_utils.WORDS, cwe_utils.sys_word_pair_random)
    cwe_utils.WORDS = big_word_list(saved[0])
    cwe_utils.sys_word_pair_random = lambda *a, **k: cwe_sys_word_pair_random_filled(*a, cwe_utils=cwe_utils, cap=samples_per_length, **k)
    state = _random.getstate()
    _random.seed(seed)
    try:
        yield
    finally:
        cwe_utils.WORDS, cwe_utils.sys_word_pair_random = saved
        _random.setstate(state)
