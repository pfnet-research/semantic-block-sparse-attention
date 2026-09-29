"""The Qwen3.5-4B checkpoint built from the Hugging Face snapshot matches the reference one.

``scripts/prepare_qwen3_5_checkpoint.py`` is run into a temporary directory,
and its output must match ``checkpoints/qwen3_5_4b/raw-keys`` in the config
the runtime parses and in every routing tensor that inference reads.  CPU
only; skipped where the snapshot or the reference checkpoint is absent.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import load_file  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLD = ROOT / "checkpoints/qwen3_5_4b/raw-keys"
_HF_HOME = pathlib.Path(os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface"))
SNAPSHOT = next(iter(sorted((_HF_HOME / "hub/models--Qwen--Qwen3.5-4B/snapshots").glob("*"))), pathlib.Path("/nonexistent"))


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> pathlib.Path:
    if not OLD.is_dir() or not SNAPSHOT.is_dir():
        pytest.skip("needs the recorded raw-keys checkpoint and the Qwen3.5-4B snapshot")
    from scripts.prepare_qwen3_5_checkpoint import build
    from scripts.prepare_nemotron_checkpoint import DEFAULT_QWEN_RAW_KEYS
    out = tmp_path_factory.mktemp("ckpt") / "raw-keys"
    build(SNAPSHOT, out, DEFAULT_QWEN_RAW_KEYS)
    return out


def test_config_is_the_recorded_one(built):
    new = json.loads((built / "config.json").read_text())
    old = json.loads((OLD / "config.json").read_text())
    # Compare the base model and settings still read by the released runtime.
    old["text_config"] = {k: v for k, v in old["text_config"].items()
                          if not k.startswith("semblo_") or k in new["text_config"]}
    assert new == old


def test_the_runtime_parses_the_same_settings(built):
    from semblo.vllm_plugin.config import SembloVllmConfig
    parse = lambda d: SembloVllmConfig.from_hf_config(json.loads((d / "config.json").read_text()))
    assert parse(built) == parse(OLD)


def test_routing_tensors_are_the_recorded_key_maps(built):
    """The recorded file also holds a query router with a zero residual and
    mixture, which the plugin drops; the key maps are the same."""
    new = load_file(str(built / "semblo_runtime.safetensors"))
    old = {n: t for n, t in load_file(str(OLD / "semblo_runtime.safetensors")).items()
           if ".semblo_query_router." not in n}
    assert sorted(new) == sorted(old)
    for name in sorted(new):
        assert new[name].dtype == old[name].dtype and new[name].shape == old[name].shape, name
        assert torch.equal(new[name], old[name]), name


def test_the_runtime_accepts_the_weights(built):
    from semblo.vllm_plugin.runtime_weights import verify_runtime_weights
    manifest = verify_runtime_weights(built)
    assert len(manifest.tensor_names) == 8
    declared = json.loads((built / "semblo_vllm_manifest.json").read_text())
    assert declared["untrained_router_arm"]["arm"] == "raw-keys"
