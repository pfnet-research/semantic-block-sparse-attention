#!/usr/bin/env python3
"""Prepare Qwen3.5-4B for the vLLM runtime with identity routing (the paper's model).

The checkpoint is built from the Hugging Face snapshot, with the same recipe as
``scripts/prepare_qwen3_5_moe_checkpoint.py`` and
``scripts/prepare_nemotron_checkpoint.py``:

- the snapshot's weight and tokenizer files as relative symlinks;
- config.json = the snapshot's, under ``SembloQwen3_5ForConditionalGeneration``,
  with the ``semblo_*`` settings of
  ``configs/checkpoints/qwen3_5_raw_keys_text_config.json``, the routing width set
  to the head dimension and the full-attention layers read from ``layer_types``;
- runtime weights: one identity key-routing map per full-attention layer.

The manifest marks the checkpoint as using identity routing, which the runtime
requires when the routing weights are not trained.

    python scripts/prepare_qwen3_5_checkpoint.py \\
        --snapshot $HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/<sha> \\
        --out checkpoints/qwen3_5_4b/raw-keys

``checkpoints/qwen3_5_4b/raw-keys`` is where the model presets of
configs/models.yaml look.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_nemotron_checkpoint import (  # noqa: E402
    DEFAULT_QWEN_RAW_KEYS,
    link_snapshot,
    write_runtime_weights,
)
from semblo.vllm_plugin.identity_router import identity_router_tensors  # noqa: E402
from semblo.vllm_plugin.runtime_weights import (  # noqa: E402
    RUNTIME_MANIFEST_FILE,
    RUNTIME_WEIGHTS_FILE,
    sha256_file,
)

ARCH = "SembloQwen3_5ForConditionalGeneration"
SCRIPT = "scripts/prepare_qwen3_5_checkpoint.py"


def build(snapshot: pathlib.Path, out: pathlib.Path, settings: pathlib.Path, force: bool = False) -> str:
    """Write the checkpoint to ``out``; returns the runtime weights' sha256."""
    snapshot = snapshot.resolve()
    cfg = json.loads((snapshot / "config.json").read_text())
    if cfg.get("model_type") != "qwen3_5" or "text_config" not in cfg:
        raise SystemExit(f"{snapshot}/config.json is not a Qwen3.5 dense config (model_type {cfg.get('model_type')!r})")
    text = cfg["text_config"]
    layer_types = list(text["layer_types"])
    if int(text["num_hidden_layers"]) != len(layer_types):
        raise SystemExit("num_hidden_layers does not match layer_types")
    attention_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    source = json.loads(settings.read_text())
    dp = {k: v for k, v in source.get("text_config", source).items() if k.startswith("semblo_")}
    dp["semblo_index_dim"] = int(text["head_dim"])
    dp["semblo_layer_indices"] = attention_layers
    dp["semblo_runtime_weights_file"] = RUNTIME_WEIGHTS_FILE
    dp["semblo_runtime_manifest_file"] = RUNTIME_MANIFEST_FILE

    if out.exists():
        if not force:
            raise SystemExit(f"{out} exists; pass --force")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    linked = link_snapshot(snapshot, out)
    config = json.loads(json.dumps(cfg))
    config["architectures"] = [ARCH]
    config["text_config"].update(dp)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    tensors = identity_router_tensors(text, dp, attention_layers)
    sha = write_runtime_weights(out, tensors, config, snapshot, script=SCRIPT)
    arm = {"arm": "raw-keys", "source": SCRIPT, "index_dim": int(text["head_dim"])}
    runtime_path = out / RUNTIME_MANIFEST_FILE
    runtime = json.loads(runtime_path.read_text())
    runtime["untrained_router_arm"] = arm
    runtime_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    vllm_manifest = {
        "architecture": ARCH, "base_checkpoint": str(snapshot), "link_mode": "symlink",
        "prepared_config_sha256": sha256_file(out / "config.json"),
        "runtime_weights": {"weights_sha256": sha, "tensor_names": sorted(tensors)},
        "untrained_router_arm": arm,
        "weights_sha256": sha,
    }
    (out / "semblo_vllm_manifest.json").write_text(
        json.dumps(vllm_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    first = attention_layers[0]
    provenance = [
        f"# raw-keys: Qwen3.5 ({snapshot.name}) prepared for the vLLM runtime by {SCRIPT}", "",
        f"Snapshot: `{snapshot}` (weights and tokenizer files are relative symlinks: {', '.join(linked[:6])}, ...)", "",
        "## config.json against the snapshot's", "",
        f"- architectures = ['{ARCH}']",
        f"- text_config: semblo_* keys from `{settings}`, then",
        f"  - semblo_index_dim = {text['head_dim']} (the head dimension; identity routing)",
        f"  - semblo_layer_indices = {attention_layers} (the full_attention entries of layer_types)",
        "- runtime weights: identity key-routing maps (the paper's raw-keys arm)",
        f"- {len(tensors)} runtime tensors, weights sha256 {sha[:16]}; shapes: " + ", ".join(
            f"{k.split('semblo_')[1]} {list(v.shape)}" for k, v in sorted(tensors.items())
            if k.startswith(f"layers.{first}.")),
    ]
    (out / "PROVENANCE.md").write_text("\n".join(provenance) + "\n")
    return sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=pathlib.Path, required=True,
                    help="the Hugging Face snapshot directory of Qwen/Qwen3.5-4B")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--settings", type=pathlib.Path, default=DEFAULT_QWEN_RAW_KEYS,
                    help="a JSON whose text_config semblo_* keys are copied")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    sha = build(a.snapshot, a.out, a.settings, force=a.force)
    print(f"raw-keys: {a.out} (runtime weights sha256 {sha[:16]})")


if __name__ == "__main__":
    main()
