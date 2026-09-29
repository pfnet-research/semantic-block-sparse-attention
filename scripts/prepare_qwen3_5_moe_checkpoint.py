#!/usr/bin/env python3
"""Prepare a Qwen3.5 MoE checkpoint (Qwen3.5-35B-A3B) for the vLLM runtime.

The same recipe as ``scripts/prepare_nemotron_checkpoint.py``, applied to the
Qwen3.5 MoE. It writes two directories under ``--out-root``:

``dense/``
    the stock model, config.json unchanged (re-serialised), weights and
    tokenizer files as relative symlinks into the Hugging Face snapshot.

``raw-keys/``
    the same model under ``SembloQwen3_5MoeForConditionalGeneration``
    with identity routing (the raw keys, as in the paper). The ``semblo_*``
    settings are copied from the Qwen3.5-4B settings into ``text_config``, the
    routing width is set to this model's head dimension, and the full-attention
    layers are listed from ``layer_types``. The runtime weights are built from
    the router modules, so their shapes are the ones the model instantiates
    (16 query heads, 2 KV heads, head_dim 256 for the 35B-A3B).

    python scripts/prepare_qwen3_5_moe_checkpoint.py \\
        --snapshot ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/<sha> \\
        --out-root checkpoints/qwen3_5_35b_a3b
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

ARCH = "SembloQwen3_5MoeForConditionalGeneration"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=pathlib.Path, required=True, help="the Hugging Face snapshot directory of Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    ap.add_argument("--qwen-raw-keys-config", type=pathlib.Path, default=DEFAULT_QWEN_RAW_KEYS,
                    help="config.json whose text_config semblo_* keys are copied (the Qwen3.5-4B raw-keys checkpoint)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    snapshot = a.snapshot.resolve()
    cfg = json.loads((snapshot / "config.json").read_text())
    if cfg.get("model_type") != "qwen3_5_moe" or "text_config" not in cfg:
        raise SystemExit(f"{snapshot}/config.json is not a Qwen3.5 MoE config (model_type {cfg.get('model_type')!r})")
    text = cfg["text_config"]
    layer_types = list(text["layer_types"])
    if int(text["num_hidden_layers"]) != len(layer_types):
        raise SystemExit("num_hidden_layers does not match layer_types")
    attention_layers = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    qwen = json.loads(a.qwen_raw_keys_config.read_text())
    qwen_text = qwen.get("text_config", qwen)
    dp = {k: v for k, v in qwen_text.items() if k.startswith("semblo_")}
    dp["semblo_index_dim"] = int(text["head_dim"])
    dp["semblo_layer_indices"] = attention_layers
    dp.pop("semblo_stop_after_layer", None)
    dp["semblo_runtime_weights_file"] = RUNTIME_WEIGHTS_FILE
    dp["semblo_runtime_manifest_file"] = RUNTIME_MANIFEST_FILE

    for kind in ("dense", "raw-keys"):
        out = a.out_root / kind
        if out.exists():
            if not a.force:
                raise SystemExit(f"{out} exists; pass --force")
            shutil.rmtree(out)
        out.mkdir(parents=True)
        linked = link_snapshot(snapshot, out)
        config = json.loads(json.dumps(cfg))
        prov = [f"# {kind}: Qwen3.5 MoE ({snapshot.name}) prepared for the vLLM runtime", "",
                f"Snapshot: `{snapshot}` (weights and tokenizer files are relative symlinks: {', '.join(linked[:6])}, ...)", "",
                "## config.json against the snapshot's", ""]
        if kind == "dense":
            prov.append("- unchanged: the same JSON, re-serialised with sorted keys")
        else:
            config["architectures"] = [ARCH]
            config["text_config"].update(dp)
            prov += [f"- architectures = ['{ARCH}']",
                     f"- text_config: semblo_* keys copied from `{a.qwen_raw_keys_config}` (the Qwen3.5-4B raw-keys checkpoint), then",
                     f"  - semblo_index_dim = {text['head_dim']} (this model's head_dim; identity routing)",
                     f"  - semblo_layer_indices = {attention_layers} (the full_attention entries of layer_types)",
                     "- runtime weights: identity key router, zero query residual, uniform mixture (the paper's raw-keys arm), built from the router modules"]
        (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if kind == "raw-keys":
            tensors = identity_router_tensors(text, dp, attention_layers)
            sha = write_runtime_weights(out, tensors, config, snapshot)
            vllm_manifest = {
                "architecture": ARCH, "base_checkpoint": str(snapshot), "link_mode": "symlink",
                "prepared_config_sha256": sha256_file(out / "config.json"),
                "runtime_weights": {"weights_sha256": sha, "tensor_names": sorted(tensors)},
                "untrained_router_arm": {"arm": "raw-keys", "source": "scripts/prepare_qwen3_5_moe_checkpoint.py", "index_dim": int(text["head_dim"])},
                "weights_sha256": sha,
            }
            (out / "semblo_vllm_manifest.json").write_text(json.dumps(vllm_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            first = attention_layers[0]
            prov.append(f"- {len(tensors)} runtime tensors, weights sha256 {sha[:16]}; shapes: " + ", ".join(
                f"{k.split('semblo_')[1]} {list(v.shape)}" for k, v in sorted(tensors.items()) if k.startswith(f"layers.{first}.")))
        (out / "PROVENANCE.md").write_text("\n".join(prov) + "\n")
        print(f"{kind}: {out}")
    print("full-attention layers:", attention_layers, "| heads", text["num_attention_heads"], "/", text["num_key_value_heads"], "| head_dim", text["head_dim"])


if __name__ == "__main__":
    main()
