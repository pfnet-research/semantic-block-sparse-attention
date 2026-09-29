#!/usr/bin/env python3
"""Prepare Nemotron 3.5 Lightning for the vLLM 0.19.1 runtime.

Writes two sibling directories under ``--out-root``, both pointing at the
Hugging Face snapshot's weight and tokenizer files by relative symlink:

``dense/``
    the stock model, config.json unchanged (re-serialised).

``raw-keys/``
    the same model under ``SembloNemotronHForCausalLM`` with identity routing
    (the raw keys, as in the paper: the router projects with the identity, adds
    no residual and mixes uniformly). The ``semblo_*`` settings are copied from
    the Qwen3.5-4B settings, the routing width is set to this model's head
    dimension, and the attention layers are listed explicitly. The runtime
    weights and their manifests are built from the router modules, so the
    shapes are the ones the model instantiates.

Every deviation from the snapshot is listed in PROVENANCE.md next to each
config.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import sys

import torch
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semblo.vllm_plugin.identity_router import identity_router_tensors  # noqa: E402
from semblo.vllm_plugin.runtime_weights import (  # noqa: E402
    RUNTIME_MANIFEST_FILE,
    RUNTIME_SCHEMA_VERSION,
    RUNTIME_WEIGHTS_FILE,
    RuntimeWeightsManifest,
    _tensor_sha256,
    semblo_config_fields,
    sha256_file,
)

BLOCK_CHARS = {"mamba": "M", "linear_attention": "M", "moe": "E", "attention": "*", "full_attention": "*", "mlp": "-"}
# The semblo_* settings of the Qwen3.5-4B checkpoint, copied into every prepared checkpoint.
DEFAULT_QWEN_RAW_KEYS = ROOT / "configs/checkpoints/qwen3_5_raw_keys_text_config.json"


def vllm_config_from_snapshot(cfg: dict) -> tuple[dict, list[int], list[str]]:
    """The snapshot config as vLLM 0.19.1 reads it, and the attention layers.

    Under transformers 5, vLLM 0.19.1 loads this model through transformers'
    ``NemotronHConfig``, which reads ``layers_block_type`` and derives the
    ``hybrid_override_pattern`` string that the vLLM model walks. The config is
    therefore used unchanged."""
    blocks = list(cfg["layers_block_type"])
    if int(cfg.get("num_hidden_layers", len(blocks))) != len(blocks):
        raise SystemExit("num_hidden_layers does not match layers_block_type")
    attention = [i for i, b in enumerate(blocks) if b in ("attention", "full_attention")]
    pattern = "".join(BLOCK_CHARS.get(b, "?") for b in blocks)
    notes = [f"config.json is the snapshot's (layer pattern {pattern}, read by transformers' NemotronHConfig under vLLM 0.19.1)"]
    return dict(cfg), attention, notes


def link_snapshot(snapshot: pathlib.Path, out: pathlib.Path) -> list[str]:
    linked = []
    for entry in sorted(snapshot.iterdir()):
        if entry.name in ("config.json",) or entry.name.startswith("."):
            continue
        dest = out / entry.name
        dest.symlink_to(os.path.relpath(entry, out), target_is_directory=entry.is_dir())
        linked.append(entry.name)
    return linked


def write_runtime_weights(out: pathlib.Path, tensors: dict[str, torch.Tensor], config: dict, snapshot: pathlib.Path,
                          script: str = "scripts/prepare_nemotron_checkpoint.py") -> str:
    names = tuple(sorted(tensors))
    tensors = {n: tensors[n] for n in names}
    weights_path = out / RUNTIME_WEIGHTS_FILE
    save_file(tensors, str(weights_path))
    manifest = RuntimeWeightsManifest(
        schema_version=RUNTIME_SCHEMA_VERSION,
        source_checkpoint=f"identity routing built by {script} from {snapshot}",
        source_config_sha256=hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        weights_file=RUNTIME_WEIGHTS_FILE,
        weights_sha256=sha256_file(weights_path),
        tensor_names=names,
        tensor_shapes={n: list(tensors[n].shape) for n in names},
        tensor_dtypes={n: str(tensors[n].dtype) for n in names},
        tensor_sha256={n: _tensor_sha256(tensors[n]) for n in names},
        semblo_config=semblo_config_fields(config),
    )
    (out / RUNTIME_MANIFEST_FILE).write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return manifest.weights_sha256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=pathlib.Path, required=True, help="the Hugging Face snapshot directory of nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16")
    ap.add_argument("--out-root", type=pathlib.Path, required=True)
    ap.add_argument("--qwen-raw-keys-config", type=pathlib.Path, default=DEFAULT_QWEN_RAW_KEYS,
                    help="config.json whose semblo_* keys are copied (the paper's Qwen3.5-4B raw-keys checkpoint)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    snapshot = a.snapshot.resolve()
    cfg = json.loads((snapshot / "config.json").read_text())
    vcfg, attention_layers, notes = vllm_config_from_snapshot(cfg)
    qwen = json.loads(a.qwen_raw_keys_config.read_text())
    qwen_text = qwen.get("text_config", qwen)
    dp = {k: v for k, v in qwen_text.items() if k.startswith("semblo_")}
    dp["semblo_index_dim"] = int(cfg["head_dim"])
    dp["semblo_layer_indices"] = attention_layers
    dp.pop("semblo_stop_after_layer", None)

    for kind in ("dense", "raw-keys"):
        out = a.out_root / kind
        if out.exists():
            if not a.force:
                raise SystemExit(f"{out} exists; pass --force")
            shutil.rmtree(out)
        out.mkdir(parents=True)
        linked = link_snapshot(snapshot, out)
        config = dict(vcfg)
        prov = [f"# {kind}: Nemotron 3.5 Lightning prepared for vLLM 0.19.1", "",
                f"Snapshot: `{snapshot}` (weights and tokenizer files are relative symlinks: {', '.join(linked[:6])}, ...)", "",
                "## config.json against the snapshot's", ""] + [f"- {n}" for n in notes]
        if kind == "dense":
            prov.append("- unchanged: the same JSON, re-serialised with sorted keys")
        else:
            config["architectures"] = ["SembloNemotronHForCausalLM"]
            config.update(dp)
            config["semblo_runtime_weights_file"] = RUNTIME_WEIGHTS_FILE
            config["semblo_runtime_manifest_file"] = RUNTIME_MANIFEST_FILE
            prov += ["- architectures = ['SembloNemotronHForCausalLM']",
                     f"- semblo_* keys copied from `{a.qwen_raw_keys_config}` (the Qwen3.5-4B raw-keys checkpoint), then",
                     f"  - semblo_index_dim = {cfg['head_dim']} (this model's head_dim; identity routing)",
                     f"  - semblo_layer_indices = {attention_layers} (the attention layers of layers_block_type)",
                     "- runtime weights: identity key router, zero query residual, uniform mixture (the paper's raw-keys arm), built from the router modules"]
        (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        if kind == "raw-keys":
            tensors = identity_router_tensors(cfg, dp, attention_layers)
            sha = write_runtime_weights(out, tensors, config, snapshot)
            vllm_manifest = {
                "architecture": config["architectures"][0], "base_checkpoint": str(snapshot), "link_mode": "symlink",
                "prepared_config_sha256": sha256_file(out / "config.json"),
                "runtime_weights": {"weights_sha256": sha, "tensor_names": sorted(tensors)},
                "untrained_router_arm": {"arm": "raw-keys", "source": "scripts/prepare_nemotron_checkpoint.py", "index_dim": int(cfg["head_dim"])},
                "weights_sha256": sha,
            }
            (out / "semblo_vllm_manifest.json").write_text(json.dumps(vllm_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            prov.append(f"- {len(tensors)} runtime tensors, weights sha256 {sha[:16]}; shapes: " + ", ".join(f"{k.split('semblo_')[1]} {list(v.shape)}" for k, v in sorted(tensors.items()) if k.startswith("layers.%d." % attention_layers[0])))
        (out / "PROVENANCE.md").write_text("\n".join(prov) + "\n")
        print(f"{kind}: {out}")
        for n in notes:
            print("   ", n[:140])
    print("attention layers:", attention_layers)


if __name__ == "__main__":
    main()
