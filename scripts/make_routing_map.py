#!/usr/bin/env python3
"""Make a checkpoint whose routing map is not the identity.

The routing-key comparisons of the paper replace the identity map W [Hkv, I, D]
of every full-attention layer (semblo/vllm_plugin/routing_map.py) with:

  nope-dims         the identity with the rotary dimensions zeroed: routing on the
                    192 non-rotary dimensions of Qwen3.5's keys
  pca               per layer and KV head, the top ``--index-dim`` eigenvectors of
                    the uncentred Gram matrix of pre-RoPE keys from a corpus
                    (``--keys-dir``, written by scripts/dump_corpus_keys.py);
                    ``--pca-rank`` below ``--index-dim`` zero-pads the rest
  random            a random semi-orthogonal map to ``--index-dim`` dimensions
  raw-keys-doubled  the identity stacked twice ([k ; k], ``--index-dim`` = 2 D),
                    for the min/max block descriptor (SEMBLO_BLOCK_DESCRIPTOR=minmax)

The source is a checkpoint made by scripts/prepare_qwen3_5_checkpoint.py.  The
output is written next to it, so its relative links to the base model stay valid:

    python scripts/make_routing_map.py --src checkpoints/qwen3_5_4b/raw-keys --arm pca \\
        --index-dim 32 --keys-dir runs/corpus_keys
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from semblo.vllm_plugin.runtime_weights import RUNTIME_WEIGHTS_FILE  # noqa: E402
from prepare_nemotron_checkpoint import write_runtime_weights  # noqa: E402

ARMS = ("nope-dims", "pca", "random", "raw-keys-doubled")


def _semi_orthogonal(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    """A [rows, cols] map with orthonormal rows (rows <= cols)."""
    a = torch.randn(cols, rows, generator=generator, dtype=torch.float64)
    q, _ = torch.linalg.qr(a)
    return q.T.contiguous().to(torch.float32)


def _pca_basis_from_gram(gram: np.ndarray, index_dim: int, rank: int | None = None) -> torch.Tensor:
    """Top-``rank`` eigenvectors of an uncentred [D, D] Gram matrix as rows [index_dim, D].

    Rows past ``rank`` are zero: the scoring kernel needs the routing vector to be
    at least 16 wide.
    """
    rank = index_dim if rank is None else int(rank)
    _, evecs = np.linalg.eigh(gram.astype(np.float64))
    top = evecs[:, -rank:][:, ::-1]
    basis = np.zeros((index_dim, gram.shape[0]), dtype=np.float64)
    basis[:rank] = top.T
    return torch.from_numpy(np.ascontiguousarray(basis)).float()


def _layer_grams(keys_dir: pathlib.Path) -> dict[int, tuple[np.ndarray, int]]:
    """layer -> (Gram [Hkv, D, D], rows) over every keys_epoch*_layer*.npz of that layer."""
    grams: dict[int, np.ndarray] = {}
    rows: dict[int, int] = {}
    for f in sorted(keys_dir.glob("keys_epoch*_layer*.npz")):
        m = re.search(r"keys_epoch(\d+)_layer(\d+)\.npz$", f.name)
        if not m:
            continue
        layer = int(m.group(2))
        with np.load(f) as z:
            if "raw" not in z:
                raise SystemExit(f"{f} has no pre-RoPE keys ('raw')")
            x = np.asarray(z["raw"], dtype=np.float32)                 # [L, Hkv, D]
        xt = np.ascontiguousarray(x.transpose(1, 0, 2))                  # [Hkv, L, D]
        g = np.matmul(xt.transpose(0, 2, 1), xt)                         # [Hkv, D, D]
        grams[layer] = grams[layer] + g if layer in grams else g
        rows[layer] = rows.get(layer, 0) + int(x.shape[0])
    if not grams:
        raise SystemExit(f"no keys_epoch*_layer*.npz under {keys_dir}")
    return {k: (grams[k], rows[k]) for k in grams}


def build(src: pathlib.Path, dst: pathlib.Path, arm: str, *, index_dim: int | None,
          seed: int, keys_dir: pathlib.Path | None, pca_rank: int | None, force: bool) -> None:
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} exists; pass --force")
        shutil.rmtree(dst)
    shutil.copytree(src, dst, symlinks=True)
    config = json.loads((dst / "config.json").read_text())
    text = config.get("text_config", config)
    tensors = load_file(str(dst / RUNTIME_WEIGHTS_FILE), device="cpu")
    generator = torch.Generator().manual_seed(seed)
    grams = _layer_grams(keys_dir) if arm == "pca" else {}
    out: dict[str, torch.Tensor] = {}
    for name, weight in sorted(tensors.items()):
        if not name.endswith("semblo_key_router.weight"):
            continue
        heads, _, head_dim = weight.shape
        eye = torch.eye(head_dim, dtype=torch.float32)
        if arm == "nope-dims":
            # Qwen3.5 rotates the first head_dim * partial_rotary_factor dimensions.
            rotary = int(round(head_dim * float(text["rope_parameters"]["partial_rotary_factor"])))
            eye[:rotary] = 0
            new = eye.expand(heads, head_dim, head_dim)
        elif arm == "raw-keys-doubled":
            new = torch.cat([eye, eye], dim=0).expand(heads, 2 * head_dim, head_dim)
        elif arm == "random":
            if index_dim is None:
                raise SystemExit("--arm random needs --index-dim")
            new = torch.stack([_semi_orthogonal(index_dim, head_dim, generator) for _ in range(heads)])
        else:
            if index_dim is None or keys_dir is None:
                raise SystemExit("--arm pca needs --index-dim and --keys-dir")
            layer = int(re.search(r"layers\.(\d+)\.", name).group(1))
            if layer not in grams:
                raise SystemExit(f"the key dump has no layer {layer} (it has {sorted(grams)})")
            gram, _rows = grams[layer]
            new = torch.stack([_pca_basis_from_gram(gram[h], index_dim, pca_rank) for h in range(heads)])
        out[name] = new.contiguous().to(torch.float32)
    text["semblo_index_dim"] = int(next(iter(out.values())).shape[1])
    (dst / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    sha = write_runtime_weights(dst, out, config, src, script="scripts/make_routing_map.py")
    note = {"arm": arm, "source": str(src), "index_dim": text["semblo_index_dim"], "seed": seed}
    if arm == "pca":
        note.update(keys_dir=str(keys_dir), keys_rows={str(k): v[1] for k, v in sorted(grams.items())},
                    pca_rank=pca_rank)
    (dst / "ROUTING_MAP.json").write_text(json.dumps(note, indent=2) + "\n")
    for stale in ("semblo_vllm_manifest.json", "PROVENANCE.md"):
        (dst / stale).unlink(missing_ok=True)
    print(f"{arm}: {dst} (index_dim {text['semblo_index_dim']}, runtime weights sha256 {sha[:16]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=pathlib.Path, required=True)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--dst", type=pathlib.Path, default=None, help="default: <src>/../<arm>[<index-dim>]")
    ap.add_argument("--index-dim", type=int, default=None)
    ap.add_argument("--pca-rank", type=int, default=None)
    ap.add_argument("--keys-dir", type=pathlib.Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dst = a.dst or a.src.parent / f"{a.arm}{a.index_dim or ''}"
    build(a.src, dst, a.arm, index_dim=a.index_dim, seed=a.seed, keys_dir=a.keys_dir,
          pca_rank=a.pca_rank, force=a.force)


if __name__ == "__main__":
    main()
