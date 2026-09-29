#!/usr/bin/env python3
"""Fit one codebook per (layer, cluster count) from a corpus (the fixed-codebook comparison).

The method fits a spherical k-means partition per prompt.  This fits the
centroids once on a corpus instead; SEMBLO_FIXED_CODEBOOK=<file> then uses them
in place of the per-prompt fit, with nothing else changed.

Input is whatever SEMBLO_DUMP_PROJECTED_KEYS wrote: one
``keys_epoch<E>_layer<L>.npz`` per (prompt, layer), holding the projected keys
exactly as ``ProductPostingIndex.build`` received them.  Points are taken in
the routing space the runtime clusters in, through the same function the
runtime calls.

    python scripts/fit_corpus_codebook.py --dumps runs/probes/corpus_keys \
        --clusters 512 --out runs/probes/corpus_codebook_c512.pt

The cluster count is not free to choose: the runtime derives it from the
prompt length (C = next power of two at or above L/u, floored by
SEMBLO_FLAT_MIN_CLUSTERS_LOG2), so the codebook has to be fitted at the C the
evaluation will actually ask for.  --clusters may be given more than once.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from semblo.inference.flat_index import (  # noqa: E402
    fit_flat_codebook, routing_space,
)

NAME = re.compile(r"keys_epoch(\d+)_layer(\d+)\.npz$")


def dumps_by_layer(root: pathlib.Path) -> dict[int, list[pathlib.Path]]:
    out: dict[int, list[pathlib.Path]] = {}
    for p in sorted(root.glob("keys_epoch*_layer*.npz")):
        m = NAME.search(p.name)
        if m:
            out.setdefault(int(m.group(2)), []).append(p)
    return out


def sample_points(path: pathlib.Path, per_prompt: int, seed: int,
                  device: str) -> torch.Tensor:
    """[Hkv, n, I] points from one prompt, in the runtime's clustering space."""
    with np.load(path, mmap_mode="r") as z:
        arr = z["projected"]                      # [L, Hkv, I], float16
        length = arr.shape[0]
        rng = np.random.default_rng(seed)
        take = min(per_prompt, length)
        # sorted so the mmap read stays roughly sequential
        rows = np.sort(rng.choice(length, size=take, replace=False))
        chunk = torch.from_numpy(np.ascontiguousarray(arr[rows]))
    keys = chunk.to(device=device, dtype=torch.float32)      # [n, Hkv, I]
    heads = keys.permute(1, 0, 2).contiguous()               # [Hkv, n, I]
    return torch.stack([routing_space(h) for h in heads])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clusters", type=int, nargs="+", required=True)
    ap.add_argument("--per-prompt-sample", type=int, default=65536)
    ap.add_argument("--iters", type=int, default=25,
                    help="more than the runtime's 8: this is fitted once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-prompts", type=int, default=0,
                    help="0 uses every dump found")
    args = ap.parse_args()

    root = pathlib.Path(args.dumps)
    by_layer = dumps_by_layer(root)
    if not by_layer:
        raise SystemExit(f"no keys_epoch*_layer*.npz under {root}")

    codebooks: dict[tuple[int, int], torch.Tensor] = {}
    meta = {
        "dumps": str(root),
        "prompts_per_layer": {},
        "per_prompt_sample": args.per_prompt_sample,
        "iters": args.iters,
        "seed": args.seed,
        # An offline codebook is only usable in the space the runtime scores
        # in, so this records which one it was fitted in.
        "space": os.environ.get("SEMBLO_ROUTING_NORMALIZE", "whole"),
    }
    for layer, paths in sorted(by_layer.items()):
        if args.max_prompts:
            paths = paths[: args.max_prompts]
        pooled = torch.cat(
            [sample_points(p, args.per_prompt_sample, args.seed + i, args.device)
             for i, p in enumerate(paths)], dim=1)          # [Hkv, N, I]
        meta["prompts_per_layer"][layer] = [p.name for p in paths]
        num_heads, num_points, index_dim = pooled.shape
        for c in args.clusters:
            if num_points < 32 * c:
                print(f"   layer {layer}: {num_points} points for C={c} is "
                      f"under 32 per centroid; the fit will be noisy")
            gen = torch.Generator(device=pooled.device)
            gen.manual_seed(args.seed + 1000 * layer + c)
            book = torch.stack([
                fit_flat_codebook(pooled[h], c, num_iters=args.iters,
                                  sample_size=num_points, generator=gen)
                for h in range(num_heads)])                  # [Hkv, C, I]
            codebooks[(layer, c)] = book.cpu()
            print(f"   layer {layer:2d}  C={c:5d}  from {num_points} points "
                  f"over {len(paths)} prompt(s) -> {tuple(book.shape)}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"codebooks": codebooks, "meta": meta}, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), "
          f"{len(codebooks)} (layer, C) entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
