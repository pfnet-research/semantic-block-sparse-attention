"""K-means++ initialization for the flat partition."""

from __future__ import annotations

from typing import Optional

import torch


def _kmeanspp_init(
    sample: torch.Tensor,
    num_centroids: int,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """k-means++ seeding on the unit sphere (D^2 sampling, cosine distance).

    Distance is 1 - cos; identical points contribute zero probability after
    the first pick, so duplicate masses draw at most one seed instead of a
    density-proportional share as with uniform seeding.
    """

    n = int(sample.shape[0])
    first = int(torch.randint(n, (1,), device=sample.device, generator=generator).item())
    chosen = [first]
    best_sim = (sample.float() @ sample[first].float().unsqueeze(-1)).squeeze(-1)
    for _ in range(num_centroids - 1):
        weights = (1.0 - best_sim).clamp_min(0.0)
        total = float(weights.sum().item())
        if total <= 0.0:
            # all remaining points coincide with a chosen seed
            idx = int(torch.randint(n, (1,), device=sample.device, generator=generator).item())
        else:
            idx = int(torch.multinomial(weights, 1, generator=generator).item())
        chosen.append(idx)
        sim = (sample.float() @ sample[idx].float().unsqueeze(-1)).squeeze(-1)
        best_sim = torch.maximum(best_sim, sim)
    return sample[torch.tensor(chosen, device=sample.device)].clone()
