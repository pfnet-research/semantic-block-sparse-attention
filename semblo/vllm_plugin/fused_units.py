"""Merged-bucket ("unit") selection, and the per-block OR of query masks.

``build_bucket_units`` and ``unit_cover_masks`` coarsen the selection unit:
consecutive buckets in the packed layout are merged into units of about
``unit_tokens`` tokens, and each query's bucket selection is expanded to the
densest covering set of units under a token budget.  The result replaces the
per-query bucket bitmask after selection, so the strict-masked union body
computes unit-granular attention without a new kernel.  This path is a
quality check only; its speed is not representative.

Enable with SEMBLO_FUSED_UNIT_TOKENS=128 (optionally
SEMBLO_FUSED_UNIT_BUDGET_X=2); requires SEMBLO_FUSED_STRICT_MASK=1.
"""

from __future__ import annotations

import torch


def build_bucket_units(bucket_sizes: torch.Tensor,
                       unit_tokens: int) -> tuple[torch.Tensor, int]:
    """[H, B] total bucket sizes -> (bucket->unit id [H, B] int32, U).

    Units are runs of consecutive bucket ids whose packed span is about
    unit_tokens: a bucket belongs to unit floor(packed_start/unit_tokens),
    so every bucket is whole and units are ~unit_tokens +- one bucket."""
    csum = bucket_sizes.long().cumsum(dim=1)
    start = csum - bucket_sizes.long()
    b2u = (start // unit_tokens).to(torch.int32)
    return b2u.contiguous(), int(b2u.max().item()) + 1


def unit_cover_masks(
    qmask: torch.Tensor,         # [Q, H, B/32] int32 per-query selection
    causal_sizes: torch.Tensor,  # [NB, H, B] int32 block snapshot
    b2u: torch.Tensor,           # [H, B] int32
    num_units: int,
    g_union: int,
    budget_tokens: int,
) -> torch.Tensor:
    """Expand each query's selected-bucket set to its covering unit set.

    Units are ranked by (selected causal mass in the unit) / (unit causal
    size) and taken greedily until the cumulative unit causal size reaches
    the budget; the unit that crosses the budget is included, as in the
    bucket budget walk.  Returns the replacement qmask (same layout).  The
    caller re-derives the block bitmask by OR over each union block's rows."""
    q_count, num_heads, n_words = qmask.shape
    b_dim = n_words * 32
    device = qmask.device
    shifts = torch.arange(32, device=device, dtype=torch.int32)
    sizes_f = causal_sizes.float()                     # [NB, H, B]
    b2u_l = b2u.long().unsqueeze(0)                    # [1, H, B]
    # per-block causal size of each unit
    unit_causal = torch.zeros(
        causal_sizes.shape[0], num_heads, num_units, device=device
    ).scatter_add_(2, b2u_l.expand(causal_sizes.shape[0], -1, -1), sizes_f)
    nb_of_q = (torch.arange(q_count, device=device) // g_union)

    out = torch.empty_like(qmask)
    slab = 1024                                        # bounds the [S, H, B] fp32 temporaries
    for lo in range(0, q_count, slab):
        hi = min(lo + slab, q_count)
        bits = ((qmask[lo:hi].unsqueeze(-1) >> shifts) & 1).bool() \
            .view(hi - lo, num_heads, -1)[..., :b_dim]
        sel_mass = bits.float() * sizes_f[nb_of_q[lo:hi]]
        gain = torch.zeros(hi - lo, num_heads, num_units, device=device) \
            .scatter_add_(2, b2u_l.expand(hi - lo, -1, -1), sel_mass)
        ucs_q = unit_causal[nb_of_q[lo:hi]]            # [S, H, U]
        density = torch.where(
            gain > 0, gain / ucs_q.clamp_min(1.0),
            torch.zeros_like(gain))
        order = density.argsort(dim=2, descending=True)
        sz_sorted = ucs_q.gather(2, order)
        csum = sz_sorted.cumsum(dim=2)
        take = ((csum - sz_sorted) < budget_tokens) & \
            (density.gather(2, order) > 0)
        chosen = torch.zeros_like(density, dtype=torch.bool) \
            .scatter_(2, order, take)                  # [S, H, U]
        new_bits = chosen.gather(2, b2u_l.expand(hi - lo, -1, -1))
        out[lo:hi] = (new_bits.view(hi - lo, num_heads, n_words, 32).int()
                      << shifts).sum(dim=-1).to(torch.int32)
    return out.contiguous()


def block_bitmask_from_qmask(qmask: torch.Tensor,
                             g_union: int) -> torch.Tensor:
    """OR the per-query words over each union block -> [NB, H, W] int32."""
    q_count = qmask.shape[0]
    nb = (q_count + g_union - 1) // g_union
    pad = nb * g_union - q_count
    qm = qmask
    if pad:
        qm = torch.cat([qm, torch.zeros(pad, *qmask.shape[1:],
                                        dtype=qmask.dtype,
                                        device=qmask.device)], dim=0)
    grouped = qm.view(nb, g_union, *qmask.shape[1:])
    out = grouped[:, 0]
    for i in range(1, g_union):
        out = out | grouped[:, i]
    return out.contiguous()
