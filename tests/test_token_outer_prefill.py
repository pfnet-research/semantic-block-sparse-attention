"""The token-outer prefill's bookkeeping, without a GPU.

reserved_layout gives every cluster whole pages in proportion to the head's
sizes, never less than one page, and never more than the store; append_rows
puts a slice's keys after their cluster's last key, in position order.
"""
from __future__ import annotations

import torch

from semblo.vllm_plugin.token_outer_prefill import append_rows, reserved_layout


def test_reserved_layout_is_proportional_paged_and_fits():
    sizes = torch.tensor([[100, 0, 300, 20], [50, 150, 0, 200]])
    width = 16 * 200
    reserve = reserved_layout(sizes, n_ref=420, total_tokens=2000, width=width, align=16)
    assert reserve.shape == sizes.shape
    assert bool((reserve % 16 == 0).all())
    assert bool((reserve >= 16).all())                       # an empty cluster still has a page
    assert bool((reserve.sum(dim=1) <= width).all())
    # the share follows the head's proportions
    want = sizes.double() * (2000 / 420)
    assert bool((reserve.double() >= want).all())
    ratio = reserve[0, 2].item() / reserve[0, 0].item()
    assert 2.5 < ratio < 3.5


def test_append_rows_follow_the_run_in_position_order():
    indptr = torch.tensor([[0, 32, 64, 96]], dtype=torch.int32)
    sizes = torch.tensor([[5, 0, 30]], dtype=torch.int32)
    ids = torch.tensor([[2, 0, 2, 1, 0]])
    rows, counts = append_rows(ids, indptr, sizes)
    assert counts.tolist() == [[2, 1, 2]]
    # cluster 0 held 5 rows at 0..4: the two newcomers take 5 and 6 in position order
    assert rows[0, 1].item() == 5 and rows[0, 4].item() == 6
    assert rows[0, 3].item() == 32                            # the empty cluster's first row
    assert rows[0, 0].item() == 94 and rows[0, 2].item() == 95
