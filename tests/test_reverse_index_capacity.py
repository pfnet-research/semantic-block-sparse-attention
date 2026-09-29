"""build_reverse_index sized by its upper bound gives the same index as when
sized by the entry count it read from the device.

Sizing by a capacity avoids reading the count back to the host.  The first
nnz entries of ``qvals`` and ``order`` must be identical, and every other
output unchanged.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("seed", [0, 1])
def test_capacity_sized_index_matches_the_count_sized_one(seed):
    from semblo.vllm_plugin.kv_outer import build_reverse_index
    torch.manual_seed(seed)
    q, heads, c = 300, 2, 128                      # C a power of two: the rank kernel aranges over the words
    words = c // 32
    # each query selects about one cluster in eight; about one in ten is not live
    sel = (torch.rand(q, heads, c, device="cuda") < 0.125)
    live = (torch.rand(heads, c, device="cuda") < 0.9).to(torch.int32).contiguous()
    shifts = torch.arange(32, device="cuda", dtype=torch.int32)
    qmask = ((sel.view(q, heads, words, 32).to(torch.int32) << shifts).sum(dim=-1)).to(torch.int32).contiguous()
    exact = build_reverse_index(qmask, live, num_queries=q)
    capped = build_reverse_index(qmask, live, num_queries=q, capacity=q * heads * c)
    nnz = int(exact[1].sum())
    assert exact[0].numel() == nnz
    assert capped[0].numel() == q * heads * c
    assert torch.equal(capped[0][:nnz], exact[0])            # qvals
    assert torch.equal(capped[3][:nnz], exact[3])            # order
    for a, b in zip(exact[1:3] + exact[4:], capped[1:3] + capped[4:]):
        assert torch.equal(a, b)
