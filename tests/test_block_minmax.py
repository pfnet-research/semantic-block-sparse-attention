"""The Quest-style block descriptor: the doubled-space dot product is the bound."""

import torch

from semblo.vllm_plugin import block_minmax


def _blocks(length, block, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(length, d, generator=g, dtype=torch.float64)
    assign = torch.arange(length) // block
    num = 1 << max(3, (max((length + block - 1) // block, 1) - 1).bit_length())
    return keys, assign, num


def test_descriptor_is_the_blockwise_max_and_min():
    keys, assign, num = _blocks(1000, 32, 16)
    space = torch.cat([keys, keys], dim=1)          # [k ; k], as the doubled router projects
    desc = block_minmax.block_minmax_descriptors(space, assign, num)
    assert desc.shape == (num, 32)
    for c in range(num):
        rows = keys[assign == c]
        if rows.numel() == 0:
            assert torch.equal(desc[c], torch.zeros(32, dtype=desc.dtype))
            continue
        assert torch.equal(desc[c, :16], rows.max(0).values)
        assert torch.equal(desc[c, 16:], rows.min(0).values)


def test_dot_product_in_the_doubled_space_is_the_quest_bound():
    keys, assign, num = _blocks(1000, 32, 16)
    space = torch.cat([keys, keys], dim=1)
    desc = block_minmax.block_minmax_descriptors(space, assign, num)
    q = torch.randn(7, 16, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    t = block_minmax.quest_query_vectors(torch.cat([q, q], dim=-1))
    scores = t @ desc.T                              # [7, C]
    ref = block_minmax.quest_bound_reference(q, desc[:, 16:], desc[:, :16])
    assert torch.allclose(scores, ref, rtol=0, atol=1e-12)
    # the bound is at least the score against the block mean
    mean_desc = torch.zeros(num, 16, dtype=torch.float64).index_add_(0, assign, keys) / torch.bincount(assign, minlength=num).clamp_min(1)[:, None]
    assert (scores >= q @ mean_desc.T - 1e-9).all()


def test_query_transform_halves():
    p = torch.tensor([[1.0, -2.0, 3.0, 1.0, -2.0, 3.0]])
    t = block_minmax.quest_query_vectors(p)
    assert torch.equal(t, torch.tensor([[1.0, 0.0, 3.0, 0.0, -2.0, 0.0]]))


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SEMBLO_BLOCK_DESCRIPTOR", raising=False)
    assert not block_minmax.enabled()
    monkeypatch.setenv("SEMBLO_BLOCK_DESCRIPTOR", "minmax")
    assert block_minmax.enabled()
