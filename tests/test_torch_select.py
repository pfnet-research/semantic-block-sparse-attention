"""Selection as a GEMM and a sort selects what the selection rule says.

The reference is written directly from the cluster score and the budget rule,
as a loop, and shares no code with the selector.
"""
from __future__ import annotations

import math

import pytest

import torch

from semblo.vllm_plugin.query_tiles import unpack_qmask
from semblo.vllm_plugin.torch_select import budget_select_bitmask


def _reference(vectors, logmix, codebooks, causal_sizes, *, temperature, beta,
               target_tokens, g_union):
    """[sel_q, Hkv, C] bool, by the rule, one query at a time."""
    sel_q, num_q_heads, rank, _ = vectors.shape
    num_kv_heads, _, num_clusters, df = codebooks.shape
    group = num_q_heads // num_kv_heads
    out = torch.zeros(sel_q, num_kv_heads, num_clusters, dtype=torch.bool)
    for q in range(sel_q):
        block = min(q // g_union, causal_sizes.shape[0] - 1)
        for h in range(num_kv_heads):
            total = torch.zeros(num_clusters, dtype=torch.float64)
            for g in range(group):
                for r in range(rank):
                    vec = vectors[q, h * group + g, r].double()
                    cent = torch.cat([codebooks[h, 0].double(),
                                      codebooks[h, 1].double()], dim=-1)
                    logits = (cent @ vec) / temperature
                    total += (torch.softmax(logits, dim=0)
                              * math.exp(float(logmix[q, h * group + g, r])))
            sizes = causal_sizes[block, h].double()
            score = total.clamp_min(1e-30).log() - beta * torch.log1p(sizes)
            score[sizes <= 0] = -float("inf")
            order = torch.argsort(score, descending=True, stable=True)
            got = 0.0
            for c in order.tolist():
                if not math.isfinite(score[c]):
                    break
                if got >= target_tokens:
                    break
                out[q, h, c] = True
                got += float(sizes[c])
    return out


def _case(seed=0, sel_q=6, group=2, num_kv_heads=2, rank=1, df=8,
          num_clusters=64, g_union=2):
    torch.manual_seed(seed)
    num_q_heads = num_kv_heads * group
    nb = (sel_q + g_union - 1) // g_union
    vectors = torch.randn(sel_q, num_q_heads, rank, 2 * df).to(torch.bfloat16)
    logmix = torch.zeros(sel_q, num_q_heads, rank)
    codebooks = torch.randn(num_kv_heads, 2, num_clusters, df).to(torch.bfloat16)
    sizes = torch.randint(0, 40, (nb, num_kv_heads, num_clusters),
                          dtype=torch.int32)
    sizes[:, :, ::7] = 0                      # some clusters are not visible
    bitmask = torch.zeros(nb, num_kv_heads, num_clusters // 32,
                          dtype=torch.int32)
    qmask = torch.zeros(sel_q, num_kv_heads, num_clusters // 32,
                        dtype=torch.int32)
    return vectors, logmix, codebooks, sizes, bitmask, qmask, g_union


def test_it_selects_what_the_rule_selects():
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case()
    counts, tokens = budget_select_bitmask(
        vectors, logmix, codebooks, sizes, bitmask,
        temperature=0.05, beta=0.25, target_tokens=200, g_union=g, qmask=qmask)
    want = _reference(vectors, logmix, codebooks, sizes,
                      temperature=0.05, beta=0.25, target_tokens=200,
                      g_union=g)
    got = unpack_qmask(qmask, sizes.shape[-1])
    # bf16 scoring can swap two clusters whose scores agree to bf16, so the
    # test requires the same count everywhere and the same set almost
    # everywhere.
    assert torch.equal(counts, want.sum(-1).to(torch.int32))
    disagree = (got != want).sum().item()
    assert disagree <= 2 * sizes.shape[1], f"{disagree} clusters differ"
    blocks = torch.arange(want.shape[0]) // g
    assert torch.equal(tokens, (want * sizes[blocks]).sum(-1).to(torch.int32))


def test_a_cluster_with_no_visible_token_is_never_selected():
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case(seed=3)
    budget_select_bitmask(
        vectors, logmix, codebooks, sizes, bitmask,
        temperature=0.05, beta=0.25, target_tokens=10_000, g_union=g,
        qmask=qmask)
    got = unpack_qmask(qmask, sizes.shape[-1])
    for q in range(got.shape[0]):
        block = q // g
        assert not got[q][sizes[block] <= 0].any()


def test_the_budget_is_reached_and_not_wildly_overshot():
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case(seed=5)
    budget = 300
    counts, tokens = budget_select_bitmask(
        vectors, logmix, codebooks, sizes, bitmask,
        temperature=0.05, beta=0.25, target_tokens=budget, g_union=g,
        qmask=qmask)
    got = unpack_qmask(qmask, sizes.shape[-1])
    for q in range(got.shape[0]):
        block = q // g
        for h in range(got.shape[1]):
            have = int((got[q, h] * sizes[block, h]).sum())
            visible = int(sizes[block, h].sum())
            assert have >= min(budget, visible)
            # whole clusters are kept, so the overshoot is at most one of them
            assert have - budget <= int(sizes[block, h].max()) or have == visible


def test_the_block_mask_is_the_union_of_its_queries():
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case(seed=7)
    budget_select_bitmask(
        vectors, logmix, codebooks, sizes, bitmask,
        temperature=0.05, beta=0.25, target_tokens=200, g_union=g, qmask=qmask)
    per_query = unpack_qmask(qmask, sizes.shape[-1])
    per_block = unpack_qmask(bitmask, sizes.shape[-1])
    for b in range(per_block.shape[0]):
        rows = per_query[b * g:(b + 1) * g]
        assert torch.equal(per_block[b], rows.any(dim=0))


def test_one_query_per_block_keeps_the_masks_equal():
    """Consensus mode: the caller pooled the queries, so g_union is one."""
    vectors, logmix, codebooks, sizes, bitmask, qmask, _ = _case(
        seed=9, sel_q=4, g_union=1)
    sizes = sizes[:1].expand(4, *sizes.shape[1:]).contiguous()
    bitmask = torch.zeros(4, sizes.shape[1], sizes.shape[2] // 32,
                          dtype=torch.int32)
    budget_select_bitmask(
        vectors, logmix, codebooks, sizes, bitmask,
        temperature=0.05, beta=0.25, target_tokens=200, g_union=1, qmask=qmask)
    assert torch.equal(bitmask, qmask)


def test_the_max_combine_ignores_the_temperature(monkeypatch):
    """Taking the best head of the raw scores makes tau a monotone scale.

    This holds with the size penalty off (beta=0).  Otherwise tau sets the
    balance between the score and the penalty.
    """
    monkeypatch.setenv("SEMBLO_SELECT_COMBINE", "max")
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case(seed=11)
    first = None
    for temperature in (0.01, 1.0, 100.0):
        bitmask.zero_()
        budget_select_bitmask(
            vectors, logmix, codebooks, sizes, bitmask,
            temperature=temperature, beta=0.0, target_tokens=200,
            g_union=g, qmask=qmask)
        got = unpack_qmask(qmask, sizes.shape[-1]).clone()
        if first is None:
            first = got
        else:
            assert torch.equal(got, first), f"tau={temperature} moved it"


def test_the_max_combine_is_not_the_sum_combine(monkeypatch):
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case(seed=13)
    monkeypatch.setenv("SEMBLO_SELECT_COMBINE", "sum")
    budget_select_bitmask(
        vectors, logmix, codebooks, sizes, bitmask,
        temperature=0.05, beta=0.0, target_tokens=200, g_union=g, qmask=qmask)
    as_sum = unpack_qmask(qmask, sizes.shape[-1]).clone()
    monkeypatch.setenv("SEMBLO_SELECT_COMBINE", "max")
    bitmask.zero_()
    budget_select_bitmask(
        vectors, logmix, codebooks, sizes, bitmask,
        temperature=0.05, beta=0.0, target_tokens=200, g_union=g, qmask=qmask)
    as_max = unpack_qmask(qmask, sizes.shape[-1])
    assert not torch.equal(as_sum, as_max)


def test_an_unknown_combine_is_refused(monkeypatch):
    import pytest

    monkeypatch.setenv("SEMBLO_SELECT_COMBINE", "median")
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case(seed=15)
    with pytest.raises(RuntimeError, match="sum or max"):
        budget_select_bitmask(
            vectors, logmix, codebooks, sizes, bitmask,
            temperature=0.05, beta=0.25, target_tokens=200, g_union=g,
            qmask=qmask)


def test_the_exact_bound_for_equal_blocks(monkeypatch):
    """top-(K/B + 1) is exact when every block holds B visible tokens.

    Adjacent blocks hold B tokens each, with one partial block where the
    causal cut falls and nothing after it.  The budget therefore covers
    ceil(K/B) blocks, or one more when the partial block is in the set, and
    only that many need to be sorted.  The bound must not be derived from the
    smallest positive size, which the partial block can make 1.
    """
    torch.manual_seed(21)
    heads, df, num_clusters, block = 2, 8, 128, 16
    sel_q, group, g_union = 8, 2, 4
    budget = 64                                   # four whole blocks
    nb = (sel_q + g_union - 1) // g_union
    vectors = torch.randn(sel_q, heads * group, 1, 2 * df).to(torch.bfloat16)
    logmix = torch.zeros(sel_q, heads * group, 1)
    codebooks = torch.randn(heads, 2, num_clusters, df).to(torch.bfloat16)
    # every block full up to the cut, one partial, then nothing
    sizes = torch.zeros(nb, heads, num_clusters, dtype=torch.int32)
    for b in range(nb):
        seen = 40 + 7 * b                         # blocks fully inside the cut
        sizes[b, :, :seen] = block
        sizes[b, :, seen] = 5                     # the block the cut falls in
    # tau=1: at 0.05 the softmax over 128 clusters saturates on random
    # vectors, many clusters share the floor score, and the two paths break
    # the tie differently.
    args = dict(temperature=1.0, beta=0.25, target_tokens=budget,
                g_union=g_union)

    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", str(block))
    qa = torch.zeros(sel_q, heads, num_clusters // 32, dtype=torch.int32)
    ba = torch.zeros(nb, heads, num_clusters // 32, dtype=torch.int32)
    ca, ta = budget_select_bitmask(vectors, logmix, codebooks, sizes, ba,
                                   qmask=qa, **args)

    monkeypatch.delenv("SEMBLO_BLOCK_TOKENS")        # the full-order path
    qb = torch.zeros_like(qa)
    bb = torch.zeros_like(ba)
    cb, tb = budget_select_bitmask(vectors, logmix, codebooks, sizes, bb,
                                   qmask=qb, **args)

    assert torch.equal(qa, qb), "the bounded walk chose a different set"
    assert torch.equal(ba, bb)
    assert torch.equal(ca, cb) and torch.equal(ta, tb)
    # and it really was bounded: four or five blocks, not a hundred
    assert int(ca.max()) <= budget // block + 1


def test_the_bound_refuses_when_the_blocks_are_not_equal(monkeypatch):
    """A block size that does not match the partition raises instead of
    selecting fewer tokens than the budget."""
    torch.manual_seed(23)
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case(seed=23)
    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", "10000")   # far too large
    with pytest.raises(RuntimeError, match="did not reach the budget"):
        budget_select_bitmask(
            vectors, logmix, codebooks, sizes, bitmask,
            temperature=0.05, beta=0.25, target_tokens=5000, g_union=g,
            qmask=qmask)


def _gpu_scorer_case():
    """A GPU case both scorers accept, or a skip."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    try:
        from semblo.vllm_plugin import torch_select as ts
    except ImportError:
        pytest.skip("needs triton")
    if ts._qtiled is None:
        pytest.skip("needs the fused scorer")
    torch.manual_seed(29)
    device = "cuda"
    # df >= 16: the scorer's tl.dot does not accept a smaller K.
    sel_q, group, heads, rank, df, C, g_union = 64, 2, 2, 1, 64, 256, 4
    nb = (sel_q + g_union - 1) // g_union
    vectors = torch.randn(sel_q, heads * group, rank, 2 * df,
                          device=device).to(torch.bfloat16)
    logmix = torch.zeros(sel_q, heads * group, rank, device=device)
    codebooks = torch.randn(heads, 2, C, df, device=device).to(torch.bfloat16)
    sizes = torch.randint(0, 40, (nb, heads, C), dtype=torch.int32,
                          device=device)
    sizes[:, :, ::7] = 0
    return ts, vectors, logmix, codebooks, sizes, nb, C, g_union


def test_the_fused_scorer_and_the_torch_one_compute_the_same_scores():
    """The fused kernel computes the same log-scores as the torch form.

    Both paths take the same bf16 products and accumulate in fp32.  Only the
    summation order differs (the kernel's online softmax runs over cluster
    tiles), so the log-scores must agree to a small tolerance everywhere.
    """
    ts, vectors, logmix, codebooks, _, _, _, _ = _gpu_scorer_case()
    kernel = ts.mixture_log_scores(vectors, logmix, codebooks,
                                   temperature=1.0, fused=True)
    plain = ts.mixture_log_scores(vectors, logmix, codebooks,
                                  temperature=1.0, fused=False)
    assert kernel.shape == plain.shape
    assert torch.isfinite(kernel).all() and torch.isfinite(plain).all()
    diff = (kernel - plain).abs()
    # log-scores are O(1..10) and summation-order noise is about 1e-5, while
    # a difference in the rule would be O(1).
    assert float(diff.max()) < 1e-2, (
        f"max |kernel - torch| log-score = {float(diff.max()):.3e} at "
        f"{tuple(int(i) for i in (diff == diff.max()).nonzero()[0])}; "
        f"mean {float(diff.mean()):.3e}")


def test_the_fused_scorer_and_the_torch_one_disagree_only_at_ties():
    """Where the two selections differ, the cluster's score is within
    summation-order noise of the budget cut."""
    ts, vectors, logmix, codebooks, sizes, nb, C, g_union = _gpu_scorer_case()
    args = dict(temperature=1.0, beta=0.0, target_tokens=200, g_union=g_union)
    sel = {}
    for tag, patched in (("kernel", ts._qtiled), ("torch", None)):
        saved, ts._qtiled = ts._qtiled, patched
        try:
            q = torch.zeros(vectors.shape[0], codebooks.shape[0], C // 32,
                            dtype=torch.int32, device=vectors.device)
            b = torch.zeros(nb, codebooks.shape[0], C // 32,
                            dtype=torch.int32, device=vectors.device)
            ts.budget_select_bitmask(vectors, logmix, codebooks, sizes, b,
                                     qmask=q, **args)
            sel[tag] = unpack_qmask(q, C).clone().bool()
        finally:
            ts._qtiled = saved
    plain = ts.mixture_log_scores(vectors, logmix, codebooks,
                                  temperature=1.0, fused=False)
    block_of = (torch.arange(vectors.shape[0], device=vectors.device)
                // g_union).clamp_max(nb - 1)
    visible = sizes[block_of].float()                       # [Q, Hkv, C]
    plain = plain.masked_fill(visible <= 0, float("-inf"))
    # the torch cut: the weakest cluster it selected, per (query, kv head)
    cut = plain.masked_fill(~sel["torch"], float("inf")).amin(dim=-1)
    disagree = sel["kernel"] != sel["torch"]
    if bool(disagree.any()):
        gap = (plain - cut[..., None]).abs()[disagree]
        assert float(gap.max()) < 1e-2, (
            f"{int(disagree.sum())} memberships differ and the widest gap to "
            f"the torch cut is {float(gap.max()):.3e}: a rule difference, "
            "not a tie")
    # and the disagreements are few in any case: at most one swapped pair
    # per (query, kv head) row
    assert int(disagree.sum()) <= 2 * disagree.shape[0] * disagree.shape[1]


def _equal_blocks_case(seed, block, num_clusters=64, sel_q=4):
    """Adjacent blocks: every visible block holds exactly ``block`` tokens."""
    vectors, logmix, codebooks, _, bitmask, qmask, g = _case(seed=seed, sel_q=sel_q, num_clusters=num_clusters)
    nb = bitmask.shape[0]
    sizes = torch.full((nb, codebooks.shape[0], num_clusters), block, dtype=torch.int32)
    return vectors, logmix, codebooks, sizes, bitmask, qmask, g


def test_the_sink_blocks_are_always_selected_inside_the_budget(monkeypatch):
    block, sink, budget = 4, 10, 32            # three sink blocks, eight blocks in all
    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", str(block))
    monkeypatch.setenv("SEMBLO_SINK_TOKENS", str(sink))
    for seed in range(4):
        vectors, logmix, codebooks, sizes, bitmask, qmask, g = _equal_blocks_case(seed, block)
        counts, tokens = budget_select_bitmask(
            vectors, logmix, codebooks, sizes, bitmask,
            temperature=0.05, beta=0.0, target_tokens=budget, g_union=g, qmask=qmask)
        got = unpack_qmask(qmask, sizes.shape[-1])
        assert got[..., :3].all(), "the first ceil(S/B) blocks must be selected for every query and head"
        assert torch.equal(counts, torch.full_like(counts, budget // block))   # the sink is inside the budget
        assert torch.equal(tokens, torch.full_like(tokens, budget))


def test_without_the_sink_the_selection_is_unchanged(monkeypatch):
    block, budget = 4, 32
    monkeypatch.setenv("SEMBLO_BLOCK_TOKENS", str(block))
    monkeypatch.delenv("SEMBLO_SINK_TOKENS", raising=False)
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _equal_blocks_case(7, block)
    budget_select_bitmask(vectors, logmix, codebooks, sizes, bitmask,
                          temperature=0.05, beta=0.0, target_tokens=budget, g_union=g, qmask=qmask)
    plain = unpack_qmask(qmask, sizes.shape[-1]).clone()
    monkeypatch.setenv("SEMBLO_SINK_TOKENS", "0")
    qmask.zero_(); bitmask.zero_()
    budget_select_bitmask(vectors, logmix, codebooks, sizes, bitmask,
                          temperature=0.05, beta=0.0, target_tokens=budget, g_union=g, qmask=qmask)
    assert torch.equal(unpack_qmask(qmask, sizes.shape[-1]), plain)


def test_the_fitted_sink_cluster_is_selected_first_inside_the_budget(monkeypatch):
    """A fitted partition with the sink: cluster 0 holds the first S positions with a zero descriptor (state.py).

    The selector takes it for every query and head.  The rest must match the loop reference with cluster 0
    removed and the budget reduced by S."""
    sink, budget = 16, 100
    monkeypatch.delenv("SEMBLO_BLOCK_TOKENS", raising=False)
    monkeypatch.setenv("SEMBLO_SINK_TOKENS", str(sink))
    for seed in range(3):
        vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case(seed=seed)
        sizes[:, :, 0] = sink                   # the sink cluster, fully visible
        codebooks[:, :, 0] = 0                  # its descriptor is zero
        counts, tokens = budget_select_bitmask(
            vectors, logmix, codebooks, sizes, bitmask,
            temperature=0.05, beta=0.25, target_tokens=budget, g_union=g, qmask=qmask)
        got = unpack_qmask(qmask, sizes.shape[-1])
        assert got[..., 0].all(), "cluster 0 must be selected for every query and head"
        rest = sizes.clone()
        rest[:, :, 0] = 0                       # out of the reference's walk
        want = _reference(vectors, logmix, codebooks, rest, temperature=0.05, beta=0.25,
                          target_tokens=budget - sink, g_union=g)
        assert torch.equal(counts, want.sum(-1).to(torch.int32) + 1)
        disagree = (got[..., 1:] != want[..., 1:]).sum().item()
        assert disagree <= 2 * sizes.shape[1], f"{disagree} clusters differ"


def test_the_fitted_sink_is_not_forced_without_the_variable(monkeypatch):
    monkeypatch.delenv("SEMBLO_BLOCK_TOKENS", raising=False)
    monkeypatch.delenv("SEMBLO_SINK_TOKENS", raising=False)
    vectors, logmix, codebooks, sizes, bitmask, qmask, g = _case(seed=2)
    sizes[:, :, 0] = 16
    codebooks[:, :, 0] = 0
    budget_select_bitmask(vectors, logmix, codebooks, sizes, bitmask,
                          temperature=0.05, beta=0.25, target_tokens=100, g_union=g, qmask=qmask)
    want = _reference(vectors, logmix, codebooks, sizes, temperature=0.05, beta=0.25,
                      target_tokens=100, g_union=g)
    got = unpack_qmask(qmask, sizes.shape[-1])
    assert (got != want).sum().item() <= 2 * sizes.shape[1]
