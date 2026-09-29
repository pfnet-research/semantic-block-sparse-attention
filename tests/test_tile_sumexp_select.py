"""SEMBLO_CONSENSUS_SCORE=sumexp: FlashPrefill's tile score (torch_select.tile_sumexp_log_scores)."""
import math

import torch

from semblo.vllm_plugin.torch_select import mixture_log_scores, tile_sumexp_log_scores


def _inputs(rows=24, hq=4, hkv=2, c=40, df=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    vectors = torch.randn(rows, hq, 1, 2 * df, generator=g).to(torch.bfloat16)
    codebooks = torch.randn(hkv, 2, c, df, generator=g).to(torch.bfloat16)
    return vectors, codebooks


def test_one_query_tiles_are_the_per_query_score():
    vectors, codebooks = _inputs()
    got = tile_sumexp_log_scores(vectors, codebooks, tile=1, temperature=4.0)
    want = mixture_log_scores(vectors, None, codebooks, temperature=4.0, fused=False)
    assert torch.allclose(got, want, atol=1e-4, rtol=0)


def test_matches_the_summed_exponentials_written_out():
    vectors, codebooks = _inputs()
    tile, tau = 6, 2.0
    got = tile_sumexp_log_scores(vectors, codebooks, tile=tile, temperature=tau, rows_per_slice=12)
    rows, hq = vectors.shape[:2]
    hkv, _, c, _ = codebooks.shape
    group = hq // hkv
    cents = torch.cat([codebooks[:, 0], codebooks[:, 1]], dim=-1).double()
    q = vectors[:, :, 0].double()
    for b in range(rows // tile):
        for h in range(hkv):
            total = torch.zeros(c, dtype=torch.float64)
            for j in range(group):
                qh = q[b * tile:(b + 1) * tile, h * group + j]           # [tile, I]
                s = torch.exp(qh @ cents[h].T / tau).sum(dim=0)        # sum_i exp(q_i . mu_c / tau)
                total += s / s.sum()                                   # normalised over the blocks
            assert torch.allclose(got[b, h].double(), total.log(), atol=1e-4, rtol=0), (b, h)


def test_a_block_one_query_wants_outranks_its_mean():
    # two queries: one points at block 0, one at block 1; block 2 is what their MEAN points at.
    # The mean-query score ranks block 2 first; the summed exponentials keep blocks 0 and 1 above it.
    df = 2
    codebooks = torch.zeros(1, 2, 3, df)
    codebooks[0, 0, 0] = torch.tensor([1.0, 0.0]); codebooks[0, 0, 1] = torch.tensor([0.0, 1.0])
    codebooks[0, 0, 2] = torch.tensor([0.75, 0.75])
    codebooks = codebooks.to(torch.bfloat16)
    vectors = torch.zeros(2, 1, 1, 2 * df)
    vectors[0, 0, 0, :2] = torch.tensor([4.0, 0.0]); vectors[1, 0, 0, :2] = torch.tensor([0.0, 4.0])
    vectors = vectors.to(torch.bfloat16)
    tile = tile_sumexp_log_scores(vectors, codebooks, tile=2, temperature=1.0)[0, 0]
    mean = mixture_log_scores(vectors.float().mean(0, keepdim=True).to(torch.bfloat16), None, codebooks,
                              temperature=1.0, fused=False)[0, 0]
    assert int(mean.argmax()) == 2
    assert tile[0] > tile[2] and tile[1] > tile[2]
    assert math.isfinite(float(tile.sum()))


def test_max_reduce_is_the_largest_logit_per_tile():
    vectors, codebooks = _inputs()
    tile, tau = 6, 2.0
    got = tile_sumexp_log_scores(vectors, codebooks, tile=tile, temperature=tau, reduce="max")
    rows, hq = vectors.shape[:2]
    hkv, _, c, _ = codebooks.shape
    group = hq // hkv
    cents = torch.cat([codebooks[:, 0], codebooks[:, 1]], dim=-1).double()
    q = vectors[:, :, 0].double()
    for b in range(rows // tile):
        for h in range(hkv):
            total = torch.zeros(c, dtype=torch.float64)
            for j in range(group):
                logit = (q[b * tile:(b + 1) * tile, h * group + j] @ cents[h].T / tau).amax(dim=0)
                total += torch.softmax(logit, dim=0)
            assert torch.allclose(got[b, h].double(), total.log(), atol=1e-4, rtol=0), (b, h)


def test_the_kernel_matches_the_torch_form_on_gpu():
    import pytest
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from semblo.vllm_plugin.tile_scores import tile_log_scores_triton
    g = torch.Generator(device="cuda").manual_seed(0)
    # the last case has logits in the tens (temperature 1), where an unstable logsumexp would overflow
    for tile, c, reduce, scale, tau in ((64, 1000, "sumexp", 0.1, 16.0), (32, 4096, "max", 0.1, 16.0),
                                        (16, 256, "sumexp", 0.1, 16.0), (64, 777, "sumexp", 1.0, 1.0),
                                        (64, 777, "max", 1.0, 1.0)):
        vectors = torch.randn(tile * 5, 16, 1, 256, generator=g, device="cuda").to(torch.bfloat16)
        codebooks = (torch.randn(4, 2, c, 128, generator=g, device="cuda") * scale).to(torch.bfloat16)
        for heads in ("perhead", "packed"):
            got = tile_log_scores_triton(vectors, codebooks, tile=tile, temperature=tau, reduce=reduce, heads=heads)
            want = tile_sumexp_log_scores(vectors, codebooks, tile=tile, temperature=tau, reduce=reduce, heads=heads)
            assert torch.isfinite(got).all()
            assert torch.allclose(got, want, atol=2e-3, rtol=0), (tile, c, reduce, heads, float((got - want).abs().max()))


def test_packed_heads_are_one_sum_over_positions_and_heads():
    # FlashPrefill V2 (PackGQA): Score(I, J) = sum_{i,h} exp(q_ih . mu_J / tau) normalised over J
    vectors, codebooks = _inputs()
    tile, tau = 6, 2.0
    got = tile_sumexp_log_scores(vectors, codebooks, tile=tile, temperature=tau, heads="packed")
    rows, hq = vectors.shape[:2]
    hkv, _, c, _ = codebooks.shape
    group = hq // hkv
    cents = torch.cat([codebooks[:, 0], codebooks[:, 1]], dim=-1).double()
    q = vectors[:, :, 0].double()
    for b in range(rows // tile):
        for h in range(hkv):
            qh = q[b * tile:(b + 1) * tile, h * group:(h + 1) * group].reshape(-1, q.shape[-1])
            s = torch.exp(qh @ cents[h].T / tau).sum(dim=0)
            assert torch.allclose(got[b, h].double(), (s / s.sum()).log(), atol=1e-4, rtol=0), (b, h)

