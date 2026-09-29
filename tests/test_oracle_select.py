"""The oracle cluster selection (SEMBLO_ORACLE_SELECT=clusters): the mass is the dense attention of
the query over the packed prompt rows below the boundary, summed per cluster, padding rows and
later positions excluded."""
import types

import pytest
import torch

from semblo.vllm_plugin import oracle_select


def _index(hkv=2, clusters=4, sizes=(5, 3, 7, 1), align=4, d=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    sizes_t = torch.tensor(sizes, dtype=torch.int64).repeat(hkv, 1)
    padded = ((sizes_t + align - 1) // align) * align
    indptr = torch.zeros(hkv, clusters + 1, dtype=torch.int64)
    indptr[:, 1:] = padded.cumsum(1)
    rows = int(indptr[0, -1]); length = int(sum(sizes))
    posting = torch.zeros(hkv, rows, dtype=torch.int32)
    order = torch.randperm(length, generator=g)                     # position of each real row, bucket-major
    for h in range(hkv):
        k = 0
        for c in range(clusters):
            for off in range(int(padded[0, c])):
                real = off < sizes[c]
                posting[h, int(indptr[h, c]) + off] = int(order[k]) if real else int(order[0])
                k += int(real)
    keys = torch.randn(rows, hkv, d, generator=g).to(torch.bfloat16)
    packed = torch.stack([keys, torch.zeros_like(keys)])
    return types.SimpleNamespace(num_kv_heads=hkv, num_buckets=clusters, bucket_indptr=indptr.to(torch.int32),
                                 bucket_sizes=sizes_t.to(torch.int32), posting_positions=posting, packed_kv=packed,
                                 prompt_length=length)


def _reference(q, index, boundary, sm_scale):
    hkv, C = index.num_kv_heads, index.num_buckets
    group = q.shape[1] // hkv
    out = torch.zeros(q.shape[0], hkv, C)
    indptr, sizes = index.bucket_indptr.tolist(), index.bucket_sizes.tolist()
    for h in range(hkv):
        rows = [(r, c) for c in range(C) for r in range(indptr[h][c], indptr[h][c] + sizes[h][c])
                if int(index.posting_positions[h, r]) < boundary]
        for i in range(q.shape[0]):
            for g in range(group):
                logits = torch.tensor([float(q[i, h * group + g].float() @ index.packed_kv[0, r, h].float()) * sm_scale for r, _ in rows])
                p = torch.softmax(logits, 0)
                for (r, c), pc in zip(rows, p):
                    out[i, h, c] += float(pc)
    return out


def test_the_oracle_mass_is_the_dense_attention_per_cluster():
    index = _index()
    q = torch.randn(3, 4, 8, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    for boundary in (16, 9):
        got = oracle_select.oracle_cluster_mass(q, index, boundary, sm_scale=0.5)
        want = _reference(q, index, boundary, 0.5)
        assert torch.allclose(got, want, atol=1e-3), (boundary, got, want)
        assert torch.allclose(got.sum(-1), torch.full((3, 2), 2.0), atol=1e-3)   # the two heads of a group each sum to one


def test_a_boundary_below_every_position_gives_no_mass():
    index = _index()
    q = torch.randn(2, 4, 8).to(torch.bfloat16)
    assert float(oracle_select.oracle_cluster_mass(q, index, 0, sm_scale=1.0).abs().sum()) == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the budget kernel is Triton")
def test_the_budget_rule_takes_the_heaviest_clusters_first():
    # a runtime-sized partition: 128 clusters (the bitmask is one int32 word per 32 clusters)
    g = torch.Generator().manual_seed(3)
    sizes = tuple(int(x) for x in torch.randint(1, 40, (128,), generator=g))
    index = _index(hkv=2, clusters=128, sizes=sizes, align=4, d=8, seed=3)
    dev = torch.device("cuda")
    for k in ("bucket_indptr", "bucket_sizes", "posting_positions", "packed_kv"):
        setattr(index, k, getattr(index, k).to(dev))
    q = torch.randn(1, 4, 8, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16).to(dev)
    prob = oracle_select.oracle_cluster_mass(q, index, index.prompt_length, sm_scale=0.5)
    sizes3 = index.bucket_sizes.unsqueeze(0).contiguous()
    bitmask = torch.zeros(1, 2, 4, dtype=torch.int32, device=dev)
    budget = 200
    counts, tokens = oracle_select.select(prob, sizes3, bitmask, target_tokens=budget, g_union=1)
    words = bitmask.cpu()
    for h in range(2):
        chosen = sorted(c for c in range(128) if (int(words[0, h, c // 32]) >> (c % 32)) & 1)
        order = prob[0, h].argsort(descending=True).tolist()
        assert chosen and chosen == sorted(order[:len(chosen)]), "the chosen clusters are the heaviest"
        visible = sum(sizes[c] for c in chosen)
        assert visible >= budget and visible - sizes[chosen[-1] if False else order[len(chosen) - 1]] < budget, \
            "clusters in score order until the budget is visible, the crossing cluster included"
        assert int(counts[0, h]) == len(chosen) and int(tokens[0, h]) == visible
