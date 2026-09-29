"""The KV-outer body with the chunk's own window folded into the combine.

Under SEMBLO_KV_OUTER_FUSED_LOCAL=1 the body (a) attends the chunk's causal
window once for all query heads and seeds the log-sum-exp combine with it,
instead of merging it afterwards, and (b) hands FA3 a page table built once
per index instead of once per chunk.  Both compute the same result, so the
two settings must agree to rounding, including for queries and heads that
selected no cluster.
"""
from __future__ import annotations


import pytest
import torch


def _needs_hopper():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("needs FlashAttention-3 (Hopper)")
    try:
        import triton  # noqa: F401
        from vllm.vllm_flash_attn import flash_attn_varlen_func  # noqa: F401
    except ImportError:
        pytest.skip("needs triton and vLLM's FlashAttention")


def _merge(out_a, lse_a, out_b, lse_b):
    """LSE merge of two partials in float64, the reference for the kernel."""
    m = torch.maximum(lse_a, lse_b)
    wa = (lse_a - m).exp()
    wb = (lse_b - m).exp()
    merged = (out_a * wa[..., None] + out_b * wb[..., None]) / (wa + wb)[..., None]
    return merged, m + (wa + wb).log()


def test_combine_seeds_with_the_local_partial():
    _needs_hopper()
    from semblo.vllm_plugin.kv_outer import combine_partials
    torch.manual_seed(0)
    device = "cuda"
    q_count, q_per_kv, head_dim, kv_head, num_kv_heads = 37, 2, 64, 1, 2
    counts = torch.randint(0, 5, (q_count,), device=device, dtype=torch.int32)
    counts[3] = 0                                   # a query with no entries
    nnz = int(counts.sum())
    start = (torch.cumsum(counts, 0) - counts).to(torch.int32)
    order = torch.randperm(nnz, device=device).to(torch.int32)
    partial = torch.randn(nnz, q_per_kv, head_dim, device=device).to(torch.bfloat16)
    lse = torch.randn(nnz * q_per_kv, device=device)
    local_o = torch.randn(q_count, num_kv_heads * q_per_kv, head_dim,
                          device=device).to(torch.bfloat16)
    local_lse = torch.randn(q_count, num_kv_heads * q_per_kv, device=device)

    # the kernel without the local partial, then the merge done by hand
    out_a = torch.zeros(q_count, num_kv_heads * q_per_kv, head_dim,
                        device=device, dtype=torch.bfloat16)
    lse_a = torch.full((q_count, num_kv_heads * q_per_kv), float("-inf"),
                       device=device)
    combine_partials(out_a, partial, lse, order, start, counts,
                     kv_head=kv_head, q_per_kv=q_per_kv, lse_out=lse_a)
    sl = slice(kv_head * q_per_kv, (kv_head + 1) * q_per_kv)
    want, want_lse = _merge(out_a[:, sl].double(), lse_a[:, sl].double(),
                            local_o[:, sl].double(), local_lse[:, sl].double())

    out_b = torch.zeros_like(out_a)
    lse_b = torch.full_like(lse_a, float("-inf"))
    combine_partials(out_b, partial, lse, order, start, counts,
                     kv_head=kv_head, q_per_kv=q_per_kv, lse_out=lse_b,
                     local_o=local_o, local_lse=local_lse)
    torch.testing.assert_close(out_b[:, sl].double(), want, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse_b[:, sl].double(), want_lse, atol=1e-3, rtol=1e-3)
    # the query with no entries is exactly the local partial
    torch.testing.assert_close(out_b[3, sl].double(), local_o[3, sl].double())
    # the other KV head's columns were not touched
    assert torch.all(out_b[:, 0:q_per_kv] == 0)


def _synthetic(seed, *, num_kv_heads=2, q_per_kv=2, head_dim=64, page=16,
               clusters=64, q_count=48, n_local=56, empty_head=None,
               device="cuda"):
    """A chunk of q_count queries over `clusters` clusters of page-multiple
    sizes.  The local window holds n_local >= q_count tokens, as in the
    runtime, where every query sees at least itself."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    sizes = torch.randint(1, 17, (clusters,), generator=g) * page   # multiples of the page
    length = int(sizes.sum())
    indptr = torch.zeros(num_kv_heads, clusters + 1, dtype=torch.int32)
    indptr[:, 1:] = torch.cumsum(sizes, 0)[None].to(torch.int32)
    causal = (torch.rand(num_kv_heads, clusters, generator=g) * (sizes[None] + 1)).floor()
    causal = torch.minimum(causal.clamp_min(0), sizes[None].expand_as(causal).float()).to(torch.int32)
    n_words = clusters // 32
    qmask = torch.randint(0, 1 << 31, (q_count, num_kv_heads, n_words), generator=g,
                          dtype=torch.int64)
    qmask = (qmask & torch.randint(0, 1 << 31, qmask.shape, generator=g)).to(torch.int32)
    qmask[:3, 1] = 0                       # three queries pick nothing in head 1
    if empty_head is not None:
        qmask[:, empty_head] = 0           # a head with no cluster at all
    packed = torch.randn(2, length, num_kv_heads, head_dim, generator=g).to(torch.bfloat16)
    q = torch.randn(q_count, num_kv_heads * q_per_kv, head_dim, generator=g).to(torch.bfloat16)
    local = torch.randn(2, n_local, num_kv_heads, head_dim, generator=g).to(torch.bfloat16)
    move = lambda t: t.to(device).contiguous()
    return dict(q=move(q), packed_kv=move(packed), qmask=move(qmask),
                causal_sizes=move(causal), bucket_indptr=move(indptr),
                local_kv=move(local), page=page)


@pytest.mark.parametrize("empty_head", [None, 0])
def test_fused_local_agrees_with_the_merge_afterwards(monkeypatch, empty_head):
    _needs_hopper()
    from semblo.vllm_plugin.kv_outer import kv_outer_attention
    case = _synthetic(1, empty_head=empty_head)
    sm_scale = case["q"].shape[-1] ** -0.5
    run = lambda: kv_outer_attention(
        case["q"], case["packed_kv"], case["qmask"], case["causal_sizes"],
        case["bucket_indptr"], sm_scale=sm_scale, page_size=case["page"],
        aligned=True, local_kv=case["local_kv"], table_cache={})[0]
    monkeypatch.setenv("SEMBLO_KV_OUTER_FUSED_LOCAL", "0")   # the merge afterwards
    monkeypatch.delenv("SEMBLO_KV_OUTER_CBATCH", raising=False)
    reference = run().float()
    monkeypatch.setenv("SEMBLO_KV_OUTER_FUSED_LOCAL", "1")
    fused = run().float()
    assert torch.isfinite(fused).all()
    torch.testing.assert_close(fused, reference, atol=3e-2, rtol=3e-2)


def test_page_table_cache_is_reused_across_chunks(monkeypatch):
    _needs_hopper()
    from semblo.vllm_plugin.kv_outer import kv_outer_attention
    case = _synthetic(2)
    sm_scale = case["q"].shape[-1] ** -0.5
    monkeypatch.setenv("SEMBLO_KV_OUTER_FUSED_LOCAL", "1")
    cache = {}
    first = kv_outer_attention(
        case["q"], case["packed_kv"], case["qmask"], case["causal_sizes"],
        case["bucket_indptr"], sm_scale=sm_scale, page_size=case["page"],
        aligned=True, local_kv=case["local_kv"], table_cache=cache)[0]
    assert len(cache) == 1
    table = next(iter(cache.values()))[3]
    # a later chunk of the same prompt sees fewer tokens per cluster; the
    # table object must be the same one and the result must still be finite
    later = case["causal_sizes"] // 2
    again = kv_outer_attention(
        case["q"], case["packed_kv"], case["qmask"], later,
        case["bucket_indptr"], sm_scale=sm_scale, page_size=case["page"],
        aligned=True, local_kv=case["local_kv"], table_cache=cache)[0]
    assert next(iter(cache.values()))[3] is table
    assert torch.isfinite(first.float()).all() and torch.isfinite(again.float()).all()
    monkeypatch.setenv("SEMBLO_KV_OUTER_FUSED_LOCAL", "0")
    plain = kv_outer_attention(
        case["q"], case["packed_kv"], case["qmask"], later,
        case["bucket_indptr"], sm_scale=sm_scale, page_size=case["page"],
        aligned=True, local_kv=case["local_kv"], table_cache={})[0]
    torch.testing.assert_close(again.float(), plain.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("with_local", [False, True])
def test_grouped_combine_is_the_per_head_combine_to_the_bit(monkeypatch, with_local):
    _needs_hopper()
    from semblo.vllm_plugin.kv_outer import combine_partials
    torch.manual_seed(2)
    device = "cuda"
    q_count, q_per_kv, head_dim, kv_head, num_kv_heads = 53, 4, 256, 2, 4
    counts = torch.randint(0, 40, (q_count,), device=device, dtype=torch.int32)
    counts[5] = 0
    nnz = int(counts.sum())
    start = (torch.cumsum(counts, 0) - counts).to(torch.int32)
    order = torch.randperm(nnz, device=device).to(torch.int32)
    partial = torch.randn(nnz, q_per_kv, head_dim, device=device).to(torch.bfloat16)
    lse = torch.randn(nnz * q_per_kv, device=device)
    local_o = torch.randn(q_count, num_kv_heads * q_per_kv, head_dim, device=device).to(torch.bfloat16)
    local_lse = torch.randn(q_count, num_kv_heads * q_per_kv, device=device)
    kw = dict(kv_head=kv_head, q_per_kv=q_per_kv)
    if with_local:
        kw.update(local_o=local_o, local_lse=local_lse)
    outs = []
    for grouped in ("0", "1"):
        monkeypatch.setenv("SEMBLO_COMBINE_GROUPED", grouped)
        out = torch.zeros(q_count, num_kv_heads * q_per_kv, head_dim, device=device, dtype=torch.bfloat16)
        lse_out = torch.full((q_count, num_kv_heads * q_per_kv), float("-inf"), device=device)
        combine_partials(out, partial, lse, order, start, counts, lse_out=lse_out, **kw)
        outs.append((out.clone(), lse_out.clone()))
    assert torch.equal(outs[0][0], outs[1][0]), "the grouped combine changed bits"
    assert torch.equal(outs[0][1], outs[1][1])
