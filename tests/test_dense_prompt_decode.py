"""The dense-decode fast path: one FA3 call over the packed prompt when the
budget is the whole prompt.

Checked against a float32 softmax over the same rows, and through the same
log-sum-exp merge the decode step uses for the generated suffix, so the LSE
convention (natural log of the scaled scores) is the one the merge expects.
"""

import types

import pytest
import torch

from semblo.vllm_plugin import fused_prefill
from semblo.vllm_plugin.paged_attention import (
    PagedSparseAttentionResult, merge_paged_attention_results)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _reference(q, k, v, scale):
    """[1, Hq, D] float32 output and [1, Hq] natural-log LSE over all rows."""
    hq, hkv = q.shape[1], k.shape[1]
    rep = hq // hkv
    kk = k.float().repeat_interleave(rep, dim=1)      # [L, Hq, D]
    vv = v.float().repeat_interleave(rep, dim=1)
    s = torch.einsum("qhd,lhd->hql", q.float(), kk) * scale   # [Hq, 1, L]
    lse = torch.logsumexp(s, dim=-1)                          # [Hq, 1]
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("hql,lhd->qhd", p, vv)                    # [1, Hq, D]
    return o, lse.transpose(0, 1)


def _index(length, k, v, *, padding_rows=0, num_kv_heads=2, device="cuda"):
    """A stand-in with the three attributes the fast path reads."""
    torch.manual_seed(0)
    total = length + padding_rows
    packed = torch.zeros(2, total, num_kv_heads, k.shape[-1], dtype=k.dtype, device=device)
    packed[0, :length] = k
    packed[1, :length] = v
    if padding_rows:
        packed[:, length:] = packed[:, :1]          # copies of a real row, as the packer leaves them
    # two buckets per head; the second one carries the padding
    indptr = torch.tensor([[0, length // 2, total]] * num_kv_heads, dtype=torch.int32, device=device)
    return types.SimpleNamespace(prompt_length=length, bucket_indptr=indptr, packed_kv=packed)


@pytest.mark.parametrize("length", [1000, 4096 + 7])
def test_the_fast_path_matches_a_dense_softmax(length, monkeypatch):
    monkeypatch.setenv("SEMBLO_DENSE_DECODE_FAST", "1")
    torch.manual_seed(1)
    hq, hkv, d = 4, 2, 128
    q = torch.randn(1, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(length, hkv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(length, hkv, d, device="cuda", dtype=torch.bfloat16)
    scale = d ** -0.5
    res = fused_prefill.dense_prompt_decode(q, _index(length, k, v), sm_scale=scale)
    assert res is not None
    o_ref, lse_ref = _reference(q, k, v, scale)
    assert res.output.shape == (1, hq, d) and res.logsumexp.shape == (1, hq)
    assert torch.allclose(res.output.float(), o_ref, atol=2e-2, rtol=2e-2)
    assert torch.allclose(res.logsumexp, lse_ref, atol=2e-3, rtol=1e-4)


def test_padding_rows_send_the_caller_back_to_the_unit_path(monkeypatch):
    monkeypatch.setenv("SEMBLO_DENSE_DECODE_FAST", "1")
    torch.manual_seed(2)
    length, hq, hkv, d = 512, 4, 2, 128
    q = torch.randn(1, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(length, hkv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(length, hkv, d, device="cuda", dtype=torch.bfloat16)
    assert fused_prefill.dense_prompt_decode(q, _index(length, k, v, padding_rows=16), sm_scale=d ** -0.5) is None
    monkeypatch.setenv("SEMBLO_DENSE_DECODE_FAST", "0")
    assert fused_prefill.dense_prompt_decode(q, _index(length, k, v), sm_scale=d ** -0.5) is None


def test_the_lse_merges_with_a_suffix_like_the_decode_step_does(monkeypatch):
    """prompt (fast path) + suffix (reference) merged by log-sum-exp equals the
    reference over prompt and suffix together."""
    monkeypatch.setenv("SEMBLO_DENSE_DECODE_FAST", "1")
    torch.manual_seed(3)
    length, suffix, hq, hkv, d = 2048, 37, 4, 2, 128
    q = torch.randn(1, hq, d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(length + suffix, hkv, d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(length + suffix, hkv, d, device="cuda", dtype=torch.bfloat16)
    scale = d ** -0.5
    prompt = fused_prefill.dense_prompt_decode(q, _index(length, k[:length], v[:length]), sm_scale=scale)
    o_suf, lse_suf = _reference(q, k[length:], v[length:], scale)
    merged = merge_paged_attention_results(
        prompt, PagedSparseAttentionResult(output=o_suf.to(q.dtype), logsumexp=lse_suf))
    o_all, lse_all = _reference(q, k, v, scale)
    assert torch.allclose(merged.output.float(), o_all, atol=2e-2, rtol=2e-2)
    assert torch.allclose(merged.logsumexp.float(), lse_all, atol=3e-3, rtol=1e-4)
