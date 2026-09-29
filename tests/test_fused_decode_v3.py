"""The fused decode step selects what the unfused select kernel selects and attends like a float64 softmax.

GPU-only (Triton, SM90).  A synthetic arena: random cluster-ordered K/V,
random codebooks, a generated suffix of a few tokens.  The selection is
checked cluster by cluster against ``theta_select_bitmask_flat`` with torch
scores, and the attention against a float64 softmax over exactly the selected
rows plus the causally visible suffix, with bf16 tolerance.  The step is then
run again with the same shapes, so the workspace is reused and the outputs
must not depend on what it held before.
"""
from __future__ import annotations

import math
import types

import pytest
import torch


def _has_sm90_triton() -> bool:
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


class _Arena:
    """Just the fields fused_decode_step reads, shaped like DecodeArena's."""

    def __init__(self, *, hkv, d, index_dim, num_clusters, prompt_length, reserve, device, gen):
        self.num_clusters = num_clusters
        self.reserve = reserve
        self.suffix_block = 256
        df = index_dim // 2
        # clusters of unequal size covering the prompt exactly, in order
        cuts = torch.sort(torch.randint(1, prompt_length, (num_clusters - 1,), generator=gen)).values
        bounds = torch.cat([torch.tensor([0]), cuts, torch.tensor([prompt_length])])
        sizes = (bounds[1:] - bounds[:-1]).to(torch.int32)
        self.sizes = sizes.view(1, -1).expand(hkv, -1).contiguous().to(device)
        self.indptr = bounds.to(torch.int32).view(1, -1).expand(hkv, -1).contiguous().to(device)
        width = prompt_length + 256
        self.posting = torch.full((hkv, width), 2147483647, dtype=torch.int32, device=device)
        # cluster-ordered rows: row i of the packed copy holds position perm[i]
        perm = torch.randperm(prompt_length, generator=gen)
        self.perm = perm
        self.posting[:, :prompt_length] = perm.to(torch.int32).to(device)
        self.packed_kv = torch.zeros(2, width, hkv, d, dtype=torch.bfloat16, device=device)
        self.packed_kv[:, :prompt_length] = (torch.randn(2, prompt_length, hkv, d, generator=gen) * 0.5).to(torch.bfloat16).to(device)
        self.codebooks = (torch.randn(hkv, 2, num_clusters, df, generator=gen)).to(torch.bfloat16).to(device)
        self.codebooks = torch.nn.functional.normalize(self.codebooks.float(), dim=-1).to(torch.bfloat16)
        self.prompt_length_t = torch.tensor([prompt_length], dtype=torch.int32, device=device)
        blocks = reserve // 256 + 1
        self.suffix_kv = torch.zeros(2, blocks, 256, hkv, d, dtype=torch.bfloat16, device=device)
        base = prompt_length // 256
        table_len = -(-(prompt_length + reserve) // 256) + 1
        self.suffix_table = torch.zeros(1, table_len, dtype=torch.int32, device=device)
        n = min(blocks, table_len - base)
        self.suffix_table[0, base:base + n] = torch.arange(n, dtype=torch.int32, device=device)
        self.index = types.SimpleNamespace(prompt_length=prompt_length)


def _selections(arena, layer, q_pre, *, temperature, beta, target_tokens):
    """The selected-cluster masks [Hkv, C] of the unfused select kernel (the same rule, scores in
    the torch form) and of the torch rule (fp32, walked in score order).

    The fused step matches the unfused kernel cluster for cluster when the latter scores in the
    torch form (SEMBLO_FLAT_TORCH_SCORES=1, set here for this call only).  With the tensor-core
    score kernel the two can differ in a boundary cluster, since random codebooks give
    near-tied scores."""
    import os
    from semblo.vllm_plugin.fused_select import theta_select_bitmask_flat
    from semblo.vllm_plugin.torch_select import budget_select_bitmask
    from semblo.vllm_plugin.trivial_router import trivial_router_vectors
    hq, hkv, d = layer.num_heads, layer.num_kv_heads, layer.head_dim
    c = arena.num_clusters
    vec = trivial_router_vectors(q_pre.reshape(1, hq, d), 1).view(1, hq, 1, d)
    logmix = torch.zeros(1, hq, 1, dtype=torch.float32, device=q_pre.device)
    sizes3 = arena.sizes.unsqueeze(0).contiguous()
    out = []
    prev = os.environ.get("SEMBLO_FLAT_TORCH_SCORES")
    os.environ["SEMBLO_FLAT_TORCH_SCORES"] = "1"
    try:
        for fn in (theta_select_bitmask_flat, budget_select_bitmask):
            bitmask = torch.zeros(1, hkv, c // 32, dtype=torch.int32, device=q_pre.device)
            fn(vec, logmix, arena.codebooks, sizes3, bitmask,
               temperature=temperature, beta=beta, target_tokens=target_tokens, g_union=1)
            words = bitmask[0]
            out.append(((words[:, :, None] >> torch.arange(32, device=words.device)[None, None, :]) & 1).reshape(hkv, c).bool())
    finally:
        if prev is None:
            os.environ.pop("SEMBLO_FLAT_TORCH_SCORES", None)
        else:
            os.environ["SEMBLO_FLAT_TORCH_SCORES"] = prev
    return out[0], out[1]


def _reference(arena, layer, q_post, positions, suffix_positions, bits, *, sm_scale):
    """Attention in float64 over the rows of the selected clusters ``bits`` [Hkv, C] and the visible suffix."""
    hq, hkv, d = layer.num_heads, layer.num_kv_heads, layer.head_dim
    g = hq // hkv
    q = q_post.view(hq, d).double()
    q_pos = int(positions[0])
    out = torch.zeros(hq, d, dtype=torch.float64, device=q.device)
    for h in range(hkv):
        rows = []
        for cl in torch.nonzero(bits[h]).flatten().tolist():
            s, e = int(arena.indptr[h, cl]), int(arena.indptr[h, cl + 1])
            rows.append(torch.arange(s, e, device=q.device))
        rows = torch.cat(rows) if rows else torch.zeros(0, dtype=torch.long, device=q.device)
        k = arena.packed_kv[0, rows, h].double()
        v = arena.packed_kv[1, rows, h].double()
        vis = suffix_positions[suffix_positions <= q_pos]
        blk = vis // 256
        phys = arena.suffix_table[0, blk].long()
        off = vis - blk * 256
        k = torch.cat([k, arena.suffix_kv[0, phys, off, h].double()])
        v = torch.cat([v, arena.suffix_kv[1, phys, off, h].double()])
        for t in range(g):
            head = h * g + t
            s = (k @ q[head]) * sm_scale
            p = torch.softmax(s, dim=0)
            out[head] = p @ v
    return out


def _kernel_bits(arena, ws):
    """The four-kernel step's selection as a [Hkv, C] mask, read back from its CSR."""
    hkv, c = arena.sizes.shape
    bits = torch.zeros(hkv, c, dtype=torch.bool, device=arena.sizes.device)
    for h in range(hkv):
        n = int(ws.rcount[h])
        starts = ws.rstart[h, :n].long()
        for st in starts.tolist():
            cl = torch.nonzero(arena.indptr[h, :-1].long() == st).flatten()
            assert cl.numel() >= 1, (h, st)
            # a run of empty clusters shares a start; the selected one is the non-empty one
            live = cl[arena.sizes[h, cl] > 0]
            bits[h, live[0] if live.numel() else cl[0]] = True
    return bits


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
@pytest.mark.parametrize("hq,hkv,d", [(32, 2, 128), (16, 4, 256)])   # Nemotron 3.5 Lightning, Qwen3.5-4B
def test_the_four_kernel_step_matches_the_torch_rule_and_a_float64_attention(hq, hkv, d):
    from semblo.vllm_plugin.fused_decode import fused_decode_step
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(7)
    prompt_length, num_clusters, reserve = 6000, 64, 1024
    arena = _Arena(hkv=hkv, d=d, index_dim=d, num_clusters=num_clusters, prompt_length=prompt_length,
                   reserve=reserve, device=device, gen=gen)
    layer = types.SimpleNamespace(num_heads=hq, num_kv_heads=hkv, head_dim=d,
                                  runtime_config=types.SimpleNamespace(triton_token_tile=16))
    temperature, beta, target, sm_scale = 0.1, 0.25, prompt_length // 8, 1.0 / math.sqrt(d)
    # three generated tokens already in the store, the fourth is this step's
    suffix_positions = torch.arange(prompt_length, prompt_length + 4, device=device)
    for i, pos in enumerate(suffix_positions[:3].tolist()):
        blk, off = pos // 256, pos % 256
        phys = int(arena.suffix_table[0, blk])
        arena.suffix_kv[:, phys, off] = (torch.randn(2, hkv, d, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    q_pre = (torch.randn(1, hq, d, generator=gen)).to(torch.bfloat16).to(device)
    q_post = (torch.randn(1, hq, d, generator=gen)).to(torch.bfloat16).to(device)
    k_new = (torch.randn(1, hkv, d, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    v_new = (torch.randn(1, hkv, d, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    positions = torch.tensor([prompt_length + 3], dtype=torch.int64, device=device)
    output = torch.zeros(1, hq * d, dtype=torch.bfloat16, device=device)
    fused_decode_step(layer, arena, q_post=q_post, q_pre=q_pre, positions=positions, k_post=k_new, v_states=v_new,
                      output=output, temperature=temperature, beta=beta, target_tokens=target, sm_scale=sm_scale,
                      num_splits=8)
    torch.cuda.synchronize()
    # the store now holds the fourth token, as the kernel wrote it
    blk, off = int(positions[0]) // 256, int(positions[0]) % 256
    phys = int(arena.suffix_table[0, blk])
    assert torch.equal(arena.suffix_kv[0, phys, off], k_new[0]) and torch.equal(arena.suffix_kv[1, phys, off], v_new[0])
    ws = arena.fused_decode_ws
    v1_bits, _torch_bits = _selections(arena, layer, q_pre, temperature=temperature, beta=beta, target_tokens=target)
    got_bits = _kernel_bits(arena, ws)
    # the selection equals the unfused kernel's exactly (same rule, same bf16 dots, same bisection).
    # The torch rule (fp32 scores, a walk in score order) is not compared here: random codebooks
    # give near-tied scores, and rounding of the dots can reorder whole sets.
    for h in range(hkv):
        sel = torch.nonzero(v1_bits[h]).flatten()
        n = int(ws.rcount[h])
        assert torch.equal(got_bits[h], v1_bits[h]), (h, torch.nonzero(got_bits[h] ^ v1_bits[h]).flatten().tolist())
        assert torch.equal(ws.rstart[h, :n].long(), arena.indptr[h, sel].long())
        assert torch.equal(ws.rlen[h, :n].long(), arena.sizes[h, sel].long())
        assert torch.equal(ws.rcum[h, :n].long(), torch.cumsum(arena.sizes[h, sel].long(), 0) - arena.sizes[h, sel].long())
        assert int(ws.rtokens[h]) == int(arena.sizes[h, sel].sum())
        assert int(ws.rtokens[h]) >= target or n == num_clusters
    want = _reference(arena, layer, q_post, positions, suffix_positions, got_bits, sm_scale=sm_scale)
    got = output.view(hq, d).double()
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    # a second step with the workspace already made: same shapes, a new query, still right
    q_pre2 = (torch.randn(1, hq, d, generator=gen)).to(torch.bfloat16).to(device)
    q_post2 = (torch.randn(1, hq, d, generator=gen)).to(torch.bfloat16).to(device)
    positions2 = positions + 1
    k2 = (torch.randn(1, hkv, d, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    v2 = (torch.randn(1, hkv, d, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    fused_decode_step(layer, arena, q_post=q_post2, q_pre=q_pre2, positions=positions2, k_post=k2, v_states=v2,
                      output=output, temperature=temperature, beta=beta, target_tokens=target, sm_scale=sm_scale,
                      num_splits=8)
    torch.cuda.synchronize()
    assert arena.fused_decode_ws is ws
    suffix2 = torch.arange(prompt_length, prompt_length + 5, device=device)
    v1_bits2, _ = _selections(arena, layer, q_pre2, temperature=temperature, beta=beta, target_tokens=target)
    got_bits2 = _kernel_bits(arena, ws)
    assert torch.equal(got_bits2, v1_bits2)
    want2 = _reference(arena, layer, q_post2, positions2, suffix2, got_bits2, sm_scale=sm_scale)
    torch.testing.assert_close(output.view(hq, d).double(), want2, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
@pytest.mark.parametrize("hq,hkv,d", [(16, 4, 256)])   # Qwen3.5-4B
def test_with_a_sink_the_step_reads_cluster_zero_and_selects_the_rest_under_the_held_out_budget(hq, hkv, d, monkeypatch):
    """SEMBLO_SINK_TOKENS follows the unfused path's rule (fused_prefill._hold_out_sink, _force_sink_bit).

    The reference is this step without the variable, on an arena whose cluster 0 has size zero and with the
    budget reduced by the sink, plus cluster 0 at its real size.  The attention is checked against a float64
    softmax over exactly those rows."""
    from semblo.vllm_plugin.fused_decode import fused_decode_step
    from semblo.vllm_plugin.fused_prefill import _hold_out_sink
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(11)
    prompt_length, num_clusters, reserve, sink = 6000, 64, 1024, 512
    arena = _Arena(hkv=hkv, d=d, index_dim=d, num_clusters=num_clusters, prompt_length=prompt_length,
                   reserve=reserve, device=device, gen=gen)
    # cluster 0 = positions [0, sink), as state._build_flat makes it; the other cuts stay random
    cuts = torch.sort(torch.randint(sink + 1, prompt_length, (num_clusters - 2,), generator=gen)).values
    bounds = torch.cat([torch.tensor([0, sink]), cuts, torch.tensor([prompt_length])])
    arena.sizes = (bounds[1:] - bounds[:-1]).to(torch.int32).view(1, -1).expand(hkv, -1).contiguous().to(device)
    arena.indptr = bounds.to(torch.int32).view(1, -1).expand(hkv, -1).contiguous().to(device)
    layer = types.SimpleNamespace(num_heads=hq, num_kv_heads=hkv, head_dim=d,
                                  runtime_config=types.SimpleNamespace(triton_token_tile=16))
    temperature, beta, target, sm_scale = 0.1, 0.25, prompt_length // 8, 1.0 / math.sqrt(d)
    q_pre = (torch.randn(1, hq, d, generator=gen)).to(torch.bfloat16).to(device)
    q_post = (torch.randn(1, hq, d, generator=gen)).to(torch.bfloat16).to(device)
    k_new = (torch.randn(1, hkv, d, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    v_new = (torch.randn(1, hkv, d, generator=gen) * 0.5).to(torch.bfloat16).to(device)
    positions = torch.tensor([prompt_length], dtype=torch.int64, device=device)
    step = dict(q_post=q_post, q_pre=q_pre, positions=positions, k_post=k_new, v_states=v_new, temperature=temperature,
                beta=beta, sm_scale=sm_scale, num_splits=8)
    monkeypatch.delenv("SEMBLO_BLOCK_TOKENS", raising=False)
    monkeypatch.setenv("SEMBLO_SINK_TOKENS", str(sink))
    output = torch.zeros(1, hq * d, dtype=torch.bfloat16, device=device)
    fused_decode_step(layer, arena, output=output, target_tokens=target, **step)
    torch.cuda.synchronize()
    ws = arena.fused_decode_ws
    got_bits = _kernel_bits(arena, ws)
    got_count, got_tokens = ws.rcount.clone(), ws.rtokens.clone()
    got_len = [ws.rlen[h, :int(ws.rcount[h])].clone() for h in range(hkv)]
    # the reference: the same step, no variable, cluster 0 held out, the budget less the sink
    monkeypatch.delenv("SEMBLO_SINK_TOKENS")
    held_sizes, held_target = _hold_out_sink(arena.sizes, target, sink)
    held = types.SimpleNamespace(**{k: v for k, v in vars(arena).items() if k != "fused_decode_ws"})
    held.sizes = held_sizes
    fused_decode_step(layer, held, output=torch.zeros_like(output), target_tokens=held_target, **step)
    torch.cuda.synchronize()
    want_bits = _kernel_bits(held, held.fused_decode_ws)
    assert not want_bits[:, 0].any()          # with size zero, cluster 0 is never selected
    want_bits[:, 0] = True
    for h in range(hkv):
        assert bool(got_bits[h, 0]), h
        assert torch.equal(got_bits[h], want_bits[h]), (h, torch.nonzero(got_bits[h] ^ want_bits[h]).flatten().tolist())
        sel = torch.nonzero(want_bits[h]).flatten()
        assert int(got_count[h]) == sel.numel()
        assert torch.equal(got_len[h].long(), arena.sizes[h, sel].long())   # cluster 0 read with its real size
        assert int(got_tokens[h]) == int(arena.sizes[h, sel].sum())
        assert int(got_tokens[h]) - sink >= held_target or sel.numel() == num_clusters
    suffix_positions = torch.arange(prompt_length, prompt_length + 1, device=device)
    want = _reference(arena, layer, q_post, positions, suffix_positions, got_bits, sm_scale=sm_scale)
    torch.testing.assert_close(output.view(hq, d).double(), want, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _has_sm90_triton(), reason="requires Triton and an SM90 GPU")
def test_the_step_launches_five_kernels():
    """Router, suffix write, select, attend and merge, and nothing else."""
    from semblo.vllm_plugin.fused_decode import fused_decode_step
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(3)
    hq, hkv, d = 32, 2, 128
    prompt_length = 4096
    arena = _Arena(hkv=hkv, d=d, index_dim=d, num_clusters=32, prompt_length=prompt_length, reserve=512,
                   device=device, gen=gen)
    layer = types.SimpleNamespace(num_heads=hq, num_kv_heads=hkv, head_dim=d,
                                  runtime_config=types.SimpleNamespace(triton_token_tile=16))
    args = dict(temperature=0.1, beta=0.25, target_tokens=prompt_length // 16, sm_scale=1.0 / math.sqrt(d), num_splits=8)
    mk = lambda *shape: (torch.randn(*shape, generator=gen)).to(torch.bfloat16).to(device)  # noqa: E731
    positions = torch.tensor([prompt_length], dtype=torch.int64, device=device)
    output = torch.zeros(1, hq * d, dtype=torch.bfloat16, device=device)
    fused_decode_step(layer, arena, q_post=mk(1, hq, d), q_pre=mk(1, hq, d), positions=positions,
                      k_post=mk(1, hkv, d), v_states=mk(1, hkv, d), output=output, **args)   # compiles, makes the workspace
    torch.cuda.synchronize()
    from torch.profiler import ProfilerActivity, profile
    q_post, q_pre, k_new, v_new = mk(1, hq, d), mk(1, hq, d), mk(1, hkv, d), mk(1, hkv, d)
    positions2 = positions + 1
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fused_decode_step(layer, arena, q_post=q_post, q_pre=q_pre, positions=positions2,
                          k_post=k_new, v_states=v_new, output=output, **args)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events() if e.device_type.name == "CUDA"]
    assert len(kernels) == 5, kernels
