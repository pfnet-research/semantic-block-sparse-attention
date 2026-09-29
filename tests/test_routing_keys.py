"""SEMBLO_ROUTING_KEYS=post: the router reads the rotated q and k, the attention's inputs are untouched.

Qwen3.5 overrides ``_project_qkv`` and ``_project_kv``, so both must apply the switch.  These tests
check that:
  * ``routing_view``: without the switch the routing slot is the pre-RoPE tensor itself; with it, the
    rotated tensor in the pre-RoPE shape, a view, and the rotated tensor is not modified;
  * ``check_routing_view`` raises when the routing tensors are not the rotated ones, and prints its
    line once per path;
  * Qwen3.5's ``_project_qkv`` and ``_project_kv`` (sliced K/V and fused fallback) route through it:
    routing slots are the rotated tensors under the switch, the attention slots (q_post, k_post, v,
    gate) are bit-identical with and without it;
  * the streamed (sliced) prefill stages the rotated keys for the index fit, hands the rotated
    queries to the selection, and prints its confirmation line.
The Qwen3.5 tests need vLLM importable (the plugin modules import it); no engine, no GPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from semblo.vllm_plugin import routing_keys


@pytest.fixture
def switch(monkeypatch: pytest.MonkeyPatch):
    """Set SEMBLO_ROUTING_KEYS for this test only, with fresh once-per-process state."""

    monkeypatch.setattr(routing_keys, "_SAID", set())
    monkeypatch.setattr(routing_keys, "_CHECKED", set())

    def set_to(post: bool) -> None:
        monkeypatch.setattr(routing_keys, "ROUTE_POST_ROPE", post)
        routing_keys._SAID.clear()
        routing_keys._CHECKED.clear()

    return set_to


# ---------------------------------------------------------------- the helper


def test_the_switch_reads_pre_or_post_and_refuses_anything_else(monkeypatch: pytest.MonkeyPatch):
    for value, post in ((None, False), ("", False), ("pre", False), ("post", True)):
        if value is None:
            monkeypatch.delenv("SEMBLO_ROUTING_KEYS", raising=False)
        else:
            monkeypatch.setenv("SEMBLO_ROUTING_KEYS", value)
        assert routing_keys._read_switch() is post
    for value in ("POST", "post_rope", "rotated"):
        monkeypatch.setenv("SEMBLO_ROUTING_KEYS", value)
        with pytest.raises(ValueError, match="SEMBLO_ROUTING_KEYS"):
            routing_keys._read_switch()


def test_routing_view_is_pre_without_the_switch_and_the_rotated_view_with_it(switch):
    gen = torch.Generator().manual_seed(0)
    pre = torch.randn(5, 2, 4, generator=gen)            # [T, H, D]
    post = torch.randn(5, 8, generator=gen)              # [T, H * D], the attention's input
    post_before = post.clone()

    switch(False)
    assert routing_keys.routing_view(pre, post) is pre
    assert routing_keys.routing_view(None, post) is None

    switch(True)
    routed = routing_keys.routing_view(pre, post)
    assert routed.shape == pre.shape
    assert torch.equal(routed, post.reshape(5, 2, 4))
    assert routed.data_ptr() == post.data_ptr()          # a view: no copy of a 512K tensor
    assert torch.equal(post, post_before)                # the attention's input is untouched
    assert routing_keys.routing_view(None, post) is None  # a layer without keys of its own
    with pytest.raises(RuntimeError, match="rotated"):
        routing_keys.routing_view(pre, None)


def test_the_check_raises_on_unrotated_routing_and_prints_once_per_path(switch, capsys):
    gen = torch.Generator().manual_seed(1)
    pre = torch.randn(5, 2, 4, generator=gen)
    post = torch.randn(5, 8, generator=gen)

    switch(False)
    routing_keys.check_routing_view("Qwen3.5", 3, "decode", (pre, post))   # off: nothing to check
    assert capsys.readouterr().out == ""

    switch(True)
    with pytest.raises(RuntimeError, match="not the rotated one"):
        routing_keys.check_routing_view("Qwen3.5", 3, "decode", (pre, post))   # routing on the unrotated tensors
    routed = routing_keys.routing_view(pre, post)
    routing_keys.check_routing_view("Qwen3.5", 3, "decode", (routed, post), (None, None))
    routing_keys.check_routing_view("Qwen3.5", 7, "decode", (routed, post))
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "[semblo] routing keys: post-RoPE (SEMBLO_ROUTING_KEYS=post) on Qwen3.5 layer 3, decode "
        "(checked: the routing tensors equal the rotated ones)"
    ]
    # a later layer is still checked, even though the line is not repeated
    with pytest.raises(RuntimeError, match="layer 9"):
        routing_keys.check_routing_view("Qwen3.5", 9, "decode", (pre, post))


# ---------------------------------------------------------------- Qwen3.5


class _Linear(nn.Module):
    """The fused projection: ``(output, None)`` like vLLM's linear layers, with a sliceable weight."""

    def __init__(self, out_features: int, in_features: int, gen: torch.Generator) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, generator=gen), requires_grad=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight), None


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, gen: torch.Generator) -> None:
        super().__init__()
        self.weight = nn.Parameter(1 + 0.1 * torch.randn(dim, generator=gen), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


class _PartialRope(nn.Module):
    """Rotates the first ``rot`` dims of each head (NeoX halves), out of place, like the stock
    module's forward_native (so _rope_mutates_inputs reads False, as in the runtime)."""

    def __init__(self, head_dim: int, rot: int) -> None:
        super().__init__()
        self.head_dim, self.rot = head_dim, rot
        self._forward_method = self.forward_native

    def forward(self, positions, query, key=None):
        return self._forward_method(positions, query, key)

    def forward_native(self, positions, query, key=None):
        return self._rotate(positions, query), (None if key is None else self._rotate(positions, key))

    def _rotate(self, positions: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(shape[0], -1, self.head_dim)
        half = self.rot // 2
        inv = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32) / half))
        ang = positions.to(torch.float32)[:, None] * inv[None]            # [T, half]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        x1, x2, rest = x[..., :half], x[..., half:self.rot], x[..., self.rot:]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1).reshape(shape)


@pytest.fixture
def plugin():
    qwen = pytest.importorskip("semblo.vllm_plugin.qwen3_5_model")
    from semblo.vllm_plugin import model

    return qwen, model


def _qwen_layer(qwen, *, gate: bool = True):
    """The pieces of SembloQwen3_5Attention its two projections read, nothing else."""

    gen = torch.Generator().manual_seed(0)
    cls = qwen.SembloQwen3_5Attention
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    hidden, hq, hkv, d = 16, 4, 2, 8
    layer.num_heads, layer.num_kv_heads, layer.head_dim = hq, hkv, d
    layer.q_size, layer.kv_size = hq * d, hkv * d
    layer.attn_output_gate = gate
    layer.layer_idx = 3
    layer.qkv_proj = _Linear((2 if gate else 1) * hq * d + 2 * hkv * d, hidden, gen)
    layer.q_norm = _RMSNorm(d, gen)
    layer.k_norm = _RMSNorm(d, gen)
    layer.rotary_emb = _PartialRope(d, rot=4)       # partial, as Qwen3.5's 0.25 of the head
    return layer


def _inputs(tokens: int = 6):
    gen = torch.Generator().manual_seed(2)
    return torch.arange(tokens) + 5, torch.randn(tokens, 16, generator=gen)


@pytest.mark.parametrize("gate", [True, False])
def test_qwen3_5_projection_routes_on_the_rotated_q_and_k(plugin, switch, gate):
    qwen, _model = plugin
    layer = _qwen_layer(qwen, gate=gate)
    positions, hidden = _inputs()

    switch(False)
    off = layer._project_qkv(positions, hidden)
    switch(True)
    on = layer._project_qkv(positions, hidden)
    q_post, k_post, v, q_pre, k_pre, g = on

    # the attention's inputs do not depend on the switch
    for i, name in ((0, "q_post"), (1, "k_post"), (2, "v"), (5, "gate")):
        assert (off[i] is None) == (on[i] is None), name
        if on[i] is not None:
            assert torch.equal(off[i], on[i]), name
    # the routing slots: pre-RoPE without the switch, the rotated tensors with it, same shapes
    assert q_pre.shape == off[3].shape == (6, 4, 8) and k_pre.shape == off[4].shape == (6, 2, 8)
    assert torch.equal(q_pre, q_post.reshape(q_pre.shape))
    assert torch.equal(k_pre, k_post.reshape(k_pre.shape))
    assert not torch.equal(off[3], q_post.reshape(q_pre.shape))
    assert not torch.equal(off[4], k_post.reshape(k_pre.shape))
    # and pre-RoPE really is pre-RoPE: rotating it gives the attention's input
    q_rot, k_rot = layer.rotary_emb(positions, off[3].flatten(-2, -1), off[4].flatten(-2, -1))
    torch.testing.assert_close(q_rot, q_post)
    torch.testing.assert_close(k_rot, k_post)


@pytest.mark.parametrize("kv_only", ["1", "0"])
def test_qwen3_5_kv_projection_routes_on_the_rotated_k(plugin, switch, monkeypatch, kv_only):
    """_project_kv feeds the streamed prefill's key store; "0" is the fused fallback."""

    qwen, _model = plugin
    monkeypatch.setenv("SEMBLO_STREAM_KV_ONLY", kv_only)
    layer = _qwen_layer(qwen)
    positions, hidden = _inputs()

    switch(False)
    k_post_off, v_off, k_pre_off = layer._project_kv(positions, hidden)
    _q, k_post_fused, v_fused, _qp, k_pre_fused, _g = layer._project_qkv(positions, hidden)
    switch(True)
    k_post, v, k_pre = layer._project_kv(positions, hidden)

    assert torch.equal(k_post, k_post_off) and torch.equal(v, v_off)
    torch.testing.assert_close(k_post, k_post_fused)
    torch.testing.assert_close(v, v_fused)
    torch.testing.assert_close(k_pre_off, k_pre_fused)
    assert k_pre.shape == k_pre_off.shape == (6, 2, 8)
    assert torch.equal(k_pre, k_post.reshape(k_pre.shape))
    assert not torch.equal(k_pre_off, k_pre)


def _session(model, layer, tokens: int):
    """A StreamedPrefillSession without its runtime-state wiring (maybe_begin_prompt)."""

    session = model.StreamedPrefillSession.__new__(model.StreamedPrefillSession)
    session.layer = layer
    session.positions = torch.arange(tokens)
    session.tokens = tokens
    session.profiling = False
    session.k_post = session.v = session.k_pre = None
    return session


def test_the_sliced_prefill_fits_and_selects_on_the_rotated_q_and_k(plugin, switch, monkeypatch, capsys):
    """In the streamed prefill (SEMBLO_PREFILL_CHUNK=32768, the default) the keys pass stages the
    index's key store through _project_kv, and the queries pass routes q_pre from _project_qkv."""

    qwen, model = plugin
    monkeypatch.setenv("SEMBLO_STREAM_KV_ONLY", "1")
    layer = _qwen_layer(qwen)
    positions, hidden = _inputs(8)
    seen: dict[str, torch.Tensor] = {}

    def run_product_attention(q_post, q_pre, flat_positions, *, query_offset):
        seen["q_post"], seen["q_pre"] = q_post.clone(), q_pre.clone()
        return torch.zeros_like(q_post)

    layer._run_product_attention = run_product_attention
    layer._finish_output = lambda attn, gate: attn

    runs = {}
    for post in (False, True):
        switch(post)
        session = _session(model, layer, 8)
        for s, e in ((0, 3), (3, 8)):
            session.keys(positions[s:e], hidden[s:e], s, e)
        session.queries(positions[:5], positions[:5], hidden[:5], 0)
        runs[post] = (session.k_post.clone(), session.v.clone(), session.k_pre.clone(),
                      seen["q_post"], seen["q_pre"], capsys.readouterr().out)

    k_post, v, k_pre, q_post, q_pre, out = runs[True]
    k_post0, v0, k_pre0, q_post0, q_pre0, out0 = runs[False]
    assert torch.equal(k_post, k_post0) and torch.equal(v, v0) and torch.equal(q_post, q_post0)
    assert torch.equal(k_pre, k_post.reshape(k_pre.shape))        # the index is fitted on rotated keys
    assert torch.equal(q_pre, q_post.reshape(q_pre.shape))        # the selection routes rotated queries
    assert not torch.equal(k_pre0, k_pre) and not torch.equal(q_pre0, q_pre)
    assert out0 == ""
    assert "[semblo] routing keys: post-RoPE (SEMBLO_ROUTING_KEYS=post) on Qwen3.5 layer 3, sliced prefill, keys pass" in out
    assert "[semblo] routing keys: post-RoPE (SEMBLO_ROUTING_KEYS=post) on Qwen3.5 layer 3, sliced prefill, queries pass" in out
    assert out.count("routing keys: post-RoPE") == 2               # once per path, not per slice
