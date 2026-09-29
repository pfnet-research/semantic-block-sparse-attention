"""The Nemotron-H port: runtime-weight names, the prepared config, the wrapper's projection."""
from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn

from semblo.vllm_plugin.runtime_weights import canonical_runtime_name


def test_runtime_names_accept_the_mixer_block():
    """Nemotron-H keeps its attention under ``mixer``; the canonical name stays
    ``self_attn`` so one runtime file serves every family."""
    assert canonical_runtime_name("model.layers.5.mixer.semblo_key_router.weight") \
        == "layers.5.self_attn.semblo_key_router.weight"
    assert canonical_runtime_name("model.layers.5.mixer.qkv_proj.weight") is None
    assert canonical_runtime_name("model.layers.5.mixer.in_proj.weight") is None


def test_the_prepared_config_lists_the_attention_layers_and_the_identity_router_shapes(tmp_path):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import prepare_nemotron_checkpoint as prep

    cfg = {"layers_block_type": ["mamba", "moe", "mamba", "attention", "moe", "attention"],
           "mtp_layers_block_type": ["attention", "moe"], "num_hidden_layers": 6,
           "chunk_size": 128, "n_groups": 8, "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128}
    out, attention, notes = prep.vllm_config_from_snapshot(cfg)
    assert out == cfg                      # nothing rewritten (transformers 5 reads this form)
    assert attention == [3, 5] and "MEM*E*" in notes[0]
    assert prep.vllm_config_from_snapshot({**cfg, "layers_block_type": ["linear_attention", "moe", "full_attention"], "num_hidden_layers": 3})[1] == [2]
    dp = {"semblo_index_dim": 128, "semblo_num_factors": 2}
    tensors = prep.identity_router_tensors(cfg, dp, attention)
    assert sorted(k for k in tensors if k.startswith("layers.3.")) == [
        "layers.3.self_attn.semblo_key_router.weight"]
    w = tensors["layers.3.self_attn.semblo_key_router.weight"]
    assert w.shape == (2, 128, 128) and torch.equal(w[1], torch.eye(128))


def _fake_nemotron_modules(monkeypatch):
    """The vLLM modules the port imports, as minimal fakes (like the plugin's own tests)."""
    from test_vllm_semblo_plugin import _import_model_module_with_fake_vllm
    model_module, _ = _import_model_module_with_fake_vllm(monkeypatch, attention_context=(object(), None, None, None))

    class FakeQKV(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8 + 4 + 4, 3))   # q 2 heads x 4, k 1 x 4, v 1 x 4

        def forward(self, x):
            return x @ self.weight.t(), None

    class FakeOut(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 3, bias=False)

        def forward(self, x):
            return self.linear(x), None

    class FakeAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.impl = types.SimpleNamespace(logits_soft_cap=None)
            self.layer_name = "model.layers.5.mixer.attn"

        def forward(self, q, k, v):
            return q

    class NemotronHAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size, self.num_heads, self.num_kv_heads, self.head_dim = 3, 2, 1, 4
            self.q_size, self.kv_size, self.scaling = 8, 4, 0.5
            self.qkv_proj, self.o_proj, self.attn = FakeQKV(), FakeOut(), FakeAttn()

    class NemotronHAttentionDecoderLayer(nn.Module):
        pass

    class NemotronHForCausalLM(nn.Module):
        pass

    nem = types.ModuleType("vllm.model_executor.models.nemotron_h")
    nem.NemotronHAttention = NemotronHAttention
    nem.NemotronHAttentionDecoderLayer = NemotronHAttentionDecoderLayer
    nem.NemotronHForCausalLM = NemotronHForCausalLM
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.nemotron_h", nem)
    monkeypatch.delitem(sys.modules, "semblo.vllm_plugin.nemotron_h_model", raising=False)
    import importlib
    return importlib.import_module("semblo.vllm_plugin.nemotron_h_model"), NemotronHAttention, model_module


def test_the_projection_returns_the_same_keys_to_the_router_and_the_attention(monkeypatch):
    """No rotary embedding: q_post/k_post are the flat views of q_pre/k_pre;
    V is a copy of its slice; K/V-only projection agrees with the fused one."""
    port, NemotronHAttention, model_module = _fake_nemotron_modules(monkeypatch)
    original = NemotronHAttention()
    layer = port.SembloNemotronHAttention.__new__(port.SembloNemotronHAttention)
    nn.Module.__init__(layer)
    layer.qkv_proj, layer.o_proj, layer.attn = original.qkv_proj, original.o_proj, original.attn
    layer.num_heads, layer.num_kv_heads, layer.head_dim = 2, 1, 4
    layer.q_size, layer.kv_size = 8, 4
    x = torch.randn(5, 3)
    q_post, k_post, v, q_pre, k_pre, gate = port.SembloNemotronHAttention._project_qkv(layer, torch.arange(5), x)
    qkv = x @ original.qkv_proj.weight.t()
    assert gate is None
    assert torch.equal(q_post, qkv[:, :8]) and torch.equal(k_post, qkv[:, 8:12]) and torch.equal(v, qkv[:, 12:].view(5, 1, 4))
    assert torch.equal(q_pre, q_post.view(5, 2, 4)) and torch.equal(k_pre, k_post.view(5, 1, 4))
    assert v.data_ptr() != qkv.data_ptr() and v.is_contiguous()
    k2, v2, k2_pre = port.SembloNemotronHAttention._project_kv(layer, torch.arange(5), x)
    assert torch.allclose(k2, k_post, atol=1e-6) and torch.allclose(v2, v, atol=1e-6) and torch.allclose(k2_pre, k_pre, atol=1e-6)
    out = port.SembloNemotronHAttention._finish_output(layer, torch.randn(5, 2, 4), None)
    assert out.shape == (5, 3)


def test_centring_the_keys_keeps_every_query_ranking_and_splits_a_shared_direction(monkeypatch):
    """Keys that share one dominant direction all normalise to (almost) the
    same unit vector, and a spherical k-means over them collapses; subtracting
    the prompt's mean key leaves q.k rankings untouched and lets the partition
    see the differences (SEMBLO_ROUTING_CENTER)."""
    from semblo.inference.flat_index import assign_flat_buckets, fit_flat_codebook, routing_space
    g = torch.Generator().manual_seed(0)
    shared = torch.randn(1, 128, generator=g) * 30.0
    own = torch.randn(4096, 128, generator=g)
    keys = shared + own
    q = torch.randn(8, 128, generator=g)
    centred = keys - keys.mean(dim=0, keepdim=True)
    # q.k - q.(k - mu) = q.mu: one constant per query, so the ranking of the
    # tokens under any query is the same (up to float rounding of near-ties)
    diff = q @ keys.t() - q @ centred.t()
    assert float((diff - diff[:, :1]).abs().max()) < 1e-3 * float(diff.abs().max())
    top_raw = (q @ keys.t()).topk(64, dim=-1).indices
    top_centred = (q @ centred.t()).topk(64, dim=-1).indices
    overlap = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top_raw, top_centred)]
    assert min(overlap) >= 62
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "whole")
    monkeypatch.setenv("SEMBLO_FLAT_DETERMINISTIC", "1")
    def largest_fraction(x):
        c = fit_flat_codebook(routing_space(x), 8, num_iters=8, sample_size=None, generator=torch.Generator().manual_seed(1))
        a = assign_flat_buckets(routing_space(x), c)
        return float(torch.bincount(a, minlength=8).max()) / x.shape[0]
    unit = torch.nn.functional.normalize(keys, dim=-1)
    assert float((unit[:256] @ unit[256:512].t()).mean()) > 0.95   # the shared direction wins the cosine
    assert largest_fraction(centred) < 0.4


def test_kmeanspp_init_and_the_whitened_partition_on_anisotropic_keys(monkeypatch):
    """k-means++ picks C distinct sample points; a whitened partition of keys
    whose variance sits in two directions is still balanced."""
    from semblo.inference.flat_index import _kmeanspp_init, fit_flat_codebook
    g = torch.Generator().manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(2048, 32, generator=g), dim=-1)
    c = _kmeanspp_init(x, 16, torch.Generator().manual_seed(1))
    assert c.shape == (16, 32)
    assert len({tuple(round(v, 6) for v in row.tolist()) for row in c}) == 16
    monkeypatch.setenv("SEMBLO_ROUTING_NORMALIZE", "whole")
    cents = fit_flat_codebook(x, 16, num_iters=4, sample_size=None, generator=torch.Generator().manual_seed(2), init="kmeanspp")
    assert cents.shape == (16, 32) and torch.isfinite(cents).all()
    with pytest.raises(ValueError):
        fit_flat_codebook(x, 16, num_iters=1, sample_size=None, init="bogus")


def test_the_int32_safe_conv_wrapper_reaches_mamba_mixer2_and_segments_a_long_call(monkeypatch):
    """The wrapper must replace the symbol in every module that imported it by
    name (mamba_mixer2 for Mamba-2, as gdn_linear_attn for GDN), and a call
    past the int32 boundary is split into segments whose later parts start
    from the conv-state cache line, with the prefix-caching block indices
    dropped."""
    import importlib
    calls = []

    def fake_conv(x, weight, bias, conv_states, query_start_loc, cache_indices=None, has_initial_state=None,
                  activation="silu", pad_slot_id=-1, metadata=None, validate_data=False, **kwargs):
        calls.append(dict(tokens=int(x.shape[1]), has_init=None if has_initial_state is None else bool(has_initial_state.all()),
                          stride=int(x.stride(1)), kw=dict(kwargs)))
        return x.clone()

    mods = {}
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.mamba",
                 "vllm.model_executor.layers.mamba.ops", "vllm.model_executor.layers.mamba.ops.causal_conv1d",
                 "vllm.model_executor.layers.mamba.gdn_linear_attn", "vllm.model_executor.layers.mamba.mamba_mixer2"):
        mods[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mods[name])
    ops = mods["vllm.model_executor.layers.mamba.ops.causal_conv1d"]
    ops.causal_conv1d_fn = fake_conv; ops.PAD_SLOT_ID = -1
    mods["vllm.model_executor.layers.mamba.gdn_linear_attn"].causal_conv1d_fn = fake_conv
    mods["vllm.model_executor.layers.mamba.mamba_mixer2"].causal_conv1d_fn = fake_conv
    monkeypatch.delitem(sys.modules, "semblo.vllm_plugin.gdn_conv_patch", raising=False)
    patch = importlib.import_module("semblo.vllm_plugin.gdn_conv_patch")
    monkeypatch.setattr(patch, "_INT32_ELEMS", 64)
    monkeypatch.setattr(patch, "_ALIGN", 4)
    patch.install()
    mm2 = mods["vllm.model_executor.layers.mamba.mamba_mixer2"]
    assert mm2.causal_conv1d_fn is not fake_conv and getattr(mm2.causal_conv1d_fn, "_semblo_int32_safe", False)
    assert mods["vllm.model_executor.layers.mamba.gdn_linear_attn"].causal_conv1d_fn is mm2.causal_conv1d_fn
    wide = torch.randn(40, 10)
    x = wide[:, 2:6].T                                   # (dim 4, tokens 40), token stride 10
    out = mm2.causal_conv1d_fn(x, None, None, None, torch.tensor([0, 40]), has_initial_state=torch.zeros(1, dtype=torch.bool),
                               block_idx_first_scheduled_token=torch.tensor([0]), block_size_to_align=128)
    assert torch.equal(out, x)
    assert [c["tokens"] for c in calls] == [12, 12, 12, 4]
    assert [c["has_init"] for c in calls] == [False, True, True, True]
    assert all(c["stride"] == 4 for c in calls)
    assert all(c["kw"].get("block_idx_first_scheduled_token") is None and c["kw"].get("block_size_to_align") == 0 for c in calls)
    calls.clear()
    mm2.causal_conv1d_fn(torch.randn(4, 8), None, None, None, torch.tensor([0, 8]), block_size_to_align=128)
    assert calls[0]["tokens"] == 8 and calls[0]["kw"]["block_size_to_align"] == 128
