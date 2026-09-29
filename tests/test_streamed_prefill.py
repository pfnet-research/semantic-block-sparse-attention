"""The streamed prefill: the same layers, walked a slice at a time, give the whole answer.

These run on the CPU against stand-ins with vLLM's signatures.  They check the
bookkeeping (the residual stream, the order of norms, the metadata a
linear-attention layer is shown for each slice and the state it carries), not
the kernels.
"""

from __future__ import annotations

import dataclasses
import types

import pytest
import torch
from torch import nn

from semblo.vllm_plugin import streamed_prefill
from semblo.vllm_plugin.streamed_prefill import (
    NemotronHLayerShape,
    PrefillStreamer,
    single_prefill,
    slice_linear_attention_metadata,
    slice_mamba2_metadata,
)


DIM = 6


@dataclasses.dataclass
class FakeMetadata:
    """The fields of vLLM's GDNAttentionMetadata a slice touches or must keep."""

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int
    has_initial_state: torch.Tensor | None = None
    non_spec_query_start_loc: torch.Tensor | None = None
    non_spec_state_indices_tensor: torch.Tensor | None = None
    spec_sequence_masks: torch.Tensor | None = None
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


def whole_prompt_metadata(tokens: int, slot: int = 3, has_state: bool = False):
    return FakeMetadata(
        num_prefills=1, num_prefill_tokens=tokens, num_decodes=0,
        num_decode_tokens=0, num_spec_decodes=0, num_spec_decode_tokens=0,
        num_actual_tokens=tokens,
        has_initial_state=torch.tensor([has_state]),
        non_spec_query_start_loc=torch.tensor([0, tokens], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([slot], dtype=torch.int32),
        nums_dict={"whole": True}, batch_ptr=torch.zeros(1), token_chunk_offset_ptr=torch.zeros(1),
    )


class FakeNorm(nn.Module):
    """GemmaRMSNorm's arithmetic and vLLM's (x, residual) -> (x, residual) contract."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim) * 0.1)

    def _norm(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + 1e-6) * (1.0 + self.weight.float())).to(x.dtype)

    def forward(self, x, residual=None):
        if residual is None:
            return self._norm(x)
        x = x + residual
        return self._norm(x), x


class FakeRecurrentAttention(nn.Module):
    """A one-line recurrence with vLLM's state contract.

    s(t) = 0.5 s(t-1) + x(t); the state is read from a cache row named by the
    metadata when ``has_initial_state`` says so, zero otherwise, and written
    back at the end -- exactly the shape of vLLM's GDN prefill path.
    """

    def __init__(self, prefix: str, cache: dict, seen: list) -> None:
        super().__init__()
        self.prefix = prefix
        self.cache = cache
        self.seen = seen

    def forward(self, hidden_states, output):
        md = streamed_prefill._forward_context().attn_metadata[self.prefix]
        n = int(md.num_actual_tokens)
        assert hidden_states.shape[0] == n, "the slice and its metadata disagree"
        assert md.non_spec_query_start_loc.tolist() == [0, n]
        assert md.num_prefills == 1 and md.num_decodes == 0
        slot = int(md.non_spec_state_indices_tensor[0])
        self.seen.append((n, bool(md.has_initial_state[0]), md.nums_dict))
        state = self.cache[slot] if bool(md.has_initial_state[0]) else torch.zeros(DIM)
        for t in range(n):
            state = 0.5 * state + hidden_states[t]
            output[t] = state
        self.cache[slot] = state


class FakeSparseAttention(nn.Module):
    """The sparse layer's streaming protocol over a position-wise map."""

    def __init__(self, seen: list) -> None:
        super().__init__()
        self.proj = nn.Linear(DIM, DIM)
        self.seen = seen

    def _positions_for_runtime(self, positions):
        return positions, positions

    @staticmethod
    def _slice_positions(rope_positions, start, stop):
        return rope_positions[start:stop]

    def begin_streamed_prefill(self, positions):
        attn = self
        seen = self.seen

        class Session:
            def __init__(self):
                self.keys_seen = []
                self.ingested = False

            def keys(self, rope, hidden, start, stop):
                assert rope.tolist() == list(range(start, stop))
                self.keys_seen.append((start, stop))

            def ingest(self):
                assert not self.ingested
                self.ingested = True
                seen.append(("ingest", list(self.keys_seen)))

            def queries(self, rope, pos, hidden, start):
                assert self.ingested, "queries before the index exists"
                assert int(pos[0]) == start and rope.tolist() == pos.tolist()
                return attn.proj(hidden)

            def finish(self):
                seen.append(("finish",))

        return Session()

    def whole(self, hidden):
        return self.proj(hidden)


class FakeLayer(nn.Module):
    def __init__(self, kind: str, prefix: str, cache: dict, seen: list) -> None:
        super().__init__()
        self.input_layernorm = FakeNorm(DIM)
        self.post_attention_layernorm = FakeNorm(DIM)
        self.mlp = nn.Sequential(nn.Linear(DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM))
        self.layer_scale = False
        if kind == "linear":
            self.linear_attn = FakeRecurrentAttention(prefix, cache, seen)
        else:
            self.self_attn = FakeSparseAttention(seen)


class FakeModel(nn.Module):
    def __init__(self, kinds, cache, seen) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(11, DIM)
        self.layers = nn.ModuleList(
            FakeLayer(k, f"model.layers.{i}.linear_attn", cache, seen)
            for i, k in enumerate(kinds)
        )
        self.norm = FakeNorm(DIM)
        self.start_layer, self.end_layer = 0, len(kinds)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)


def stock_forward(model, input_ids, store):
    """vLLM's Qwen3NextModel.forward and Qwen3NextDecoderLayer.forward, whole."""

    hidden = model.embed_input_ids(input_ids)
    residual = None
    for layer in model.layers:
        if residual is None:
            residual = hidden
            hidden = layer.input_layernorm(hidden)
        else:
            hidden, residual = layer.input_layernorm(hidden, residual)
        out = torch.empty_like(hidden)
        if hasattr(layer, "linear_attn"):
            layer.linear_attn(hidden_states=hidden, output=out)
        else:
            out = layer.self_attn.whole(hidden)
        hidden, residual = layer.post_attention_layernorm(out, residual)
        hidden = layer.mlp(hidden)
    hidden, _ = model.norm(hidden, residual)
    return hidden


def _use_store(monkeypatch, store):
    context = types.SimpleNamespace(attn_metadata=store)
    monkeypatch.setattr(streamed_prefill, "_forward_context", lambda: context)
    monkeypatch.setattr(
        streamed_prefill, "_conv_metadata",
        lambda starts, device: ({"slice": starts.tolist()}, None, None),
    )


@pytest.mark.parametrize("chunk", [3, 8, 16])
def test_the_streamed_prefill_equals_the_whole_forward(monkeypatch, chunk):
    torch.manual_seed(0)
    tokens = 8
    kinds = ["linear", "linear", "sparse", "linear", "sparse"]
    input_ids = torch.randint(0, 11, (tokens,))
    positions = torch.arange(tokens)
    prefixes = [f"model.layers.{i}.linear_attn" for i, k in enumerate(kinds) if k == "linear"]

    # The whole answer, from one call per layer with the whole prompt's metadata.
    cache_whole, seen_whole = {}, []
    model = FakeModel(kinds, cache_whole, seen_whole)
    whole_md = whole_prompt_metadata(tokens)
    _use_store(monkeypatch, {p: whole_md for p in prefixes})
    expected = stock_forward(model, input_ids, None)

    # The same weights, walked a slice at a time.
    cache, seen = {}, []
    for layer in model.layers:
        block = getattr(layer, "linear_attn", None)
        if block is not None:
            block.cache, block.seen = cache, seen
        else:
            layer.self_attn.seen = seen
    store = {p: whole_md for p in prefixes}
    _use_store(monkeypatch, store)
    streamer = PrefillStreamer(
        model, chunk=chunk,
        sparse_layers={i: model.layers[i].self_attn for i, k in enumerate(kinds) if k == "sparse"},
        say=lambda _: None,
    )
    assert streamer.wants(tokens) == (tokens >= chunk)
    got = streamer.run(input_ids, positions)

    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    # The store is handed back whole, and the carried state ends where the
    # whole run's did.
    assert all(store[p] is whole_md for p in prefixes)
    torch.testing.assert_close(cache[3], cache_whole[3])

    # What each linear-attention layer was shown: one slice per chunk, the
    # first without a state and every later one with, and metadata cut for
    # the slice rather than the whole prompt's.
    spans = [(s, min(s + chunk, tokens)) for s in range(0, tokens, chunk)]
    per_layer = [(e - s, s > 0, {"slice": [0, e - s]}) for s, e in spans]
    linear_seen = [x for x in seen if isinstance(x[0], int)]
    assert linear_seen == per_layer * kinds.count("linear")
    # And the sparse layers saw every slice's keys before any query.
    assert [x for x in seen if x[0] == "ingest"] == [("ingest", spans)] * kinds.count("sparse")


def test_a_slice_keeps_the_rows_it_continues_and_renames_only_its_own_extent():
    whole = whole_prompt_metadata(10, slot=7)
    first = slice_linear_attention_metadata(
        whole, 4, True, "cpu", conv_metadata=lambda s, d: ("nums", "bp", "tco"))
    later = slice_linear_attention_metadata(
        whole, 4, False, "cpu", conv_metadata=lambda s, d: ("nums", "bp", "tco"))
    for md in (first, later):
        assert md.num_actual_tokens == 4 and md.num_prefill_tokens == 4
        assert md.non_spec_query_start_loc.tolist() == [0, 4]
        assert md.non_spec_state_indices_tensor.tolist() == [7]
        assert md.num_prefills == 1 and md.num_decodes == 0
        assert (md.nums_dict, md.batch_ptr, md.token_chunk_offset_ptr) == ("nums", "bp", "tco")
    assert first.has_initial_state.tolist() == [False]
    assert later.has_initial_state.tolist() == [True]
    # The whole is untouched: dataclasses.replace copies.
    assert whole.num_actual_tokens == 10 and whole.nums_dict == {"whole": True}


def test_a_prompt_that_already_has_state_keeps_it_in_its_first_slice(monkeypatch):
    """The first slice inherits the whole's has_initial_state, read, not assumed."""

    torch.manual_seed(1)
    kinds = ["linear"]
    cache, seen = {}, []
    model = FakeModel(kinds, cache, seen)
    whole_md = whole_prompt_metadata(4, slot=2, has_state=True)
    cache[2] = torch.ones(DIM)
    _use_store(monkeypatch, {"model.layers.0.linear_attn": whole_md})
    PrefillStreamer(model, chunk=2, sparse_layers={}, say=lambda _: None).run(
        torch.zeros(4, dtype=torch.long), torch.arange(4))
    assert [x[1] for x in seen] == [True, True]


def test_only_a_long_single_prefill_is_wanted(monkeypatch):
    model = FakeModel(["linear"], {}, [])
    streamer = PrefillStreamer(model, chunk=4, sparse_layers={}, say=lambda _: None)
    prefix = "model.layers.0.linear_attn"

    _use_store(monkeypatch, None)                      # a dummy run: the profile run above all
    assert streamer.wants(4) and not streamer.wants(3)

    _use_store(monkeypatch, {prefix: whole_prompt_metadata(4)})
    assert streamer.wants(4)

    decode = dataclasses.replace(whole_prompt_metadata(4), num_decodes=1)
    _use_store(monkeypatch, {prefix: decode})
    assert not streamer.wants(4)
    assert not single_prefill(decode)

    two = dataclasses.replace(whole_prompt_metadata(4), num_prefills=2)
    _use_store(monkeypatch, {prefix: two})
    assert not streamer.wants(4)

    spec = dataclasses.replace(whole_prompt_metadata(4), spec_sequence_masks=torch.tensor([True]))
    _use_store(monkeypatch, {prefix: spec})
    assert not streamer.wants(4)


def test_a_stock_full_attention_layer_cannot_be_sliced():
    model = FakeModel(["linear", "sparse"], {}, [])
    with pytest.raises(ValueError, match="neither a linear-attention layer nor a sparse one"):
        PrefillStreamer(model, chunk=4, sparse_layers={}, say=lambda _: None)
    with pytest.raises(ValueError, match="positive slice size"):
        PrefillStreamer(model, chunk=0, sparse_layers={1: model.layers[1].self_attn})


def test_vllm_ops_run_their_cuda_kernels_only_while_the_block_is_open():
    from semblo.vllm_plugin.streamed_prefill import vllm_cuda_kernels

    class Op(nn.Module):
        """The shape of vLLM's CustomOp: a dispatch attribute and two implementations."""

        def __init__(self, on_cuda: bool = False):
            super().__init__()
            self._forward_method = self.forward_cuda if on_cuda else self.forward_native

        def forward(self, x):
            return self._forward_method(x)

        def forward_native(self, x):
            return x + 1

        def forward_cuda(self, x):
            return x + 100

    class Plain(nn.Module):
        def forward(self, x):
            return x

    model = nn.Sequential(Op(), Op(on_cuda=True), Plain(), nn.Sequential(Op()))
    x = torch.zeros(1)
    assert [m(x).item() for m in model if isinstance(m, Op)] == [1, 100]
    with vllm_cuda_kernels(model, kernels="cuda") as swapped:
        assert swapped == 2, "the two native ops, not the one already on its CUDA path"
        assert [m(x).item() for m in model if isinstance(m, Op)] == [100, 100]
        assert model[3][0](x).item() == 100
    assert [m(x).item() for m in model if isinstance(m, Op)] == [1, 100]
    assert model[3][0](x).item() == 1


def test_the_rotary_modules_can_take_their_own_implementation():
    from semblo.vllm_plugin.streamed_prefill import vllm_cuda_kernels

    class FakeRotaryEmbedding(nn.Module):
        def __init__(self):
            super().__init__()
            self._forward_method = self.forward_native

        def forward(self, positions, q, k=None):
            return self._forward_method(positions, q, k)

        def forward_native(self, positions, q, k=None):
            return q * 3, None if k is None else k * 3

        def forward_cuda(self, positions, q, k=None):
            return q * 300, None if k is None else k * 300

    class Act(nn.Module):
        def __init__(self):
            super().__init__()
            self._forward_method = self.forward_native

        def forward(self, x):
            return self._forward_method(x)

        def forward_native(self, x):
            return x + 1

        def forward_cuda(self, x):
            return x + 100

    model = nn.Sequential(FakeRotaryEmbedding(), Act())
    rope, act = model[0], model[1]
    q = torch.full((2, 4), 0.5, dtype=torch.bfloat16)
    pos = torch.arange(2)

    with vllm_cuda_kernels(model, kernels="cuda", rope="fp32") as n:
        assert n == 2
        out, _ = rope(pos, q)
        assert out.dtype == torch.bfloat16 and float(out[0, 0]) == 1.5
        assert rope._forward_method.__name__ == "forward_native", "out of place, and it says so"
        assert float(act(torch.zeros(1))) == 100
    assert float(rope(pos, q)[0][0, 0]) == 1.5 and float(act(torch.zeros(1))) == 1

    with vllm_cuda_kernels(model, kernels="native") as n:
        assert n == 0
    with vllm_cuda_kernels(model, kernels="cuda", rope="native") as n:
        assert n == 1 and float(rope(pos, q)[0][0, 0]) == 1.5 and float(act(torch.zeros(1))) == 100

    with pytest.raises(ValueError, match="SEMBLO_STREAM_KERNELS"):
        with vllm_cuda_kernels(model, kernels="triton"):
            pass


def test_the_default_runs_native_forms_through_inductor(monkeypatch):
    """The default is the compiled model's numerics, one compile per module."""

    from semblo.vllm_plugin.streamed_prefill import vllm_cuda_kernels

    class Op(nn.Module):
        def __init__(self):
            super().__init__()
            self._forward_method = self.forward_native

        def forward(self, x):
            return self._forward_method(x)

        def forward_native(self, x):
            return x * 2 + 1

        def forward_cuda(self, x):
            return x * 200

    model = nn.Sequential(Op(), Op())
    monkeypatch.delenv("SEMBLO_STREAM_KERNELS", raising=False)
    streamer = PrefillStreamer(FakeModel(["linear"], {}, []), chunk=4, sparse_layers={}, say=lambda _: None)
    assert streamer.kernels == "compiled" and streamer.rope == ""
    x = torch.full((3,), 2.0)
    with vllm_cuda_kernels(model) as swapped:      # the default: compiled native forms
        assert swapped == 2
        assert model[0]._forward_method.__name__ == "forward_native"
        assert torch.equal(model(x), torch.full((3,), 11.0))   # (2*2+1)*2+1, the native maths
        assert model[0]._semblo_compiled_native is model[0]._forward_method
    with vllm_cuda_kernels(model):
        assert model[0]._forward_method is model[0]._semblo_compiled_native, "compiled once per module"
    assert torch.equal(model(x), torch.full((3,), 11.0))


# -- the Nemotron-H shape: norm -> mixer, Mamba-2 state, per-token MoE ----------


@dataclasses.dataclass
class FakeMamba2Metadata:
    """The fields of vLLM's Mamba2AttentionMetadata a slice renames or must keep."""

    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int
    has_initial_states_p: torch.Tensor | None = None
    query_start_loc_p: torch.Tensor | None = None
    state_indices_tensor_p: torch.Tensor | None = None
    num_spec_decodes: int = 0
    spec_sequence_masks: torch.Tensor | None = None
    cu_chunk_seqlen_p: torch.Tensor | None = None
    last_chunk_indices_p: torch.Tensor | None = None
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None
    prep_initial_states: bool = False
    chunk_size: int = 0
    seq_idx_p: torch.Tensor | None = None


def whole_mamba2_metadata(tokens: int, slot: int = 5, chunk_size: int = 4):
    return FakeMamba2Metadata(
        num_prefills=1, num_prefill_tokens=tokens, num_decodes=0, num_decode_tokens=0,
        num_reqs=1, has_initial_states_p=torch.tensor([False]),
        query_start_loc_p=torch.tensor([0, tokens], dtype=torch.int32),
        state_indices_tensor_p=torch.tensor([slot], dtype=torch.int32),
        cu_chunk_seqlen_p=torch.tensor([0, tokens]), last_chunk_indices_p=torch.tensor([0]),
        nums_dict={"whole": True}, batch_ptr=torch.zeros(1), token_chunk_offset_ptr=torch.zeros(1),
        prep_initial_states=False, chunk_size=chunk_size, seq_idx_p=torch.zeros(tokens),
    )


class FakeMambaMixer2(nn.Module):
    """The Mamba-2 mixer's contract: state read when BOTH flags say so, written at the end."""

    def __init__(self, prefix: str, cache: dict, seen: list) -> None:
        super().__init__()
        self.prefix = prefix
        self.cache = cache
        self.seen = seen

    def forward(self, hidden_states):
        md = streamed_prefill._forward_context().attn_metadata[self.prefix]
        n = int(md.num_prefill_tokens + md.num_decode_tokens)
        assert hidden_states.shape[0] == n, "the slice and its metadata disagree"
        assert md.query_start_loc_p.tolist() == [0, n]
        assert md.num_prefills == 1 and md.num_decodes == 0 and md.num_reqs == 1
        assert md.cu_chunk_seqlen_p[-1].item() == n, "the chunk table is not the slice's"
        slot = int(md.state_indices_tensor_p[0])
        reads = bool(md.has_initial_states_p[0]) and bool(md.prep_initial_states)
        self.seen.append((n, bool(md.has_initial_states_p[0]), bool(md.prep_initial_states)))
        state = self.cache[slot] if reads else torch.zeros(DIM)
        output = torch.empty_like(hidden_states)
        for t in range(n):
            state = 0.5 * state + hidden_states[t]
            output[t] = state
        self.cache[slot] = state
        return output


class FakeMoE(nn.Module):
    """A per-token mixer with vLLM's layer-counter hazard made visible."""

    def __init__(self, name: str, seen: list) -> None:
        super().__init__()
        self.name = name
        self.seen = seen
        self.net = nn.Sequential(nn.Linear(DIM, 2 * DIM), nn.GELU(), nn.Linear(2 * DIM, DIM))

    def forward(self, hidden_states):
        ctx = streamed_prefill._forward_context()
        self.seen.append(("moe", self.name, getattr(ctx, "all_moe_layers", None)))
        return self.net(hidden_states)


class FakeNemotronLayer(nn.Module):
    def __init__(self, kind: str, idx: int, cache: dict, seen: list) -> None:
        super().__init__()
        self.norm = FakeNorm(DIM)
        if kind == "mamba":
            self.mixer = FakeMambaMixer2(f"model.layers.{idx}.mixer", cache, seen)
        elif kind == "moe":
            self.mixer = FakeMoE(f"model.layers.{idx}.mixer", seen)
        else:
            self.mixer = FakeSparseAttention(seen)


class FakeNemotronModel(nn.Module):
    def __init__(self, kinds, cache, seen) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(11, DIM)
        self.layers = nn.ModuleList(
            FakeNemotronLayer(k, i, cache, seen) for i, k in enumerate(kinds))
        self.norm_f = FakeNorm(DIM)
        self.start_layer, self.end_layer = 0, len(kinds)

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)


def stock_nemotron_forward(model, input_ids):
    """vLLM's NemotronHModel.forward and its decoder layers, whole."""

    hidden = model.embed_input_ids(input_ids)
    residual = None
    for layer in model.layers:
        if residual is None:
            residual = hidden
            hidden = layer.norm(hidden)
        else:
            hidden, residual = layer.norm(hidden, residual)
        mixer = layer.mixer
        hidden = mixer.whole(hidden) if isinstance(mixer, FakeSparseAttention) else mixer(hidden)
    hidden, _ = model.norm_f(hidden, residual)
    return hidden


def _use_nemotron_store(monkeypatch, store, moe_layers=None):
    context = types.SimpleNamespace(attn_metadata=store, all_moe_layers=moe_layers, moe_layer_index=0)
    monkeypatch.setattr(streamed_prefill, "_forward_context", lambda: context)
    monkeypatch.setattr(
        streamed_prefill, "_conv_metadata",
        lambda starts, device: ({"slice": starts.tolist()}, None, None),
    )
    monkeypatch.setattr(
        streamed_prefill, "_chunk_metadata",
        lambda starts, chunk_size: (starts.clone(), torch.tensor([0]), torch.zeros(int(starts[-1]))),
    )
    return context


@pytest.mark.parametrize("chunk", [4, 8, 16])
def test_the_nemotron_streamed_prefill_equals_the_whole_forward(monkeypatch, chunk):
    """Mamba-2 state through the cache rows, MoE per slice, attention whole; same answer."""

    torch.manual_seed(0)
    kinds = ["mamba", "moe", "mamba", "sparse", "moe", "mamba"]
    tokens = 14
    cache_whole, cache_stream = {}, {}
    seen_whole, seen_stream = [], []
    model = FakeNemotronModel(kinds, cache_whole, seen_whole)
    ids = torch.randint(0, 11, (tokens,))
    md = {f"model.layers.{i}.mixer": whole_mamba2_metadata(tokens) for i, k in enumerate(kinds) if k == "mamba"}
    _use_nemotron_store(monkeypatch, md, moe_layers=["model.layers.1.mixer", "model.layers.4.mixer"])
    with torch.no_grad():
        want = stock_nemotron_forward(model, ids)
    for layer in model.layers:
        if isinstance(layer.mixer, FakeMambaMixer2):
            layer.mixer.cache, layer.mixer.seen = cache_stream, seen_stream
        elif isinstance(layer.mixer, FakeMoE):
            layer.mixer.seen = seen_stream
        else:
            layer.mixer.seen = seen_stream
    streamer = PrefillStreamer(model, chunk=chunk, sparse_layers={3: model.layers[3].mixer},
                               say=lambda _: None, shape=NemotronHLayerShape())
    ctx = streamed_prefill._forward_context()
    with torch.no_grad():
        got = streamer.run(ids, torch.arange(tokens))
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
    assert cache_stream[5].allclose(cache_whole[5]), "the final Mamba-2 state differs"
    # The first slice starts from nothing; every later one reads the state with both flags set.
    spans = [(s, min(s + chunk, tokens)) for s in range(0, tokens, chunk)]
    mamba = [x for x in seen_stream if isinstance(x[0], int)]
    assert mamba == [(e - s, i > 0, i > 0) for _ in range(3) for i, (s, e) in enumerate(spans)]
    # The MoE op saw no layer list while the prefill streamed, and the list is back afterwards.
    assert all(x[2] is None for x in seen_stream if x[0] == "moe")
    assert ctx.all_moe_layers == ["model.layers.1.mixer", "model.layers.4.mixer"] and ctx.moe_layer_index == 0
    # The whole metadata is back in the store.
    for k, v in md.items():
        assert ctx.attn_metadata[k] is v


def test_a_mamba2_slice_renames_its_extent_and_arms_both_state_flags():
    whole = whole_mamba2_metadata(12, slot=9, chunk_size=4)
    chunk_calls = []

    def chunk_metadata(starts, chunk_size):
        chunk_calls.append((starts.tolist(), chunk_size))
        return ("cu", "last", "seq")

    first = slice_mamba2_metadata(whole, 5, True, "cpu",
                                  conv_metadata=lambda s, d: ("nums", "bp", "tco"), chunk_metadata=chunk_metadata)
    later = slice_mamba2_metadata(whole, 5, False, "cpu",
                                  conv_metadata=lambda s, d: ("nums", "bp", "tco"), chunk_metadata=chunk_metadata)
    for md in (first, later):
        assert md.num_prefill_tokens == 5 and md.num_prefills == 1 and md.num_decodes == 0
        assert md.query_start_loc_p.tolist() == [0, 5]
        assert md.state_indices_tensor_p.tolist() == [9] and md.chunk_size == 4
        assert (md.cu_chunk_seqlen_p, md.last_chunk_indices_p, md.seq_idx_p) == ("cu", "last", "seq")
        assert (md.nums_dict, md.batch_ptr, md.token_chunk_offset_ptr) == ("nums", "bp", "tco")
    assert first.has_initial_states_p.tolist() == [False] and first.prep_initial_states is False
    assert later.has_initial_states_p.tolist() == [True] and later.prep_initial_states is True
    assert chunk_calls == [([0, 5], 4), ([0, 5], 4)], "the chunk table is cut for the slice"
    assert whole.num_prefill_tokens == 12 and whole.prep_initial_states is False


def test_a_nemotron_layer_without_a_mixer_cannot_be_sliced():
    model = FakeNemotronModel(["mamba", "sparse"], {}, [])
    with pytest.raises(ValueError, match="no mixer and is not a sparse one"):
        PrefillStreamer(model, chunk=4, sparse_layers={}, say=lambda _: None, shape=NemotronHLayerShape())


def test_a_dummy_run_streams_zeros_through_a_stateful_block(monkeypatch):
    """No metadata (the profile run): the Mamba-2 mixer's output is not trusted."""

    class Garbage(nn.Module):
        prefix = "model.layers.0.mixer"

        def forward(self, hidden_states):
            return torch.full_like(hidden_states, float("nan"))

    class MambaGarbage(Garbage):
        pass

    MambaGarbage.__name__ = "MambaMixer2"
    model = FakeNemotronModel(["moe"], {}, [])
    layer = FakeNemotronLayer("moe", 0, {}, [])
    layer.mixer = MambaGarbage()
    model.layers = nn.ModuleList([layer])
    _use_nemotron_store(monkeypatch, None)
    out = PrefillStreamer(model, chunk=2, sparse_layers={}, say=lambda _: None,
                          shape=NemotronHLayerShape()).run(torch.zeros(4, dtype=torch.long), torch.arange(4))
    assert torch.isfinite(out).all()


def test_a_fused_moe_op_keeps_its_own_dispatch_while_the_block_is_open():
    """vLLM's fused MoE is a CustomOp, but its forward is opaque either way; it is not swapped."""

    class FusedMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self._forward_method = self.forward_cuda

        def forward_native(self, x):
            return x

        def forward_cuda(self, x):
            return x

    class SiluAndMul(FusedMoE):
        pass

    model = nn.Module()
    model.moe = FusedMoE()
    model.act = SiluAndMul()
    before = (model.moe._forward_method, model.act._forward_method)
    with streamed_prefill.vllm_cuda_kernels(model, kernels="compiled") as swapped:
        assert swapped == 1
        assert model.moe._forward_method is before[0]
        assert model.act._forward_method is not before[1]
    assert (model.moe._forward_method, model.act._forward_method) == before
