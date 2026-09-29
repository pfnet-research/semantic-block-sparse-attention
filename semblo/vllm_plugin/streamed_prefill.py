"""A long prefill, run as ordinary Python: layer by layer, each layer a slice at a time.

Why this is not a branch inside the compiled model: vLLM traces the inner
model with Dynamo once and then drops every guard
(``TorchCompileWithNoGuardsWrapper`` passes ``skip_all_guards_unsafe`` as
the guard filter, and ``VllmBackend.__call__`` asserts it is entered once).
A Python branch is therefore decided once, at trace time, and that decision
applies to every later call, including decode steps.

So the long prefill does not go through the compiled model.  The decision is
made in the outer forward, which vLLM does not compile, and the long prefill
runs the same modules as plain Python.  The compiled model keeps the
projections captured, the sparse core behind its operator, and nothing in
``splitting_ops`` beyond that operator.

The streamed prefill is layer-major (every token through layer 0, then every
token through layer 1, and so on) because a sparse layer needs the whole
prompt's keys before it can score any query: the index is fitted on all of
them.  Within a layer every step runs over slices of ``chunk`` tokens:

    linear-attention layer    norm -> GDN, its state carried through vLLM's
                              own cache from slice to slice -> norm -> MLP
    sparse full-attention     pass 1: norm, project, keep K and V only
    layer                     the barrier: fit the index, pack the copy
                              pass 2: project Q, select, attend, gate,
                              project out -> norm -> MLP

The only full-length tensors are the two hidden-width buffers the residual
stream needs, the K/V a sparse layer keeps until its index exists, and the
cluster-ordered copy.  Q/QKV, the MLP intermediate and the GDN projection,
convolution and recurrence buffers are all slice-sized.

Linear-attention state: a slice is one more step of a recurrence vLLM
already steps.  The layer reads ``ssm_state[idx]``, zeroes it where
``has_initial_state`` is false, and writes ``last_recurrent_state`` back;
the convolution reads and writes ``conv_state`` the same way, as between
the steps of vLLM's own chunked prefill.  A slice needs its own metadata:
its length, ``[0, length]`` as its query start locations,
``has_initial_state`` true for every slice but the first, and the
convolution's bookkeeping rebuilt from those.

Element-wise work: outside the compiled model nothing fuses the native
element-wise chains.  For the duration of a streamed prefill every vLLM
``CustomOp`` under the model is therefore switched to another
implementation, and switched back afterwards so that the compiled model
sees its usual dispatch.  The default runs each op's ``forward_native``
through Inductor, once per module, which gives the same fusion and fp32
intermediates as the compiled model and so the closest numerics an eager
path can get.  SEMBLO_STREAM_KERNELS=cuda uses vLLM's CUDA kernels instead,
which is slightly faster.

The result is not bitwise equal to a whole-prompt forward: a recurrence or
a GEMM over slices uses different reduction trees and tiling.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any, Callable, Iterator

import torch


def _forward_context() -> Any:
    """vLLM's per-forward context; imported here so tests can stand in for it."""

    from vllm.forward_context import get_forward_context

    return get_forward_context()


def _conv_metadata(starts_cpu: torch.Tensor, device) -> tuple[Any, Any, Any]:
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    return compute_causal_conv1d_metadata(starts_cpu, device=device)


def single_prefill(md: Any) -> bool:
    """One request, all of it prefill, nothing speculative: the case a slice is legal in."""

    return (
        getattr(md, "num_prefills", 0) == 1
        and getattr(md, "num_decodes", 0) == 0
        and getattr(md, "num_spec_decodes", 0) == 0
        and getattr(md, "spec_sequence_masks", None) is None
    )


def slice_linear_attention_metadata(
    md: Any,
    length: int,
    first: bool,
    device,
    conv_metadata: Callable[..., tuple[Any, Any, Any]] | None = None,
) -> Any:
    """The same metadata, describing one slice of one single-request prefill.

    Everything not named here is kept, in particular the state indices,
    which say which cache row the slice continues.
    """

    if conv_metadata is None:
        conv_metadata = _conv_metadata      # resolved at call time, so a test can replace it
    starts_cpu = torch.tensor([0, int(length)], dtype=torch.int32)
    nums_dict, batch_ptr, token_chunk_offset_ptr = conv_metadata(starts_cpu, device)
    return dataclasses.replace(
        md,
        num_prefill_tokens=int(length),
        num_actual_tokens=int(length),
        non_spec_query_start_loc=starts_cpu.to(device),
        has_initial_state=torch.tensor([not first], dtype=torch.bool, device=device),
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


def _chunk_metadata(starts: torch.Tensor, chunk_size: int) -> tuple[Any, Any, Any]:
    from vllm.v1.attention.backends.mamba2_attn import compute_varlen_chunk_metadata

    return compute_varlen_chunk_metadata(starts, int(chunk_size))


def slice_mamba2_metadata(
    md: Any,
    length: int,
    first: bool,
    device,
    conv_metadata: Callable[..., tuple[Any, Any, Any]] | None = None,
    chunk_metadata: Callable[..., tuple[Any, Any, Any]] | None = None,
) -> Any:
    """The same Mamba-2 metadata, describing one slice of one single-request prefill.

    vLLM's Mamba-2 mixer reads more of its metadata than the GDN one: the
    prefill's own query offsets, the logical-chunk tables its SSD kernel
    iterates (recomputed for the slice), the conv kernel's program tables,
    and two flags for the initial state.  ``has_initial_states_p`` alone
    does not make the mixer read the state it wrote for the previous slice:
    ``prep_initial_states`` gates that read, and the builder sets it False
    for a prompt starting at position 0, so it is set here for every slice
    but the first.  The state indices and the physical chunk size are kept.
    """

    if conv_metadata is None:
        conv_metadata = _conv_metadata
    if chunk_metadata is None:
        chunk_metadata = _chunk_metadata
    starts_cpu = torch.tensor([0, int(length)], dtype=torch.int32)
    starts = starts_cpu.to(device)
    nums_dict, batch_ptr, token_chunk_offset_ptr = conv_metadata(starts_cpu, device)
    cu_chunk_seqlen, last_chunk_indices, seq_idx = chunk_metadata(starts, md.chunk_size)
    return dataclasses.replace(
        md,
        num_prefills=1,
        num_prefill_tokens=int(length),
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=1,
        query_start_loc_p=starts,
        has_initial_states_p=torch.tensor([not first], dtype=torch.bool, device=device),
        prep_initial_states=bool(not first),
        cu_chunk_seqlen_p=cu_chunk_seqlen,
        last_chunk_indices_p=last_chunk_indices,
        seq_idx_p=seq_idx,
        nums_dict=nums_dict,
        batch_ptr=batch_ptr,
        token_chunk_offset_ptr=token_chunk_offset_ptr,
    )


def is_mamba2_metadata(md: Any) -> bool:
    """Mamba-2 metadata carries the flag the GDN metadata has no need of."""

    return hasattr(md, "prep_initial_states")


def _compiled_native(module: Any) -> Callable[..., Any]:
    """``forward_native`` through Inductor, once per module: the compiled model's numerics.

    Inductor fuses a pointwise chain and keeps fp32 between its steps,
    rounding to bf16 once at the end, as the compiled model does.  The eager
    native form rounds after every step.  The wrapper is named
    ``forward_native`` so that the router's in-place check treats it as out
    of place.
    """

    fn = getattr(module, "_semblo_compiled_native", None)
    if fn is None:
        compiled = torch.compile(module.forward_native)

        def forward_native(*args, **kwargs):
            return compiled(*args, **kwargs)

        module._semblo_compiled_native = fn = forward_native
    return fn


def _fp32_rope(module: Any) -> Callable[..., Any]:
    """The native rotary embedding in fp32, one rounding to the input dtype at the end."""

    def forward_native(positions, query, key=None, *args, **kwargs):
        dtype = query.dtype
        q, k = module.forward_native(
            positions, query.float(), None if key is None else key.float(), *args, **kwargs
        )
        return q.to(dtype), None if k is None else k.to(dtype)

    return forward_native


def _is_rope(module: Any) -> bool:
    return "RotaryEmbedding" in type(module).__name__


def _is_moe(module: Any) -> bool:
    return "MoE" in type(module).__name__


@contextlib.contextmanager
def vllm_cuda_kernels(model: Any, *, kernels: str = "compiled", rope: str = "") -> Iterator[int]:
    """Every CustomOp under ``model`` runs the chosen implementation until the block ends.

    ``kernels``: "cuda" uses vLLM's fused kernels (``forward_cuda``);
    "compiled" runs ``forward_native`` through Inductor, once per module (the
    compiled model's numerics at fused speed); "native" changes nothing.
    ``rope`` overrides the choice for the rotary modules alone, with one more
    option, "fp32".

    vLLM's CustomOp picks its implementation once, at construction, from the
    compilation config: with Inductor on, every op runs ``forward_native`` so
    that Inductor can fuse it inside the compiled graph.  Outside that graph
    that choice is slow.  The swap is on the instance attribute the op
    dispatches through, and it is undone on exit because the compiled model
    must trace the native forms.
    """

    if kernels not in ("cuda", "compiled", "native"):
        raise ValueError("SEMBLO_STREAM_KERNELS must be cuda, compiled or native")
    if rope not in ("", "cuda", "compiled", "native", "fp32"):
        raise ValueError("SEMBLO_STREAM_ROPE must be cuda, compiled, native or fp32")
    swapped: list[tuple[Any, Any]] = []
    for module in model.modules():
        current = getattr(module, "_forward_method", None)
        if current is None or not hasattr(module, "forward_native"):
            continue
        if _is_moe(module):
            # vLLM's fused MoE is a CustomOp too, but its forward is one
            # opaque custom op inside the compiled model as well, so there is
            # nothing for Inductor to fuse.  Compiling its forward_native
            # also breaks models whose MoE layers have different intermediate
            # widths (Nemotron 3.5 Lightning), so it keeps its own dispatch.
            continue
        mode = rope if (rope and _is_rope(module)) else kernels
        if mode == "native":
            continue
        if mode == "cuda":
            target = getattr(module, "forward_cuda", None)
            if target is None:
                continue
            if getattr(current, "__func__", None) is getattr(target, "__func__", None):
                continue
        elif mode == "compiled":
            target = _compiled_native(module)
        else:
            target = _fp32_rope(module)
        module._forward_method = target
        swapped.append((module, current))
    try:
        yield len(swapped)
    finally:
        for module, current in swapped:
            module._forward_method = current


@contextlib.contextmanager
def _moe_layers_named_by_themselves() -> Iterator[None]:
    """vLLM's MoE op finds its layer by a per-forward counter; a slice loop breaks it.

    Without opaque custom-op types (torch < 2.11) vLLM's fused-MoE op is
    passed the literal name "from_forward_context" and looks its layer up as
    ``all_moe_layers[moe_layer_index]``, incrementing the counter once per
    MoE layer per forward.  Called once per slice, the counter would hand
    the wrong layer's weights to later slices.  With ``all_moe_layers``
    cleared for the duration of the streamed prefill the op finds its layer
    by its own prefix instead.  vLLM rebuilds the context for every forward,
    so nothing leaks into the compiled trace.
    """

    try:
        ctx = _forward_context()
    except Exception:       # no forward context (a unit test): nothing to guard
        yield
        return
    saved = getattr(ctx, "all_moe_layers", None)
    if saved is None:
        yield
        return
    ctx.all_moe_layers = None
    index = getattr(ctx, "moe_layer_index", 0)
    try:
        yield
    finally:
        ctx.all_moe_layers = saved
        ctx.moe_layer_index = index


def _norm(module: Any, x: torch.Tensor, residual: torch.Tensor | None = None):
    """A norm through vLLM's CUDA path when it has one; the module call otherwise."""

    method = getattr(module, "forward_cuda", None)
    if method is None or not x.is_cuda:
        method = module
    return method(x, residual) if residual is not None else method(x)


class Qwen3NextLayerShape:
    """How a Qwen3-Next / Qwen3.5 decoder layer is built, as the streamer walks it.

    Each layer is ``input_layernorm -> (linear_attn | self_attn) ->
    post_attention_layernorm -> mlp``, with optional layer scales; the block
    that carries state between slices is ``linear_attn`` (vLLM's GDN, whose
    forward writes into ``output``), and the model's final norm is ``norm``.
    """

    name = "qwen3_next"

    @staticmethod
    def block(layer: Any) -> Any | None:
        """The sliced block of a non-sparse layer, or None if the layer has none."""

        return getattr(layer, "linear_attn", None)

    @staticmethod
    def stateful(block: Any) -> bool:
        return True

    @staticmethod
    def call(block: Any, h: torch.Tensor) -> torch.Tensor:
        a = torch.empty_like(h)
        block(hidden_states=h, output=a)
        return a

    @staticmethod
    def enter(layer, hidden, residual, s, e, first):
        """The input norm of one slice; returns the branch input and the residual."""

        if first:
            r = residual[s:e]
            return _norm(layer.input_layernorm, r), r
        return _norm(layer.input_layernorm, hidden[s:e], residual[s:e])

    @staticmethod
    def leave(layer, a, r):
        """From the attention output of one slice to the layer's output."""

        scaled = bool(getattr(layer, "layer_scale", False))
        if scaled:
            a = a * (layer.attn_layer_scale.to(a.dtype)[0] + 1)
        h, r = _norm(layer.post_attention_layernorm, a, r)
        h = layer.mlp(h)
        if scaled:
            h = h * (layer.ffn_layer_scale.to(h.dtype)[0] + 1)
        return h, r

    @staticmethod
    def final_norm(model: Any) -> Any:
        return model.norm

    @staticmethod
    def describe(layer_idx: int) -> str:
        return f"layer {layer_idx} is neither a linear-attention layer nor a sparse one"


class NemotronHLayerShape:
    """How a Nemotron-H decoder layer is built: ``norm -> mixer``, nothing after.

    Every layer, whether its mixer is Mamba-2, an MoE, a dense MLP or
    attention, is one pre-norm and one mixer whose output is the layer's
    output; the MLP and MoE are layers of their own, not the second half of
    an attention layer.  The mixers that carry state between slices are the
    Mamba-2 ones (vLLM's ``MambaMixer2``, whose forward returns its output);
    an MoE or MLP mixer is per-token and takes no metadata.  The model's
    final norm is ``norm_f``.
    """

    name = "nemotron_h"

    @staticmethod
    def block(layer: Any) -> Any | None:
        """The mixer, unless it is an attention: that one reads the whole prompt's metadata.

        A stock attention mixer that is not registered as a sparse layer is
        refused rather than sliced: fed a slice, it would attend only within
        the slice.
        """

        mixer = getattr(layer, "mixer", None)
        if mixer is None or "Attention" in type(mixer).__name__:
            return None
        return mixer

    @staticmethod
    def stateful(block: Any) -> bool:
        return "Mamba" in type(block).__name__

    @staticmethod
    def call(block: Any, h: torch.Tensor) -> torch.Tensor:
        return block(h)

    @staticmethod
    def enter(layer, hidden, residual, s, e, first):
        if first:
            r = residual[s:e]
            return _norm(layer.norm, r), r
        return _norm(layer.norm, hidden[s:e], residual[s:e])

    @staticmethod
    def leave(layer, a, r):
        return a, r

    @staticmethod
    def final_norm(model: Any) -> Any:
        return model.norm_f

    @staticmethod
    def describe(layer_idx: int) -> str:
        return f"layer {layer_idx} has no mixer and is not a sparse one"


class PrefillStreamer:
    """Runs one long prefill over ``model``'s layers without entering its compiled forward."""

    def __init__(
        self,
        model: Any,
        *,
        chunk: int,
        sparse_layers: dict[int, Any],
        say: Callable[[str], None] = print,
        shape: Any = None,
    ) -> None:
        if int(chunk) <= 0:
            raise ValueError("a streamed prefill needs a positive slice size")
        self.model = model
        self.chunk = int(chunk)
        self.sparse_layers = dict(sparse_layers)
        self.shape = shape if shape is not None else Qwen3NextLayerShape()
        self._say = say
        self._said: set[int] = set()
        import os
        self.kernels = os.environ.get("SEMBLO_STREAM_KERNELS", "compiled")
        self.rope = os.environ.get("SEMBLO_STREAM_ROPE", "")
        self._metadata_cache: dict[tuple[int, bool], Any] = {}
        self._probe_prefix: str | None = None
        # Every layer has to be one this loop knows how to slice.  A stock
        # full-attention layer is not: its attention reads the whole
        # prompt's metadata and would attend only within a slice.
        for layer_idx, layer in enumerate(model.layers):
            if layer_idx in self.sparse_layers:
                continue
            block = self.shape.block(layer)
            if block is None:
                raise ValueError(
                    f"{self.shape.describe(layer_idx)}; the streamed prefill cannot slice it"
                )
            if self._probe_prefix is None and self.shape.stateful(block):
                self._probe_prefix = str(block.prefix)

    # -- the decision -----------------------------------------------------

    def wants(self, num_tokens: int) -> bool:
        """Is this forward a long single-request prefill (or a dummy run of one)?

        A dummy run carries no attention metadata.  The important one is
        vLLM's profile run, the longest forward the engine sees and the one it
        sizes its KV pool from: streaming it makes the profiled footprint the
        streamed footprint.  A real forward is streamed only when it is one
        request, all prefill, nothing speculative.
        """

        if num_tokens < self.chunk:
            return False
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return False
        store = _forward_context().attn_metadata
        if store is None:
            return True
        if not isinstance(store, dict) or self._probe_prefix is None:
            return False
        md = store.get(self._probe_prefix)
        return md is not None and single_prefill(md)

    # -- the forward --------------------------------------------------------

    def run(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self.model
        import os as _os
        if _os.environ.get("SEMBLO_PREFILL_ORDER", "layer") == "token":
            return self._run_token_outer(input_ids, positions, inputs_embeds)
        if inputs_embeds is not None:
            hidden = inputs_embeds
        else:
            hidden = model.embed_input_ids(input_ids)
        tokens = int(hidden.shape[0])
        spans = [(s, min(s + self.chunk, tokens)) for s in range(0, tokens, self.chunk)]
        if tokens not in self._said:
            self._said.add(tokens)
            self._say(f"[semblo] streaming a {tokens:,}-token prefill in {len(spans)} "
                      f"slices of {self.chunk:,}, outside the compiled model")

        # Slices of this forward's metadata are cut once and shared by every
        # linear-attention layer; the next forward cuts its own.
        self._metadata_cache.clear()
        with vllm_cuda_kernels(model, kernels=self.kernels, rope=self.rope) as swapped, \
                _moe_layers_named_by_themselves():
            if tokens in self._said and swapped and "kernels" not in self._said:
                self._said.add("kernels")
                self._say(f"[semblo] {swapped} vLLM ops swapped while a prefill streams: "
                          f"kernels={self.kernels} rope={self.rope or self.kernels}")
            residual: torch.Tensor | None = None
            start = int(getattr(model, "start_layer", 0))
            end = int(getattr(model, "end_layer", len(model.layers)))
            for layer_idx in range(start, end):
                layer = model.layers[layer_idx]
                attn = self.sparse_layers.get(layer_idx)
                if attn is None:
                    hidden, residual = self._sliced_layer(layer, hidden, residual, spans)
                else:
                    hidden, residual = self._sparse_layer(
                        layer, attn, positions, hidden, residual, spans
                    )
            assert residual is not None
            final_norm = self.shape.final_norm(model)
            for s, e in spans:
                out, _ = _norm(final_norm, hidden[s:e], residual[s:e])
                hidden[s:e] = out
        return hidden

    # -- the token-outer forward (token_outer_prefill.py) ------------------

    def _run_token_outer(self, input_ids, positions, inputs_embeds):
        """One slice through every layer, then the next: chunked prefill's order.

        Activations are slice-sized; a sparse layer's session fixes its index
        on the prompt's head and appends every later slice's keys to the
        fixed clusters.  The output buffer is the one full-length tensor,
        because the caller indexes the last position out of it.
        """
        model = self.model
        tokens = int(input_ids.shape[0]) if input_ids is not None else int(inputs_embeds.shape[0])
        spans = [(s, min(s + self.chunk, tokens)) for s in range(0, tokens, self.chunk)]
        if tokens not in self._said:
            self._said.add(tokens)
            self._say(f"[semblo] token-outer prefill of {tokens:,} tokens in {len(spans)} slices of "
                      f"{self.chunk:,}, outside the compiled model")
        self._metadata_cache.clear()
        sessions = {idx: attn.begin_token_outer_prefill(positions, self.chunk)
                    for idx, attn in self.sparse_layers.items()}
        out = None
        with vllm_cuda_kernels(model, kernels=self.kernels, rope=self.rope) as swapped, \
                _moe_layers_named_by_themselves():
            if tokens in self._said and swapped and "kernels" not in self._said:
                self._said.add("kernels")
                self._say(f"[semblo] {swapped} vLLM ops swapped while a prefill streams: "
                          f"kernels={self.kernels} rope={self.rope or self.kernels}")
            store = _forward_context().attn_metadata
            start = int(getattr(model, "start_layer", 0))
            end = int(getattr(model, "end_layer", len(model.layers)))
            final_norm = self.shape.final_norm(model)
            for i, (s, e) in enumerate(spans):
                hidden = inputs_embeds[s:e] if inputs_embeds is not None \
                    else model.embed_input_ids(input_ids[s:e])
                if out is None:
                    out = hidden.new_empty((tokens, hidden.shape[-1]))
                residual = None
                for layer_idx in range(start, end):
                    layer = model.layers[layer_idx]
                    session = sessions.get(layer_idx)
                    if session is None:
                        hidden, residual = self._sliced_one(layer, hidden, residual, s, e, i == 0, store)
                    else:
                        hidden, residual = self._sparse_one(layer, session, hidden, residual, s, e)
                o, _ = _norm(final_norm, hidden, residual)
                out[s:e] = o
        for session in sessions.values():
            session.finish()
        return out

    def _sliced_one(self, layer, hidden, residual, s, e, first_slice, store):
        """A non-sparse layer on one slice-sized (hidden, residual) pair."""
        block = self.shape.block(layer)
        stateful = self.shape.stateful(block)
        hidden, residual, first = self._split_residual(hidden, residual)
        whole = None
        dummy = False
        if stateful:
            if isinstance(store, dict):
                whole = store.get(block.prefix)
                if whole is None:
                    raise RuntimeError(f"no attention metadata for {block.prefix}")
            else:
                dummy = store is None
        try:
            h, r = self._enter(layer, hidden, residual, 0, e - s, first)
            if whole is not None:
                store[block.prefix] = self._sliced(whole, e - s, first_slice, h.device)
            a = self.shape.call(block, h)
            if dummy:
                a = torch.zeros_like(a)
            return self._leave(layer, a, r)
        finally:
            if whole is not None:
                store[block.prefix] = whole

    def _sparse_one(self, layer, session, hidden, residual, s, e):
        hidden, residual, first = self._split_residual(hidden, residual)
        h, r = self._enter(layer, hidden, residual, 0, e - s, first)
        a = session.slice(h, s, e)
        return self._leave(layer, a, r)

    # -- one layer, sliced --------------------------------------------------

    def _enter(self, layer, hidden, residual, s, e, first):
        return self.shape.enter(layer, hidden, residual, s, e, first)

    def _leave(self, layer, a, r):
        return self.shape.leave(layer, a, r)

    @staticmethod
    def _split_residual(hidden, residual):
        """vLLM's first layer aliases the residual to its input and norms into a new buffer."""

        if residual is None:
            return torch.empty_like(hidden), hidden, True
        return hidden, residual, False

    def _sliced(self, whole, length: int, first: bool, device):
        """One slice's metadata, cut from this forward's whole and reused across its layers.

        Keyed by the whole's identity as well, so a slice cut for one request
        is never handed to the next: the state indices inside it name that
        request's cache rows.  The first slice inherits the whole's own
        has-initial-state flag.  The GDN and the Mamba-2 metadata are cut by
        their own functions.
        """

        key = (id(whole), int(length), bool(first))
        md = self._metadata_cache.get(key)
        if md is None:
            if is_mamba2_metadata(whole):
                flag = whole.has_initial_states_p
                starts_with_state = first and flag is not None and bool(flag[0].item())
                md = slice_mamba2_metadata(whole, length, first and not starts_with_state, device)
            else:
                starts_with_state = first and bool(whole.has_initial_state[0].item())
                md = slice_linear_attention_metadata(
                    whole, length, first and not starts_with_state, device
                )
            self._metadata_cache[key] = md
        return md

    def _sliced_layer(self, layer, hidden, residual, spans):
        """A non-sparse layer, one slice at a time.

        A stateful block (GDN, Mamba-2) is shown its slice's metadata and
        carries its state through vLLM's own cache rows; a per-token block
        (an MoE or MLP mixer) is called on the slice.  In a dummy run there
        is no metadata: vLLM's Mamba-2 mixer then returns without writing its
        output, so the block's result is replaced by zeros rather than letting
        uninitialised memory run through the norms.
        """

        block = self.shape.block(layer)
        stateful = self.shape.stateful(block)
        hidden, residual, first = self._split_residual(hidden, residual)
        store = _forward_context().attn_metadata
        whole = None
        dummy = False
        if stateful:
            if isinstance(store, dict):
                whole = store.get(block.prefix)
                if whole is None:
                    raise RuntimeError(f"no attention metadata for {block.prefix}")
            else:
                dummy = store is None
        try:
            for i, (s, e) in enumerate(spans):
                h, r = self._enter(layer, hidden, residual, s, e, first)
                if whole is not None:
                    store[block.prefix] = self._sliced(whole, e - s, i == 0, h.device)
                a = self.shape.call(block, h)
                if dummy:
                    a = torch.zeros_like(a)
                h, r = self._leave(layer, a, r)
                hidden[s:e] = h
                residual[s:e] = r
        finally:
            if whole is not None:
                store[block.prefix] = whole
        return hidden, residual

    _linear_layer = _sliced_layer

    def _sparse_layer(self, layer, attn, positions, hidden, residual, spans):
        hidden, residual, first = self._split_residual(hidden, residual)
        rope_positions, flat_positions = attn._positions_for_runtime(positions)
        session = attn.begin_streamed_prefill(flat_positions)
        # Pass 1: the input norm of every slice, kept in ``hidden`` for pass
        # 2, and the keys and values of every slice, kept by the session.
        for s, e in spans:
            h, r = self._enter(layer, hidden, residual, s, e, first)
            hidden[s:e] = h
            if not first:
                residual[s:e] = r
            session.keys(attn._slice_positions(rope_positions, s, e), h, s, e)
        # The barrier: the index is fitted on the whole prompt's keys.
        session.ingest()
        # Pass 2: the queries of every slice, against the finished index.
        for s, e in spans:
            a = session.queries(
                attn._slice_positions(rope_positions, s, e),
                flat_positions[s:e],
                hidden[s:e],
                s,
            )
            h, r = self._leave(layer, a, residual[s:e])
            hidden[s:e] = h
            residual[s:e] = r
        session.finish()
        return hidden, residual
