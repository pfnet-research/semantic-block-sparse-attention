"""Qwen3.5 integration for the semblo vLLM runtime.

The runtime (index build, block selection, packed attention) is architecture
independent and lives on ``SembloAttention``.  This module subclasses it
and replaces only the projections: the qkv layout (query fused with an output
gate), per-head QK norms, no V norm, partial RoPE through vLLM's rotary module,
a sigmoid output gate, and no KV sharing.  The linear-attention layers are not
modified.
"""

from __future__ import annotations

import os

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from semblo.vllm_plugin.attention_op import register_layer
from semblo.vllm_plugin.config import (
    SembloVllmConfig,
)
from semblo.vllm_plugin.model import (
    SembloAttention,
    SembloRuntimeState,
    _norm_forward,
    _split_the_graph_at_sparse_attention,
    _text_config,
    _rope_mutates_inputs,
)
from semblo.vllm_plugin.routing_keys import routing_view
from semblo.vllm_plugin.identity_router import load_identity_router_into_module
from semblo.vllm_plugin.paper_defaults import (
    qwen3_5_attention_layers,
    runtime_config_for_model,
)
from semblo.vllm_plugin.runtime_weights import (
    filter_base_weight_iterator,
    load_runtime_weights_into_module,
)
from semblo.vllm_plugin.streamed_prefill import PrefillStreamer
from semblo.vllm_plugin.validation import DEFERRED_CHECKS


class SembloQwen3_5Attention(SembloAttention):
    """Qwen3.5 full-attention wrapper over the shared runtime."""

    ROUTING_MODEL_NAME = "Qwen3.5"

    def __init__(
        self,
        original: Qwen3NextAttention,
        *,
        runtime_config: SembloVllmConfig,
        runtime_state: SembloRuntimeState,
        layer_idx: int,
    ) -> None:
        nn.Module.__init__(self)
        if not hasattr(original, "qkv_proj"):
            raise RuntimeError(
                "this integration targets the vLLM 0.19.1 Qwen3_5 qkv_proj layout"
            )
        # Keep vLLM's child-module names so the inherited weight loader works.
        self.qkv_proj = original.qkv_proj
        self.o_proj = original.o_proj
        self.q_norm = original.q_norm
        self.k_norm = original.k_norm
        self.rotary_emb = original.rotary_emb
        self.attn = original.attn
        register_layer(self.attn.layer_name, self)

        self.config = original.config
        self.hidden_size = original.hidden_size
        self.num_heads = original.num_heads
        self.num_kv_heads = original.num_kv_heads
        self.head_dim = original.head_dim
        self.q_size = original.q_size
        self.kv_size = original.kv_size
        self.scaling = original.scaling
        self.attn_output_gate = bool(getattr(original, "attn_output_gate", False))
        # Structural facts the shared runtime branches on.
        self.v_norm = None
        self.use_k_eq_v = False
        self.is_sliding = False
        self.is_kv_shared_layer = False

        self._init_semblo_runtime(
            runtime_config=runtime_config,
            runtime_state=runtime_state,
            layer_idx=layer_idx,
            # Qwen3.5 has no KV sharing: every full layer is its own source.
            source_layer_idx=layer_idx,
            uses_sparse_attention=True,
            builds_product_index=True,
            attention_logits_soft_cap=None,
        )

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> None:
        """Write the layer result into ``output``, as vLLM's Qwen3.5 decoder layer expects.

        The output projection allocates its own tensor, so the result is
        copied into the caller's buffer.  The copy runs on the compiled side
        of the graph split and costs one captured kernel."""

        del kwargs
        output.copy_(self._sparse_layer_forward(positions, hidden_states))

    def _positions_for_runtime(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # vLLM passes Qwen3.5 attention the M-RoPE position grid [3, T].  For
        # text the three rows are identical.  The runtime (prompt detection,
        # causal masks, key positions) uses the 1-D row, and the rotary module
        # gets the full grid.
        if positions.ndim == 2:
            return positions, positions[0]
        return positions, positions

    def _project_qkv(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            q_gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            output_gate = gate.reshape(gate.shape[0], -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q = q.unflatten(-1, (self.num_heads, self.head_dim))
            output_gate = None

        q_pre_rope = _norm_forward(self.q_norm, q.contiguous())
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k_pre_rope = _norm_forward(self.k_norm, k)
        q_flat = q_pre_rope.flatten(-2, -1)
        k_flat = k_pre_rope.flatten(-2, -1)
        # ``positions`` is the M-RoPE grid (see ``_positions_for_runtime``).
        #
        # q_flat and k_flat are views of the pre-RoPE tensors used for
        # routing, so they are cloned only if the rotary module rotates in
        # place.  An unconditional clone would cost about 5 GB per sparse
        # layer at 512K.
        if _rope_mutates_inputs(self.rotary_emb):
            q_in, k_in = q_flat.clone(), k_flat.clone()
        else:
            q_in, k_in = q_flat, k_flat
        q_post_rope, k_post_rope = self.rotary_emb(positions, q_in, k_in)
        # Copy V rather than view it.  A view would keep the whole fused
        # 2Q+K+V projection alive (ten times the size of V) through the
        # attention, index build and pack.
        v_states = v.unflatten(-1, (self.num_kv_heads, self.head_dim)).contiguous()
        # With SEMBLO_ROUTING_KEYS=post, routing uses the rotated q and k
        # (routing_keys.py).
        q_pre_rope = routing_view(q_pre_rope, q_post_rope)
        k_pre_rope = routing_view(k_pre_rope, k_post_rope)
        return (
            q_post_rope,
            k_post_rope,
            v_states,
            q_pre_rope,
            k_pre_rope,
            output_gate,
        )

    def _project_kv(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """K and V alone, from the last 2 x kv_size rows of the fused weight.

        The first pass of a streamed prefill needs only K and V, which are a
        small fraction of the fused projection's columns.  Only an unquantised
        2-D weight is sliced; anything else goes through the fused path.
        """

        weight = getattr(self.qkv_proj, "weight", None)
        rows = 2 * self.kv_size
        if (
            os.environ.get("SEMBLO_STREAM_KV_ONLY", "1") != "1"
            or weight is None or weight.ndim != 2 or weight.dtype != hidden_states.dtype
            or weight.shape[0] <= rows or hasattr(self.qkv_proj, "weight_scale")
        ):
            return super()._project_kv(positions, hidden_states)
        bias = getattr(self.qkv_proj, "bias", None)
        kv = F.linear(
            hidden_states, weight[-rows:], None if bias is None else bias[-rows:]
        )
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        k_pre_rope = _norm_forward(self.k_norm, k.unflatten(-1, (self.num_kv_heads, self.head_dim)))
        k_flat = k_pre_rope.flatten(-2, -1)
        # The rotary module rotates a query and a key together.  It gets a
        # throwaway copy of K as the query, and K itself is copied only if the
        # rotation is in place.
        k_in = k_flat.clone() if _rope_mutates_inputs(self.rotary_emb) else k_flat
        _q, k_post_rope = self.rotary_emb(positions, k_flat.clone(), k_in)
        v_states = v.unflatten(-1, (self.num_kv_heads, self.head_dim)).contiguous()
        # The routing keys that the index is fitted on, as in _project_qkv.
        return k_post_rope, v_states, routing_view(k_pre_rope, k_post_rope)

    def _finish_output(
        self, attn_output: torch.Tensor, gate: torch.Tensor | None
    ) -> torch.Tensor:
        flat = attn_output.view(attn_output.shape[0], -1)
        if gate is not None:
            # ``attn_output`` is not read elsewhere, so the product is written
            # into it in place, saving a query-width tensor.  The sigmoid is
            # out of place because ``gate`` may be a view of the fused
            # projection.
            flat = flat.mul_(torch.sigmoid(gate))
        return self.o_proj(flat)[0]

    def _official_dense_attention(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """vLLM's Qwen3.5 attention, up to but not including ``o_proj``.

        Qwen3.5 has no KV sharing, so this is not reached in practice.  It
        implements the base class's interface.
        """

        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            q_gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim)
            q, _gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(q.shape[0], -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).flatten(-2, -1)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).flatten(-2, -1)
        # The caller applies the gate and the output projection.
        q, k = self.rotary_emb(positions, q, k)
        return self.attn(q, k, v)


class _SembloQwen3_5Installer:
    """Replaces the full-attention layers of a vLLM Qwen3.5 model in place.

    Shared by the dense and the MoE variants, which vLLM builds from the same
    decoder layer and ``Qwen3NextAttention``.  Only the parent class differs,
    and ``super()`` resolves to it through the MRO.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        _split_the_graph_at_sparse_attention(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # A stock checkpoint gets the paper's settings and identity routers
        # built in memory.  A prepared checkpoint is read as is
        # (paper_defaults.py).
        runtime_config = runtime_config_for_model(
            vllm_config.model_config.hf_config,
            max_model_len=int(vllm_config.model_config.max_model_len),
            attention_layers=qwen3_5_attention_layers,
        )
        runtime_config.validate_vllm_config(vllm_config)
        self.semblo_runtime_config = runtime_config
        self.semblo_hf_config = vllm_config.model_config.hf_config
        if runtime_config.runtime_checks:
            DEFERRED_CHECKS.allow()
        else:
            DEFERRED_CHECKS.turn_off()
        model_path = Path(str(vllm_config.model_config.model)).expanduser()
        if (
            runtime_config.mode == "sparse"
            and not runtime_config.identity_router
            and not model_path.is_dir()
        ):
            raise ValueError(
                "semblo sparse vLLM requires a local prepared model directory"
            )
        self.semblo_checkpoint_dir = (
            model_path.resolve() if model_path.exists() else model_path
        )
        self.semblo_runtime_state = SembloRuntimeState()
        self._prefill_streamer: PrefillStreamer | None = None
        if runtime_config.mode == "dense_passthrough":
            return

        text = _text_config(vllm_config.model_config.hf_config)
        wrapped_layers = set(runtime_config.layer_indices)
        if not wrapped_layers:
            raise ValueError("no semblo full-attention layers were configured")
        self.semblo_runtime_state.reset_layer_idx = min(wrapped_layers)

        self.semblo_runtime_state.max_model_len = int(
            vllm_config.model_config.max_model_len)
        layers = self.language_model.model.layers
        for layer_idx in sorted(wrapped_layers):
            layer = layers[layer_idx]
            if text.layer_types[layer_idx] != "full_attention":
                raise ValueError(
                    f"semblo layer {layer_idx} is not a full-attention layer"
                )
            original = getattr(layer, "self_attn", None)
            if not isinstance(original, Qwen3NextAttention):
                raise TypeError(
                    f"layer {layer_idx} is not the official vLLM Qwen3_5 attention"
                )
            layer.self_attn = SembloQwen3_5Attention(
                original,
                runtime_config=runtime_config,
                runtime_state=self.semblo_runtime_state,
                layer_idx=layer_idx,
            )
            self.semblo_runtime_state.register_source_module(
                layer_idx, layer.self_attn
            )
            _limit_pool_to_window(original, runtime_config)
        chunk = int(runtime_config.prefill_chunk)
        if chunk > 0:
            self._prefill_streamer = PrefillStreamer(
                self.language_model.model,
                chunk=chunk,
                sparse_layers={
                    idx: layers[idx].self_attn for idx in sorted(wrapped_layers)
                },
            )
            print(f"[semblo] long prefills stream in slices of {chunk:,} tokens, "
                  "outside the compiled model", flush=True)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Route a long prefill to the streamer, and everything else to vLLM.

        vLLM compiles the inner model, not this method.  A long single-request
        prefill runs here as plain Python over the same modules and never
        enters the compiled forward (see streamed_prefill.py).  Decode and
        all other calls go through the compiled model unchanged.
        """

        streamer = self._prefill_streamer
        if (
            streamer is not None
            and intermediate_tensors is None
            and streamer.wants(int(positions.shape[-1]))
        ):
            return streamer.run(input_ids, positions, inputs_embeds)
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loaded = super().load_weights(filter_base_weight_iterator(weights))
        if self.semblo_runtime_config.mode != "sparse":
            return loaded
        from vllm.distributed import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        if self.semblo_runtime_config.identity_router:
            loaded.update(load_identity_router_into_module(
                self,
                self.semblo_hf_config,
                self.semblo_runtime_config,
                tp_rank=get_tensor_model_parallel_rank(),
                tp_size=get_tensor_model_parallel_world_size(),
            ))
            return loaded
        runtime_loaded = load_runtime_weights_into_module(
            self,
            self.semblo_checkpoint_dir,
            require_exact_match=True,
            allow_unused_layer_parameters=(
                self.semblo_runtime_config.runtime_allow_unused_layers
            ),
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(),
        )
        loaded.update(runtime_loaded)
        return loaded


class SembloQwen3_5ForConditionalGeneration(
    _SembloQwen3_5Installer, Qwen3_5ForConditionalGeneration
):
    """Qwen3.5 (dense FFN): Qwen3.5-2B / 4B / 9B."""


class SembloQwen3_5MoeForConditionalGeneration(
    _SembloQwen3_5Installer, Qwen3_5MoeForConditionalGeneration
):
    """Qwen3.5 MoE: Qwen3.5-35B-A3B (256 experts, 8 active, a shared expert).

    The MoE lives in ``layer.mlp``, which is not modified.  The expert weights
    go through vLLM's loader; the base-weight filter removes only
    ``semblo_*`` tensors.
    """


def _limit_pool_to_window(attn_module, runtime_config) -> None:
    """Shrink or remove vLLM's KV pool for these layers.

    Once the packed prompt K/V exists, the prompt is read from it, and vLLM's
    pool is read only for the generated suffix.  Two options avoid storing
    the prompt twice:

    - ``own_prompt_cache``: the layer declares no KV cache at all, and the
      generated suffix is kept in the plugin's own store.
    - ``prompt_kv_window``: the layer declares a sliding window, so vLLM's
      pool holds only a window per request.  The window must cover the
      generated suffix.  vLLM's ``Attention.get_kv_cache_spec`` returns a
      SlidingWindowSpec exactly when ``sliding_window`` is set.  The sparse
      path never calls ``attn.forward``, so the window's mask never applies.

    The paper's configuration uses ``own_prompt_cache``.
    """

    if getattr(runtime_config, "own_prompt_cache", False):
        attn = getattr(attn_module, "attn", None)
        if attn is None:
            raise TypeError("own_prompt_cache needs a vLLM Attention layer")
        # vLLM collects specs with `if spec := attn.get_kv_cache_spec(...)`
        # and skips layers that return None, so this layer gets no KV
        # allocation.  Bound on the instance so no other model is affected.
        attn.get_kv_cache_spec = lambda vllm_config, _a=attn: None
        print(f"[semblo] own_prompt_cache: {getattr(attn, 'layer_name', '?')} "
              f"declares no KV cache", flush=True)
        return

    window = int(getattr(runtime_config, "prompt_kv_window", 0) or 0)
    if window <= 0:
        return
    attn = getattr(attn_module, "attn", None)
    if attn is None or not hasattr(attn, "sliding_window"):
        raise TypeError(
            "prompt_kv_window needs a vLLM Attention with a sliding_window "
            "attribute; this build has "
            f"{type(attn).__name__ if attn is not None else 'no attn'}"
        )
    before = getattr(attn, "sliding_window", "<missing>")
    attn.sliding_window = window
    # Logged so the setting can be verified from the output.
    print(f"[semblo] prompt_kv_window: {getattr(attn, 'layer_name', '?')} "
          f"sliding_window {before} -> {attn.sliding_window}", flush=True)
