"""Nemotron-H model for the semblo vLLM runtime (Nemotron 3.5 Lightning).

The runtime (index build, block ranking, sparse attention, decode) is
architecture independent and lives on the shared attention wrapper class.
This module subclasses it and replaces only the projections.  Nemotron-H
attention has one fused ``qkv_proj``, no query/key norm, no output gate and
no rotary embedding (position enters only through the Mamba-2 layers), so the
routing keys are the attention keys themselves.  The Mamba-2 and MoE layers
are unchanged.

Two structural differences from Qwen3.5:

* The decoder layer keeps its attention under ``mixer`` and calls it with the
  hidden states only. The runtime needs the positions, so the layer's class
  is swapped for one that passes them in.
* The config has no ``text_config`` and lists its layers as
  ``layers_block_type``, so the sparse layers are selected from that list.

The attention decoder layer's ``forward`` is adapted from vLLM's
``NemotronHAttentionDecoderLayer.forward`` (see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import os

from collections.abc import Iterable
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from vllm.config import VllmConfig
from vllm.model_executor.models.nemotron_h import (
    NemotronHAttention,
    NemotronHAttentionDecoderLayer,
    NemotronHForCausalLM,
)

from semblo.vllm_plugin.attention_op import register_layer
from semblo.vllm_plugin.config import (
    SembloVllmConfig,
)
from semblo.vllm_plugin.model import (
    SembloAttention,
    SembloRuntimeState,
    _split_the_graph_at_sparse_attention,
    _text_config,
)
from semblo.vllm_plugin.identity_router import load_identity_router_into_module
from semblo.vllm_plugin.paper_defaults import (
    NEMOTRON_BUDGET_DIVISOR,
    NEMOTRON_ENV,
    nemotron_h_attention_layers,
    runtime_config_for_model,
)
from semblo.vllm_plugin.runtime_weights import (
    filter_base_weight_iterator,
    load_runtime_weights_into_module,
)
from semblo.vllm_plugin.streamed_prefill import NemotronHLayerShape, PrefillStreamer
from semblo.vllm_plugin.validation import DEFERRED_CHECKS


class SembloNemotronHAttention(SembloAttention):
    """Nemotron-H attention wrapper over the shared runtime."""

    # No rotary embedding: pre-RoPE and post-RoPE keys are the same tensor,
    # so SEMBLO_ROUTING_KEYS has no effect on this model.
    ROUTING_MODEL_NAME = "Nemotron-H"

    def __init__(
        self,
        original: NemotronHAttention,
        *,
        runtime_config: SembloVllmConfig,
        runtime_state: SembloRuntimeState,
        layer_idx: int,
    ) -> None:
        nn.Module.__init__(self)
        if not hasattr(original, "qkv_proj"):
            raise RuntimeError(
                "this integration targets the vLLM 0.19.1 NemotronH qkv_proj layout"
            )
        # Keep vLLM's child-module names so the base-weight loader still finds
        # qkv_proj and o_proj under ``mixer``.
        self.qkv_proj = original.qkv_proj
        self.o_proj = original.o_proj
        self.attn = original.attn
        register_layer(self.attn.layer_name, self)

        self.config = getattr(original, "config", None)
        self.hidden_size = original.hidden_size
        self.num_heads = original.num_heads
        self.num_kv_heads = original.num_kv_heads
        self.head_dim = original.head_dim
        self.q_size = original.q_size
        self.kv_size = original.kv_size
        self.scaling = original.scaling
        # Attributes the shared runtime branches on. There are no norms and
        # no rotary module (see _project_qkv).
        self.q_norm = None
        self.k_norm = None
        self.v_norm = None
        self.rotary_emb = None
        self.attn_output_gate = False
        self.use_k_eq_v = False
        self.is_sliding = False
        self.is_kv_shared_layer = False

        self._init_semblo_runtime(
            runtime_config=runtime_config,
            runtime_state=runtime_state,
            layer_idx=layer_idx,
            # no KV sharing: every attention layer is its own source
            source_layer_idx=layer_idx,
            uses_sparse_attention=True,
            builds_product_index=True,
            attention_logits_soft_cap=None,
        )

    def forward(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Sparse attention forward.

        ``positions`` is passed in by SembloNemotronHAttentionDecoderLayer.
        The runtime needs it for prompt detection and causal masking.
        """

        del kwargs
        if positions is None:
            raise RuntimeError(
                "SembloNemotronHAttention needs the positions; the decoder "
                "layer must be SembloNemotronHAttentionDecoderLayer"
            )
        return self._sparse_layer_forward(positions, hidden_states)

    def _positions_for_runtime(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # a plain 1-D position row; there is no rotary grid to keep apart
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
        """Project Q, K and V.

        Without a rotary embedding, the tensors used for attention and for
        routing are the same, returned as flat and per-head views of one
        projection. Nothing downstream writes into them, so the views avoid a
        copy of Q (4.3 GB at 512K).

        V is copied out of the fused buffer. A view would keep the whole
        Q+K+V buffer alive for as long as V is needed.
        """

        del positions
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_heads = q.unflatten(-1, (self.num_heads, self.head_dim))
        k_heads = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v_states = v.unflatten(-1, (self.num_kv_heads, self.head_dim)).contiguous()
        return q, k, v_states, q_heads, k_heads, None

    def _project_kv(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project K and V only, using the last 2 x kv_size rows of the fused
        weight. Falls back to the fused projection when the weight cannot be
        sliced."""

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
        k_heads = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v_states = v.unflatten(-1, (self.num_kv_heads, self.head_dim)).contiguous()
        return k, v_states, k_heads

    def _finish_output(
        self, attn_output: torch.Tensor, gate: torch.Tensor | None
    ) -> torch.Tensor:
        del gate
        return self.o_proj(attn_output.view(attn_output.shape[0], -1))[0]

    def _official_dense_attention(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """vLLM's Nemotron-H attention, up to but not including ``o_proj``.

        Required by the base class. Not reached for this model, which has no
        KV sharing."""

        del positions
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        return self.attn(q, k, v)


class SembloNemotronHAttentionDecoderLayer(NemotronHAttentionDecoderLayer):
    """vLLM's decoder layer, passing the positions to the wrapped mixer.

    Installed by swapping the class of the existing layer object, so modules,
    names and loaded weights stay where vLLM put them. Only ``forward``
    changes."""

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        del kwargs
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)
        hidden_states = self.mixer(hidden_states=hidden_states, positions=positions)
        return hidden_states, residual


class SembloNemotronHForCausalLM(NemotronHForCausalLM):
    """vLLM's Nemotron-H with the attention layers replaced in place."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        _split_the_graph_at_sparse_attention(vllm_config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # A stock model gets the paper's settings (paper_defaults.py).
        runtime_config = runtime_config_for_model(
            vllm_config.model_config.hf_config,
            max_model_len=int(vllm_config.model_config.max_model_len),
            attention_layers=nemotron_h_attention_layers,
            budget_divisor=NEMOTRON_BUDGET_DIVISOR,
            model_env=NEMOTRON_ENV,
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
        # transformers >= 5 names the block "full_attention". vLLM's config
        # class says "attention".
        block_types = ["attention" if b == "full_attention" else b
                       for b in list(text.layers_block_type)]
        wrapped_layers = set(runtime_config.layer_indices)
        if not wrapped_layers:
            raise ValueError("no semblo attention layers were configured")
        self.semblo_runtime_state.reset_layer_idx = min(wrapped_layers)
        self.semblo_runtime_state.max_model_len = int(
            vllm_config.model_config.max_model_len)
        from semblo.vllm_plugin.qwen3_5_model import _limit_pool_to_window

        layers = self.model.layers
        for layer_idx in sorted(wrapped_layers):
            layer = layers[layer_idx]
            if block_types[layer_idx] != "attention":
                raise ValueError(
                    f"semblo layer {layer_idx} is a {block_types[layer_idx]} "
                    "layer, not an attention layer"
                )
            if not isinstance(layer, NemotronHAttentionDecoderLayer):
                raise TypeError(f"layer {layer_idx} is not the official NemotronH attention decoder layer")
            original = getattr(layer, "mixer", None)
            if not isinstance(original, NemotronHAttention):
                raise TypeError(
                    f"layer {layer_idx} is not the official vLLM NemotronH attention"
                )
            layer.mixer = SembloNemotronHAttention(
                original,
                runtime_config=runtime_config,
                runtime_state=self.semblo_runtime_state,
                layer_idx=layer_idx,
            )
            layer.__class__ = SembloNemotronHAttentionDecoderLayer
            self.semblo_runtime_state.register_source_module(
                layer_idx, layer.mixer
            )
            _limit_pool_to_window(original, runtime_config)
        chunk = int(runtime_config.prefill_chunk)
        if chunk > 0:
            # Streamed prefill: the Mamba-2, MoE and MLP layers process the
            # prompt one slice at a time (Mamba-2 carries its conv and SSM
            # state through vLLM's cache rows, as in vLLM's chunked prefill),
            # while the attention layers see the whole prompt. This keeps
            # transient memory slice-sized. The slice is a multiple of the
            # Mamba-2 chunk so every slice is tiled as the whole prompt would be.
            mamba_chunk = int(getattr(text, "chunk_size", 0) or 0)
            if mamba_chunk > 0 and chunk % mamba_chunk != 0:
                raise ValueError(
                    f"SEMBLO_PREFILL_CHUNK={chunk} is not a multiple of the Mamba-2 "
                    f"chunk size {mamba_chunk}"
                )
            self._prefill_streamer = PrefillStreamer(
                self.model,
                chunk=chunk,
                sparse_layers={idx: layers[idx].mixer for idx in sorted(wrapped_layers)},
                shape=NemotronHLayerShape(),
            )
            print(f"[semblo] long prefills stream in slices of {chunk:,} tokens, "
                  "outside the compiled model (Nemotron-H: Mamba-2 state carried "
                  "through vLLM's cache rows)", flush=True)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Stream a long single-request prefill, otherwise call vLLM's forward.

        vLLM compiles the inner ``NemotronHModel``, not this method, so the
        branch is decided in Python on every forward (see streamed_prefill.py).
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
