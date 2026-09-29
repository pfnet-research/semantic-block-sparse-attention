from __future__ import annotations

"""vLLM model integration for semblo sparse global attention.

This module is imported lazily by vLLM's model registry. Supported configurations:

* NVIDIA H200/H100 (SM90), BF16/FP16;
* tensor parallelism without KV-head replication, pipeline parallel size one;
* one active request;
* one non-chunked, non-prefix-cached prompt prefill;
* semantic-block selection for the configured full-attention layers.

Other layers use vLLM's implementations. Sparse layers keep a cluster-ordered
prompt KV store; with own_prompt_cache, this replaces their vLLM prompt cache.
"""

from dataclasses import dataclass
import os
import time

_FULL_GRAPH_CALLS: dict = {}
from typing import Any

import torch
from torch import nn

from semblo.vllm_plugin.routing_map import SembloKeyRouter, SembloQueryState
from semblo.vllm_plugin.attention_op import (
    SPLITTING_OP_NAME,
    sparse_attention_op,
)
from semblo.vllm_plugin.config import (
    SembloVllmConfig,
)
from semblo.vllm_plugin import fused_prefill as _fused_prefill
from semblo.vllm_plugin import arena as _arena
from semblo.vllm_plugin import fused_decode as _fused_decode
from semblo.vllm_plugin.decode_graph import (
    DecodeStepGraph,
    graphs_enabled,
    tensor_signature,
)
from semblo.vllm_plugin.paged_attention import (
    PostingTiles,
    PagedSparseAttentionResult,
    merge_paged_attention_results,
    paged_posting_attention,
)
from semblo.vllm_plugin import routing_keys as _routing_keys
from semblo.vllm_plugin.validation import (
    DEFERRED_CHECKS,
    SLOT_BUCKET_COLLAPSE,
)

# Above this many queries the tensor path's single large projection GEMM wins.
FUSED_ROUTER_MAX_QUERIES = 64
# SEMBLO_ROUTING_KEYS=post (route on the rotated q and k) is handled in
# routing_keys.py: every projection, including the subclasses' overrides,
# passes its pre-RoPE slots through routing_view.

# Number of times the sparse attention path has run.  Output tokens alone
# cannot tell the sparse path from the dense one, so benchmarks read this
# counter to confirm that the plugin did not fall back to dense attention.
# attention_op explains why the layer is an operator opaque to torch.compile.
SPARSE_ATTENTION_CALLS = 0


def sparse_attention_calls() -> int:
    """Number of sparse attention calls since the process started."""

    return SPARSE_ATTENTION_CALLS
from semblo.vllm_plugin.state import (
    SembloRuntimeState,
    ProductPostingIndex,
    _PACK_ALIGN,
    record_pack_seconds,
)

from vllm.config import VllmConfig
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    unified_kv_cache_update,
)


def _text_config(config: Any) -> Any:
    return getattr(config, "text_config", config)


def _dump_projected_keys(
    directory: str,
    product: torch.Tensor,
    positions: torch.Tensor,
    layer_idx: int,
    epoch: int,
    raw_keys: torch.Tensor | None = None,
) -> None:
    """Save one prompt's k-means input (``SEMBLO_DUMP_PROJECTED_KEYS=<dir>``).

    The input of scripts/fit_corpus_codebook.py and of the PCA map of
    scripts/make_routing_map.py.  ``SEMBLO_DUMP_PROJECTED_KEYS_SAMPLE=N`` keeps
    N rows per prompt, the same rows for every layer.
    """

    import numpy as np
    from pathlib import Path

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    sample = int(os.environ.get("SEMBLO_DUMP_PROJECTED_KEYS_SAMPLE", "0"))
    if sample and product.shape[0] > sample:
        gen = torch.Generator(device="cpu").manual_seed(int(epoch))
        rows = torch.randperm(int(product.shape[0]), generator=gen)[:sample]
        rows = rows.sort().values.to(product.device)
        product = product.index_select(0, rows)
        positions = positions.index_select(0, rows)
        if raw_keys is not None:
            raw_keys = raw_keys.index_select(0, rows)
    arrays = {
        "projected": product.detach().to(torch.float16).cpu().numpy(),
        "positions": positions.detach().cpu().numpy(),
    }
    if raw_keys is not None:
        arrays["raw"] = raw_keys.detach().to(torch.float16).cpu().numpy()
    np.savez(path / f"keys_epoch{epoch:05d}_layer{layer_idx}.npz", **arrays)


@dataclass(frozen=True)
class _PagedContext:
    kv_cache: torch.Tensor
    block_table: torch.Tensor
    block_size: int


# Distinct probe sizes for kind, blocks, tokens per block, KV heads, head
# dimension.  The block size must stay a multiple of 16 because the backends
# reject anything else.
_KV_CACHE_PROBE_SHAPE = (2, 3, 16, 5, 7)


def _profile_packed_kv_enabled(config: SembloVllmConfig) -> bool:
    """Should the profiling pass reserve the packed prompt K/V?

    ``SEMBLO_PROFILE_PACKED_KV`` overrides the config setting.
    """

    override = os.environ.get("SEMBLO_PROFILE_PACKED_KV")
    if override is not None:
        return override == "1"
    return bool(getattr(config, "profile_packed_kv", False))


def _rope_mutates_inputs(rope: Any) -> bool:
    """Does this rotary module write into the tensors it is handed?

    It depends on which implementation the CustomOp dispatch picked.  vLLM's
    forward_cuda rotates the query and key in place, while forward_native
    returns new tensors.  The router reads the pre-RoPE tensors, so an
    in-place rotation must be preceded by a clone, which costs about 4.3 GB
    per sparse layer at 512K.  Anything not known to be native is treated as
    in-place.
    """

    if os.environ.get("SEMBLO_ROPE_FORCE_CLONE") == "1":
        return True          # always clone
    method = getattr(rope, "_forward_method", None)
    return getattr(method, "__name__", "") != "forward_native"


def _norm_forward(norm: Any, x: torch.Tensor) -> torch.Tensor:
    """Run a norm through vLLM's CUDA implementation when it has one.

    Inside the opaque operator nothing fuses the native implementation, so
    the module's own dispatch is slow here.  Falls back to the module call on
    CPU and for test doubles.
    """

    method = getattr(norm, "forward_cuda", None)
    if method is not None and x.is_cuda:
        return method(x)
    return norm(x)


_KV_VIEW_CACHE: dict[tuple[int, int, tuple[int, ...]], torch.Tensor] = {}



_SKIPPED_CACHE_UPDATES: dict = {}


def record_skipped_cache_update(layer_idx: int) -> None:
    """Count the standard cache writes that own_prompt_cache suppresses."""

    _SKIPPED_CACHE_UPDATES[int(layer_idx)] = \
        _SKIPPED_CACHE_UPDATES.get(int(layer_idx), 0) + 1


def skipped_cache_updates() -> dict:
    return dict(_SKIPPED_CACHE_UPDATES)

def _canonical_kv_cache_view(attn_layer: Any, kv_cache: torch.Tensor) -> torch.Tensor:
    """View the paged cache as ``[2,num_blocks,block,Hkv,D]`` without copying.

    vLLM backends disagree on axis order: FlashAttention leads with the
    key/value pair while the Triton backend places it after the block axis.
    The backend is asked for its own shape with distinct probe sizes, and the
    cache is permuted so the kernel reads it through explicit strides.

    The view is cached per layer and cache storage, because it is needed on
    every decode step and vLLM allocates a layer's cache once per engine.
    """

    cache_key = (id(attn_layer), kv_cache.data_ptr(), tuple(map(int, kv_cache.shape)))
    cached = _KV_VIEW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    backend = getattr(attn_layer, "attn_backend", None)
    if backend is None:
        raise RuntimeError("the vLLM attention layer does not expose its backend")
    _kind, num_blocks, block_size, num_kv_heads, head_dim = _KV_CACHE_PROBE_SHAPE
    probe = tuple(
        int(size)
        for size in backend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_dim
        )
    )
    if sorted(probe) != sorted(_KV_CACHE_PROBE_SHAPE):
        raise RuntimeError(
            f"unsupported vLLM KV-cache layout {probe}; the semblo kernel "
            "needs a permutation of [2,num_blocks,block,Hkv,D]"
        )
    if kv_cache.ndim != 5:
        raise RuntimeError(
            "expected a five-dimensional vLLM paged KV cache, got shape "
            f"{tuple(int(size) for size in kv_cache.shape)}"
        )
    axis_of = {size: axis for axis, size in enumerate(probe)}
    order = tuple(axis_of[size] for size in _KV_CACHE_PROBE_SHAPE)
    canonical = kv_cache.permute(order)
    if int(canonical.shape[0]) != 2:
        raise RuntimeError(
            "expected vLLM paged KV cache with shape [2,num_blocks,block,Hkv,D]"
        )
    _KV_VIEW_CACHE[cache_key] = canonical
    return canonical


def _source_paged_context(source: "SembloAttention") -> _PagedContext:
    if source.runtime_config.own_prompt_cache:
        # This layer has no vLLM cache or block table.  The suffix store
        # supplies a cache tensor, a block table and a block size, addressed
        # by absolute position like the vLLM cache.
        state = source.runtime_state
        buf = state.suffix_kv.get(source.source_layer_idx)
        if buf is None or state.suffix_block_table is None:
            raise RuntimeError(
                "own_prompt_cache: the suffix store is read before any "
                "generated token was written to it"
            )
        return _PagedContext(kv_cache=buf,
                             block_table=state.suffix_block_table,
                             block_size=int(state.suffix_block_size))
    metadata, attn_layer, kv_cache, _slot_mapping = get_attention_context(
        source.attn.layer_name
    )
    if metadata is None:
        raise RuntimeError("vLLM attention metadata is unavailable")
    block_table = getattr(metadata, "block_table", None)
    if block_table is None:
        raise RuntimeError(
            "semblo sparse attention requires a vLLM backend exposing block_table"
        )
    if block_table.ndim != 2 or block_table.shape[0] != 1:
        raise RuntimeError("the first integration requires a batch-1 block table")
    canonical = _canonical_kv_cache_view(attn_layer, kv_cache)
    return _PagedContext(
        kv_cache=canonical,
        block_table=block_table,
        block_size=int(canonical.shape[2]),
    )


_SAID: set[str] = set()


def _say_once(key: str, message: str) -> None:
    """Print a message once per distinct key."""

    if key not in _SAID:
        _SAID.add(key)
        print(message, flush=True)


class SembloAttention(nn.Module):
    """Sparse attention for one full-attention layer.

    The model-specific subclasses (qwen3_5_model.py, nemotron_h_model.py)
    set the projections and head sizes and call ``_init_semblo_runtime``.
    """

    def _init_semblo_runtime(
        self,
        *,
        runtime_config: SembloVllmConfig,
        runtime_state: SembloRuntimeState,
        layer_idx: int,
        source_layer_idx: int,
        uses_sparse_attention: bool,
        builds_product_index: bool,
        attention_logits_soft_cap: float | None,
    ) -> None:
        """Architecture-independent wiring shared with the Qwen3.5 wrapper.

        Requires ``num_heads/num_kv_heads/head_dim`` to be set already."""

        self.layer_idx = int(layer_idx)
        self.source_layer_idx = int(source_layer_idx)
        self.uses_sparse_attention = bool(uses_sparse_attention)
        self.builds_product_index = bool(builds_product_index)
        self.runtime_config = runtime_config
        self.runtime_state = runtime_state
        self.attention_logits_soft_cap = attention_logits_soft_cap
        # Per-layer selection budget; layers absent from the map use the
        # global target.  Resolved once here so the hot path reads one int.
        self.layer_target_tokens = runtime_config.layer_target_tokens.get(
            self.layer_idx, runtime_config.target_tokens
        )
        # A decode-only budget.  0 keeps decode on the prefill budget.  The
        # per-layer map takes precedence.
        self.layer_decode_target_tokens = (
            self.layer_target_tokens
            if self.layer_idx in runtime_config.layer_target_tokens
            or not runtime_config.decode_target_tokens
            else runtime_config.decode_target_tokens
        )
        # Set per call from the phase; every selection site reads this.
        self._effective_target = self.layer_target_tokens
        # The decode step of this layer, captured once per request (see
        # decode_graph.py).
        self._decode_graph = DecodeStepGraph(f"layer {self.layer_idx}")
        self._decode_buffers_cache: tuple | None = None

        if self.builds_product_index and runtime_config.mode == "sparse":
            # The routing map (routing_map.py): the identity unless a map file
            # is loaded.  Queries go through the source layer's map.
            self.semblo_key_router = SembloKeyRouter(
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                index_dim=runtime_config.index_dim,
                num_factors=runtime_config.num_factors,
            )
            if runtime_config.router_parameter_dtype == "float32":
                self.semblo_key_router.float()

    def _project_kv(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """K and V only, for pass 1 of a streamed prefill.

        The default runs the fused projection and drops Q.  A model whose
        fused weight can be sliced overrides this and projects K/V alone.
        """

        _q_post, k_post, v_states, _q_pre, k_pre, _gate = self._project_qkv(
            positions, hidden_states
        )
        return k_post, v_states, k_pre

    def _source_module(self) -> "SembloAttention":
        source = self.runtime_state.get_source_module(self.source_layer_idx)
        if not isinstance(source, SembloAttention):
            raise RuntimeError("registered index source has an unexpected type")
        return source

    def _fit_index(self, product, positions):
        """Fit the partition and posting lists for ``positions`` from their
        routing keys.  The whole prompt goes through here once."""
        if positions.numel() < 128:
            raise ValueError("sparse prefill requires at least 128 prompt tokens after chat formatting")
        return ProductPostingIndex.build(
            product,
            positions,
            num_centroids=self.runtime_config.num_centroids,
            kmeans_iters=self.runtime_config.kmeans_iters,
            kmeans_sample_size=self.runtime_config.kmeans_sample_size,
            kmeans_init=self.runtime_config.kmeans_init,
            kmeans_preprocess=self.runtime_config.kmeans_preprocess,
            # Offset by the layer so that different layers do not share a
            # random draw, while a given prompt still maps to the same index
            # on every request.
            codebook_seed=self.runtime_config.codebook_seed + self.source_layer_idx,
            # Only read when SEMBLO_FIXED_CODEBOOK names a fixed codebook,
            # which is keyed by (layer, cluster count).
            layer_idx=self.source_layer_idx,
            flat_unit_tokens=self.runtime_config.flat_unit_tokens,
            flat_min_clusters_log2=self.runtime_config.flat_min_clusters_log2,
            page_align=self.runtime_config.fa3_page_align,
        )

    def _build_or_append_index(
        self,
        k_pre_rope: torch.Tensor,
        positions: torch.Tensor,
    ) -> ProductPostingIndex:
        if not self.builds_product_index:
            raise RuntimeError("only layers with their own K/V can build an index")
        if self.runtime_state.step_first_position != 0:
            index = self.runtime_state.get(self.source_layer_idx)
            self.runtime_state.record_step_positions(index, positions)
            return index
        product = self.semblo_key_router(k_pre_rope.unsqueeze(0))[0]
        dump_dir = os.environ.get("SEMBLO_DUMP_PROJECTED_KEYS")
        if dump_dir:
            _dump_projected_keys(dump_dir, product, positions, self.source_layer_idx,
                                 self.runtime_state.prompt_epoch, raw_keys=k_pre_rope)
        index = self._fit_index(product, positions)
        if not DEFERRED_CHECKS.off and DEFERRED_CHECKS.active:
            # A collapsed index, where one bucket holds most of the keys,
            # makes selection pick the whole context and overrun the capacity
            # workspace.  Real prompts keep their largest bucket far below
            # half the context, so this reports such a build on the next step.
            DEFERRED_CHECKS.record_max(
                SLOT_BUCKET_COLLAPSE,
                index.bucket_sizes,
                message="index bucket collapse: one bucket holds most of the prompt",
                limit=max(int(index.prompt_length) // 2, 8192),
            )
        self.runtime_state.put(self.source_layer_idx, index)
        return index

    def _query_state(
        self,
        q_pre_rope: torch.Tensor,
        source: "SembloAttention",
    ) -> SembloQueryState:
        """The routing vectors of [Q, Hq, D] pre-RoPE queries.

        A query goes through the map of its KV head in the source layer and
        is normalized like the keys (``SEMBLO_ROUTING_NORMALIZE``).  One
        routing vector per head, with a uniform mixture.
        """
        weight = source.semblo_key_router.weight                  # [Hkv, I, D]
        cached = getattr(self, "_routing_map", None)
        if cached is None or cached[0] != (weight.data_ptr(), q_pre_rope.dtype):
            q_to_kv = torch.div(
                torch.arange(self.num_heads, device=weight.device) * self.num_kv_heads,
                self.num_heads, rounding_mode="floor")
            per_q = weight.to(dtype=q_pre_rope.dtype).index_select(0, q_to_kv)
            # Decided once per layer, never per call: torch.equal on CUDA
            # tensors synchronizes the device, which would invalidate a
            # decode-step graph capture.
            identity = bool(per_q.shape[1] == per_q.shape[2] and torch.equal(
                per_q, torch.eye(per_q.shape[1], dtype=per_q.dtype,
                                 device=per_q.device).expand_as(per_q)))
            print(f"[semblo] routing map layer {getattr(self, 'layer_idx', '?')}: "
                  + ("identity" if identity else f"linear {tuple(weight.shape)}"), flush=True)
            cached = self._routing_map = ((weight.data_ptr(), q_pre_rope.dtype), per_q, identity)
        _, per_q, identity = cached
        mode = os.environ.get("SEMBLO_ROUTING_NORMALIZE", "whole") or "whole"
        factors = 1 if mode == "whole" else int(self.runtime_config.num_factors)
        queries, heads = q_pre_rope.shape[0], q_pre_rope.shape[1]
        if identity:
            projected = q_pre_rope
        else:
            projected = torch.einsum(
                "qhd,hid->qhi", q_pre_rope.to(torch.bfloat16), per_q.to(torch.bfloat16))
        dim = projected.shape[-1]
        from semblo.vllm_plugin.trivial_router import trivial_router_vectors
        if mode == "none":
            # Not normalized: for block representatives that are pooled keys
            # rather than points on a sphere, where the temperature carries
            # the whole scale.
            from semblo.vllm_plugin import block_minmax
            if block_minmax.enabled():
                # Quest-style scoring: [q ; q] -> [max(q, 0) ; min(q, 0)]
                # against the block's [max ; min] descriptor (block_minmax.py).
                projected = block_minmax.quest_query_vectors(projected)
            vectors = projected.contiguous().view(1, queries, heads, 1, dim)
        elif mode.startswith("split:"):
            from semblo.inference.flat_index import routing_space
            vectors = routing_space(projected).to(projected.dtype) \
                .contiguous().view(1, queries, heads, 1, dim)
        else:
            vectors = trivial_router_vectors(projected, factors).view(
                1, queries, heads, 1, dim)
        mixture = torch.ones(1, queries, heads, 1, dtype=torch.float32,
                             device=q_pre_rope.device)
        return SembloQueryState(vectors=vectors, mixture_weights=mixture)

    def _merge_generated_suffix(
        self,
        prompt_result: PagedSparseAttentionResult,
        q: torch.Tensor,
        start: int,
        end: int,
        index: ProductPostingIndex,
        context: _PagedContext,
        q_positions: torch.Tensor,
    ) -> PagedSparseAttentionResult:
        """Attend densely over the generated suffix and merge by log-sum-exp.

        The suffix is never packed or bucketed: it lives only in the paged
        cache, and keeping it there makes appending a decode token O(1).
        """

        if end - start != 1:
            raise RuntimeError(
                "the initial vLLM decode path supports one generated query at a time"
            )
        posting, starts, lengths, counts, selected = \
            self.runtime_state.suffix_tiles_for_step(
                index,
                num_kv_heads=self.num_kv_heads,
                tile_size=self.runtime_config.triton_token_tile,
                pool=self._index_buffer_pool() is not None,
            )
        suffix_posting = posting
        suffix_tiles = PostingTiles(starts, lengths, counts, selected)
        suffix_result = paged_posting_attention(
            q[start:end],
            context.kv_cache,
            suffix_posting,
            suffix_tiles,
            q_positions,
            torch.zeros_like(q_positions, dtype=torch.int32),
            context.block_table,
            block_size=context.block_size,
            sm_scale=self.scaling,
            softcap=self.attention_logits_soft_cap,
            token_tile=self.runtime_config.triton_token_tile,
            num_splits=self.runtime_config.decode_num_splits,
            return_logsumexp=True,
        )
        return merge_paged_attention_results(prompt_result, suffix_result)

    def _fused_decode_fn(self):
        """The fused decode implementation this configuration selects, or None.

        The conditions are the ones the chunk loop checks before taking its
        fused branch, hoisted so the captured path can ask the same question
        without entering the loop.
        """

        if not (
            self.runtime_config.mode == "sparse"
            and self.runtime_config.fused_select_block_queries > 0
            and not self.attention_logits_soft_cap
        ):
            return None
        selected = os.environ.get("SEMBLO_FUSED_DECODE")
        if selected == "1":
            return _fused_prefill.fused_decode_attention
        return None

    def _decode_buffers(self, q: torch.Tensor) -> tuple:
        """Per-request buffers for the captured decode step.

        A captured graph replays the addresses it was captured with, so the
        step's output and its query metadata cannot be freshly allocated each
        time.  They are made once per request and rewritten in place.
        """

        # Keyed by shape rather than by request: they are written before they
        # are read, so a capture can outlive a request when the index's
        # buffers are pooled too.
        cached = self._decode_buffers_cache
        if cached is None or cached[0] != tuple(q.shape):
            cached = (
                tuple(q.shape),
                torch.empty_like(q),
                torch.empty(q.shape[0], dtype=torch.int32, device=q.device),
                torch.zeros(q.shape[0], dtype=torch.int32, device=q.device),
            )
            self._decode_buffers_cache = cached
            self._decode_graph.reset()
        return cached[1], cached[2], cached[3]

    def _decode_body(self, decode_fn, q, q_pre_rope, positions, index, context, source,
                     out, q_positions, request_ids, posting, suffix_tiles) -> None:
        """One generated token through the sparse core, into ``out``.

        Straight-line and free of host reads: the per-layer capture and the
        whole-step (vLLM FULL) capture both record exactly this."""
        # vLLM writes the step's position into a buffer it keeps; copying
        # rather than converting keeps the destination address fixed.
        q_positions.copy_(positions)
        query_state = self._query_state(q_pre_rope, source)
        prompt_result = decode_fn(
            q,
            query_state,
            index,
            q_positions,
            temperature=self.runtime_config.selection_temperature,
            beta=self.runtime_config.bucket_size_penalty,
            target_tokens=int(self._effective_target),
            sm_scale=self.scaling,
            num_splits=self.runtime_config.packed_decode_num_splits * 2,
        )
        suffix_result = paged_posting_attention(
            q,
            context.kv_cache,
            posting,
            suffix_tiles,
            q_positions,
            request_ids,
            context.block_table,
            block_size=context.block_size,
            sm_scale=self.scaling,
            softcap=self.attention_logits_soft_cap,
            token_tile=self.runtime_config.triton_token_tile,
            num_splits=self.runtime_config.decode_num_splits,
            return_logsumexp=True,
        )
        merged = merge_paged_attention_results(prompt_result, suffix_result)
        out.copy_(merged.output)

    def _arena(self, like: torch.Tensor):
        cfg = self.runtime_config
        # The cluster rule exactly as the flat build reads it (state.py): the
        # setting, overridden by the environment.  The arena must agree with
        # the index on the number of clusters.
        unit_tokens = int(os.environ.get("SEMBLO_FLAT_UNIT_TOKENS", cfg.flat_unit_tokens))
        if unit_tokens <= 0:
            raise RuntimeError("full-graph decode needs the flat index (SEMBLO_FLAT_UNIT_TOKENS or "
                               "semblo_flat_unit_tokens > 0)")
        from semblo.vllm_plugin import suffix_flush as _flush
        room = int(self.runtime_state.max_model_len) - int(cfg.token_capacity)
        # Sized to the room the model has for generated tokens unless
        # SEMBLO_SUFFIX_RESERVE says otherwise.
        reserve = _arena.suffix_reserve(room)
        extra = dict(reserve=reserve)
        if _flush.enabled():
            # The flush bounds the exact suffix at keep + every, and the packed
            # store must hold every position the model can reach.
            if _flush.keep_exact() + _flush.flush_every() > reserve:
                raise RuntimeError(
                    f"SEMBLO_SUFFIX_KEEP_EXACT + SEMBLO_SUFFIX_FLUSH_EVERY must not exceed the suffix reserve "
                    f"({reserve})")
            if os.environ.get("SEMBLO_FUSED_DECODE") != "3":
                raise RuntimeError("the suffix flush needs the four-kernel decode step (SEMBLO_FUSED_DECODE=3)")
            if os.environ.get("SEMBLO_ROUTING_CENTER") == "1":
                raise RuntimeError("the suffix flush does not re-centre keys (SEMBLO_ROUTING_CENTER=1)")
            extra.update(packed_capacity=int(self.runtime_state.max_model_len),
                         table_positions=int(self.runtime_state.max_model_len) + reserve,
                         keep_kpre=True)
        elif room > reserve:
            # The suffix tiles and store cover the reservation and nothing
            # past it: a longer generation would be silently unattended.
            raise RuntimeError(
                f"full-graph decode reserves {reserve} generated positions "
                f"(SEMBLO_SUFFIX_RESERVE) but max_model_len {self.runtime_state.max_model_len} "
                f"minus the token capacity {cfg.token_capacity} allows {room}")
        return _arena.arena_for(
            self.runtime_state, self.source_layer_idx,
            capacity=int(cfg.token_capacity), unit_tokens=unit_tokens,
            min_clusters_log2=max(3, int(os.environ.get("SEMBLO_FLAT_MIN_CLUSTERS_LOG2",
                                                        cfg.flat_min_clusters_log2))),
            num_kv_heads=self.num_kv_heads, head_dim=self.head_dim, index_dim=int(cfg.index_dim),
            page_align=int(os.environ.get("SEMBLO_FA3_ALIGN", cfg.fa3_page_align)),
            kv_dtype=like.dtype, desc_dtype=like.dtype, device=like.device, **extra)

    def _adopt_into_arena(self) -> None:
        """After a request's prefill: its index moves into the layer's arena."""
        index = self.runtime_state.source_indexes.get(self.source_layer_idx)
        if index is None or index.packed_kv is None:
            return
        from semblo.vllm_plugin import suffix_flush

        if (self.runtime_config.own_prompt_cache and not suffix_flush.enabled()
                and int(index.prompt_length) != int(self.runtime_config.token_capacity)):
            raise ValueError(
                "full-graph decode without suffix flushing requires the prompt length "
                "to equal semblo_vllm_token_capacity; use PIECEWISE for variable-length prompts")
        arena = self._arena(index.packed_kv)
        arena.adopt(index, self.runtime_state.prompt_epoch)

    def _fused_decode_v3_check(self, q_pre, source) -> None:
        """Once per layer, before any capture: the four-kernel step's preconditions.

        It routes with the query itself (the trivial identity router, whole
        vector, uniform mixture), reads the suffix from the arena's own
        store, and has no descriptor and no learned map.  Anything else must
        take the general path (SEMBLO_FUSED_DECODE=1), so this raises rather
        than falling back.
        """
        if getattr(self, "_fused_decode_v3_ok", False):
            return
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("the fused decode step's check must run before the capture (no warm-up step ran)")
        cfg = self.runtime_config
        if not cfg.own_prompt_cache:
            raise RuntimeError("SEMBLO_FUSED_DECODE=3 reads the generated suffix from the arena's store: it needs semblo_vllm_own_prompt_cache")
        if int(cfg.flat_unit_tokens if os.environ.get("SEMBLO_FLAT_UNIT_TOKENS") is None else os.environ["SEMBLO_FLAT_UNIT_TOKENS"]) <= 0:
            raise RuntimeError("SEMBLO_FUSED_DECODE=3 needs the flat index")
        state = self._query_state(q_pre.reshape(-1, self.num_heads, self.head_dim), source)
        vectors = state.vectors
        if vectors.shape[-2] != 1 or vectors.shape[-1] != self.head_dim:
            raise RuntimeError(f"SEMBLO_FUSED_DECODE=3 needs one whole-vector routing query per head (got {tuple(vectors.shape)})")
        routing_map = getattr(self, "_routing_map", None)
        if routing_map is None or not routing_map[2]:
            raise RuntimeError("SEMBLO_FUSED_DECODE=3 needs the identity routing map")
        if (os.environ.get("SEMBLO_ROUTING_NORMALIZE", "whole") or "whole") != "whole":
            raise RuntimeError("SEMBLO_FUSED_DECODE=3 normalises the whole routing vector (SEMBLO_ROUTING_NORMALIZE=whole)")
        self._fused_decode_v3_ok = True
        print(f"[semblo] fused decode step (four kernels) engaged in layer {getattr(self, 'layer_idx', '?')}", flush=True)

    def _full_graph_decode_step(self, positions, q_post, k_post, v_states, q_pre, output, k_pre=None) -> None:
        """A decode step that vLLM may be capturing: the arena body, no host state.

        The same code runs for the warm-up (an empty arena), the capture, and
        any eager decode step; a real request's prefill has filled the arena
        by the time its first token is generated."""
        source = self._source_module()
        arena = self._arena(q_post)
        index = arena.index
        # A replayed graph never reaches this Python, so a count that keeps
        # growing during generation means the step runs eagerly.
        _FULL_GRAPH_CALLS[self.layer_idx] = _FULL_GRAPH_CALLS.get(self.layer_idx, 0) + 1
        if self.layer_idx == min(_FULL_GRAPH_CALLS) and _FULL_GRAPH_CALLS[self.layer_idx] % 16 == 0:
            print(f"[semblo] full-graph decode step ran eagerly {_FULL_GRAPH_CALLS[self.layer_idx]} times in layer {self.layer_idx}"
                  f" (capturing now: {torch.cuda.is_current_stream_capturing()})", flush=True)
        if os.environ.get("SEMBLO_FUSED_DECODE") == "3":
            # The four-kernel step (fused_decode.py): the suffix write, the
            # selection with its CSR, the attention over prompt and suffix in
            # one grid, and the merge into ``output``.  Its preconditions are
            # checked once per layer before any capture.
            self._fused_decode_v3_check(q_pre, source)
            if not self.is_kv_shared_layer:
                if k_post is None or v_states is None:
                    raise RuntimeError("physical sparse layers must produce K/V")
                record_skipped_cache_update(self.source_layer_idx)
            self._effective_target = self._target_for_step(True)
            _fused_decode.fused_decode_step(
                self, arena,
                q_post=q_post, q_pre=q_pre, positions=positions,
                k_post=None if self.is_kv_shared_layer else k_post,
                v_states=None if self.is_kv_shared_layer else v_states,
                k_pre=None if self.is_kv_shared_layer else k_pre,
                output=output,
                temperature=self.runtime_config.selection_temperature,
                beta=self.runtime_config.bucket_size_penalty,
                target_tokens=int(self._effective_target),
                sm_scale=self.scaling,
                num_splits=self.runtime_config.packed_decode_num_splits * 2,
            )
            return
        if not self.is_kv_shared_layer:
            if k_post is None or v_states is None:
                raise RuntimeError("physical sparse layers must produce K/V")
            if not self.runtime_config.own_prompt_cache:
                unified_kv_cache_update(
                    k_post.view(-1, self.num_kv_heads, self.head_dim),
                    v_states,
                    self.attn.layer_name,
                )
            else:
                record_skipped_cache_update(self.source_layer_idx)
                arena.write_suffix(
                    positions,
                    k_post.view(-1, self.num_kv_heads, self.head_dim),
                    v_states.view(-1, self.num_kv_heads, self.head_dim),
                )
        context = _source_paged_context(source)
        q = q_post.view(-1, self.num_heads, self.head_dim).contiguous()
        self._effective_target = self._target_for_step(True)
        decode_fn = self._fused_decode_fn()
        if decode_fn is None:
            raise RuntimeError("full-graph decode needs the fused decode path (sparse mode, fused_select_block_queries > 0)")
        out, q_positions, request_ids = self._decode_buffers(q)
        tile = int(self.runtime_config.triton_token_tile)
        posting, starts, lengths, counts, selected = index.generated_suffix_tiles(
            num_kv_heads=self.num_kv_heads, tile_size=tile)
        # The reservation's tiles are static, so the suffix kernel is told
        # how many tiles the suffix has reached.  This is computed on the
        # device from the position, so a replayed graph reads the right
        # number every step.
        live = ((positions[:1].to(torch.int32) - arena.prompt_length_t) // tile + 1).clamp_(1, int(starts.shape[-1]))
        counts.copy_(live.view(1, 1).expand_as(counts))
        self._decode_body(decode_fn, q, q_pre, positions, index, context, source,
                          out, q_positions, request_ids, posting,
                          PostingTiles(starts, lengths, counts, selected))
        output.copy_(out.view(output.shape))

    def _captured_decode_step(
        self,
        decode_fn,
        q: torch.Tensor,
        q_pre_rope: torch.Tensor,
        positions: torch.Tensor,
        index,
        context: _PagedContext,
        source: "SembloAttention",
    ) -> torch.Tensor:
        """One generated token through the sparse core, captured and replayed.

        This is the chunk loop's fused decode branch without the loop: one
        query, one chunk, no per-step allocation and no host-side value that
        varies from step to step, because a captured body has to be straight
        line code.
        """

        out, q_positions, request_ids = self._decode_buffers(q)
        posting, starts, lengths, counts, selected = \
            self.runtime_state.suffix_tiles_for_step(
                index,
                num_kv_heads=self.num_kv_heads,
                tile_size=self.runtime_config.triton_token_tile,
            )
        suffix_tiles = PostingTiles(starts, lengths, counts, selected)

        def body() -> None:
            self._decode_body(decode_fn, q, q_pre_rope, positions, index, context, source,
                              out, q_positions, request_ids, posting, suffix_tiles)

        self._decode_graph.run(
            (
                # Not the request's identity or epoch: every request makes a
                # new index object, so either would force a re-capture.  The
                # tensor addresses below are the real guard.
                ("num_buckets", int(index.num_buckets)),
                ("is_flat", bool(getattr(index, "is_flat", False))),
                tensor_signature(
                    q=q,
                    q_pre=q_pre_rope,
                    positions=positions,
                    out=out,
                    q_positions=q_positions,
                    request_ids=request_ids,
                    packed_kv=index.packed_kv,
                    posting_positions=index.posting_positions,
                    bucket_sizes=index.bucket_sizes,
                    bucket_indptr=index.bucket_indptr,
                    codebooks=index.codebooks,
                    kv_cache=context.kv_cache,
                    block_table=context.block_table,
                    suffix_posting=posting,
                    suffix_starts=starts,
                    suffix_lengths=lengths,
                    suffix_counts=counts,
                ),
            ),
            body,
        )
        return out.flatten(-2, -1)

    def _target_for_step(self, is_decode: bool) -> int:
        """The selection budget for this call.

        ``_effective_target`` lives on the layer object, which outlives the
        request, so it is set on every call rather than carried over.
        """

        return self.layer_decode_target_tokens if is_decode else self.layer_target_tokens

    def _run_product_attention(
        self,
        q_post_rope: torch.Tensor,
        q_pre_rope: torch.Tensor,
        positions: torch.Tensor,
        out: torch.Tensor | None = None,
        query_offset: int = 0,
    ) -> torch.Tensor:
        """Sparse attention for the given queries, written into ``out`` if given.

        ``query_offset`` is where these queries start in the prompt.  The
        loop below indexes the tensors it was given, and the fused body is
        told the first and last position of each chunk, from which it cuts
        each row's causal length.  Those positions are the local indices plus
        ``query_offset``.
        """
        global SPARSE_ATTENTION_CALLS
        SPARSE_ATTENTION_CALLS += 1
        source = self._source_module()
        index = self.runtime_state.get(self.source_layer_idx)
        context = _source_paged_context(source)
        q = q_post_rope.view(-1, self.num_heads, self.head_dim).contiguous()
        is_decode = self.runtime_state.step_first_position >= int(index.prompt_length)
        self._effective_target = self._target_for_step(is_decode)
        use_packed = (
            self.runtime_config.pack_prompt_kv
            and self.runtime_config.mode == "sparse"
            and index.packed_kv is not None
        )

        if is_decode and use_packed and positions.numel() == 1 and graphs_enabled():
            decode_fn = self._fused_decode_fn()
            if decode_fn is not None:
                return self._captured_decode_step(
                    decode_fn, q, q_pre_rope, positions, index, context, source
                )

        # Write into the caller's output buffer rather than allocating a
        # second one (4 GiB at 512K).  The captured decode step above returns
        # its result by value.
        if out is not None:
            output = out.view(-1, self.num_heads, self.head_dim)
        else:
            output = torch.empty_like(q)
        if not use_packed or self.runtime_config.fused_select_block_queries <= 0:
            raise RuntimeError(
                "the sparse path needs the packed prompt K/V and the fused "
                "selection (semblo_fused_select_block_queries > 0)")
        fused_snapshot = None
        step = self.runtime_config.query_chunk_size
        n_queries = positions.numel()
        for start in range(0, n_queries, step):
            end = min(n_queries, start + step)
            q_positions = positions[start:end].to(torch.int32)
            query_state = self._query_state(q_pre_rope[start:end], source)
            if is_decode:
                decode_fn = self._fused_decode_fn()
                if decode_fn is None:
                    raise RuntimeError("decode needs SEMBLO_FUSED_DECODE=1 or 3")
                prompt_result = decode_fn(
                    q[start:end],
                    query_state,
                    index,
                    q_positions,
                    temperature=self.runtime_config.selection_temperature,
                    beta=self.runtime_config.bucket_size_penalty,
                    target_tokens=int(self._effective_target),
                    sm_scale=self.scaling,
                    num_splits=self.runtime_config.packed_decode_num_splits * 2,
                )
                prompt_result = self._merge_generated_suffix(
                    prompt_result, q, start, end, index, context, q_positions,
                )
                output[start:end] = prompt_result.output
                continue
            # Prefill: block selection followed by the block-union gather
            # body.  Prefill positions are a contiguous arange, so the
            # chunk's last position is end - 1 + query_offset.
            if os.environ.get("SEMBLO_FUSED_CONSENSUS") == "1":
                # Block-shared selection: route one pooled query per union
                # block.  The attention body still runs per query.
                g = self.runtime_config.fused_select_block_queries
                qp = q_pre_rope[start:end]
                nb_c = (qp.shape[0] + g - 1) // g
                pad_c = nb_c * g - qp.shape[0]
                if pad_c:
                    tail = qp[-1:].expand(pad_c, *qp.shape[1:])
                    qp = torch.cat([qp, tail], dim=0)
                if os.environ.get("SEMBLO_CONSENSUS_SCORE", "mean") in ("sumexp", "max"):
                    # FlashPrefill's tile score: every query is routed, and
                    # the selector forms one score row per block.
                    query_state = self._query_state(qp, source)
                else:
                    pooled = qp.reshape(nb_c, g, *qp.shape[1:]).mean(dim=1)
                    query_state = self._query_state(pooled, source)
            if fused_snapshot is None:
                # The token-outer prefill keeps its own running counts.
                fused_snapshot = getattr(index, "_token_outer_snapshot", None) \
                    or _fused_prefill.CausalSnapshot(index)
            if os.environ.get("SEMBLO_CHECK_POSITIONS") == "1":
                # The body is told a position and given indices.  Check that
                # they agree (one host read per chunk).
                first = int(q_positions[0])
                if first != start + query_offset:
                    raise RuntimeError(
                        f"query {start} of this call sits at position "
                        f"{first}, but the body is being told "
                        f"{start + query_offset}: the causal cut would "
                        f"be taken at the wrong place")
            output[start:end] = _fused_prefill.fused_chunk_attention(
                q[start:end],
                query_state,
                index,
                q_positions,
                fused_snapshot,
                start + query_offset,
                end - 1 + query_offset,
                temperature=self.runtime_config.selection_temperature,
                beta=self.runtime_config.bucket_size_penalty,
                target_tokens=int(self._effective_target),
                queries_per_block=self.runtime_config.fused_select_block_queries,
                sm_scale=self.scaling,
                layer_idx=self.layer_idx,
                fa3_page_size=self.runtime_config.fa3_page_size,
                kv_outer_default=self.runtime_config.kv_outer,
                dense_prefix_x=4,
            )
        if not is_decode and getattr(index, "original_kv", None) is not None \
                and os.environ.get("SEMBLO_KV_OUTER_GATHER", "0") != "1":
            # SEMBLO_KV_OUTER_GATHER=1 reads them on every slice of the
            # streamed path, so they then live as long as the index.
            index.original_kv = None
        if not is_decode and getattr(index, "original_k_pre", None) is not None:
            index.original_k_pre = None
        if not is_decode and output.is_cuda:
            # Hand cached-but-free blocks back to the driver when they pile up
            # within a prompt, which can otherwise exhaust memory at 512K.
            # Off by default because a release costs ~170 ms.  With
            # SEMBLO_EMPTY_CACHE_ABOVE_GIB=N it runs only when the cached free
            # memory exceeds N GiB.
            above = float(os.environ.get("SEMBLO_EMPTY_CACHE_ABOVE_GIB", "0"))
            if above > 0:
                cached = (torch.cuda.memory_reserved()
                          - torch.cuda.memory_allocated()) / float(1 << 30)
                if cached > above:
                    torch.cuda.empty_cache()
                    if os.environ.get("SEMBLO_EMPTY_CACHE_LOG", "1") == "1":
                        after = (torch.cuda.memory_reserved()
                                 - torch.cuda.memory_allocated()) / float(1 << 30)
                        print(f"[semblo] cache release after layer "
                              f"{getattr(self, 'layer_idx', '?')}: cached-free "
                              f"{cached:.1f} -> {after:.1f} GiB", flush=True)
        return output.flatten(-2, -1)

    def _index_buffer_pool(self):
        """Where this layer's index should allocate, or None for fresh memory.

        Reusing the previous request's buffers keeps the addresses the
        captured decode step replays, which avoids a re-capture per request.
        Off unless ``SEMBLO_INDEX_POOL=1``, because it makes a request's index
        memory outlive the request.  Every pooled buffer is fully written
        before it is read.
        """

        if _arena.full_graph_decode_enabled():
            # The cluster-ordered copy is written straight into the layer's
            # arena, whose address the captured decode step replays, so the
            # adoption after prefill has nothing to copy.
            def arena_pool(shape, dtype, device):
                arena = self._arena(torch.empty(0, dtype=dtype, device=device))
                return arena.packed_rows(int(shape[1]), tuple(shape), dtype)

            return arena_pool
        if os.environ.get("SEMBLO_INDEX_POOL") != "1":
            return None
        state = self.runtime_state
        layer = self.source_layer_idx

        def pool(shape, dtype, device):
            return state.pooled_buffer(f"{layer}:packed_kv", shape, dtype, device)

        return pool

    def _adopt_index_buffers(self, index) -> None:
        """Move the index's smaller arrays into engine-lifetime buffers too."""

        if os.environ.get("SEMBLO_INDEX_POOL") != "1":
            return
        self.runtime_state.adopt_index_buffers(self.source_layer_idx, index)

    def _reserve_packed_kv_for_profiling(self, like: torch.Tensor) -> None:
        """Let the profiling pass see the memory a served request will take.

        vLLM sizes its KV pool from the peak allocation of a dummy forward
        pass.  On that pass this layer takes the dense path, so the packed
        prompt K/V it allocates when serving (2.15 GB per sparse layer at a
        1M-token capacity) is invisible, and the KV pool can take that room.

        Off by default: the profiled peak is dominated by a transient that
        the served request never coexists with, so the reservation can make
        shorter contexts fail to start.  See the setting's note in config.py.

        Only the packed K/V is reserved.  The index's own arrays are two
        orders of magnitude smaller.  The reservation is held across the
        layers of the pass, so the peak contains all of them at once.
        """

        if not _profile_packed_kv_enabled(self.runtime_config):
            return
        if not (
            self.builds_product_index
            and self.runtime_config.pack_prompt_kv
            and self.runtime_config.mode == "sparse"
            and like.is_cuda
        ):
            return
        if self.layer_idx in self.runtime_state.profiling_reservation:
            return
        # Only the pass that decides the pool size, which runs before the
        # pool exists.  vLLM's later warm-ups also arrive without attention
        # metadata, but by then the pool is allocated.
        _metadata, _layer, kv_cache, _slots = get_attention_context(
            self.attn.layer_name
        )
        if kv_cache is not None and kv_cache.numel():
            return
        # ``pack_prompt_kv`` allocates [2, capacity + pad, Hkv, D].  The pad
        # makes the token axis a whole number of the largest page it may be
        # viewed as.
        capacity = int(self.runtime_config.token_capacity)
        padded = capacity + (-capacity) % _PACK_ALIGN
        self.runtime_state.profiling_reservation[self.layer_idx] = torch.empty(
            2, padded, self.num_kv_heads, self.head_dim,
            dtype=like.dtype, device=like.device,
        )

    def _release_profiling_reservation(self) -> None:
        """Drop the reservation once the pass has seen every layer's share.

        It must survive until the last sparse layer of the pass, so the peak
        contains all layers at once.  It must not survive the pass itself:
        vLLM measures the profile from the peak and from what is still
        allocated at the end, so memory left standing is counted twice.
        """

        state = self.runtime_state
        if not state.profiling_reservation or not state.source_modules:
            return
        # The last source layer is the last one that reserves, and layers run
        # in order, so by the time it is reached every share is in the peak.
        if self.layer_idx >= max(state.source_modules):
            state.profiling_reservation.clear()

    def _is_memory_profiling_run(self) -> bool:
        """Report whether vLLM is sizing the KV cache instead of serving.

        ``profile_run`` executes the model once with dummy tokens and no
        attention metadata.  Its positions do not form a single prompt, so a
        request-local index must not be built from them.
        """

        if self.runtime_config.own_prompt_cache:
            # This layer is in no KV-cache group, so the per-layer lookup
            # below would raise KeyError.  vLLM's profiling pass carries no
            # attention metadata at all, while a real step carries a dict of
            # it.  The runtime state's step position cannot be used here,
            # because maybe_begin_prompt sets it after this check.
            from vllm.forward_context import get_forward_context
            return get_forward_context().attn_metadata is None
        metadata, _attn_layer, _kv_cache, _slot_mapping = get_attention_context(
            self.attn.layer_name
        )
        return metadata is None

    def _sparse_layer_forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Everything torch.compile can see, with the core as an operator.

        The projection, the norms, the rope and the output projection do not
        depend on the per-request index, so they are compiled and captured
        here, as vLLM does for its own attention layers.  Only the core goes
        through the operator.

        The buffer the operator fills is allocated here, in the traced caller.
        Under piecewise capture it comes from the CUDA graph pool and keeps
        one address, which the captured piece downstream reads on every
        replay.

        There is no branch on the token count: vLLM traces this function once
        and drops every guard, so a branch would be fixed at trace time.  A
        long prefill is diverted before the compiled model instead
        (streamed_prefill.py).
        """

        rope_positions, positions = self._positions_for_runtime(positions)
        q_post, k_post, v_states, q_pre, k_pre, gate = self._project_qkv(
            rope_positions, hidden_states
        )
        attn_output = torch.empty_like(q_post)
        sparse_attention_op(
            positions,
            hidden_states,
            q_post,
            k_post,
            v_states,
            q_pre,
            k_pre,
            attn_output,
            self.attn.layer_name,
        )
        return self._finish_output(attn_output, gate)

    def attention_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        q_post: torch.Tensor,
        k_post: torch.Tensor | None,
        v_states: torch.Tensor | None,
        q_pre: torch.Tensor,
        k_pre: torch.Tensor | None,
        output: torch.Tensor,
    ) -> None:
        """The attention core, reached only through the custom operator.

        Everything here is invisible to torch.compile: the host reads, the
        request-local index lookup, and the profiling branch below, which
        must be decided per call rather than once at compile time.  The
        result is written into ``output``, whose address the captured pieces
        around this layer depend on.

        ``output`` holds the attention result, not the layer result: the
        output projection is on the traced side of the split.
        """

        if self._is_memory_profiling_run():
            # Match the dense model's profiled activation footprint and leave
            # the runtime state untouched.  The traced projection already
            # produced what the official one would have, except on a
            # KV-sharing layer, which has no K or V of its own.
            self._reserve_packed_kv_for_profiling(q_post)
            if self.runtime_config.own_prompt_cache:
                # This layer declared no KV cache, so the official
                # attention's cache update would fail its metadata lookup.
                # The official layer writes zeros during profiling anyway,
                # and the tensors that make up the footprint already exist.
                output.zero_()
            elif k_post is None:
                dense = self._official_dense_attention(positions, hidden_states)
                output.copy_(dense.view(output.shape))
            else:
                dense = self.attn(q_post, k_post, v_states)
                output.copy_(dense.view(output.shape))
            self._release_profiling_reservation()
            return

        if _routing_keys.ROUTE_POST_ROPE:
            # Check that the traced projection honours
            # SEMBLO_ROUTING_KEYS=post, once per layer and path.
            _routing_keys.check_routing_view(
                self.ROUTING_MODEL_NAME, self.layer_idx,
                "decode" if positions.numel() == 1 else "whole-prompt prefill",
                (q_pre, q_post), (k_pre, k_post))

        if _arena.full_graph_decode_enabled() and positions.numel() == 1:
            # One generated token under vLLM's FULL cudagraph: no host
            # bookkeeping, the body over the layer's fixed-address arena.
            self._full_graph_decode_step(positions, q_post, k_post, v_states, q_pre, output, k_pre=k_pre)
            return

        self.runtime_state.maybe_begin_prompt(
            self.layer_idx, positions,
            engage_above_tokens=self.runtime_config.engage_above_tokens)

        if self.runtime_state.dense_request:
            # The whole request is served by the stock attention: no index,
            # no packed copy, no captured decode step.  The stock layer needs
            # vLLM's paged cache, which own_prompt_cache does not declare.
            if self.runtime_config.own_prompt_cache:
                raise RuntimeError(
                    f"a prompt of {positions.numel()} tokens is below the engage "
                    f"threshold ({self.runtime_config.engage_above_tokens}) and would "
                    "run the stock attention, but semblo_vllm_own_prompt_cache "
                    "declared no paged cache for this layer; set "
                    "SEMBLO_ENGAGE_ABOVE_TOKENS=0 or drop own_prompt_cache at this length")
            if k_post is None:
                # A KV-sharing layer has no K/V of its own.  The official
                # attention recomputes the shared ones from hidden_states.
                dense = self._official_dense_attention(positions, hidden_states)
            else:
                dense = self.attn(q_post, k_post, v_states)
            output.copy_(dense.view(output.shape))
            return

        if not self.uses_sparse_attention:
            attn_output = self.attn(q_post, k_post, v_states)
            if self.builds_product_index:
                if k_pre is None:
                    raise RuntimeError("a product-index source must produce keys")
                self._build_or_append_index(k_pre, positions)
            output.copy_(attn_output.view(output.shape))
            return

        self._ingest_prompt_kv(k_post, v_states, k_pre, positions)

        sparse_output = self._run_product_attention(
            q_post, q_pre, positions, out=output
        )
        if sparse_output.data_ptr() != output.data_ptr():
            # Only the captured decode step returns a buffer of its own.
            output.copy_(sparse_output.view(output.shape))
        if _arena.full_graph_decode_enabled() and positions.numel() > 1:
            self._adopt_into_arena()

    def begin_streamed_prefill(self, positions: torch.Tensor) -> "StreamedPrefillSession":
        """This layer's share of a prefill that is run outside the compiled model.

        The streamer (streamed_prefill.py) walks the prompt in slices twice
        through this layer: ``keys`` for every slice, then ``ingest`` once,
        then ``queries`` for every slice.  See the session for what each does.
        """

        return StreamedPrefillSession(self, positions)

    def begin_token_outer_prefill(self, positions: torch.Tensor, chunk: int):
        """This layer's share of a prefill that walks the prompt slice by slice
        through every layer, with the index fixed on the prompt's head
        (token_outer_prefill.py, SEMBLO_PREFILL_ORDER=token)."""

        from semblo.vllm_plugin.token_outer_prefill import TokenOuterPrefillSession
        return TokenOuterPrefillSession(self, positions, chunk)

    @staticmethod
    def _slice_positions(rope_positions: torch.Tensor, start: int, stop: int):
        """Take rows [start, stop) of whatever shape the rotary module wants.

        Qwen3.5 uses an M-RoPE grid of shape [3, T], sliced on the last axis.
        Nemotron-H passes a plain [T].
        """

        if rope_positions.ndim == 2:
            return rope_positions[:, start:stop]
        return rope_positions[start:stop]

    def _ingest_prompt_kv(self, k_post, v_states, k_pre, positions, *,
                          prompt_order_views: bool = True) -> None:
        """Store this step's K/V, build the index, and pack the prompt's copy.

        The streamed prefill calls this between its two passes: the index has
        to exist before any query is scored, and it is built from K alone.

        ``prompt_order_views``: leave views of this call's K/V and pre-RoPE
        keys on the index for the chunk loop (the local window slice and the
        prefix indexes).  The streamed path passes False, because a view
        would keep the full prompt's K/V alive through the second pass.
        """

        if not self.is_kv_shared_layer:
            if k_post is None or v_states is None:
                raise RuntimeError("physical sparse layers must produce K/V")
            if not self.runtime_config.own_prompt_cache:
                unified_kv_cache_update(
                    k_post.view(-1, self.num_kv_heads, self.head_dim),
                    v_states,
                    self.attn.layer_name,
                )
            else:
                # This layer declared no KV cache.  The prompt is served from
                # the cluster-ordered copy packed below, and the generated
                # suffix gets the layer's own store.
                record_skipped_cache_update(self.source_layer_idx)
                self._write_own_suffix(k_post, v_states, positions)
            if self.builds_product_index:
                if k_pre is None:
                    raise RuntimeError("a product-index source must produce pre-RoPE keys")
                index = self._build_or_append_index(k_pre, positions)
                if (
                    self.runtime_config.pack_prompt_kv
                    and self.runtime_config.mode == "sparse"
                    and index.packed_kv is None
                ):
                    # The router reads pre-RoPE keys, but attention reads what
                    # vLLM stored: post-RoPE K and V.  Pack those.
                    collect = self.runtime_config.collect_selection_stats
                    if collect and positions.is_cuda:
                        torch.cuda.synchronize()
                    pack_started = time.perf_counter()
                    index.pack_prompt_kv(
                        k_post.view(-1, self.num_kv_heads, self.head_dim),
                        v_states.view(-1, self.num_kv_heads, self.head_dim),
                        pool=self._index_buffer_pool(),
                    )
                    gather_ablation = os.environ.get("SEMBLO_KV_OUTER_GATHER", "0") == "1"
                    if (prompt_order_views or gather_ablation) and (
                            gather_ablation
                            or os.environ.get("SEMBLO_KV_OUTER_FUSED_LOCAL", "1") == "1"
                            or os.environ.get("SEMBLO_PREFIX_INDEX_DOUBLING", "0") == "1"):
                        # Views of this forward's own tensors.  The chunk loop
                        # slices its local window from them, and the reference
                        # is dropped once the prompt has been attended.  With
                        # SEMBLO_KV_OUTER_GATHER=1 the streamed path keeps
                        # them too.
                        index.original_kv = (
                            k_post.view(-1, self.num_kv_heads, self.head_dim),
                            v_states.view(-1, self.num_kv_heads, self.head_dim),
                        )
                    if prompt_order_views \
                            and os.environ.get("SEMBLO_PREFIX_INDEX_DOUBLING", "0") == "1":
                        index.original_k_pre = k_pre.view(
                            -1, self.num_kv_heads, self.head_dim)
                    self._adopt_index_buffers(index)
                    if self.runtime_config.own_prompt_cache and _arena.full_graph_decode_enabled():
                        # The arena's fixed-address store, mapped from this
                        # prompt's end.
                        self._arena(k_post).install_suffix_store(int(index.prompt_length))
                    elif self.runtime_config.own_prompt_cache:
                        # Allocated now rather than on the first generated
                        # token: the reader builds its context during prefill
                        # too, and a captured decode graph replays fixed
                        # addresses.  Its size is max_model_len minus this
                        # prompt.
                        self._ensure_own_suffix_store(index, k_post)
                    if collect:
                        if positions.is_cuda:
                            torch.cuda.synchronize()
                        record_pack_seconds(time.perf_counter() - pack_started)

    def _ensure_own_suffix_store(self, index, like) -> None:
        """Allocate this layer's suffix store, sized by what can still be generated."""

        prompt_length = int(index.prompt_length)
        cfg = int(self.runtime_state.max_model_len)
        if cfg <= prompt_length:
            raise RuntimeError(
                f"own_prompt_cache: max_model_len {cfg} leaves no room for a "
                f"suffix after a {prompt_length}-token prompt"
            )
        previous = self.runtime_state.suffix_kv.get(self.source_layer_idx)
        buf = self.runtime_state.ensure_suffix_store(
            self.source_layer_idx,
            prompt_length=prompt_length,
            max_model_len=cfg,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=like.dtype,
            device=like.device,
        )
        if previous is not buf:
            # A captured decode step replays fixed addresses, and this
            # buffer moves when the prompt length changes.  The graph is
            # otherwise rebuilt only when the decode buffer's shape changes,
            # which never happens at batch one, so without this reset the
            # replay would read the previous request's suffix.
            self._decode_graph.reset()

    def _write_own_suffix(self, k_post, v_states, positions) -> None:
        """Store one generated token's K/V, under own_prompt_cache.

        Only the suffix.  Prompt positions go into the cluster-ordered copy,
        so writing them here would keep a second full-size copy.
        """

        index = self.runtime_state.source_indexes.get(self.source_layer_idx)
        if index is None:
            return                       # still inside the prompt's own step
        prompt_length = int(index.prompt_length)
        if int(self.runtime_state.step_first_position) < prompt_length:
            return
        self.runtime_state.write_suffix_kv(
            self.source_layer_idx, positions,
            k_post.view(-1, self.num_kv_heads, self.head_dim),
            v_states.view(-1, self.num_kv_heads, self.head_dim),
        )

class StreamedPrefillSession:
    """One sparse layer's share of a streamed prefill: keys first, then queries.

    The streamer hands this layer the prompt twice, a slice at a time:

        keys     project the slice and keep only its K and V, until the whole
                 prompt's keys are staged.
        ingest   the barrier: the index is fitted on every key, the
                 cluster-ordered copy is packed, and the K/V staging is dropped.
        queries  project the slice again, keep only Q and the gate, select,
                 attend, gate and project out: this slice's rows of the layer
                 output.

    The projection runs once per pass (about +3% of a 512K prefill).  The index
    is still fitted on the whole prompt, so the same clusters are chosen.

    A profiling pass (no attention metadata) does the same projections, so
    vLLM sizes its pool from the streamed footprint, but it touches no
    runtime state and writes zeros, like the traced path's profiling branch.
    """

    def __init__(self, layer: "SembloAttention", positions: torch.Tensor) -> None:
        self.layer = layer
        self.positions = positions
        self.tokens = int(positions.numel())
        self.profiling = layer._is_memory_profiling_run()
        self.k_post: torch.Tensor | None = None
        self.v: torch.Tensor | None = None
        self.k_pre: torch.Tensor | None = None
        if not self.profiling:
            layer.runtime_state.maybe_begin_prompt(layer.layer_idx, positions)

    def keys(self, rope_positions, hidden_states, start: int, stop: int) -> None:
        layer = self.layer
        k_post, v_states, k_pre = layer._project_kv(rope_positions, hidden_states)
        if k_post is None or v_states is None or k_pre is None:
            raise RuntimeError("the streamed prefill needs a layer with its own K/V")
        if not self.profiling:
            # SEMBLO_ROUTING_KEYS=post check on the keys the index is fitted
            # from.
            _routing_keys.check_routing_view(
                layer.ROUTING_MODEL_NAME, layer.layer_idx,
                "sliced prefill, keys pass (the key store the index is fitted from)",
                (k_pre, k_post))
        if os.environ.get("SEMBLO_CHECK_KV_PROJECTION") == "1":
            # The K/V-only projection against the fused one it replaces.
            _q, k2, v2, _qp, kp2, _g = layer._project_qkv(rope_positions, hidden_states)
            for name, a, b in (("k_post", k_post, k2), ("v", v_states, v2), ("k_pre", k_pre, kp2)):
                if not torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2):
                    raise RuntimeError(
                        f"the K/V-only projection disagrees with the fused one on {name}: "
                        f"max |diff| {(a.float() - b.float()).abs().max().item():.4g}")
        if self.k_post is None:
            self.k_post = k_post.new_empty((self.tokens,) + tuple(k_post.shape[1:]))
            self.v = v_states.new_empty((self.tokens,) + tuple(v_states.shape[1:]))
            self.k_pre = k_pre.new_empty((self.tokens,) + tuple(k_pre.shape[1:]))
        self.k_post[start:stop] = k_post
        self.v[start:stop] = v_states
        self.k_pre[start:stop] = k_pre

    def ingest(self) -> None:
        if self.k_post is None:
            raise RuntimeError("no slice was projected before the index was asked for")
        if self.profiling:
            self.layer._reserve_packed_kv_for_profiling(self.k_post)
        else:
            # Label the work with the custom op's name, so a profile charges
            # it the same way as on the compiled path.
            with torch.profiler.record_function("semblo::attention"):
                # No prompt-order views, so the staging can be freed below.
                # The prefix indexes are therefore not available on this path.
                self.layer._ingest_prompt_kv(self.k_post, self.v, self.k_pre, self.positions,
                                             prompt_order_views=False)
        # Free the full-length staging before the queries pass.
        self.k_post = self.v = self.k_pre = None

    def queries(self, rope_positions, positions, hidden_states, start: int) -> torch.Tensor:
        layer = self.layer
        q_post, _k_post, _v, q_pre, _k_pre, gate = layer._project_qkv(
            rope_positions, hidden_states
        )
        if self.profiling:
            attn = torch.zeros_like(q_post)
        else:
            # SEMBLO_ROUTING_KEYS=post check on the routed queries.
            _routing_keys.check_routing_view(
                layer.ROUTING_MODEL_NAME, layer.layer_idx,
                "sliced prefill, queries pass", (q_pre, q_post))
            with torch.profiler.record_function("semblo::attention"):
                attn = layer._run_product_attention(
                    q_post, q_pre, positions, query_offset=start
                )
        return layer._finish_output(attn.view(hidden_states.shape[0], -1), gate)

    def finish(self) -> None:
        if self.profiling:
            self.layer._release_profiling_reservation()
        if _arena.full_graph_decode_enabled() and not self.profiling:
            self.layer._adopt_into_arena()

def _split_the_graph_at_sparse_attention(vllm_config: VllmConfig) -> None:
    """Ask vLLM's compiler to break its graph around the sparse layer.

    A captured CUDA graph records fixed addresses, and this layer reads an
    index that is built per request, so the layer cannot be inside one.
    Everything else in the model can: the piecewise mode captures the pieces
    between the operators listed here and runs the operators themselves
    normally.

    This appends at model construction rather than in the configuration
    hook.  vLLM fills ``splitting_ops`` with its own attention operators only
    when the field is unset, so setting it earlier would replace that list.
    By the time a model is constructed the field is filled, and the compiler
    has not yet run.
    """

    compilation = getattr(vllm_config, "compilation_config", None)
    if compilation is None:
        return
    ops = getattr(compilation, "splitting_ops", None)
    if ops is None:
        # Compilation is off, so there is no graph to split.
        return
    if SPLITTING_OP_NAME not in ops:
        ops.append(SPLITTING_OP_NAME)
