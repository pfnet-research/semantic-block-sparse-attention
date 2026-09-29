from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Literal


VllmSparseMode = Literal["dense_passthrough", "all_tokens", "sparse"]


def _text_config(config: Any) -> Any:
    if isinstance(config, dict):
        return config.get("text_config", config)
    return getattr(config, "text_config", config)


_STALE_CHUNK_VARIABLES = (
    "SEMBLO_PREFILL_PROJECTION_CHUNK",
    "SEMBLO_MLP_CHUNK",
    "SEMBLO_GDN_CHUNK",
    "SEMBLO_MLP_SPLIT_GRAPH",
)


def _prefill_chunk(text: Any) -> int:
    """The slice size of a streamed prefill, from the environment or the config.

    The per-block chunk variables in ``_STALE_CHUNK_VARIABLES`` are refused
    rather than ignored, so that a run setting one does not silently run whole.
    """

    for stale in _STALE_CHUNK_VARIABLES:
        if stale in os.environ:
            raise ValueError(
                f"{stale} is gone: the whole long prefill now streams in slices "
                "of SEMBLO_PREFILL_CHUNK tokens (streamed_prefill.py)"
            )
    return int(os.environ.get(
        "SEMBLO_PREFILL_CHUNK", _value(text, "semblo_vllm_prefill_chunk", 0)))


def _flag(config: Any, name: str, default: bool) -> bool:
    """Read a boolean setting that may arrive as the string "False".

    ``--hf-text-override`` passes its right-hand side through as text, and
    ``bool("False")`` is True.
    """

    raw = _value(config, name, default)
    if isinstance(raw, str):
        low = raw.strip().lower()
        if low in ("false", "0", "no", "off", ""):
            return False
        if low in ("true", "1", "yes", "on"):
            return True
        raise ValueError(f"{name}: cannot read {raw!r} as a boolean")
    return bool(raw)


def _value(config: Any, name: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _layer_target_tokens(raw: Any) -> dict[int, int]:
    """Parse the per-layer selection-budget map (layer index -> tokens)."""

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            "semblo_target_tokens_per_layer must be a mapping "
            "of layer index to token budget"
        )
    parsed: dict[int, int] = {}
    for key, value in raw.items():
        parsed[int(key)] = int(value)
    return parsed


def _runtime_checks_setting(text: Any) -> bool:
    """Read the runtime-checks setting; SEMBLO_RUNTIME_CHECKS overrides the config."""

    override = os.environ.get("SEMBLO_RUNTIME_CHECKS")
    if override is not None:
        return override.strip().lower() not in {"0", "false", "off", ""}
    return bool(_value(text, "semblo_vllm_runtime_checks", True))


@dataclass(frozen=True)
class SembloVllmConfig:
    """Runtime-only configuration for the vLLM integration.

    The supported setting is narrow: one active request, non-chunked prompt
    prefill, and no prefix caching.  These constraints are validated rather
    than silently falling back to a different routing lifecycle.
    """

    mode: VllmSparseMode
    layer_indices: tuple[int, ...]
    index_dim: int
    num_factors: int
    num_centroids: int
    kmeans_iters: int
    kmeans_sample_size: int
    kmeans_init: str
    # Assignment-side preprocessing before the per-prompt k-means: "none" or
    # "whiten" (per-factor whitening of the partition only; the stored
    # centroids stay raw-space group means).
    kmeans_preprocess: str
    codebook_seed: int
    collect_selection_stats: bool
    pack_prompt_kv: bool
    # Whether the memory-profiling pass allocates the packed prompt K/V a
    # served request will need, so that vLLM's KV pool is sized with it in
    # mind.  vLLM sizes the pool from the peak of a dummy forward pass, and on
    # that pass this layer takes the dense path, so the packed buffer is
    # otherwise invisible.
    #
    # The packed buffer holds 2 (K and V) x capacity x the rank's KV head
    # count x head dimension x 2 bytes per layer, the same size as vLLM's own
    # KV pool for the same context (2.15 GB per layer at 512K for Qwen3.5-4B).
    #
    # Off by default: the profiled peak is dominated by an activation transient
    # that never coexists with the served request, so adding the buffer to it
    # over-reserves.  Turn it on if the engine starts but the first long
    # request fails on an allocation made by the model, and raise
    # gpu_memory_utilization with it.  vLLM's ``kv_cache_memory_bytes``, which
    # pins the pool size directly, is usually the simpler option.
    profile_packed_kv: bool
    # Fused prefill fast path (theta-select + block-union body).  Prefill
    # only, needs pack_prompt_kv, no softcap.  0 disables it.
    fused_select_block_queries: int
    router_parameter_dtype: str
    selection_temperature: float
    top_centroids_per_factor: int
    target_tokens: int
    # Optional per-layer overrides of the selection budget, keyed by decoder
    # layer index.  Layers absent from the map use ``target_tokens``.
    # ``token_capacity`` must cover the largest per-layer value.
    layer_target_tokens: dict[int, int]
    max_probe_buckets: int
    # Mean tokens per block in the flat partition, i.e. C = L / flat_unit_tokens.
    # 0 keeps the product lattice.  It sets both the attention body's
    # efficiency (a block is one varlen sequence whose mean causal length is
    # flat_unit_tokens / 2 regardless of L) and the merge volume (partials per
    # query = budget / that length).  SEMBLO_FLAT_UNIT_TOKENS overrides it.
    flat_unit_tokens: int
    # Floor on the flat block count, as a power of two.  It keeps a short
    # prompt from collapsing to a handful of blocks, but it also caps the
    # block size: at 512K a 2,048-token block needs C = 256, and a floor of
    # 512 would halve the block size instead.
    flat_min_clusters_log2: int
    # The FA3 attention body: page size for the block table (0 = the Triton
    # union body), the packed-KV alignment, and whether the body iterates
    # blocks on the outside and gathers their queries (KV-outer) rather than
    # iterating query tiles.  The defaults are the paper's configuration.
    # SEMBLO_FA3_BODY, SEMBLO_FA3_ALIGN and SEMBLO_KV_OUTER override them.
    own_prompt_cache: bool
    # Prompts no longer than this run the stock dense attention for the whole
    # request; the sparse path engages above it.  0 means always engage.
    engage_above_tokens: int
    prompt_kv_window: int
    fa3_page_size: int
    fa3_page_align: int
    kv_outer: bool
    # Size-aware ranking: a bucket scores score - bucket_size_penalty *
    # log1p(size).  A negative value is a size bonus.
    bucket_size_penalty: float
    query_chunk_size: int
    token_capacity: int
    decode_num_splits: int
    # How many programs share the packed-K/V read at decode; the layer doubles
    # it.  Separate from ``decode_num_splits`` because the two kernels behave
    # differently.  With one split the decode grid is only a few blocks on a
    # 132-SM GPU; at 32 the per-layer decode body is several times faster
    # (about 0.14 ms against 0.02 ms at 256K).
    packed_decode_num_splits: int
    triton_token_tile: int
    # The search path's shape checks.  True runs them deferred (device-side
    # folds, read once per step).  False removes them from the decode path,
    # saving about two dozen kernel launches per step; a failure they would
    # have named then surfaces as a wrong result or a crashing kernel.
    runtime_checks: bool
    # How prompt keys become bucket codes.  "product": per-factor spherical
    # k-means on the two halves of the projected key.  "residual": the first
    # stage clusters the whole vector, the second clusters the residual, and
    # the bucket is the code pair.  The selector is unchanged because the
    # query is duplicated, so the factor-sum score becomes q.(c1+c2).
    quantizer: str
    # Permit the runtime weights file to carry router tensors for layers the
    # runtime config excludes (they are skipped with a warning).  False keeps
    # the exact-match check.
    runtime_allow_unused_layers: bool
    runtime_weights_file: str | None
    runtime_manifest_file: str | None

    @classmethod
    def from_hf_config(
        cls, config: Any, *, identity_router: bool = False
    ) -> "SembloVllmConfig":
        """Parse the ``semblo_*`` keys of an HF config (object or dict).

        ``identity_router`` declares that the routers are the identity map
        built in memory (a stock model, see paper_defaults.py), so no runtime
        weights file may be named.
        """

        text = _text_config(config)
        enabled = bool(_value(text, "semblo_enabled", False))
        mode = str(_value(text, "semblo_vllm_mode", "sparse" if enabled else "dense_passthrough"))
        if mode not in {"dense_passthrough", "all_tokens", "sparse"}:
            raise ValueError(f"unsupported semblo_vllm_mode: {mode!r}")
        raw_layers = _value(text, "semblo_layer_indices", None)
        if raw_layers is None:
            raw_layers = tuple(
                idx
                for idx, layer_type in enumerate(_value(text, "layer_types", ()))
                if layer_type == "full_attention"
            )
        layers = tuple(int(value) for value in raw_layers)
        cfg = cls(
            mode=mode,
            layer_indices=layers,
            index_dim=int(_value(text, "semblo_index_dim", 128)),
            num_factors=int(_value(text, "semblo_num_factors", 2)),
            num_centroids=int(_value(text, "semblo_num_centroids", 64)),
            kmeans_iters=int(_value(text, "semblo_kmeans_iters", 4)),
            kmeans_sample_size=int(_value(text, "semblo_kmeans_sample_size", 8192)),
            kmeans_init=str(_value(text, "semblo_kmeans_init", "random")),
            kmeans_preprocess=str(
                _value(text, "semblo_kmeans_preprocess", "none")
            ),
            codebook_seed=int(
                _value(text, "semblo_vllm_codebook_seed", 0)
            ),
            collect_selection_stats=bool(
                _value(text, "semblo_vllm_collect_selection_stats", False)
            ),
            pack_prompt_kv=bool(
                _value(text, "semblo_vllm_pack_prompt_kv", False)
            ),
            profile_packed_kv=bool(
                _value(text, "semblo_vllm_profile_packed_kv", False)
            ),
            fused_select_block_queries=int(
                _value(text, "semblo_fused_select_block_queries", 0)
            ),
            router_parameter_dtype=str(
                _value(text, "semblo_router_parameter_dtype", "float32")
            ),
            selection_temperature=float(
                _value(text, "semblo_selection_temperature", 0.05)
            ),
            top_centroids_per_factor=int(
                _value(text, "semblo_top_centroids_per_factor", 16)
            ),
            # The environment wins, as for the block size.  This keeps the
            # token budget out of hf_overrides, and so out of vLLM's compile
            # cache key: changing the budget does not recompile the model.
            target_tokens=int(os.environ.get(
                "SEMBLO_TARGET_TOKENS",
                _value(text, "semblo_target_tokens", 4096))),
            # Setting this to the prompt length makes decode dense over the
            # K/V the sparse prefill produced.
            decode_target_tokens=int(os.environ.get(
                "SEMBLO_DECODE_TARGET_TOKENS",
                _value(text, "semblo_decode_target_tokens", 0))),
            prefill_chunk=_prefill_chunk(text),
            dense_tail_tokens=int(os.environ.get(
                "SEMBLO_DENSE_TAIL_TOKENS",
                _value(text, "semblo_dense_tail_tokens", 0))),
            flat_unit_tokens=int(
                _value(text, "semblo_flat_unit_tokens", 0)),
            flat_min_clusters_log2=int(
                _value(text, "semblo_flat_min_clusters_log2",
                       9)),
            # Ask vLLM to allocate nothing for the sparse layers: the
            # block-ordered copy becomes the only store for the prompt.  This
            # is valid because that copy is filled from the layer's own
            # activations and the prompt is never read back from the pool.
            own_prompt_cache=_flag(
                text, "semblo_vllm_own_prompt_cache", False),
            # Under whole-step capture (arena.py) the captured graph is the
            # sparse decode step, so every request must engage the index.
            engage_above_tokens=(0 if os.environ.get("SEMBLO_FULL_GRAPH_DECODE") == "1" else int(os.environ.get(
                "SEMBLO_ENGAGE_ABOVE_TOKENS",
                _value(text, "semblo_vllm_engage_above_tokens", 131072)))),
            # 0 keeps vLLM's full-length pool.  A positive value tells vLLM
            # the sparse layers are sliding-window layers of that many tokens.
            prompt_kv_window=int(
                _value(text, "semblo_vllm_prompt_kv_window", 0)),
            fa3_page_size=int(
                _value(text, "semblo_fa3_page_size", 16)),
            fa3_page_align=int(
                _value(text, "semblo_fa3_page_align", 16)),
            kv_outer=bool(
                _value(text, "semblo_kv_outer", True)),
            layer_target_tokens=_layer_target_tokens(
                _value(text, "semblo_target_tokens_per_layer", None)
            ),
            max_probe_buckets=int(
                _value(text, "semblo_max_probe_buckets", 1024)
            ),
            bucket_size_penalty=float(
                _value(text, "semblo_bucket_size_penalty", 0.0)
            ),
            query_chunk_size=int(_value(text, "semblo_vllm_query_chunk_size", 256)),
            token_capacity=int(_value(text, "semblo_vllm_token_capacity", 8192)),
            decode_num_splits=int(_value(text, "semblo_vllm_decode_num_splits", 8)),
            packed_decode_num_splits=int(
                _value(text, "semblo_vllm_packed_decode_num_splits", 32)
            ),
            triton_token_tile=int(_value(text, "semblo_vllm_token_tile", 16)),
            runtime_checks=_runtime_checks_setting(text),
            quantizer=str(_value(text, "semblo_vllm_quantizer", "product")),
            runtime_allow_unused_layers=bool(
                _value(text, "semblo_runtime_allow_unused_layers", False)
            ),
            runtime_weights_file=(
                str(_value(text, "semblo_runtime_weights_file", "")) or None
            ),
            runtime_manifest_file=(
                str(_value(text, "semblo_runtime_manifest_file", "")) or None
            ),
            identity_router=bool(identity_router),
        )
        cfg.validate()
        return cfg
    # Prefill and decode can run on different budgets.  0 means decode
    # uses the prefill budget.
    decode_target_tokens: int = 0
    # Tokens per slice when a long prefill runs outside the compiled model,
    # layer by layer and slice by slice (streamed_prefill.py).  0 runs every
    # prefill through the compiled model whole.
    prefill_chunk: int = 0
    # The last N query positions of a prefill read every cluster instead of
    # the budget.  0 keeps every position on the budget.
    dense_tail_tokens: int = 0
    # The routers are the identity map built in memory (identity_router.py)
    # instead of read from ``runtime_weights_file``, so a stock Hugging Face
    # model can be served without a prepared directory (paper_defaults.py).
    identity_router: bool = False


    def validate(self) -> None:
        if self.quantizer not in {"product", "residual"}:
            raise ValueError("semblo_vllm_quantizer must be product or residual")
        if self.num_factors != 2:
            raise ValueError("vLLM runtime currently implements exactly two product factors")
        if self.index_dim <= 0 or self.index_dim % self.num_factors:
            raise ValueError("index_dim must be positive and divisible by num_factors")
        if self.num_centroids <= 0:
            raise ValueError("num_centroids must be positive")
        if not 1 <= self.top_centroids_per_factor <= self.num_centroids:
            raise ValueError("top_centroids_per_factor must be in [1, num_centroids]")
        if (
            self.mode == "sparse"
            and self.top_centroids_per_factor != self.num_centroids
        ):
            # Exhaustive factor proposals keep every joint bucket reachable,
            # which guarantees a non-empty causal selection for early query
            # positions without a per-chunk GPU synchronization.  Empty
            # retrieval is an error, so a narrower search could abort a request.
            raise ValueError(
                "the H200 runtime requires exhaustive factor proposals: "
                f"top_centroids_per_factor={self.top_centroids_per_factor} must equal "
                f"num_centroids={self.num_centroids}"
            )
        if self.selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        if self.router_parameter_dtype not in {"model", "float32"}:
            raise ValueError("router_parameter_dtype must be 'model' or 'float32'")
        if self.target_tokens <= 0 or self.token_capacity < self.target_tokens:
            raise ValueError("token_capacity must be at least target_tokens")
        for layer_idx, layer_target in self.layer_target_tokens.items():
            if layer_target <= 0:
                raise ValueError(
                    f"per-layer target_tokens for layer {layer_idx} must be positive"
                )
            if self.token_capacity < layer_target:
                raise ValueError(
                    f"token_capacity {self.token_capacity} is below the per-layer "
                    f"target_tokens {layer_target} for layer {layer_idx}"
                )
        if self.max_probe_buckets <= 0 or self.query_chunk_size <= 0:
            raise ValueError("probe and query-chunk sizes must be positive")
        if self.prefill_chunk < 0:
            raise ValueError("semblo_vllm_prefill_chunk must be 0 or positive")
        if self.own_prompt_cache:
            if not self.pack_prompt_kv:
                raise ValueError(
                    "own_prompt_cache needs pack_prompt_kv: without the "
                    "cluster-ordered copy the prompt has no store at all"
                )
            if self.mode != "sparse":
                raise ValueError(
                    "own_prompt_cache is only meaningful in sparse mode"
                )
            if self.profile_packed_kv:
                # The profiling reservation reads the layer's KV cache, which
                # own_prompt_cache removes.
                raise ValueError(
                    "own_prompt_cache and the profiling reservation cannot be "
                    "used together: the reservation reads the layer's KV cache"
                )
        if self.prompt_kv_window:
            # The window is only safe because the prompt is read from the
            # packed copy.  Without it, the prompt is read from vLLM's pool,
            # and a window would silently return the null block for older
            # tokens.
            if not self.pack_prompt_kv:
                raise ValueError(
                    "prompt_kv_window needs pack_prompt_kv: without the "
                    "packed copy the prompt is read from vLLM's pool"
                )
            if self.mode != "sparse":
                raise ValueError(
                    "prompt_kv_window is only meaningful in sparse mode"
                )
            if self.prompt_kv_window < 1024:
                raise ValueError(
                    "prompt_kv_window must cover the generated suffix plus a "
                    "block; below 1024 tokens that is not worth asserting"
                )
        if self.decode_num_splits not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError("decode_num_splits must be one of 1,2,4,8,16,32,64")
        if self.packed_decode_num_splits not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError("packed_decode_num_splits must be one of 1,2,4,8,16,32,64")
        if self.triton_token_tile not in {8, 16, 32}:
            raise ValueError("triton_token_tile must be 8, 16, or 32")
        if self.identity_router:
            if self.runtime_weights_file or self.runtime_manifest_file:
                raise ValueError(
                    "identity routers are built in memory only for a config "
                    "that names no runtime weights file; this one names "
                    f"{self.runtime_weights_file!r}"
                )
        elif self.mode == "sparse" and (
            not self.runtime_weights_file or not self.runtime_manifest_file
        ):
            raise ValueError(
                "sparse mode requires semblo_runtime.safetensors and its manifest"
            )
        if self.mode == "sparse" and not self.identity_router and (
            self.runtime_weights_file != "semblo_runtime.safetensors"
            or self.runtime_manifest_file != "semblo_runtime.json"
        ):
            raise ValueError(
                "the current loader expects semblo_runtime.safetensors "
                "and semblo_runtime.json in the prepared model directory"
            )

    def validate_vllm_config(self, vllm_config: Any) -> None:
        if self.mode == "dense_passthrough":
            return
        parallel = vllm_config.parallel_config
        scheduler = vllm_config.scheduler_config
        cache = vllm_config.cache_config
        compilation = vllm_config.compilation_config
        tp_size = int(parallel.tensor_parallel_size)
        if tp_size != 1:
            # Router tensors are head-major, index build, selection and
            # packing operate on the rank-local KV-head shard, and the loader
            # slices each rank's head block, so the only requirement is an
            # even split.  KV-head replication (TP above the KV-head count)
            # is not implemented.
            text_config = getattr(
                vllm_config.model_config, "hf_text_config", None
            )
            kv_heads = int(getattr(text_config, "num_key_value_heads", 0) or 0)
            if kv_heads < tp_size or kv_heads % tp_size != 0:
                raise ValueError(
                    "semblo sparse mode with tensor_parallel_size="
                    f"{tp_size} requires the KV-head count ({kv_heads}) to be "
                    "divisible by it; KV-head replication is unimplemented"
                )
        if int(getattr(parallel, "pipeline_parallel_size", 1)) != 1:
            raise ValueError("semblo vLLM sparse mode currently requires pipeline_parallel_size=1")
        if int(scheduler.max_num_seqs) != 1:
            raise ValueError("semblo vLLM sparse mode currently requires max_num_seqs=1")
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError(
                "semblo vLLM sparse mode does not yet support speculative decoding"
            )
        if bool(getattr(scheduler, "enable_chunked_prefill", False)):
            raise ValueError("prompt-global codebooks require enable_chunked_prefill=False")
        if bool(getattr(cache, "enable_prefix_caching", False)):
            raise ValueError("prefix caching is not yet compatible with request-local product indexes")
        if bool(getattr(cache, "kv_sharing_fast_prefill", False)):
            raise ValueError("kv_sharing_fast_prefill must be disabled for sparse global layers")
        max_batched = int(getattr(scheduler, "max_num_batched_tokens", 0) or 0)
        max_model_len = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
        if max_batched and max_model_len and max_batched < max_model_len:
            raise ValueError(
                "prompt-global routing requires max_num_batched_tokens >= max_model_len "
                "for the initial non-chunked implementation"
            )
        # Piecewise capture is supported: the sparse layer is listed in
        # ``splitting_ops`` and stays outside the captured pieces.  The
        # operator writes into a buffer its traced caller allocates inside the
        # captured region, so downstream pieces replay valid addresses.
        #
        # Full capture would take the sparse layer itself into a graph, and
        # the layer holds a request-local index and Python-side suffix state
        # that replay would freeze.  It is refused unless the decode step runs
        # over the fixed-address arena (SEMBLO_FULL_GRAPH_DECODE).
        enforce_eager = bool(getattr(vllm_config.model_config, "enforce_eager", False))
        cudagraph_mode = getattr(compilation, "cudagraph_mode", None)
        if not enforce_eager and cudagraph_mode is not None:
            mode_name = str(cudagraph_mode).upper()
            if "NONE" in mode_name or "OFF" in mode_name:
                pass
            elif "PIECEWISE" in mode_name and "FULL" not in mode_name:
                pass
            elif os.environ.get("SEMBLO_FULL_GRAPH_DECODE") == "1":
                # The decode step runs over a fixed-address arena (arena.py)
                # and may be captured whole.  engage_above_tokens is forced to
                # 0 so that every request takes the captured sparse step.
                #
                # vLLM compiles the model at the first dummy run.  With
                # max_num_seqs=1 the decode-only FULL capture has the single
                # size 1, and if nothing ran the compiled model before it
                # (a streamed prefill runs outside the compiled model),
                # Dynamo traces at one token and specialises the token
                # dimension, and vLLM's piecewise backend fails with an
                # IndexError.  A warm-up at a few tokens, listed as a compile
                # size, makes the first trace symbolic.  vLLM drops a compile
                # size that is also a capture size from the warm-up, so the
                # smallest size from two up that is not a capture size is used.
                sizes = list(getattr(compilation, "compile_sizes", None) or [])
                captured = set(getattr(compilation, "cudagraph_capture_sizes", None) or [])
                if not any(isinstance(s, int) and s >= 2 and s not in captured for s in sizes):
                    size = 2
                    while size in captured:
                        size += 1
                    compilation.compile_sizes = sizes + [size]
                    print(f"[semblo] full-graph decode: compile_sizes += [{size}], so the first compile "
                          "is a warm-up at that many tokens and not the capture's one token", flush=True)
            elif not os.environ.get("SEMBLO_ALLOW_CUDA_GRAPHS"):
                # SEMBLO_ALLOW_CUDA_GRAPHS bypasses this check and is not a
                # supported mode.
                raise ValueError(
                    f"semblo sparse mode cannot use cudagraph_mode "
                    f"{mode_name}: capturing the sparse layer would bake in "
                    "the addresses of a request-local index and freeze its "
                    "Python-side suffix state. Use 'PIECEWISE', which keeps "
                    "the layer outside the captured pieces, or 'NONE'"
                )

