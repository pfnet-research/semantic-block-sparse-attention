"""The paper's configuration as the plugin's defaults.

Two kinds of default live here. Both fill in only what the user left unset.

``PAPER_ENV``
    The runtime switches of the paper's RULER runs (``paper_env`` in
    ``scripts/run_paper_cell.sh``): prefill in 32K slices and single-launch
    decode without decode-step capture. They are set when vLLM loads the
    plugin, so an exported value always wins.

``PAPER_TEXT_CONFIG``
    The ``semblo_*`` settings of a prepared Qwen3.5 checkpoint
    (``configs/checkpoints/qwen3_5_raw_keys_text_config.json``) with the
    paper's ``hf_overrides`` applied. At model construction they are written
    into the text config of a stock Hugging Face model (one whose config
    names no runtime weights file), together with settings derived from the
    model (routing width, sparse layers) and from ``max_model_len`` (token
    capacity and budget). A prepared checkpoint is left unchanged.

Some switches exist both as an environment variable and as a config key, and
the environment variable takes precedence. So that a mere default does not
override a key the user set, the variables filled in here are recorded in
``DEFAULTED_ENV_MARKER`` (inherited by child processes) and are dropped in
favour of an explicit key at model construction
(``yield_defaulted_env_to_config``).
"""

from __future__ import annotations

import os

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from semblo.vllm_plugin.config import SembloVllmConfig, _text_config, _value


# The environment of the paper's RULER runs, without data paths, vLLM's own
# variables, the determinism switch, and the budget (SEMBLO_TARGET_TOKENS),
# which is derived from the context length instead.
# tests/test_paper_defaults.py checks this against scripts/run_paper_cell.sh.
PAPER_ENV: dict[str, str] = {
    "SEMBLO_FUSED_UMAX": "L",
    "SEMBLO_FUSED_DECODE": "1",
    "SEMBLO_FUSED_STRICT_MASK": "0",
    "SEMBLO_FUSED_ROWMASK": "0",
    "SEMBLO_FA3_BODY": "16",
    "SEMBLO_FA3_ALIGN": "16",
    "SEMBLO_KV_OUTER": "1",
    "SEMBLO_FLAT_MIN_CLUSTERS_LOG2": "7",
    "SEMBLO_FLAT_UNIT_TOKENS": "2048",
    "SEMBLO_ROUTING_NORMALIZE": "whole",
    "SEMBLO_SINK_TOKENS": "0",
    "SEMBLO_PREFILL_CHUNK": "32768",
    "SEMBLO_ENGAGE_ABOVE_TOKENS": "0",
    "SEMBLO_EMPTY_CACHE_ABOVE_GIB": "16",
    "SEMBLO_DECODE_GRAPH": "0",
}

# Comma-separated names of the PAPER_ENV variables set by apply_env_defaults.
# Stored in the environment so that the engine process vLLM spawns also knows
# which values are defaults.
DEFAULTED_ENV_MARKER = "SEMBLO_PAPER_ENV_DEFAULTED"

# Environment variables that override a config key wherever they are read
# (config.py, state.py, fused_prefill.py, token_outer_prefill.py, model.py).
ENV_SHADOWING_CONFIG: dict[str, str] = {
    "SEMBLO_FA3_BODY": "semblo_fa3_page_size",
    "SEMBLO_FA3_ALIGN": "semblo_fa3_page_align",
    "SEMBLO_KV_OUTER": "semblo_kv_outer",
    "SEMBLO_FLAT_MIN_CLUSTERS_LOG2": "semblo_flat_min_clusters_log2",
    "SEMBLO_FLAT_UNIT_TOKENS": "semblo_flat_unit_tokens",
    "SEMBLO_PREFILL_CHUNK": "semblo_vllm_prefill_chunk",
    "SEMBLO_ENGAGE_ABOVE_TOKENS": "semblo_vllm_engage_above_tokens",
}

# The semblo_* settings a stock model receives: those of the prepared Qwen3.5
# checkpoint (configs/checkpoints/qwen3_5_raw_keys_text_config.json), with the
# paper's hf_overrides applied (marked "paper"). Settings derived per model are
# not listed: the routing width, sparse layers, token capacity and budget.
PAPER_TEXT_CONFIG: dict[str, Any] = {
    "semblo_bucket_size_penalty": 0.25,             # paper (checkpoint: 0.0)
    "semblo_enabled": True,
    "semblo_fused_select_block_queries": 32,        # paper
    "semblo_kmeans_iters": 4,
    "semblo_kmeans_sample_size": 8192,
    "semblo_max_probe_buckets": 16384,              # paper (checkpoint: 1024)
    "semblo_num_centroids": 128,                    # paper (checkpoint: 64)
    "semblo_num_factors": 2,
    "semblo_selection_temperature": 0.1,            # paper (checkpoint: 0.05)
    "semblo_stop_after_layer": None,
    "semblo_top_centroids_per_factor": 128,         # paper (checkpoint: 64)
    "semblo_vllm_codebook_seed": 0,                 # paper
    "semblo_vllm_decode_num_splits": 8,
    "semblo_vllm_mode": "sparse",
    "semblo_vllm_own_prompt_cache": True,           # paper
    "semblo_vllm_pack_prompt_kv": True,
    "semblo_vllm_query_chunk_size": 8192,           # paper (checkpoint: 1024)
}

# The token capacity is max_model_len rounded down to a multiple of this, and
# the budget is the capacity over BUDGET_DIVISOR (the paper's K = L / 16).
CAPACITY_MULTIPLE = 16384
BUDGET_DIVISOR = 16

# The paper's settings for Nemotron 3.5 Lightning: keys centred per prompt, a
# longer k-means fit, and a budget of L / 8. Applied where unset when a
# Nemotron model is built.
NEMOTRON_ENV: dict[str, str] = {
    "SEMBLO_ROUTING_CENTER": "1",
    "SEMBLO_FLAT_FIT_ITERS": "32",
    "SEMBLO_FLAT_FIT_PER_C": "128",
}
NEMOTRON_BUDGET_DIVISOR = 8

_STOCK_MARKER_KEY = "semblo_runtime_weights_file"


def apply_env_defaults() -> tuple[str, ...]:
    """Set every unset ``PAPER_ENV`` variable and return the names set.

    An exported value is never replaced. The names set here, and those a
    parent process set that still hold their default value, are recorded in
    ``DEFAULTED_ENV_MARKER``.
    """

    inherited = defaulted_env_names()
    applied: list[str] = []
    for name, value in PAPER_ENV.items():
        if name not in os.environ:
            os.environ[name] = value
            applied.append(name)
    recorded = sorted(inherited | set(applied))
    if recorded:
        os.environ[DEFAULTED_ENV_MARKER] = ",".join(recorded)
    return tuple(applied)


def defaulted_env_names() -> frozenset[str]:
    """The ``PAPER_ENV`` variables whose current value was set as a default."""

    raw = os.environ.get(DEFAULTED_ENV_MARKER, "")
    names = {name for name in raw.split(",") if name}
    return frozenset(
        name for name in names
        if name in PAPER_ENV and os.environ.get(name) == PAPER_ENV[name]
    )


def _has_key(text: Any, name: str) -> bool:
    if isinstance(text, Mapping):
        return name in text
    return hasattr(text, name)


def _set_key(text: Any, name: str, value: Any) -> None:
    if isinstance(text, dict):
        text[name] = value
    else:
        setattr(text, name, value)


def yield_defaulted_env_to_config(config: Any) -> tuple[str, ...]:
    """Drop each defaulted variable whose config key the user set explicitly.

    Returns the dropped names. Only variables in ``ENV_SHADOWING_CONFIG`` whose
    value came from ``apply_env_defaults`` are affected. An exported variable
    still overrides the config key. Must run before the runtime config is
    parsed.
    """

    text = _text_config(config)
    defaulted = defaulted_env_names()
    dropped: list[str] = []
    for name, key in ENV_SHADOWING_CONFIG.items():
        if name in defaulted and _has_key(text, key):
            del os.environ[name]
            dropped.append(name)
            print(f"[semblo] {key} is set in the config; the default "
                  f"{name}={PAPER_ENV[name]} is dropped", flush=True)
    if dropped:
        remaining = sorted(defaulted - set(dropped))
        os.environ[DEFAULTED_ENV_MARKER] = ",".join(remaining)
    return tuple(dropped)


def is_stock_config(config: Any) -> bool:
    """True for a model not prepared by scripts/prepare_*_checkpoint.py (no runtime file named)."""

    return not _has_key(_text_config(config), _STOCK_MARKER_KEY)


def stock_token_capacity(max_model_len: int) -> int:
    """``max_model_len`` rounded down to a multiple of ``CAPACITY_MULTIPLE``."""

    capacity = (int(max_model_len) // CAPACITY_MULTIPLE) * CAPACITY_MULTIPLE
    if capacity <= 0:
        raise ValueError(
            f"max_model_len {max_model_len} is below {CAPACITY_MULTIPLE}, so "
            "the default token capacity would be zero; set "
            "semblo_vllm_token_capacity and semblo_target_tokens in the "
            "text config (hf_overrides)"
        )
    return capacity


def qwen3_5_attention_layers(text: Any) -> list[int]:
    """The full-attention layers of a Qwen3.5 text config (``layer_types``)."""

    layer_types = list(_value(text, "layer_types", None) or ())
    layers = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
    if not layers:
        raise ValueError("the text config lists no full_attention layer in layer_types")
    return layers


def nemotron_h_attention_layers(text: Any) -> list[int]:
    """The attention layers of a Nemotron-H config (``layers_block_type``)."""

    blocks = list(_value(text, "layers_block_type", None) or ())
    layers = [i for i, kind in enumerate(blocks) if kind in ("attention", "full_attention")]
    if not layers:
        raise ValueError("the config lists no attention layer in layers_block_type")
    return layers


def stock_text_config_values(
    config: Any,
    *,
    attention_layers: Sequence[int],
    max_model_len: int,
    budget_divisor: int = BUDGET_DIVISOR,
) -> dict[str, Any]:
    """The ``semblo_*`` keys a stock config lacks, with their paper values.

    Keys already present (e.g. from hf_overrides) are not returned. Derived keys:
    ``semblo_index_dim`` = the model's ``head_dim`` (identity routing),
    ``semblo_layer_indices`` = ``attention_layers``,
    ``semblo_vllm_token_capacity`` = ``stock_token_capacity(max_model_len)``,
    ``semblo_target_tokens`` = the capacity (explicit or derived) // ``budget_divisor`` (16; 8 for Nemotron).
    Does not modify ``config``.
    """

    text = _text_config(config)
    head_dim = _value(text, "head_dim", None)
    if head_dim is None:
        raise ValueError("the config has no head_dim; the routing width cannot be derived")
    wanted: dict[str, Any] = dict(PAPER_TEXT_CONFIG)
    wanted["semblo_index_dim"] = int(head_dim)
    wanted["semblo_layer_indices"] = [int(i) for i in attention_layers]
    if _has_key(text, "semblo_vllm_token_capacity"):
        capacity = int(_value(text, "semblo_vllm_token_capacity", 0))
    else:
        capacity = stock_token_capacity(max_model_len)
        wanted["semblo_vllm_token_capacity"] = capacity
    wanted["semblo_target_tokens"] = capacity // budget_divisor
    return {key: value for key, value in wanted.items() if not _has_key(text, key)}


def fill_stock_text_config(
    config: Any,
    *,
    attention_layers: Sequence[int],
    max_model_len: int,
    budget_divisor: int = BUDGET_DIVISOR,
) -> dict[str, Any]:
    """Write ``stock_text_config_values`` into the text config and return them."""

    values = stock_text_config_values(
        config, attention_layers=attention_layers, max_model_len=max_model_len, budget_divisor=budget_divisor)
    text = _text_config(config)
    for key, value in values.items():
        _set_key(text, key, value)
    return values


def runtime_config_for_model(
    hf_config: Any,
    *,
    max_model_len: int,
    attention_layers: Callable[[Any], list[int]],
    budget_divisor: int = BUDGET_DIVISOR,
    model_env: Mapping[str, str] | None = None,
) -> SembloVllmConfig:
    """The runtime config of a vLLM model, stock or prepared.

    Called by the model classes at construction. A prepared checkpoint (whose
    config names ``semblo_runtime_weights_file``) is parsed as it is and loads
    its routers from that file. A stock model gets the paper's settings for
    every key it lacks and identity routers built in memory. ``model_env``
    holds model-specific environment defaults (``NEMOTRON_ENV``), set where
    unset.
    """

    applied = [name for name in (model_env or {}) if name not in os.environ]
    for name in applied:
        os.environ[name] = model_env[name]
    if applied:
        print(f"[semblo] model defaults: {', '.join(f'{n}={os.environ[n]}' for n in applied)}", flush=True)
    yield_defaulted_env_to_config(hf_config)
    stock = is_stock_config(hf_config)
    if stock:
        filled = fill_stock_text_config(
            hf_config,
            attention_layers=attention_layers(_text_config(hf_config)),
            max_model_len=max_model_len,
            budget_divisor=budget_divisor,
        )
        text = _text_config(hf_config)
        print(
            "[semblo] stock model: the paper's settings filled in "
            f"({len(filled)} keys; token capacity "
            f"{int(_value(text, 'semblo_vllm_token_capacity', 0)):,}, "
            f"semblo_target_tokens {int(_value(text, 'semblo_target_tokens', 0)):,}, "
            f"layers {list(_value(text, 'semblo_layer_indices', []))})",
            flush=True,
        )
    return SembloVllmConfig.from_hf_config(hf_config, identity_router=stock)
