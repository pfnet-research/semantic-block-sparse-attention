"""The paper's configuration as the plugin's defaults (paper_defaults.py).

The environment defaults and the stock-model settings are checked against
scripts/run_paper_cell.sh and
configs/checkpoints/qwen3_5_raw_keys_text_config.json.  The identity routers
built in memory are checked against the prepared checkpoint's runtime file.
CPU only.
"""

from __future__ import annotations

import json
import os
import pathlib
import re

import pytest
import torch
from torch import nn

from semblo.vllm_plugin import paper_defaults as pd
from semblo.vllm_plugin.config import SembloVllmConfig

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_paper_cell.sh"
RAW_KEYS_SETTINGS = ROOT / "configs/checkpoints/qwen3_5_raw_keys_text_config.json"
PREPARED = ROOT / "checkpoints/qwen3_5_4b/raw-keys"

# Variables of the script's sparse run that are not method defaults: data
# paths, and the budget, which a stock model derives from the context length.
_NOT_DEFAULTS = {"SEMBLO_HOTPOT_JSON", "SEMBLO_WORDLIST", "SEMBLO_RULER_CACHE_DIR", "SEMBLO_TARGET_TOKENS"}
# Text-config keys a stock model derives instead of copying.
_DERIVED = {"semblo_index_dim", "semblo_layer_indices", "semblo_vllm_token_capacity", "semblo_target_tokens"}
_FILE_KEYS = {"semblo_runtime_weights_file", "semblo_runtime_manifest_file"}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No PAPER_ENV variable, no marker and no budget override set."""

    monkeypatch.setattr(os, "environ", os.environ.copy())
    for name in [*pd.PAPER_ENV, pd.DEFAULTED_ENV_MARKER, "SEMBLO_TARGET_TOKENS",
                 "SEMBLO_FULL_GRAPH_DECODE", "SEMBLO_DECODE_TARGET_TOKENS", "SEMBLO_DENSE_TAIL_TOKENS",
                 "SEMBLO_RUNTIME_CHECKS"]:
        os.environ.pop(name, None)


def _function_body(text: str, name: str) -> str:
    start = text.index(f"{name} () {{")
    end = text.index("\n}\n", start)
    return text[start:end]


def _script_sparse_env() -> dict[str, str]:
    text = SCRIPT.read_text()
    env: dict[str, str] = {}
    for line in _function_body(text, "paper_env").splitlines():
        if line.strip().startswith("export "):
            for name, value in re.findall(r"\b(SEMBLO_[A-Z0-9_]+)=(\S+)", line):
                env[name] = value
    ruler = _function_body(text, "paper_ruler")
    ours = ruler[ruler.index('if [ "$arm" = ours ]'):ruler.index("python -m semblo.lm_eval_runner")]
    for name, value in re.findall(r"\b(SEMBLO_[A-Z0-9_]+)=(\S+)", ours):
        env[name] = value
    return env


def test_paper_env_is_the_scripts_sparse_run_environment():
    env = _script_sparse_env()
    assert _NOT_DEFAULTS <= set(env)
    assert pd.PAPER_ENV == {k: v for k, v in env.items() if k not in _NOT_DEFAULTS}
    # the quality configuration: sliced prefill, one-launch decode, no decode capture
    assert pd.PAPER_ENV["SEMBLO_PREFILL_CHUNK"] == "32768"
    assert pd.PAPER_ENV["SEMBLO_FUSED_DECODE"] == "1"
    assert pd.PAPER_ENV["SEMBLO_DECODE_GRAPH"] == "0"


def test_paper_text_config_is_the_prepared_settings_under_the_papers_overrides():
    text = SCRIPT.read_text()
    overrides: dict[str, object] = {}
    for var in ("PAPER_RULE", "PAPER_BASE"):
        fragment = re.search(rf"^{var}='(.*)'$", text, re.MULTILINE).group(1)
        overrides.update(json.loads("{" + fragment + "}"))
    ruler = _function_body(text, "paper_ruler")
    hf = re.search(r'"hf_overrides=(.*)"', ruler[ruler.index('if [ "$arm" = ours ]'):]).group(1)
    literal = dict(re.findall(r'\\"(semblo_[a-z0-9_]+)\\":(true|false|[0-9.]+)', hf))
    overrides.update({k: json.loads(v) for k, v in literal.items()})
    # the clustering seed is a per-run argument in the script; the default is the first
    assert '\\"semblo_vllm_codebook_seed\\":$seed' in hf
    overrides["semblo_vllm_codebook_seed"] = 0
    overrides.pop("semblo_vllm_token_capacity", None)   # derived (checked below)
    settings = json.loads(RAW_KEYS_SETTINGS.read_text())["text_config"]
    expected = {k: v for k, v in settings.items() if k not in _DERIVED | _FILE_KEYS}
    expected.update(overrides)
    assert pd.PAPER_TEXT_CONFIG == expected
    assert not set(pd.PAPER_TEXT_CONFIG) & (_DERIVED | _FILE_KEYS)


def test_apply_env_defaults_never_overrides_an_explicit_value(clean_env):
    os.environ["SEMBLO_PREFILL_CHUNK"] = "0"
    os.environ["SEMBLO_FLAT_UNIT_TOKENS"] = "1024"
    applied = pd.apply_env_defaults()
    assert "SEMBLO_PREFILL_CHUNK" not in applied and "SEMBLO_FLAT_UNIT_TOKENS" not in applied
    assert os.environ["SEMBLO_PREFILL_CHUNK"] == "0"
    assert os.environ["SEMBLO_FLAT_UNIT_TOKENS"] == "1024"
    for name, value in pd.PAPER_ENV.items():
        if name not in ("SEMBLO_PREFILL_CHUNK", "SEMBLO_FLAT_UNIT_TOKENS"):
            assert os.environ[name] == value and name in applied
    assert pd.defaulted_env_names() == frozenset(applied)
    # a second call (the engine process inherits the values) keeps them defaults
    assert pd.apply_env_defaults() == ()
    assert pd.defaulted_env_names() == frozenset(applied)


def test_a_defaulted_variable_gives_way_to_an_explicit_config_key(clean_env):
    os.environ["SEMBLO_FA3_ALIGN"] = "16"           # exported: still wins over the config
    pd.apply_env_defaults()
    text = {"semblo_flat_unit_tokens": 512, "semblo_fa3_page_align": 32}
    assert pd.yield_defaulted_env_to_config({"text_config": text}) == ("SEMBLO_FLAT_UNIT_TOKENS",)
    assert "SEMBLO_FLAT_UNIT_TOKENS" not in os.environ
    assert os.environ["SEMBLO_FA3_ALIGN"] == "16"
    assert os.environ["SEMBLO_PREFILL_CHUNK"] == "32768"       # no config key: the default stays
    assert "SEMBLO_FLAT_UNIT_TOKENS" not in pd.defaulted_env_names()


@pytest.mark.parametrize("max_model_len,capacity", [
    (525312, 524288), (1049600, 1048576), (132096, 131072), (262144, 262144), (16384, 16384),
])
def test_capacity_is_max_model_len_rounded_down(max_model_len, capacity):
    assert pd.stock_token_capacity(max_model_len) == capacity


def test_capacity_below_one_multiple_is_refused():
    with pytest.raises(ValueError, match="token capacity would be zero"):
        pd.stock_token_capacity(16383)


def _qwen3_5_4b_like() -> dict:
    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 8
    return {
        "architectures": ["SembloQwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "head_dim": 256, "num_attention_heads": 16, "num_key_value_heads": 4,
            "num_hidden_layers": 32, "layer_types": layer_types,
        },
    }


def test_a_stock_config_is_filled_with_the_paper_settings(clean_env):
    config = _qwen3_5_4b_like()
    assert pd.is_stock_config(config)
    filled = pd.fill_stock_text_config(
        config, attention_layers=pd.qwen3_5_attention_layers(config["text_config"]), max_model_len=525312)
    expected = dict(pd.PAPER_TEXT_CONFIG)
    expected.update(semblo_index_dim=256, semblo_layer_indices=[3, 7, 11, 15, 19, 23, 27, 31],
                    semblo_vllm_token_capacity=524288, semblo_target_tokens=32768)
    assert filled == expected
    assert {k: v for k, v in config["text_config"].items() if k.startswith("semblo_")} == expected
    # the layers are the prepared checkpoint's
    settings = json.loads(RAW_KEYS_SETTINGS.read_text())["text_config"]
    assert filled["semblo_layer_indices"] == settings["semblo_layer_indices"]


def test_an_explicit_key_is_kept_and_the_budget_follows_an_explicit_capacity(clean_env):
    config = _qwen3_5_4b_like()
    config["text_config"].update(semblo_selection_temperature=0.2, semblo_vllm_token_capacity=262144)
    values = pd.stock_text_config_values(
        config, attention_layers=pd.qwen3_5_attention_layers(config["text_config"]), max_model_len=1049600)
    assert "semblo_selection_temperature" not in values
    assert "semblo_vllm_token_capacity" not in values
    assert values["semblo_target_tokens"] == 262144 // 16
    config["text_config"]["semblo_target_tokens"] = 4096
    assert "semblo_target_tokens" not in pd.stock_text_config_values(
        config, attention_layers=[3], max_model_len=1049600)


def test_the_stock_runtime_config(clean_env):
    pd.apply_env_defaults()
    runtime = pd.runtime_config_for_model(
        _qwen3_5_4b_like(), max_model_len=525312, attention_layers=pd.qwen3_5_attention_layers)
    assert runtime.identity_router and runtime.mode == "sparse"
    assert runtime.runtime_weights_file is None and runtime.runtime_manifest_file is None
    assert runtime.token_capacity == 524288 and runtime.target_tokens == 32768
    assert runtime.layer_indices == (3, 7, 11, 15, 19, 23, 27, 31) and runtime.index_dim == 256
    assert (runtime.selection_temperature, runtime.bucket_size_penalty) == (0.1, 0.25)
    assert (runtime.num_centroids, runtime.top_centroids_per_factor, runtime.max_probe_buckets) == (128, 128, 16384)
    assert (runtime.fused_select_block_queries, runtime.query_chunk_size) == (32, 8192)
    assert runtime.own_prompt_cache and runtime.pack_prompt_kv and runtime.codebook_seed == 0
    assert runtime.prefill_chunk == 32768 and runtime.engage_above_tokens == 0


def test_the_budget_from_the_environment_still_wins(clean_env):
    os.environ["SEMBLO_TARGET_TOKENS"] = "8192"
    runtime = pd.runtime_config_for_model(
        _qwen3_5_4b_like(), max_model_len=525312, attention_layers=pd.qwen3_5_attention_layers)
    assert runtime.target_tokens == 8192 and runtime.token_capacity == 524288


def test_nemotron_layers_and_a_flat_config():
    config = {"layers_block_type": ["mamba", "moe", "mamba", "attention", "moe", "full_attention"],
              "head_dim": 128, "num_attention_heads": 32, "num_key_value_heads": 2}
    assert pd.nemotron_h_attention_layers(config) == [3, 5]
    values = pd.stock_text_config_values(config, attention_layers=[3, 5], max_model_len=132096)
    assert values["semblo_index_dim"] == 128 and values["semblo_layer_indices"] == [3, 5]
    assert values["semblo_vllm_token_capacity"] == 131072 and values["semblo_target_tokens"] == 8192


def _prepared_like() -> dict:
    config = _qwen3_5_4b_like()
    settings = json.loads(RAW_KEYS_SETTINGS.read_text())["text_config"]
    config["text_config"].update(settings)
    return config


def test_a_prepared_config_is_left_untouched(clean_env):
    config = _prepared_like()
    before = json.loads(json.dumps(config))
    assert not pd.is_stock_config(config)
    runtime = pd.runtime_config_for_model(config, max_model_len=525312, attention_layers=pd.qwen3_5_attention_layers)
    assert config == before
    assert not runtime.identity_router
    assert runtime == SembloVllmConfig.from_hf_config(before)
    assert runtime.runtime_weights_file == "semblo_runtime.safetensors"


def test_identity_is_refused_for_a_config_that_names_a_runtime_file():
    with pytest.raises(ValueError, match="identity routers are built in memory only"):
        SembloVllmConfig.from_hf_config(_prepared_like(), identity_router=True)


def test_a_sparse_config_without_a_runtime_file_still_needs_one(clean_env):
    config = _prepared_like()
    for key in _FILE_KEYS:
        del config["text_config"][key]
    with pytest.raises(ValueError, match="sparse mode requires semblo_runtime.safetensors"):
        SembloVllmConfig.from_hf_config(config)


def test_a_named_runtime_file_that_is_missing_still_raises(tmp_path):
    from semblo.vllm_plugin.runtime_weights import load_runtime_weights_into_module

    with pytest.raises(FileNotFoundError, match="runtime weights manifest is missing"):
        load_runtime_weights_into_module(nn.Module(), tmp_path)


def _router_host(runtime: SembloVllmConfig, text: dict) -> nn.Module:
    """A module with the router parameters under the vLLM model's names."""

    from semblo.vllm_plugin.routing_map import SembloKeyRouter

    layers = nn.ModuleDict()
    for layer in runtime.layer_indices:
        attn = nn.Module()
        attn.semblo_key_router = SembloKeyRouter(
            num_kv_heads=text["num_key_value_heads"], head_dim=text["head_dim"],
            index_dim=runtime.index_dim, num_factors=runtime.num_factors)
        with torch.no_grad():
            for parameter in attn.parameters():
                parameter.fill_(float("nan"))
        block = nn.Module()
        block.self_attn = attn
        layers[str(layer)] = block
    host = nn.Module()
    host.model = nn.Module()
    host.model.layers = layers
    return host


def _prepared_tensors() -> dict[str, torch.Tensor]:
    weights = PREPARED / "semblo_runtime.safetensors"
    if not weights.is_file():
        pytest.skip(f"needs the prepared checkpoint {PREPARED}")
    from safetensors.torch import load_file

    return load_file(str(weights))


def test_identity_tensors_are_the_prepared_checkpoints():
    from semblo.vllm_plugin.identity_router import identity_router_tensors

    recorded = _prepared_tensors()
    config = json.loads((PREPARED / "config.json").read_text())
    text = config["text_config"]
    dp = {k: v for k, v in text.items() if k.startswith("semblo_")}
    built = identity_router_tensors(text, dp, text["semblo_layer_indices"])
    # The prepared file also holds a query router with a zero residual and
    # mixture, which the loader drops.
    recorded = {n: t for n, t in recorded.items() if ".semblo_query_router." not in n}
    assert sorted(built) == sorted(recorded)
    for name in sorted(built):
        assert built[name].dtype == recorded[name].dtype, name
        assert built[name].shape == recorded[name].shape, name
        assert torch.equal(built[name], recorded[name]), name


def test_the_in_memory_routers_load_as_the_file_does(clean_env):
    """A stock 4B-shaped config loads, through the shared loader, exactly the
    parameters the prepared checkpoint's file loads."""

    from semblo.vllm_plugin.identity_router import load_identity_router_into_module
    from semblo.vllm_plugin.runtime_weights import load_runtime_weights_into_module

    _prepared_tensors()
    config = _qwen3_5_4b_like()
    runtime = pd.runtime_config_for_model(config, max_model_len=525312, attention_layers=pd.qwen3_5_attention_layers)
    text = config["text_config"]
    stock, prepared = _router_host(runtime, text), _router_host(runtime, text)
    loaded_stock = load_identity_router_into_module(stock, config, runtime)
    loaded_file = load_runtime_weights_into_module(prepared, PREPARED)
    assert loaded_stock == loaded_file and len(loaded_stock) == 8
    stock_state, file_state = stock.state_dict(), prepared.state_dict()
    assert sorted(stock_state) == sorted(file_state)
    for name in stock_state:
        assert torch.equal(stock_state[name], file_state[name]), name


def test_in_memory_routers_are_refused_for_a_prepared_config(clean_env):
    from semblo.vllm_plugin.identity_router import load_identity_router_into_module

    runtime = SembloVllmConfig.from_hf_config(_prepared_like())
    with pytest.raises(ValueError, match="does not declare identity routers"):
        load_identity_router_into_module(nn.Module(), _prepared_like(), runtime)


def test_nemotron_budget_is_an_eighth_of_the_capacity():
    from semblo.vllm_plugin.paper_defaults import NEMOTRON_BUDGET_DIVISOR, stock_text_config_values

    text = {"head_dim": 128, "layers_block_type": ["mamba", "attention", "mamba"]}
    values = stock_text_config_values(text, attention_layers=[1], max_model_len=525312,
                                      budget_divisor=NEMOTRON_BUDGET_DIVISOR)
    assert values["semblo_vllm_token_capacity"] == 524288
    assert values["semblo_target_tokens"] == 65536


def test_model_env_defaults_only_fill_unset_names(monkeypatch):
    import semblo.vllm_plugin.paper_defaults as pd

    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("SEMBLO_ROUTING_CENTER", raising=False)
    monkeypatch.setenv("SEMBLO_FLAT_FIT_ITERS", "8")
    monkeypatch.delenv("SEMBLO_FLAT_FIT_PER_C", raising=False)
    monkeypatch.setattr(pd, "yield_defaulted_env_to_config", lambda config: ())
    monkeypatch.setattr(pd, "is_stock_config", lambda config: False)
    monkeypatch.setattr(pd.SembloVllmConfig, "from_hf_config", classmethod(lambda cls, config, identity_router=False: "cfg"))
    assert pd.runtime_config_for_model({}, max_model_len=525312, attention_layers=lambda t: [1],
                                       budget_divisor=pd.NEMOTRON_BUDGET_DIVISOR, model_env=pd.NEMOTRON_ENV) == "cfg"
    assert os.environ["SEMBLO_ROUTING_CENTER"] == "1"
    assert os.environ["SEMBLO_FLAT_FIT_ITERS"] == "8"          # exported: kept
    assert os.environ["SEMBLO_FLAT_FIT_PER_C"] == "128"
