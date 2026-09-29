"""SEMBLO_TARGET_TOKENS, when set, overrides the token budget of the config.

The budget is read from the environment so that changing it does not change
hf_overrides, which is part of the key of vLLM's compiled graph. Every budget
then runs on the same compiled model.
"""
import importlib
import os
import types

import pytest


@pytest.fixture
def config_module():
    mod = importlib.import_module("semblo.vllm_plugin.config")
    return importlib.reload(mod)


def _build(config_module, text, env=None):
    old = dict(os.environ)
    try:
        os.environ.pop("SEMBLO_TARGET_TOKENS", None)
        os.environ.update(env or {})
        return config_module.SembloVllmConfig.from_hf_config(
            types.SimpleNamespace(text_config=text))
    finally:
        os.environ.clear()
        os.environ.update(old)


# token_capacity has to cover the budget or the config refuses to build.
BASE = {"semblo_target_tokens": 4096,
        "semblo_vllm_token_capacity": 262144}


def test_the_config_is_used_when_the_environment_is_silent(config_module):
    assert _build(config_module, BASE).target_tokens == 4096


def test_the_environment_overrides_the_config(config_module):
    cfg = _build(config_module, BASE, {"SEMBLO_TARGET_TOKENS": "16384"})
    assert cfg.target_tokens == 16384


def test_the_environment_works_when_the_config_says_nothing(config_module):
    cfg = _build(config_module, {"semblo_vllm_token_capacity": 262144},
                 {"SEMBLO_TARGET_TOKENS": "8192"})
    assert cfg.target_tokens == 8192


def test_the_capacity_check_still_fires_on_an_environment_budget(config_module):
    # The capacity check runs after the environment override is applied.
    with pytest.raises(ValueError, match="token_capacity"):
        _build(config_module,
               {"semblo_vllm_token_capacity": 4096},
               {"SEMBLO_TARGET_TOKENS": "16384"})
