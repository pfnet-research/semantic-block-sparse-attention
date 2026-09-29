"""Decode can run on its own token budget.

Setting it to the prompt length makes decode read every cluster, i.e. dense
attention over the K/V that a sparse prefill produced.  This separates errors
in the prompt's representations from errors in decode-time selection.
"""

import pytest

from semblo.vllm_plugin.config import SembloVllmConfig
pytest.importorskip("vllm")
from semblo.vllm_plugin.model import SembloAttention


class _Cfg:
    """The shape from_hf_config expects: an object with a text_config dict."""

    def __init__(self, text):
        self.text_config = text


def _cfg(text):
    return _Cfg(text)


def _text(**kw):
    base = {
        "semblo_target_tokens": 8192,
        "semblo_flat_unit_tokens": 2048,
        "semblo_max_probe_buckets": 16384,
        "semblo_vllm_token_capacity": 524288,
    }
    base.update(kw)
    return base


class _Layer:
    """The three lines of SembloAttention.__init__ that decide the budget."""

    def __init__(self, runtime_config, layer_idx):
        self.layer_idx = layer_idx
        self.layer_target_tokens = runtime_config.layer_target_tokens.get(
            layer_idx, runtime_config.target_tokens
        )
        self.layer_decode_target_tokens = (
            self.layer_target_tokens
            if layer_idx in runtime_config.layer_target_tokens
            or not runtime_config.decode_target_tokens
            else runtime_config.decode_target_tokens
        )


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("SEMBLO_TARGET_TOKENS", "SEMBLO_DECODE_TARGET_TOKENS"):
        monkeypatch.delenv(k, raising=False)


def test_unset_means_decode_uses_the_prefill_budget(monkeypatch):
    monkeypatch.setenv("SEMBLO_TARGET_TOKENS", "16384")
    rc = SembloVllmConfig.from_hf_config(_cfg(_text()))
    assert rc.decode_target_tokens == 0
    layer = _Layer(rc, 3)
    assert layer.layer_target_tokens == 16384
    assert layer.layer_decode_target_tokens == 16384


def test_the_environment_sets_a_decode_only_budget(monkeypatch):
    monkeypatch.setenv("SEMBLO_TARGET_TOKENS", "32768")
    monkeypatch.setenv("SEMBLO_DECODE_TARGET_TOKENS", "524288")
    rc = SembloVllmConfig.from_hf_config(_cfg(_text()))
    layer = _Layer(rc, 3)
    assert layer.layer_target_tokens == 32768, "prefill keeps its budget"
    assert layer.layer_decode_target_tokens == 524288, "decode reads everything"


def test_a_per_layer_budget_still_wins(monkeypatch):
    """A per-layer budget takes precedence over the global decode budget."""
    monkeypatch.setenv("SEMBLO_DECODE_TARGET_TOKENS", "524288")
    rc = SembloVllmConfig.from_hf_config(
        _cfg(_text(semblo_target_tokens_per_layer={"3": 4096})))
    layer3, layer4 = _Layer(rc, 3), _Layer(rc, 4)
    assert layer3.layer_decode_target_tokens == 4096
    assert layer4.layer_decode_target_tokens == 524288


def test_the_text_config_can_carry_it_too(monkeypatch):
    rc = SembloVllmConfig.from_hf_config(
        _cfg(_text(semblo_decode_target_tokens=262144)))
    assert rc.decode_target_tokens == 262144


def test_a_prefill_after_a_decode_gets_the_prefill_budget_back(monkeypatch):
    """The budget is chosen per step, not carried over from the previous step.

    The layer object outlives the request, so a decode step must not leave
    its budget in place for a later prefill.  This calls the real method.
    """

    monkeypatch.setenv("SEMBLO_TARGET_TOKENS", "4096")
    monkeypatch.setenv("SEMBLO_DECODE_TARGET_TOKENS", "524288")
    rc = SembloVllmConfig.from_hf_config(_cfg(_text()))
    layer = _Layer(rc, 3)
    rule = SembloAttention._target_for_step

    assert rule(layer, False) == 4096, "the first prefill selects on its own budget"
    assert rule(layer, True) == 524288, "decode reads everything"
    assert rule(layer, False) == 4096, "and the NEXT prefill is back on 4,096"


def test_the_dense_tail_is_read_from_the_environment(monkeypatch):
    """The last N prefill positions read every cluster instead of the budget.

    In long-context tasks the question usually follows the document, so the
    last positions are the ones that know what is being asked.  Letting only
    them attend densely is cheap.
    """

    monkeypatch.setenv("SEMBLO_TARGET_TOKENS", "32768")
    rc = SembloVllmConfig.from_hf_config(_cfg(_text()))
    assert rc.dense_tail_tokens == 0, "off unless asked for"

    monkeypatch.setenv("SEMBLO_DENSE_TAIL_TOKENS", "2048")
    rc = SembloVllmConfig.from_hf_config(_cfg(_text()))
    assert rc.dense_tail_tokens == 2048
    assert rc.target_tokens == 32768, "the tail does not touch the budget"


def test_the_dense_tail_can_come_from_the_text_config(monkeypatch):
    rc = SembloVllmConfig.from_hf_config(
        _cfg(_text(semblo_dense_tail_tokens=4096)))
    assert rc.dense_tail_tokens == 4096
