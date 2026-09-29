"""vLLM's YaRN on multimodal RoPE is checked against Transformers when used, and corrected only where it is off."""
from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
vllm = pytest.importorskip("vllm")

from transformers import Qwen3_5TextConfig  # noqa: E402
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS  # noqa: E402

import semblo  # noqa: E402,F401  (installs the check)
from semblo import mrope_yarn_fix  # noqa: E402

YARN = {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000,
        "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144,
        "attn_factor": 0.8782488562869419}


def _config(rope_parameters: dict) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(head_dim=256, num_attention_heads=16, num_key_value_heads=4,
                             rope_parameters=dict(rope_parameters), max_position_embeddings=525312)


def _rope(cfg):
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope
    _ROPE_DICT.clear()   # get_rope caches modules by their arguments
    with set_current_vllm_config(VllmConfig()):
        return get_rope(head_size=cfg.head_dim, max_position=525312, rope_parameters=cfg.rope_parameters,
                        dtype=torch.float32)


@pytest.fixture
def fresh(monkeypatch):
    """The check not yet run."""
    monkeypatch.setattr(mrope_yarn_fix, "_correct", None)
    yield


def test_the_installed_vllm_is_corrected_to_transformers(fresh):
    cfg = _config(YARN)
    reference, _ = ROPE_INIT_FUNCTIONS["yarn"](cfg, "cpu")
    rope = _rope(cfg)
    got = rope._compute_inv_freq(rope.scaling_factor).float()
    torch.testing.assert_close(got, reference.float(), rtol=1e-5, atol=0)
    raw = mrope_yarn_fix._original(rope, rope.scaling_factor).float()
    assert not torch.allclose(raw, reference.float(), rtol=1e-4), "this vLLM already agrees; the test expects 0.19.1"
    assert mrope_yarn_fix.correction_in_use() is True


def test_a_vllm_that_agrees_is_left_alone(fresh, monkeypatch):
    # a vLLM whose own function already computes the corrected frequencies
    monkeypatch.setattr(mrope_yarn_fix, "_original", mrope_yarn_fix._corrected_inv_freq)
    called = []
    monkeypatch.setattr(mrope_yarn_fix, "_corrected_inv_freq", lambda *a: called.append(1))
    cfg = _config(YARN)
    reference, _ = ROPE_INIT_FUNCTIONS["yarn"](cfg, "cpu")
    rope = _rope(cfg)
    torch.testing.assert_close(rope._compute_inv_freq(rope.scaling_factor).float(), reference.float(), rtol=1e-5, atol=0)
    assert mrope_yarn_fix.correction_in_use() is False and not called


def test_a_disagreement_the_correction_does_not_explain_stops_the_run(fresh, monkeypatch):
    monkeypatch.setattr(mrope_yarn_fix, "_corrected_inv_freq", lambda rope, arg: torch.ones(32))
    with pytest.raises(RuntimeError, match="does not make them agree"):
        _rope(_config(YARN))


def test_a_rope_without_yarn_does_not_run_the_check(fresh):
    plain = {k: v for k, v in YARN.items() if k in ("mrope_interleaved", "mrope_section", "rope_theta", "partial_rotary_factor")}
    plain["rope_type"] = "default"
    _rope(_config(plain))
    assert mrope_yarn_fix.correction_in_use() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_the_check_runs_under_a_cuda_default_device(fresh):
    # vLLM builds its model under a CUDA default device, while the reference is computed on the CPU.
    cfg = _config(YARN)
    reference, _ = ROPE_INIT_FUNCTIONS["yarn"](cfg, "cpu")
    with torch.device("cuda"):
        rope = _rope(cfg)
        got = rope._compute_inv_freq(rope.scaling_factor).float().cpu()
    torch.testing.assert_close(got, reference.float(), rtol=1e-5, atol=0)
    assert mrope_yarn_fix.correction_in_use() is True
