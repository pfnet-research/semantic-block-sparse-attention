"""YaRN on vLLM's multimodal RoPE, checked against Transformers when it is used and corrected if vLLM is off.

vLLM 0.19.1's ``MRotaryEmbedding`` enlarges its cos/sin cache to four times the maximum position (for video
positions) and passes that enlarged value to its base class as ``max_position_embeddings``.  With YaRN,
``YaRNScalingRotaryEmbedding._compute_inv_freq`` then puts it into ``yarn_find_correction_range`` where the model's
``original_max_position_embeddings`` belongs.  For Qwen3.5 (rotary dim 64, theta 1e7, original 262,144, factor 4)
the ramp moves from dims 14-22 (Transformers' ``ROPE_INIT_FUNCTIONS["yarn"]``, and SGLang) to 16-24, and inv_freq
pairs 15-23 are interpolated too little.  The upstream fix is vllm-project/vllm#58879.

``install()`` runs when this package is imported, for the plugin and the evaluation entry points, dense and sparse
alike. It wraps ``MRotaryEmbedding._compute_inv_freq``. The first time vLLM computes YaRN frequencies for a
multimodal RoPE, the wrapper builds a reference case (Qwen3.5's YaRN through vLLM's own ``get_rope``) and compares
vLLM's inverse frequencies with Transformers':

* If they agree, vLLM's frequencies are used as they are.
* If they disagree, the correction is switched on and the case is compared again. The correction computes the
  correction range from the maximum position vLLM was given before enlarging it. The cache stays four times as
  long. If the frequencies still disagree, the run stops with an error.

The decision depends on what vLLM computes, not on its version number. A RoPE without YaRN never triggers the check.
"""
from __future__ import annotations

import contextlib
import threading
from typing import Callable

import torch

# The reference case: Qwen3.5's text RoPE beyond its native window (factor-4 YaRN from 262,144 positions).
_REFERENCE_ROPE = {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "rope_type": "yarn",
                   "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0,
                   "original_max_position_embeddings": 262144, "attn_factor": 0.8782488562869419}
_REFERENCE_HEAD_DIM = 256
_REFERENCE_MAX_POSITION = 525312
_RTOL = 1e-5

_original: Callable[..., torch.Tensor] | None = None   # vLLM's MRotaryEmbedding._compute_inv_freq
_correct: bool | None = None                             # None until the check has run
_checking = threading.local()


def _corrected_inv_freq(rope, scaling_factor: float) -> torch.Tensor:
    """vLLM's YaRN inverse frequencies with the correction range from the original maximum position."""
    from vllm.model_executor.layers.rotary_embedding.yarn_scaling_rope import YaRNScalingRotaryEmbedding

    enlarged = rope.max_position_embeddings
    rope.max_position_embeddings = rope.cache_max_position_num // 4   # the maximum position vLLM was given
    try:
        return YaRNScalingRotaryEmbedding._compute_inv_freq(rope, scaling_factor)
    finally:
        rope.max_position_embeddings = enlarged


def _reference_case() -> tuple[object, torch.Tensor]:
    """(vLLM's rope module for the reference case, Transformers' YaRN inverse frequencies for it)."""
    from transformers import Qwen3_5TextConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from vllm.config import VllmConfig, get_current_vllm_config_or_none, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT, get_rope

    cfg = Qwen3_5TextConfig(head_dim=_REFERENCE_HEAD_DIM, num_attention_heads=16, num_key_value_heads=4,
                            rope_parameters=dict(_REFERENCE_ROPE), max_position_embeddings=_REFERENCE_MAX_POSITION)
    reference, _ = ROPE_INIT_FUNCTIONS["yarn"](cfg, "cpu")
    cached = dict(_ROPE_DICT)   # get_rope caches modules by their arguments; keep the reference case out of it
    _ROPE_DICT.clear()
    try:
        # Inside vLLM's model construction a config is current and is used as it is; only a bare call sets one.
        context = (contextlib.nullcontext() if get_current_vllm_config_or_none() is not None
                   else set_current_vllm_config(VllmConfig()))
        with context:
            rope = get_rope(head_size=cfg.head_dim, max_position=_REFERENCE_MAX_POSITION,
                            rope_parameters=cfg.rope_parameters, dtype=torch.float32)
    finally:
        _ROPE_DICT.clear()
        _ROPE_DICT.update(cached)
    return rope, reference.float()


def _decide() -> bool:
    """Whether vLLM's YaRN frequencies need the correction; raises if the correction does not make them right."""
    _checking.active = True
    try:
        rope, reference = _reference_case()
        # vLLM builds its model under a CUDA default device; compare on the CPU, where the reference is.
        as_computed = _original(rope, rope.scaling_factor).float().cpu()
        if torch.allclose(as_computed, reference, rtol=_RTOL, atol=0):
            return False
        corrected = _corrected_inv_freq(rope, rope.scaling_factor).float().cpu()
        if not torch.allclose(corrected, reference, rtol=_RTOL, atol=0):
            raise RuntimeError(
                "vLLM's YaRN inverse frequencies for multimodal RoPE differ from Transformers', and correcting the "
                "correction range does not make them agree; refusing to run with a RoPE nobody checked "
                f"(max relative difference {((corrected - reference).abs() / reference).max().item():.3g})")
        return True
    finally:
        _checking.active = False


def correction_in_use() -> bool | None:
    """True or False once the check has run (the first YaRN multimodal RoPE), None before."""
    return _correct


def _compute_inv_freq(self, arg: float) -> torch.Tensor:
    """MRotaryEmbedding._compute_inv_freq: ``arg`` is the base without YaRN and the scaling factor with it (vLLM's
    YaRN cos/sin cache passes ``self.scaling_factor``)."""
    global _correct
    if getattr(self, "scaling_factor", None) is None or getattr(_checking, "active", False):
        return _original(self, arg)
    if _correct is None:
        _correct = _decide()
        if _correct:
            print("[semblo] vLLM's multimodal-RoPE YaRN differs from Transformers'; using the correction range from "
                  "the original maximum position", flush=True)
    return _corrected_inv_freq(self, arg) if _correct else _original(self, arg)


def install() -> None:
    """Wrap MRotaryEmbedding._compute_inv_freq once (a no-op without vLLM)."""
    global _original
    if _original is not None:
        return
    try:
        from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
    except ImportError:
        return
    _original = MRotaryEmbedding._compute_inv_freq
    MRotaryEmbedding._compute_inv_freq = _compute_inv_freq
