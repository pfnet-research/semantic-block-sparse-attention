from __future__ import annotations

"""vLLM general-plugin entry point.

The model module is registered lazily so importing ``semblo`` does not
require vLLM for offline analysis.
"""

# custom architecture -> (module:class, official architecture whose
# MODELS_CONFIG_MAP entry it inherits, whether that entry is required)
_REGISTRATIONS = {
    "SembloQwen3_5ForConditionalGeneration": (
        "semblo.vllm_plugin.qwen3_5_model:"
        "SembloQwen3_5ForConditionalGeneration",
        "Qwen3_5ForConditionalGeneration",
        False,  # uniform head dims; the hook is inherited when present
    ),
    "SembloQwen3_5MoeForConditionalGeneration": (
        "semblo.vllm_plugin.qwen3_5_model:"
        "SembloQwen3_5MoeForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        False,  # the same attention as the dense Qwen3.5; the MoE is in the MLP
    ),
    "SembloNemotronHForCausalLM": (
        "semblo.vllm_plugin.nemotron_h_model:SembloNemotronHForCausalLM",
        "NemotronHForCausalLM",
        False,  # head_dim 128 everywhere; no backend policy to inherit
    ),
}


def register() -> None:
    # Must run before any semblo module reads the environment: fills in the
    # paper's settings for every variable left unset (paper_defaults.py).
    from semblo.vllm_plugin.paper_defaults import apply_env_defaults

    apply_env_defaults()
    from vllm import ModelRegistry

    for custom, (target, _official, _required) in _REGISTRATIONS.items():
        ModelRegistry.register_model(custom, target)
    register_model_config_hook()
    # Installed unconditionally, so dense models in this process benefit too.
    # vLLM's GDN prefill conv kernel overflows int32 token offsets on long
    # strided inputs (see gdn_conv_patch).
    try:
        from semblo.vllm_plugin.gdn_conv_patch import install

        install()
    except ImportError:
        # A vLLM build without the GDN module has nothing to patch.
        pass
    # Host hook around the model runner's step for suffix_flush.py. It is a
    # no-op unless SEMBLO_SUFFIX_FLUSH_EVERY is set.
    from semblo.vllm_plugin import suffix_flush

    suffix_flush.install_hook()


def register_model_config_hook() -> None:
    """Give each custom architecture the ``MODELS_CONFIG_MAP`` entry of the
    official architecture it wraps.

    The map is keyed by architecture name, so a renamed architecture would
    otherwise miss vLLM's per-model config hook.
    """

    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    for custom, (_target, official_name, required) in _REGISTRATIONS.items():
        official = MODELS_CONFIG_MAP.get(official_name)
        if official is None:
            if required:
                raise RuntimeError(
                    "this vLLM build has no MODELS_CONFIG_MAP entry for "
                    f"{official_name}; the semblo architecture cannot "
                    "inherit its attention-backend policy"
                )
            continue
        MODELS_CONFIG_MAP[custom] = official
