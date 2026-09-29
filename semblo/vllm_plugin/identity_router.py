"""The identity routing map, for a model without a routing map file.

The method routes on the pre-RoPE keys themselves.  The prepare scripts
(scripts/prepare_*_checkpoint.py) write this map to
``semblo_runtime.safetensors``; for a stock model it is built here in memory
and loaded through the same loader.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from semblo.vllm_plugin.config import SembloVllmConfig, _text_config, _value
from semblo.vllm_plugin.runtime_weights import load_runtime_tensors_into_module


def identity_router_tensors(
    cfg: Mapping[str, Any], dp: Mapping[str, Any], layers: Sequence[int]
) -> dict[str, torch.Tensor]:
    """The identity routing map of every layer in ``layers``.

    ``cfg`` holds the model's ``num_key_value_heads`` and ``head_dim``; ``dp``
    the ``semblo_*`` settings (``semblo_index_dim`` must equal ``head_dim``).
    Returns ``layers.<i>.self_attn.semblo_key_router.weight`` [Hkv, D, D],
    float32 on the CPU.
    """

    num_kv = int(cfg["num_key_value_heads"])
    head_dim = int(cfg["head_dim"]); index_dim = int(dp["semblo_index_dim"])
    if index_dim != head_dim:
        raise ValueError(f"identity routing needs index_dim == head_dim ({index_dim} != {head_dim})")
    weight = torch.eye(head_dim).expand(num_kv, head_dim, head_dim).contiguous()
    return {f"layers.{layer}.self_attn.semblo_key_router.weight": weight.clone() for layer in layers}


def load_identity_router_into_module(
    module: nn.Module,
    hf_config: Any,
    runtime_config: SembloVllmConfig,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> set[str]:
    """Build the identity routers of ``runtime_config``'s layers and load them.

    The tensors are loaded with ``require_exact_match=True``, so any mismatch
    between the routers the model instantiates and those built here is an
    error, as it is for a runtime weights file.
    """

    if not runtime_config.identity_router:
        raise ValueError(
            "the runtime config does not declare identity routers; its routers "
            "are read from the runtime weights file"
        )
    text = _text_config(hf_config)
    cfg = {
        name: _value(text, name, None)
        for name in ("num_attention_heads", "num_key_value_heads", "head_dim")
    }
    missing = sorted(name for name, value in cfg.items() if value is None)
    if missing:
        raise ValueError(f"the config lacks {', '.join(missing)}; identity routers cannot be shaped")
    dp = {"semblo_index_dim": runtime_config.index_dim}
    tensors = identity_router_tensors(cfg, dp, runtime_config.layer_indices)
    names = tuple(sorted(tensors))
    print(
        f"[semblo] identity routing built in memory ({len(names)} tensors for "
        f"layers {list(runtime_config.layer_indices)}); no runtime weights file "
        "is read",
        flush=True,
    )
    return load_runtime_tensors_into_module(
        module,
        tensors,
        names,
        require_exact_match=True,
        allow_unused_layer_parameters=False,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )
