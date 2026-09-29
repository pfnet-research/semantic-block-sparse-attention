"""The sparse attention layer as an operator that torch.compile does not trace into.

The layer's forward pass asks whether vLLM is serving or profiling the KV
cache, and takes a dense path when profiling.  The answer comes from the
attention metadata, which is absent during the warm-up pass that
torch.compile records, so a traced layer would compile the dense branch and
replay it for every later request.  Behind an operator the branch is taken
per call, as in eager mode, while the rest of the model still compiles.

The layer is reached through a table keyed by vLLM's unique per-layer name,
the way vLLM reaches its own attention layers, because an operator's
arguments must be tensors and plain values.  The operator is registered once
per process, since a second registration of the same schema is an error.

The operator writes into a buffer allocated by its caller instead of
returning a fresh tensor.  Piecewise CUDA graph capture requires this: a
captured graph replays the device addresses it was captured with, and vLLM's
``CUDAGraphWrapper`` copies nothing on replay, so the downstream piece reads
the result at its capture-time address.  A buffer allocated by the traced
caller lives in the captured region and keeps one address across replays.
vLLM's ``unified_attention_with_output`` does the same.

The operator takes projected tensors, not hidden states.  Only the attention
core depends on the per-request index.  Keeping the QKV projection, norms,
rotary embedding and output projection in the layer's traced ``forward``
leaves them in the compiled, captured region, which matters because decode is
bound by the number of operations the host issues.
"""

from __future__ import annotations

from typing import Any

import torch


_OP_NAMESPACE = "semblo"
_OP_NAME = "attention"

# The name vLLM's compiler uses to split a graph at this operator.  It keeps
# the operator outside the captured regions, since the layer's index is built
# per request, while everything around it is captured.
SPLITTING_OP_NAME = f"{_OP_NAMESPACE}::{_OP_NAME}"

_LAYERS: dict[str, Any] = {}


def register_layer(layer_name: str, layer: Any) -> None:
    """Make ``layer`` reachable from the operator under ``layer_name``."""

    _LAYERS[str(layer_name)] = layer


def registered_layer(layer_name: str) -> Any:
    layer = _LAYERS.get(str(layer_name))
    if layer is None:
        raise RuntimeError(
            f"no semblo layer is registered as {layer_name!r}; the "
            "operator is reachable only through a layer that registered itself"
        )
    return layer


def _attention(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    q_post: torch.Tensor,
    k_post: torch.Tensor | None,
    v_states: torch.Tensor | None,
    q_pre: torch.Tensor,
    k_pre: torch.Tensor | None,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    registered_layer(layer_name).attention_forward(
        positions, hidden_states, q_post, k_post, v_states, q_pre, k_pre, output
    )


def _attention_fake(
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    q_post: torch.Tensor,
    k_post: torch.Tensor | None,
    v_states: torch.Tensor | None,
    q_pre: torch.Tensor,
    k_pre: torch.Tensor | None,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


def _register() -> None:
    library = torch.library.Library(_OP_NAMESPACE, "FRAGMENT")
    # ``Tensor(a!) output`` declares the mutation, so the compiler keeps the
    # write ordered against the downstream reads of the same buffer.
    library.define(
        f"{_OP_NAME}(Tensor positions, Tensor hidden_states, "
        "Tensor q_post, Tensor? k_post, Tensor? v_states, "
        "Tensor q_pre, Tensor? k_pre, "
        "Tensor(a!) output, str layer_name) -> ()"
    )
    # Named backends rather than a composite key: the operator has to stay
    # opaque, and a composite registration would also claim the meta kernel
    # that the fake implementation below provides.
    library.impl(_OP_NAME, _attention, "CPU")
    library.impl(_OP_NAME, _attention, "CUDA")
    library._register_fake(_OP_NAME, _attention_fake)
    # The registration lives as long as this object does.
    global _LIBRARY
    _LIBRARY = library


_LIBRARY: torch.library.Library | None = None
_register()

# Bound once rather than resolved through ``torch.ops`` on every call, to save
# host time on the decode path.
sparse_attention_op = getattr(torch.ops, _OP_NAMESPACE).attention
"""Call the operator; the layer's ``forward`` goes through here."""

