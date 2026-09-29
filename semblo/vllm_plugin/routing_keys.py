"""Which query and key the router reads: pre-RoPE (the method) or post-RoPE (an ablation).

Every model's projection, ``_project_qkv``, returns six tensors,
``(q_post, k_post, v, q_pre, k_pre, gate)``, and the streamed prefill's K/V-only
projection, ``_project_kv``, returns ``(k_post, v, k_pre)``.  The attention reads
the "post" slots.  Everything that routes reads the "pre" slots: the key store
the index is fitted from, the query state (prefill and decode), the decode
step's index append and the suffix flush's key store.

``SEMBLO_ROUTING_KEYS=post`` makes the router read the rotated tensors
instead.  The switch is one function, ``routing_view``, which each projection
applies to its "pre" slots just before returning them, so the attention's
inputs are never touched.  Because each model defines its own projections,
``check_routing_view`` verifies on the eager paths (the streamed prefill's two
passes, the attention operator) that the routing tensor a layer returned is
the rotated one, raises if not, and prints one line per path.
"""

from __future__ import annotations

import os

import torch


def _read_switch() -> bool:
    """``SEMBLO_ROUTING_KEYS``: ``pre`` (default) or ``post``.  Any other value raises."""

    value = os.environ.get("SEMBLO_ROUTING_KEYS", "") or "pre"
    if value not in ("pre", "post"):
        raise ValueError(f"SEMBLO_ROUTING_KEYS={value!r}: expected 'pre' or 'post'")
    return value == "post"


ROUTE_POST_ROPE: bool = _read_switch()

_SAID: set[str] = set()
_CHECKED: set[tuple[str, int]] = set()


def routing_view(
    pre: torch.Tensor | None, post: torch.Tensor | None
) -> torch.Tensor | None:
    """The tensor the router reads in place of ``pre``.

    pre   [T, H, D]            the normed, unrotated q or k (None for a layer
                               without keys of its own)
    post  [T, H * D] or [T, H, D]  the same tensor after the rotary embedding;
                               the attention's input, which the caller returns
                               unchanged

    Returns ``pre`` itself, or under ``SEMBLO_ROUTING_KEYS=post`` ``post`` in
    ``pre``'s shape [T, H, D].  The latter is a view whenever ``post`` is
    contiguous (it is for every rotary module here), so it costs nothing and
    the unrotated tensor can be freed.  Nothing downstream writes into a
    routing tensor.  This runs inside the traced projection, so it does no
    host work beyond reading one global, which the compiler guards on.
    """

    if not ROUTE_POST_ROPE or pre is None:
        return pre
    if post is None:
        raise RuntimeError(
            "SEMBLO_ROUTING_KEYS=post routes on the rotated tensor, and the projection produced none"
        )
    return post.reshape(pre.shape)


def check_routing_view(
    model: str,
    layer_idx: int,
    path: str,
    *pairs: tuple[torch.Tensor | None, torch.Tensor | None],
) -> None:
    """Under ``SEMBLO_ROUTING_KEYS=post``: is the routing tensor the rotated one?

    ``pairs`` are ``(routing tensor as returned, rotated tensor)``, e.g.
    ``(q_pre, q_post), (k_pre, k_post)``.  A pair whose routing tensor is None
    is skipped.  Once per (path, layer) the tensors are compared exactly, and a
    mismatch (a projection that ignored the switch) raises.  Once per path and
    process one line is printed.  Does nothing when the switch is off.

    ``torch.equal`` reads a device value, so this belongs on eager paths only.
    Inside a CUDA graph capture the comparison is skipped (the printed line
    says so) and left for the next eager call.
    """

    if not ROUTE_POST_ROPE:
        return
    pairs = tuple((r, p) for r, p in pairs if r is not None)
    if not pairs:
        return
    key = (path, int(layer_idx))
    capturing = any(
        r.is_cuda for r, _ in pairs
    ) and torch.cuda.is_current_stream_capturing()
    if key not in _CHECKED and not capturing:
        for routed, rotated in pairs:
            if rotated is None or not torch.equal(routed.reshape(rotated.shape), rotated):
                raise RuntimeError(
                    f"SEMBLO_ROUTING_KEYS=post, but {model} layer {layer_idx} returned a "
                    f"routing tensor on the {path} that is not the rotated one: its "
                    "projection does not pass its pre-RoPE slots through routing_view"
                )
        _CHECKED.add(key)
    if path not in _SAID:
        _SAID.add(path)
        how = (
            "not checked: inside a CUDA graph capture" if capturing
            else "checked: the routing tensors equal the rotated ones"
        )
        print(
            f"[semblo] routing keys: post-RoPE (SEMBLO_ROUTING_KEYS=post) on {model} "
            f"layer {layer_idx}, {path} ({how})",
            flush=True,
        )
