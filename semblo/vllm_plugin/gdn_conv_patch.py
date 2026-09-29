"""Work around an int32 offset overflow in vLLM's GDN prefill conv kernel.

``_causal_conv1d_fwd_kernel`` computes token offsets as int32
``token_index * stride_x_token``.  The GDN layer passes ``mixed_qkv`` as a
narrow slice of the in-projection output, so ``stride_x_token`` is the full
projection row (12288 for Qwen3.5-4B at TP=1, not the conv dim 8192) and the
product wraps for prompts longer than 2^31 / 12288 = 174,762 tokens.  The
wrapped offset points about 4 GB before the tensor.  Under
``expandable_segments`` that address is usually still mapped, so the kernel
silently reads and writes other tensors and the hidden states turn NaN.
Under the default allocator it faults.

This wrapper makes every launch int32-safe without changing the kernel:

  1. If the input is a strided slice, repack it contiguously, so the token
     stride drops to ``dim``.
  2. If ``tokens * dim`` still reaches 2^31 (262,144 tokens at TP=1), run
     the kernel over token segments and let the conv-state cache carry the
     (width-1)-token tail across calls.  This is bit-equivalent to one call
     because the conv is causal and local.

Segmentation supports single-sequence batches only, which is all that
``max_num_seqs=1`` produces at these lengths.

Mamba-2 (Nemotron-H) calls the same kernel from ``mamba_mixer2`` and passes
``hidden_states_B_C`` as a slice of its in-projection (10,304 wide for
Nemotron 3.5 Lightning, so the wrap is at 208,404 tokens).  Its call also
carries prefix-caching block indices.  A segmented call drops them, which is
correct only with prefix caching off, as this plugin requires.
"""

from __future__ import annotations

import torch

_INT32_ELEMS = 2**31
# Leave headroom below the boundary and keep segment starts aligned.
_ALIGN = 1024


def _token_stride(x: torch.Tensor) -> int:
    # Channel-last (dim, tokens): stride(0) == 1, stride(1) = row width.
    return max(int(x.stride(0)), int(x.stride(1)))


_PREFIX_CACHE_KWARGS = ("block_idx_first_scheduled_token", "block_idx_last_scheduled_token",
                        "initial_state_idx", "num_computed_tokens")


def install() -> None:
    import importlib
    from vllm.model_executor.layers.mamba import gdn_linear_attn as _gdn
    from vllm.model_executor.layers.mamba.ops import causal_conv1d as _ops

    original = _ops.causal_conv1d_fn
    if getattr(original, "_semblo_int32_safe", False):
        return
    # Every module that imported the symbol by name at load time.
    importers = [_gdn]
    for name in ("mamba_mixer2", "mamba_mixer", "short_conv"):
        try:
            importers.append(importlib.import_module(f"vllm.model_executor.layers.mamba.{name}"))
        except ImportError:
            pass

    def safe_causal_conv1d_fn(
        x,
        weight,
        bias,
        conv_states,
        query_start_loc,
        cache_indices=None,
        has_initial_state=None,
        activation="silu",
        pad_slot_id=_ops.PAD_SLOT_ID,
        metadata=None,
        validate_data=False,
        **kwargs,
    ):
        def call(x_, qsl, has_init, meta):
            return original(
                x_, weight, bias, conv_states, qsl,
                cache_indices=cache_indices,
                has_initial_state=has_init,
                activation=activation,
                pad_slot_id=pad_slot_id,
                metadata=meta,
                validate_data=validate_data,
                **kwargs,
            )

        dim, total = x.shape
        if total * _token_stride(x) < _INT32_ELEMS:
            return call(x, query_start_loc, has_initial_state, metadata)

        if _token_stride(x) > dim:
            # Repack the strided slice as contiguous channel-last.
            x = x.T.contiguous().T
        if total * dim < _INT32_ELEMS:
            # The precomputed metadata batch layout is still valid (same
            # sequences, same lengths); only the strides changed.
            return call(x, query_start_loc, has_initial_state, metadata)

        batch = int(query_start_loc.numel()) - 1
        if batch != 1:
            # Cannot happen under max_num_seqs=1 at these lengths.
            raise RuntimeError(
                "int32-safe causal_conv1d segmentation supports a single "
                f"sequence, got batch={batch} with {total} tokens"
            )
        seg = (_INT32_ELEMS // dim - _ALIGN) // _ALIGN * _ALIGN
        out = torch.empty(total, dim, dtype=x.dtype, device=x.device).T
        device = x.device
        # Mamba-2 passes the prefix-caching block indices of the whole
        # sequence, which do not describe a segment.  Prefix caching is off,
        # so a segmented call runs without them (the conv-state cache line
        # still carries the tail).
        if any(kwargs.get(k) is not None for k in _PREFIX_CACHE_KWARGS) or kwargs.get("block_size_to_align"):
            for k in _PREFIX_CACHE_KWARGS:
                kwargs[k] = None
            kwargs["block_size_to_align"] = 0
        for start in range(0, total, seg):
            end = min(start + seg, total)
            xi = x[:, start:end]
            qsl = torch.tensor([0, end - start], device=device,
                               dtype=query_start_loc.dtype)
            if start == 0:
                has_init = has_initial_state
            else:
                # The previous call left this sequence's conv-state cache
                # line holding the last (width-1) tokens of its segment.
                has_init = torch.ones(1, device=device, dtype=torch.bool)
            out[:, start:end] = call(xi, qsl, has_init, None)
        return out

    safe_causal_conv1d_fn._semblo_int32_safe = True
    _ops.causal_conv1d_fn = safe_causal_conv1d_fn
    for module in importers:
        if getattr(module, "causal_conv1d_fn", None) is original:
            module.causal_conv1d_fn = safe_causal_conv1d_fn
