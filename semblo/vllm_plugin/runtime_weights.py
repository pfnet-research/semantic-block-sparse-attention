from __future__ import annotations

"""Load the routing map of a prepared model directory.

vLLM loads the base model's weights from the directory as usual; the routing
map of every full-attention layer (routing_map.py) is read from a separate
``semblo_runtime.safetensors``, whose manifest records its shapes and hashes.
"""

from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn
from safetensors.torch import load_file

LOGGER = logging.getLogger(__name__)


RUNTIME_WEIGHTS_FILE = "semblo_runtime.safetensors"
RUNTIME_MANIFEST_FILE = "semblo_runtime.json"
RUNTIME_SCHEMA_VERSION = 1

_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)$")
_ATTENTION_BLOCK_NAMES = ("self_attn", "mixer")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_runtime_name(name: str) -> str | None:
    """Return a prefix-independent name for a semblo parameter.

    Examples from the two model implementations are normalized to the same
    value::

        model.language_model.layers.4.self_attn.semblo_key_router.weight
        model.layers.4.self_attn.semblo_key_router.weight

    Both become::

        layers.4.self_attn.semblo_key_router.weight
    """

    # The attention block is ``self_attn`` in Qwen3.5 and ``mixer`` in
    # Nemotron-H.  The canonical name uses ``self_attn`` for both.
    for block in _ATTENTION_BLOCK_NAMES:
        marker = f".{block}.semblo_"
        marker_index = name.find(marker)
        if marker_index < 0:
            continue
        prefix = name[:marker_index]
        match = _LAYER_PATTERN.search(prefix)
        if match is None:
            return None
        suffix = name[marker_index + len(f".{block}.") :]
        return f"layers.{int(match.group(1))}.self_attn.{suffix}"
    return None


def is_runtime_parameter(name: str) -> bool:
    return canonical_runtime_name(name) is not None


def semblo_config_fields(config: Mapping[str, object]) -> dict[str, object]:
    text = config.get("text_config", config)
    if not isinstance(text, Mapping):
        raise ValueError("checkpoint text_config must be a JSON object")
    return {
        str(key): value
        for key, value in text.items()
        if str(key).startswith("semblo_")
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().contiguous().cpu()
    data = cpu.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class RuntimeWeightsManifest:
    schema_version: int
    source_checkpoint: str
    source_config_sha256: str
    weights_file: str
    weights_sha256: str
    tensor_names: tuple[str, ...]
    tensor_shapes: dict[str, list[int]]
    tensor_dtypes: dict[str, str]
    tensor_sha256: dict[str, str]
    semblo_config: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_checkpoint": self.source_checkpoint,
            "source_config_sha256": self.source_config_sha256,
            "weights_file": self.weights_file,
            "weights_sha256": self.weights_sha256,
            "tensor_names": list(self.tensor_names),
            "tensor_shapes": self.tensor_shapes,
            "tensor_dtypes": self.tensor_dtypes,
            "tensor_sha256": self.tensor_sha256,
            "semblo_config": self.semblo_config,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "RuntimeWeightsManifest":
        if int(raw.get("schema_version", -1)) != RUNTIME_SCHEMA_VERSION:
            raise ValueError(
                "unsupported semblo runtime weights schema: "
                f"{raw.get('schema_version')!r}"
            )
        return cls(
            schema_version=RUNTIME_SCHEMA_VERSION,
            source_checkpoint=str(raw["source_checkpoint"]),
            source_config_sha256=str(raw["source_config_sha256"]),
            weights_file=str(raw["weights_file"]),
            weights_sha256=str(raw["weights_sha256"]),
            tensor_names=tuple(str(value) for value in raw["tensor_names"]),
            tensor_shapes={
                str(key): [int(value) for value in values]
                for key, values in dict(raw["tensor_shapes"]).items()
            },
            tensor_dtypes={
                str(key): str(value)
                for key, value in dict(raw["tensor_dtypes"]).items()
            },
            tensor_sha256={
                str(key): str(value)
                for key, value in dict(raw["tensor_sha256"]).items()
            },
            semblo_config=dict(raw["semblo_config"]),
        )


def read_runtime_manifest(directory: str | Path) -> RuntimeWeightsManifest:
    path = Path(directory).expanduser().resolve() / RUNTIME_MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"runtime weights manifest is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return RuntimeWeightsManifest.from_dict(raw)


def verify_runtime_weights(directory: str | Path) -> RuntimeWeightsManifest:
    root = Path(directory).expanduser().resolve()
    manifest = read_runtime_manifest(root)
    weights_path = root / manifest.weights_file
    if not weights_path.is_file():
        raise FileNotFoundError(f"runtime weights file is missing: {weights_path}")
    actual_file_hash = sha256_file(weights_path)
    if actual_file_hash != manifest.weights_sha256:
        raise ValueError(
            "runtime weights file hash mismatch: "
            f"expected {manifest.weights_sha256}, got {actual_file_hash}"
        )
    tensors = load_file(str(weights_path), device="cpu")
    if tuple(sorted(tensors)) != tuple(sorted(manifest.tensor_names)):
        raise ValueError("runtime tensor names do not match the manifest")
    for name in manifest.tensor_names:
        tensor = tensors[name]
        if list(tensor.shape) != manifest.tensor_shapes[name]:
            raise ValueError(f"runtime tensor shape mismatch for {name}")
        if str(tensor.dtype) != manifest.tensor_dtypes[name]:
            raise ValueError(f"runtime tensor dtype mismatch for {name}")
        if _tensor_sha256(tensor) != manifest.tensor_sha256[name]:
            raise ValueError(f"runtime tensor content hash mismatch for {name}")
    return manifest


def _canonical_parameter_map(module: nn.Module) -> dict[str, tuple[str, nn.Parameter]]:
    result: dict[str, tuple[str, nn.Parameter]] = {}
    for actual_name, parameter in module.named_parameters():
        canonical = canonical_runtime_name(actual_name)
        if canonical is None:
            continue
        if canonical in result:
            raise ValueError(
                f"multiple model parameters map to runtime name {canonical!r}"
            )
        result[canonical] = (actual_name, parameter)
    return result


_UNUSED_LAYER_PARAMETER = re.compile(r"^layers\.\d+\.self_attn\.semblo_")


def load_runtime_weights_into_module(
    module: nn.Module,
    directory: str | Path,
    *,
    require_exact_match: bool = True,
    allow_unused_layer_parameters: bool = False,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> set[str]:
    """Load a runtime router file into either the HF model or the vLLM model.

    The file and its manifest are verified (a missing file raises), then the
    tensors go through ``load_runtime_tensors_into_module`` in the manifest's
    order; see there for the keyword arguments.
    """

    root = Path(directory).expanduser().resolve()
    manifest = verify_runtime_weights(root)
    tensors = load_file(str(root / manifest.weights_file), device="cpu")
    return load_runtime_tensors_into_module(
        module,
        tensors,
        manifest.tensor_names,
        require_exact_match=require_exact_match,
        allow_unused_layer_parameters=allow_unused_layer_parameters,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )


_QUERY_ROUTER = ".self_attn.semblo_query_router."


def _without_query_router(
    tensors: Mapping[str, torch.Tensor], tensor_names: Sequence[str]
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Drop the query-router tensors of a file written by an earlier version.

    Queries go through the key-side map (routing_map.py).  A query router
    whose residual and mixture are zero adds nothing to that, so its tensors
    are dropped; any other query router raises.
    """

    kept: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if _QUERY_ROUTER not in name:
            kept[name] = tensor
            continue
        field = name.rsplit(".", 1)[-1]
        if field in ("down", "up", "mixture_weight", "mixture_bias") and bool(tensor.any()):
            raise ValueError(
                f"{name} is not zero: this file holds a trained query router, "
                "which the plugin does not support")
    return kept, [name for name in tensor_names if name in kept]


def load_runtime_tensors_into_module(
    module: nn.Module,
    tensors: Mapping[str, torch.Tensor],
    tensor_names: Sequence[str],
    *,
    require_exact_match: bool = True,
    allow_unused_layer_parameters: bool = False,
    tp_rank: int = 0,
    tp_size: int = 1,
) -> set[str]:
    """Copy runtime router tensors, keyed by canonical name, into ``module``.

    ``tensors`` maps canonical names (``layers.<i>.self_attn.semblo_*``) to
    full-size tensors.  ``tensor_names`` is the order they are copied in and
    must name every key of ``tensors``.  Returns the model's parameter names
    that were written.

    ``allow_unused_layer_parameters`` skips the tensors of layers the model
    has no router for (layers left out of ``semblo_layer_indices`` are
    dense).  Only whole per-layer router groups may be skipped.  Any other
    mismatch raises.

    ``tp_rank``/``tp_size`` shard head-major router tensors for tensor
    parallelism.  Every semblo parameter carries its head count in
    dimension 0 (KV heads for the key side, query heads for the query side),
    and vLLM assigns each rank one contiguous block of heads, so a file
    tensor whose leading dimension is ``tp_size`` times the parameter's is
    sliced to this rank's block.  Head-independent tensors (equal shapes)
    are replicated unchanged.
    """

    if sorted(tensor_names) != sorted(tensors):
        raise ValueError("runtime tensor names do not match the tensors given")
    tensors, tensor_names = _without_query_router(tensors, tensor_names)
    targets = _canonical_parameter_map(module)
    sidecar_names = set(tensors)
    target_names = set(targets)
    missing_in_model = sorted(sidecar_names - target_names)
    missing_in_file = sorted(target_names - sidecar_names)
    skipped: set[str] = set()
    if missing_in_model and allow_unused_layer_parameters:
        skipped = {
            name for name in missing_in_model if _UNUSED_LAYER_PARAMETER.match(name)
        }
        missing_in_model = sorted(set(missing_in_model) - skipped)
        if skipped:
            LOGGER.warning(
                "skipping %d runtime router tensors for layers the runtime "
                "config excludes: %s",
                len(skipped),
                ", ".join(sorted(skipped)[:8]),
            )
    if missing_in_model:
        raise ValueError(
            "runtime weights contain parameters that are absent from the model: "
            + ", ".join(missing_in_model[:8])
        )
    if require_exact_match and missing_in_file:
        raise ValueError(
            "runtime weights are missing model parameters: "
            + ", ".join(missing_in_file[:8])
        )

    loaded: set[str] = set()
    with torch.no_grad():
        for canonical in tensor_names:
            if canonical in skipped:
                continue
            actual_name, parameter = targets[canonical]
            tensor = tensors[canonical]
            if tuple(tensor.shape) != tuple(parameter.shape):
                local = int(parameter.shape[0]) if parameter.ndim > 0 else 0
                if (
                    tp_size > 1
                    and tensor.ndim == parameter.ndim
                    and local > 0
                    and int(tensor.shape[0]) == local * tp_size
                    and tuple(tensor.shape[1:]) == tuple(parameter.shape[1:])
                ):
                    tensor = tensor[tp_rank * local : (tp_rank + 1) * local]
                else:
                    raise ValueError(
                        f"shape mismatch for {canonical}: file {tuple(tensor.shape)}, "
                        f"model {tuple(parameter.shape)} (tp_size={tp_size})"
                    )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(actual_name)
    return loaded


def filter_base_weight_iterator(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Remove router tensors from the normal vLLM weight iterator.

    The router tensors are loaded only from ``semblo_runtime.safetensors``,
    whether or not the base checkpoint also contains them.
    """

    for name, tensor in weights:
        if not is_runtime_parameter(name):
            yield name, tensor
