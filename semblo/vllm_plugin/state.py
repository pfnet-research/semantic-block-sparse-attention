from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from semblo.vllm_plugin.validation import DEFERRED_CHECKS, SLOT_DECODE_ORDER


# The packed KV token axis is padded to this, so it can be viewed as pages of
# any size up to 256 (FA3's block table indexes pages, and a view needs the
# axis to be a whole number of them).  256 rows of 4 heads x 256 dims x bf16
# is 512 KB, against the 1.06 GB the buffer already occupies at 256K.
_PACK_ALIGN = 256
_SAID: dict[str, bool] = {}     # one-time log lines


def _sink_tokens() -> int:
    """SEMBLO_SINK_TOKENS for the fitted partition (0 when unset)."""
    return int(os.environ.get("SEMBLO_SINK_TOKENS", "0") or "0")


def _check_sink_partition(sink: int, length: int, fit_length: int, num_clusters: int,
                          min_size: int, max_size: int, hier_span: int, fit_prefix: int) -> None:
    """Refuse the settings a reserved sink cluster does not compose with."""
    if sink % 16:
        raise RuntimeError(f"SEMBLO_SINK_TOKENS={sink}: a multiple of 16 keeps the sink cluster page-aligned")
    if 0 < fit_prefix <= sink:
        raise RuntimeError("SEMBLO_FIT_PREFIX_TOKENS must exceed SEMBLO_SINK_TOKENS")
    if length <= sink or fit_length - sink < 2 * num_clusters:
        raise RuntimeError(f"SEMBLO_SINK_TOKENS={sink} leaves too few keys ({length:,} in the prompt, "
                           f"{fit_length:,} to fit) for {num_clusters - 1} clusters")
    if min_size or max_size:
        raise RuntimeError("SEMBLO_SINK_TOKENS and the cluster size split/merge cannot be used together")
    if hier_span > 0:
        raise RuntimeError("SEMBLO_SINK_TOKENS and SEMBLO_HIER_SPAN cannot be used together")
    if os.environ.get("SEMBLO_PAGE_CLUSTERS", "0") not in ("", "0", "1"):
        raise RuntimeError("SEMBLO_SINK_TOKENS and SEMBLO_PAGE_CLUSTERS cannot be used together")


def _page_align_order(order, counts, positions, align):
    """Pad each bucket's slice so buckets START on a page boundary.

    The FA3 body reads whole pages, and a page straddling a bucket boundary
    would drag in tokens at arbitrary positions -- including future ones -- so
    fa3_pages.py drops any straddling page.  With unaligned starts that costs
    each bucket BOTH its leading and its trailing fragment, about `align`
    tokens; aligning the starts removes the leading one and halves the loss.

    `counts` (the true, unpadded sizes) is left alone, so the padding rows sit
    outside every bucket's [start, start + size) and no reader can name them.
    They still need a valid entry in the posting list, since pack_prompt_kv
    gathers through it, so they repeat row 0 -- whose K/V is then never read.

    Returns (padded order, padded indptr, position -> padded row).
    """
    device = order.device
    length = int(order.numel())
    counts64 = counts.to(torch.int64)
    padded = ((counts64 + align - 1) // align) * align
    zero = counts64.new_zeros(1)
    ip_orig = torch.cat([zero, counts64.cumsum(0)])
    ip_pad = torch.cat([zero, padded.cumsum(0)])
    total = int(ip_pad[-1])
    ar = torch.arange(total, device=device)
    bucket = torch.searchsorted(ip_pad, ar, right=True) - 1
    off = ar - ip_pad[bucket]
    real = off < counts64[bucket]
    src = (ip_orig[bucket] + off).clamp_max(length - 1)
    new_order = torch.where(real, order[src], order[0])
    # position -> padded row, built from the REAL rows only: the padding rows
    # repeat row 0, so scattering the padded list would overwrite that
    # position's true mapping
    bucket_of_row = torch.searchsorted(ip_orig, torch.arange(length,
                                                             device=device),
                                       right=True) - 1
    padded_row = (torch.arange(length, device=device)
                  + (ip_pad[bucket_of_row] - ip_orig[bucket_of_row]))
    inverse = torch.zeros(length, dtype=torch.int32, device=device)
    inverse[positions[order].long()] = padded_row.to(torch.int32)
    return new_order, ip_pad.to(torch.int32), inverse


def _collapse_assignment_to_pages(
    space: torch.Tensor,        # [L, index_dim] the clustering space
    centroids: torch.Tensor,    # [C, index_dim]
    positions: torch.Tensor,    # [L] original token positions
    page: int,
    assign_fn,
) -> torch.Tensor:
    """Give every token of a page the same cluster.

    With page-granular clusters the prompt could be served from vLLM's own
    paged cache instead of a cluster-ordered copy.  A page goes to the
    centroid nearest its mean, computed in the same half-normalised space
    the fit used, so the page is placed as one point of that space rather
    than by a vote among its tokens.
    """

    device = space.device
    page_of_row = (positions.long() // page)
    num_pages = int(page_of_row.max().item()) + 1
    sums = torch.zeros(num_pages, space.shape[1], dtype=torch.float32,
                       device=device)
    sums.index_add_(0, page_of_row, space.float())
    counts = torch.bincount(page_of_row, minlength=num_pages).clamp_min(1)
    means = sums / counts.unsqueeze(1).float()
    # Back onto the sphere half by half: the fit space is two unit halves
    # concatenated, and an average of unit vectors is not one.
    halves = means.chunk(2, dim=-1)
    means = torch.cat(
        [torch.nn.functional.normalize(h, dim=-1) for h in halves], dim=-1)
    page_assign = assign_fn(means.to(space.dtype), centroids)
    return page_assign[page_of_row]


@dataclass
class SelectionStats:
    """Running totals for one prompt's hard retrieval.

    Selected tokens, bucket intervals and work tiles are counted together:
    a work tile is padded to a fixed width, so counting only selected tokens
    understates what the kernel processes.

    Totals accumulate on the device so collecting them adds no synchronization
    to the forward pass; read them once after the request finishes.
    """

    selected_tokens: torch.Tensor | None = None
    processed_slots: torch.Tensor | None = None
    work_tiles: torch.Tensor | None = None
    ranked_candidate_buckets: torch.Tensor | None = None
    touched_kv_blocks: torch.Tensor | None = None
    sampled_chunk_selected: torch.Tensor | None = None
    packed_runs: dict | None = None
    calls: int = 0

    def record(
        self,
        tiles: Any,
        *,
        tile_size: int,
        ranked_candidates: torch.Tensor,
        touched_blocks: torch.Tensor | None = None,
        packed_runs: dict[str, float] | None = None,
    ) -> None:
        selected = tiles.selected_tokens.sum(dtype=torch.int64)
        counts = tiles.counts.sum(dtype=torch.int64)
        buckets = ranked_candidates.sum(dtype=torch.int64)
        blocks = (
            touched_blocks.sum(dtype=torch.int64)
            if touched_blocks is not None
            else None
        )
        if self.selected_tokens is None:
            self.selected_tokens = selected.clone()
            self.processed_slots = counts * int(tile_size)
            self.work_tiles = counts.clone()
            self.ranked_candidate_buckets = buckets.clone()
            self.touched_kv_blocks = None if blocks is None else blocks.clone()
            if blocks is not None:
                self.sampled_chunk_selected = selected.clone()
        else:
            self.selected_tokens += selected
            self.processed_slots += counts * int(tile_size)
            self.work_tiles += counts
            self.ranked_candidate_buckets += buckets
            if blocks is not None and self.touched_kv_blocks is None:
                self.touched_kv_blocks = blocks.clone()
                self.sampled_chunk_selected = selected.clone()
        if packed_runs is not None and self.packed_runs is None:
            self.packed_runs = dict(packed_runs)
        self.calls += 1

    def summary(self) -> dict[str, float | int]:
        if self.selected_tokens is None:
            return {"calls": 0}
        selected = int(self.selected_tokens.item())
        slots = int(self.processed_slots.item())
        return {
            "calls": self.calls,
            "selected_valid_tokens": selected,
            "processed_tile_slots": slots,
            "work_tiles": int(self.work_tiles.item()),
            "ranked_candidate_buckets": int(self.ranked_candidate_buckets.item()),
            "touched_kv_blocks": (
                int(self.touched_kv_blocks.item())
                if self.touched_kv_blocks is not None
                else None
            ),
            "slots_per_valid_token": (slots / selected) if selected else 0.0,
            "packed_bucket_runs": self.packed_runs,
            "sampled_chunk_selected_tokens": (
                int(self.sampled_chunk_selected.item())
                if self.sampled_chunk_selected is not None
                else None
            ),
            "sampled_tokens_per_touched_kv_block": (
                int(self.sampled_chunk_selected.item())
                / int(self.touched_kv_blocks.item())
                if self.touched_kv_blocks is not None
                and int(self.touched_kv_blocks.item())
                else None
            ),
        }


# Diagnostic accumulator for the single-request configuration this integration
# supports.  Disabled unless the prepared config sets
# ``semblo_vllm_collect_selection_stats``.
SELECTION_STATS = SelectionStats()

# One entry per freshly built prompt index (one per layer per request),
# gated by the same flag: bucket occupancy, size skew and factor
# dependence.  Drained by the caller.
INDEX_BUILD_STATS: list[dict] = []


def _entropy_bits(probabilities: torch.Tensor) -> float:
    live = probabilities[probabilities > 0]
    return float(-(live * live.log2()).sum().item())


def record_index_build_stats(
    index: "ProductPostingIndex", *, source_layer_idx: int, build_seconds: float
) -> None:
    """Occupancy of one built index; every host read here is prefill-only."""

    sizes = index.bucket_sizes.to(torch.float64)
    num_heads, num_buckets = sizes.shape
    entry: dict[str, object] = {
        "source_layer_idx": int(source_layer_idx),
        "prompt_length": int(index.prompt_length),
        "num_non_empty_buckets": int(index.num_non_empty_buckets),
        "build_seconds": round(float(build_seconds), 4),
    }
    occupied = sizes[sizes > 0]
    if occupied.numel():
        quantiles = torch.quantile(
            occupied,
            torch.tensor([0.5, 0.9], dtype=torch.float64, device=occupied.device),
        )
        entry["bucket_size_max"] = int(occupied.max().item())
        entry["bucket_size_mean"] = round(float(occupied.mean().item()), 1)
        entry["bucket_size_p50"] = round(float(quantiles[0].item()), 1)
        entry["bucket_size_p90"] = round(float(quantiles[1].item()), 1)
    centroids = int(index.codebooks.shape[2])
    if num_heads == 1 and num_buckets == centroids * centroids:
        joint = sizes[0].reshape(centroids, centroids)
        total = joint.sum()
        if float(total.item()) > 0:
            joint = joint / total
            first = _entropy_bits(joint.sum(dim=1))
            second = _entropy_bits(joint.sum(dim=0))
            both = _entropy_bits(joint.reshape(-1))
            entry["factor_entropy_bits"] = [round(first, 3), round(second, 3)]
            entry["joint_entropy_bits"] = round(both, 3)
            entry["factor_mutual_information_bits"] = round(first + second - both, 3)
    INDEX_BUILD_STATS.append(entry)


def record_pack_seconds(seconds: float) -> None:
    """Attach the prompt-K/V packing time to the most recent build entry."""

    if INDEX_BUILD_STATS:
        INDEX_BUILD_STATS[-1]["pack_seconds"] = round(float(seconds), 4)


def _report_index_sizes(kind: str, sizes: torch.Tensor) -> None:
    """One line of partition statistics per built index.

    Gated by SEMBLO_INDEX_SIZE_REPORT because it synchronises; the numbers
    (median / p99 / max bucket, fraction of tokens in oversized buckets)
    are what decides whether a partition is kernel-friendly."""
    if os.environ.get("SEMBLO_INDEX_SIZE_REPORT") != "1":
        return
    live = sizes[sizes > 0].float()
    if live.numel() == 0:
        return
    total = float(live.sum())
    big = float(live[live > 512].sum())
    print(
        f"INDEXSIZE {kind} buckets={int(sizes.shape[1])} "
        f"non_empty={int(live.numel())} "
        f"p50={float(live.median()):.0f} "
        f"p99={float(torch.quantile(live, 0.99)):.0f} "
        f"max={float(live.max()):.0f} "
        f"frac_tokens_over_512={big / max(total, 1):.3f}",
        flush=True,
    )

def _bucket_slot_table(
    bucket_sizes: torch.Tensor, bucket_indptr: torch.Tensor, slots: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Number the non-empty buckets so per-bucket work stays that small.

    Returns the slot of each bucket (``-1`` when it holds nothing), the packed
    offset each slot starts at, and which bucket each slot is.
    """

    non_empty = bucket_sizes > 0
    slot = (non_empty.cumsum(dim=-1) - 1).to(torch.int32)
    bucket_slot = torch.where(non_empty, slot, torch.full_like(slot, -1))
    num_kv_heads, num_buckets = map(int, bucket_sizes.shape)
    slot_start = torch.zeros(
        num_kv_heads, slots + 1, dtype=torch.int32, device=bucket_sizes.device
    )
    slot_bucket = torch.zeros_like(slot_start)
    destination = torch.where(non_empty, slot.to(torch.int64), slots)
    slot_start.scatter_(1, destination, bucket_indptr[:, :-1].to(torch.int32))
    slot_bucket.scatter_(
        1,
        destination,
        torch.arange(num_buckets, device=bucket_sizes.device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(num_kv_heads, -1),
    )
    return (
        bucket_slot.contiguous(),
        slot_start[:, :slots].contiguous(),
        slot_bucket[:, :slots].contiguous(),
    )


def _descriptor_features(
    space: torch.Tensor,      # [L, I], half-normalised
    weight: torch.Tensor,     # [Dm, I]
    bias: torch.Tensor,       # [Dm]
    activation: str,
) -> torch.Tensor:
    """psi applied to one KV head's keys.  Returns [L, Dm], float32.

    The activation is applied BEFORE the sum, which is the only place it can
    matter: a sum of linear features is n_c (W zbar_c + b) and carries just
    the mean and the count.  Kept in float32 because the sum runs over the
    whole cluster and an exp activation makes the terms large.
    """

    raw = space.float() @ weight.float().T + bias.float()
    if activation == "linear":
        return raw
    if activation == "softplus":
        return torch.nn.functional.softplus(raw)
    if activation == "relu":
        return torch.nn.functional.relu(raw)
    if activation == "exp":
        return torch.exp(raw.clamp(max=15.0))
    raise ValueError(f"unknown descriptor activation {activation!r}")


_FIXED_CODEBOOK_CACHE: dict[str, Any] = {}


def _load_fixed_codebook(
    layer_idx: int | None,
    num_clusters: int,
    num_kv_heads: int,
    index_dim: int,
):
    """The fixed-codebook baseline (SEMBLO_FIXED_CODEBOOK), or None.

    Replaces the per-prompt spherical k-means with centroids fitted once on a
    corpus.  The assignment, posting lists, scoring and kernels are unchanged.

    The file holds ``{"codebooks": {(layer, C): [Hkv, C, I]}, "meta": {...}}``.
    A missing entry raises rather than falling back to the per-prompt fit.
    """
    path = os.environ.get("SEMBLO_FIXED_CODEBOOK")
    if not path:
        return None
    if layer_idx is None:
        raise RuntimeError(
            "SEMBLO_FIXED_CODEBOOK is set but the index build was not told which "
            "layer it is in; the codebook is per (layer, cluster count)")
    if path not in _FIXED_CODEBOOK_CACHE:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if "codebooks" not in blob:
            raise RuntimeError(f"{path} has no 'codebooks' entry")
        _FIXED_CODEBOOK_CACHE[path] = blob
    blob = _FIXED_CODEBOOK_CACHE[path]
    key = (int(layer_idx), int(num_clusters))
    book = blob["codebooks"].get(key)
    if book is None:
        have = sorted(blob["codebooks"].keys())
        raise RuntimeError(
            f"{path} has no codebook for layer {layer_idx} with "
            f"{num_clusters} clusters; it holds {have}.  The cluster count "
            f"follows the prompt length, so fit the codebook at the length "
            f"the evaluation actually runs at.")
    if tuple(book.shape) != (num_kv_heads, num_clusters, index_dim):
        raise RuntimeError(
            f"{path} layer {layer_idx}: codebook is {tuple(book.shape)}, "
            f"expected {(num_kv_heads, num_clusters, index_dim)}")
    return book


@dataclass
class ProductPostingIndex:
    """Request-local product index aligned to logical token positions.

    Original K/V remain exclusively in vLLM's paged cache.  This object stores
    only codebooks and an inverted list of logical positions.
    """

    codebooks: torch.Tensor  # [Hkv, 2, M, Df]
    posting_positions: torch.Tensor  # [Hkv, L], bucket-major, int32
    bucket_indptr: torch.Tensor  # [Hkv, M**2 + 1], int32
    bucket_sizes: torch.Tensor  # [Hkv, M**2], int32
    prompt_length: int
    max_position: int
    packed_kv: torch.Tensor | None = None  # [2,L,Hkv,D] in bucket order
    # built on first use by inverse_positions(), or at build time when the
    # layout is page-aligned (padding rows repeat row 0, so the padded posting
    # list can no longer be inverted by scattering it)
    _inverse_positions: torch.Tensor | None = None
    # bucket starts are multiples of this in the packed layout; 0 = unaligned.
    # fa3_pages can then keep a bucket's leading partial page, halving what
    # page trimming gives up.
    page_align: int = 0
    # The unquantized router projections phi(k), stored in the same per-head
    # posting order as ``posting_positions``.  Only kept when the two-stage
    # phi scorer needs token-level scores (fp16: L x I is 32MB at 128K).
    packed_projected_keys: torch.Tensor | None = None  # [Hkv,L,I]
    # Sum of psi over the members of each cluster, [Hkv, C, Dm].  None unless
    # the cluster descriptor is configured.
    cluster_descriptors: torch.Tensor | None = None
    num_non_empty_buckets: int = 0
    # Non-empty buckets are 8-21% of the product code, so anything that works
    # per bucket is done over these slots rather than over all M**2 of them.
    bucket_slot: torch.Tensor | None = None   # [Hkv,M**2] int32, -1 if empty
    slot_start: torch.Tensor | None = None    # [Hkv,slots] int32, packed offset
    slot_bucket: torch.Tensor | None = None   # [Hkv,slots] int32, bucket of a slot
    # Positions of the generated suffix, laid down once and grown geometrically.
    suffix_positions: torch.Tensor | None = None
    suffix_tiles: tuple | None = None  # (starts, lengths, counts, selected, tile_size)
    # Flat partition (SEMBLO_FLAT_UNIT_TOKENS > 0): ``codebooks`` holds
    # [Hkv, 2, C, Df] cluster descriptors (the two halves of one descriptor,
    # kept split to match the query layout) and a bucket id is a cluster id,
    # so num_buckets is C rather than C**2.
    flat_clusters: int = 0
    # Coarse level over the flat partition (SEMBLO_HIER_SPAN).  Cluster ids are
    # renumbered so group g owns ids [g*hier_span, (g+1)*hier_span); this
    # holds one descriptor per group in the same [2, ., Df] half-split layout
    # as ``codebooks``.  None when the coarse level is off.
    group_descriptors: torch.Tensor | None = None  # [Hkv, 2, C/span, Df]
    # [Hkv, 2, C/span] max member distance, per half: the coarse score is an
    # upper bound q.mu + ||q||.r, not the mean's score (see group_half_radii)
    group_radii: torch.Tensor | None = None
    hier_span: int = 0

    @property
    def num_groups(self) -> int:
        return 0 if not self.hier_span else self.flat_clusters // self.hier_span

    @property
    def num_kv_heads(self) -> int:
        return int(self.codebooks.shape[0])

    @property
    def num_centroids(self) -> int:
        return int(self.codebooks.shape[2])

    @property
    def num_buckets(self) -> int:
        if self.flat_clusters:
            return int(self.flat_clusters)
        return self.num_centroids**2

    @property
    def is_flat(self) -> bool:
        return bool(self.flat_clusters)

    @classmethod
    def build(
        cls,
        product_keys: torch.Tensor,
        positions: torch.Tensor,
        *,
        num_centroids: int,
        kmeans_iters: int,
        kmeans_sample_size: int,
        codebook_seed: int | None = None,
        quantizer: str = "product",
        kmeans_init: str = "random",
        centroid_refiner: Any = None,
        kmeans_preprocess: str = "none",
        flat_unit_tokens: int = 0,
        flat_min_clusters_log2: int = 9,
        page_align: int = 0,
        # The cluster descriptor's key-side parameters.  None keeps the
        # centroid-only scoring of the method.
        descriptor_weight: torch.Tensor | None = None,   # [Hkv, Dm, I]
        descriptor_bias: torch.Tensor | None = None,     # [Hkv, Dm]
        descriptor_activation: str = "softplus",
        # Only used to look up a fixed codebook (SEMBLO_FIXED_CODEBOOK).
        layer_idx: int | None = None,
    ) -> "ProductPostingIndex":
        """Fit the prompt-global product codebook and posting lists.

        The k-means draws its sample, its initial centroids, and any empty
        cluster replacement at random.  Left on the global RNG, the codebook for
        a given prompt depends on how many requests preceded it in the process,
        so the same prompt served twice selects different tokens.
        ``codebook_seed`` makes the fit depend on the prompt alone.
        """

        if product_keys.ndim != 3:
            raise ValueError("product_keys must have shape [L,Hkv,I]")
        length, num_kv_heads, _ = product_keys.shape
        if positions.shape != (length,):
            raise ValueError("positions must have shape [L]")
        if length == 0:
            raise ValueError("cannot build an index for an empty prompt")
        if not torch.equal(positions, torch.arange(length, device=positions.device, dtype=positions.dtype)):
            raise ValueError("initial vLLM integration requires contiguous prompt positions 0..L-1")

        generator: torch.Generator | None = None
        if codebook_seed is not None:
            generator = torch.Generator(device=product_keys.device)
            generator.manual_seed(int(codebook_seed))

        # the environment overrides the setting
        flat_target = int(os.environ.get("SEMBLO_FLAT_UNIT_TOKENS",
                                         flat_unit_tokens))
        if flat_target > 0:
            return cls._build_flat(
                product_keys, positions, target_size=flat_target,
                kmeans_iters=kmeans_iters, kmeans_sample_size=kmeans_sample_size,
                generator=generator,
                min_clusters_log2=flat_min_clusters_log2,
                page_align=page_align,
                descriptor_weight=descriptor_weight,
                descriptor_bias=descriptor_bias,
                descriptor_activation=descriptor_activation,
                layer_idx=layer_idx,
            )
        raise ValueError(
            "the index needs the flat partition: set SEMBLO_FLAT_UNIT_TOKENS or "
            "semblo_flat_unit_tokens to the mean block size")

    @classmethod
    def _build_flat(
        cls,
        product_keys: torch.Tensor,
        positions: torch.Tensor,
        *,
        target_size: int,
        kmeans_iters: int,
        kmeans_sample_size: int,
        generator: torch.Generator | None,
        min_clusters_log2: int = 9,
        page_align: int = 0,
        descriptor_weight: torch.Tensor | None = None,   # [Hkv, Dm, I]
        descriptor_bias: torch.Tensor | None = None,     # [Hkv, Dm]
        descriptor_activation: str = "softplus",
        layer_idx: int | None = None,
    ) -> "ProductPostingIndex":
        """Flat partition at ~target_size tokens per cluster.

        Same outputs as the product build (posting order, indptr, sizes,
        slot tables); the difference is that a bucket id is a cluster id and
        ``codebooks`` stores per-cluster descriptors split into the two
        halves to match the query layout.
        """

        from semblo.inference.flat_index import (
            assign_flat_buckets, balanced_group_order, fit_flat_codebook,
            flat_cluster_count, group_half_radii, routing_space,
            split_merge_sizes,
        )

        length, num_kv_heads, index_dim = product_keys.shape
        # C must be a power of two: the union-build and streaming-select
        # kernels index the bucket axis with tl.arange(0, B) and
        # tl.arange(0, B // 32), and Triton requires power-of-two ranges.
        # SEMBLO_FLAT_MIN_CLUSTERS_LOG2 sets a floor on the cluster count;
        # the count is capped by the number of prompt tokens.
        # Positional blocks (SEMBLO_BLOCK_TOKENS=B).  The partition is not
        # fitted: token i belongs to block i // B, and the representative of a
        # block is the plain mean of its keys in the routing space.  Scoring,
        # selection, gather and attention are the same code as the method's.
        block_tokens = int(os.environ.get("SEMBLO_BLOCK_TOKENS", "0"))
        min_c_log2 = max(3, int(os.environ.get(
            "SEMBLO_FLAT_MIN_CLUSTERS_LOG2", min_clusters_log2)))
        if block_tokens > 0:
            # exactly ceil(L / B) blocks, rounded up to a power of two for the
            # kernels' bucket axis; the cluster floor and the target size do
            # not apply, the block size decides.
            blocks = (int(length) + block_tokens - 1) // block_tokens
            num_clusters = 1 << max(3, (max(blocks, 1) - 1).bit_length())
        else:
            raw_c = flat_cluster_count(length, target_size)
            num_clusters = 1 << max(min_c_log2, (max(raw_c, 1) - 1).bit_length())
            num_clusters = min(num_clusters, 1 << (length.bit_length() - 1))
            # the token-outer prefill fits on the prompt's head but wants the
            # count the whole prompt would get (token_outer_prefill.py)
            forced = int(os.environ.get("SEMBLO_FLAT_CLUSTERS", "0"))
            if forced > 0:
                num_clusters = forced
        min_size = int(os.environ.get("SEMBLO_FLAT_MIN_TOKENS", "0"))
        max_size = int(os.environ.get("SEMBLO_FLAT_MAX_TOKENS", "0"))

        codebooks: list[torch.Tensor] = []
        posting_rows: list[torch.Tensor] = []
        indptr_rows: list[torch.Tensor] = []
        size_rows: list[torch.Tensor] = []
        inverse_rows: list[torch.Tensor] = []
        group_rows: list[torch.Tensor] = []
        radius_rows: list[torch.Tensor] = []
        descriptor_rows: list[torch.Tensor] = []
        fixed_codebook = _load_fixed_codebook(layer_idx, num_clusters,
                                              num_kv_heads, index_dim)
        hier_span = int(os.environ.get("SEMBLO_HIER_SPAN", "0"))
        report = os.environ.get("SEMBLO_INDEX_SIZE_REPORT") == "1"
        if report:
            import time as _time
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
        center_mode = os.environ.get("SEMBLO_ROUTING_CENTER", "0")
        # the adjacent-block baseline's representative is the plain mean of a
        # block's raw keys by definition; centring is a property of the
        # clustered partition and does not apply there
        center = center_mode in ("1", "whiten") and block_tokens == 0
        whiten = center_mode == "whiten" and block_tokens == 0
        flat_init = os.environ.get("SEMBLO_FLAT_INIT", "random")
        # SEMBLO_FLAT_METRIC=euclidean is the no-normalisation ablation of the
        # routing space: the k-means runs on the keys as they are,
        # its centroids are plain means, and the query is scored raw against
        # them -- so it only makes sense with SEMBLO_ROUTING_NORMALIZE=none, and
        # only on the plain per-prompt fit.
        flat_metric = os.environ.get("SEMBLO_FLAT_METRIC", "cosine") or "cosine"
        from semblo.inference.flat_index import METRICS
        if flat_metric not in METRICS:
            raise RuntimeError(f"SEMBLO_FLAT_METRIC={flat_metric!r}: expected one of {METRICS}")
        if flat_metric != "cosine":
            if (os.environ.get("SEMBLO_ROUTING_NORMALIZE", "whole") or "whole") != "none":
                raise RuntimeError("SEMBLO_FLAT_METRIC=euclidean needs SEMBLO_ROUTING_NORMALIZE=none: "
                                   "a Euclidean fit over normalised keys is not the no-normalisation arm")
            if whiten or fixed_codebook is not None or block_tokens:
                raise RuntimeError("SEMBLO_FLAT_METRIC=euclidean is only implemented for the plain per-prompt fit")
        key_stats = os.environ.get("SEMBLO_KEY_STATS") == "1"
        if center and fixed_codebook is not None:
            raise RuntimeError("SEMBLO_ROUTING_CENTER re-centres the keys per prompt; "
                               "an offline codebook (SEMBLO_FIXED_CODEBOOK) was fitted without it")
        for kv_head in range(num_kv_heads):
            keys = product_keys[:, kv_head]
            if center or key_stats:
                # The mean key of this prompt and head.  Nemotron 3.5's keys
                # (no rotary embedding) share one dominant direction: after
                # normalisation every key points nearly the same way and
                # k-means puts the prompt in one cluster.  Subtracting the
                # mean changes no token's rank under any query (q.k and
                # q.(k - mu) differ by the constant q.mu) and leaves the part
                # of the key that tells tokens apart.  Off by default.
                mu = keys.float().mean(dim=0, keepdim=True)
                if key_stats:
                    kf = keys.float()
                    norms = kf.norm(dim=-1)
                    finite = torch.isfinite(kf).all(dim=-1)
                    q = torch.quantile(norms[finite][:: max(1, int(finite.sum()) // 65536)].float(),
                                       torch.tensor([0.5, 0.99, 1.0], device=norms.device)) if bool(finite.any()) else torch.zeros(3)
                    ratio = float(mu.norm()) / max(float(norms.mean()), 1e-9)
                    print(f"[semblo] key stats layer {layer_idx} kv head {kv_head}: |mean key| / mean |key| = {ratio:.3f}"
                          f"  |key| p50 {float(q[0]):.3g} p99 {float(q[1]):.3g} max {float(q[2]):.3g}"
                          f"  non-finite keys {int((~finite).sum())} of {int(finite.numel())}"
                          f"{' (centred)' if center else ''}", flush=True)
                if center:
                    keys = (keys.float() - mu).to(keys.dtype)
            # the space both sides of the score agree on; SEMBLO_ROUTING_NORMALIZE
            # picks the transform (whole by default, half, or none)
            space = routing_space(keys)
            # ~32 sample points per centroid is enough for a stable
            # partition, so the sample and the iteration count scale with C.
            per_centroid = int(os.environ.get("SEMBLO_FLAT_FIT_PER_C", "32"))
            fit_sample = min(length, max(int(kmeans_sample_size),
                                         per_centroid * num_clusters))
            fit_iters = int(os.environ.get("SEMBLO_FLAT_FIT_ITERS", "8"))
            # SEMBLO_FIT_PREFIX_TOKENS=P fits the centroids on the first P
            # positions only and still assigns every key: the partition a
            # chunked prefill would have, whose index is fixed once the
            # prefix has passed and later keys join existing clusters.
            fit_prefix = int(os.environ.get("SEMBLO_FIT_PREFIX_TOKENS", "0"))
            fit_space = space if fit_prefix <= 0 else space[:min(length, fit_prefix)]
            if fit_prefix > 0 and kv_head == 0 and not _SAID.get("fit_prefix"):
                _SAID["fit_prefix"] = True
                print(f"[semblo] centroids fitted on the first {fit_space.shape[0]:,} of {length:,} positions "
                      f"(SEMBLO_FIT_PREFIX_TOKENS={fit_prefix}); every key is assigned", flush=True)
            if _sink_tokens() > 0 and block_tokens <= 0 and (fixed_codebook is not None or whiten):
                raise RuntimeError("SEMBLO_SINK_TOKENS reserves a cluster of the per-prompt fit; it is not defined for "
                                   "a fixed codebook or the whitened fit")
            if block_tokens > 0:
                if fixed_codebook is not None:
                    raise RuntimeError(
                        "SEMBLO_BLOCK_TOKENS and SEMBLO_FIXED_CODEBOOK are two "
                        "different partitions; pick one")
                # Adjacent blocks: the assignment is arithmetic and the
                # representative is the mean of the block's keys.  index_add_
                # over the block id is one pass over the prompt, so this
                # replaces the k-means entirely (no fit, no iterations).
                assign = torch.div(
                    torch.arange(length, device=space.device),
                    block_tokens, rounding_mode="floor",
                ).to(torch.int32)
                sums = torch.zeros(num_clusters, index_dim,
                                   dtype=torch.float32, device=space.device)
                sums.index_add_(0, assign.long(), space.float())
                counts = torch.bincount(
                    assign, minlength=num_clusters).to(torch.float32)
                # Trailing blocks are empty when L is not a multiple of B;
                # they get a zero representative and are excluded from
                # selection anyway (n_c = 0).
                cents = (sums / counts.clamp_min(1.0)[:, None]).to(space.dtype)
                from semblo.vllm_plugin import block_minmax
                if block_minmax.enabled():
                    # Quest-style min/max descriptor (SEMBLO_BLOCK_DESCRIPTOR=
                    # minmax): the block's coordinate-wise maximum in the
                    # first half of the doubled routing space and its minimum
                    # in the second; the query side rectifies its halves
                    # (block_minmax.py), so the same dot product scores the
                    # query-aware bound.  Needs the raw-keys-doubled router.
                    cents = block_minmax.block_minmax_descriptors(
                        space, assign, num_clusters)
                    del sums
            elif fixed_codebook is not None:
                # Fixed codebook: the centroids do not depend on the
                # prompt, only the assignment does.
                cents = fixed_codebook[kv_head].to(
                    device=space.device, dtype=space.dtype)
                assign = assign_flat_buckets(space, cents)
            elif whiten:
                # The PARTITION is fitted in the whitened space (centred keys
                # times the inverse square root of their covariance, from a
                # sample), so k-means stops cutting along the few directions
                # that carry most of the variance; the cluster descriptors
                # the queries are scored against are the group means in the
                # routing space itself, so the score keeps its meaning.
                from semblo.inference.flat_index import _segment_sum
                sample_n = min(length, max(int(kmeans_sample_size), 4 * index_dim))
                sel = torch.randperm(length, device=keys.device, generator=generator)[:sample_n]
                kf = keys.float()
                cov = (kf[sel].T @ kf[sel]) / max(1, sample_n - 1)
                evals, evecs = torch.linalg.eigh(cov)
                wmat = evecs @ torch.diag(evals.clamp_min(1e-4 * float(evals.max())).rsqrt()) @ evecs.T
                space_w = torch.nn.functional.normalize(kf @ wmat, dim=-1)
                cents_w = fit_flat_codebook(
                    space_w, num_clusters, num_iters=fit_iters,
                    sample_size=fit_sample, generator=generator, init=flat_init)
                assign = assign_flat_buckets(space_w, cents_w)
                counts_w = torch.bincount(assign, minlength=num_clusters)
                sums = _segment_sum(space.float(), assign, num_clusters, counts_w)
                cents = torch.nn.functional.normalize(
                    sums / counts_w.clamp_min(1)[:, None].float(), dim=-1).to(space.dtype)
                del space_w, cents_w, sums, kf
            elif _sink_tokens() > 0:
                # AN ATTENTION SINK IN THE FITTED PARTITION (SEMBLO_SINK_TOKENS=S).
                # The first S positions form cluster 0,
                # which the selectors take first inside the budget
                # (fused_prefill.force_sink_cluster); the other positions are
                # fitted into the remaining C-1 clusters, ids 1..C-1.  C does not
                # change, so no kernel's power-of-two assumption moves.  Cluster
                # 0's descriptor is zero: it is never scored into a selection and
                # never chosen by _nearest_cluster (suffix flush, token-outer
                # append), so no later key joins it.
                sink = _sink_tokens()
                _check_sink_partition(sink, length, fit_space.shape[0], num_clusters,
                                      min_size, max_size, hier_span, fit_prefix)
                rest = fit_flat_codebook(
                    fit_space[sink:], num_clusters - 1, num_iters=fit_iters,
                    sample_size=min(fit_sample, fit_space.shape[0] - sink),
                    generator=generator, init=flat_init, metric=flat_metric)
                cents = torch.cat([rest.new_zeros(1, rest.shape[1]), rest], dim=0)
                assign = torch.cat([
                    torch.zeros(sink, dtype=torch.long, device=space.device),
                    assign_flat_buckets(space[sink:], rest, metric=flat_metric).long() + 1])
                if kv_head == 0 and not _SAID.get("sink"):
                    _SAID["sink"] = True
                    print(f"[semblo] attention sink: positions 0..{sink - 1} form cluster 0, "
                          f"the other {length - sink:,} fill clusters 1..{num_clusters - 1} "
                          f"(SEMBLO_SINK_TOKENS={sink})", flush=True)
            else:
                cents = fit_flat_codebook(
                    fit_space, num_clusters, num_iters=fit_iters,
                    sample_size=fit_sample, generator=generator, init=flat_init,
                    metric=flat_metric)
                assign = assign_flat_buckets(space, cents, metric=flat_metric)
            page_clusters = int(os.environ.get("SEMBLO_PAGE_CLUSTERS", "0"))
            if page_clusters > 1:
                if min_size or max_size:
                    raise RuntimeError(
                        "SEMBLO_PAGE_CLUSTERS and the size split/merge cannot be "
                        "used together: the split reassigns individual tokens "
                        "and would break the page-constant assignment")
                assign = _collapse_assignment_to_pages(
                    space, cents, positions, page_clusters,
                    assign_flat_buckets)
            if min_size or max_size:
                assign, cents = split_merge_sizes(
                    assign, space, cents,
                    min_size=min_size or 1,
                    max_size=max_size or length,
                    generator=generator)
                # keep the bucket axis at a fixed width across heads
                pad = num_clusters - cents.shape[0]
                if pad > 0:
                    cents = torch.cat(
                        [cents, torch.zeros(pad, cents.shape[1],
                                            dtype=cents.dtype,
                                            device=cents.device)], dim=0)
                elif pad < 0:
                    raise RuntimeError(
                        f"split produced {cents.shape[0]} > {num_clusters} "
                        "clusters; raise the cluster budget")
            # Coarse level (SEMBLO_HIER_SPAN > 0).  The token PARTITION is
            # untouched -- this only renumbers the clusters so that every
            # group of `span` consecutive cluster ids is a semantic
            # neighbourhood, and records one descriptor per group.  The
            # scorer can then look at C/span groups first and open only the
            # surviving ones, which is what takes the selection cost off its
            # L^2 track.  Renumbering the clusters also renumbers the packed
            # KV (the posting order is built from `assign` right below), so
            # a group is one contiguous run of packed rows as well.
            if hier_span > 0 and os.environ.get(
                    "SEMBLO_ROUTING_NORMALIZE", "whole") not in ("half", ""):
                raise RuntimeError(
                    "SEMBLO_HIER_SPAN normalises its group descriptors per half; "
                    "it has not been ported to the other routing spaces")
            if hier_span > 0 and num_clusters % hier_span == 0 \
                    and num_clusters // hier_span >= 2:
                corder, gdesc = balanced_group_order(cents, hier_span)
                inv = torch.empty_like(corder)
                inv[corder] = torch.arange(num_clusters, device=corder.device)
                assign = inv[assign]
                cents = cents[corder]
                g0, g1 = gdesc.chunk(2, dim=-1)
                gdesc_h = torch.stack([
                    torch.nn.functional.normalize(g0.float(), dim=-1),
                    torch.nn.functional.normalize(g1.float(), dim=-1),
                ], dim=0)
                f0, f1 = cents.chunk(2, dim=-1)
                fdesc_h = torch.stack([
                    torch.nn.functional.normalize(f0.float(), dim=-1),
                    torch.nn.functional.normalize(f1.float(), dim=-1),
                ], dim=0)
                group_rows.append(gdesc_h.to(product_keys.dtype))
                radius_rows.append(
                    group_half_radii(fdesc_h, gdesc_h).to(torch.float32))
            order = torch.argsort(assign, stable=True)
            counts = torch.bincount(assign, minlength=num_clusters).to(
                torch.int32)
            if key_stats:
                print(f"[semblo] cluster stats layer {layer_idx} kv head {kv_head}: {num_clusters} clusters, "
                      f"largest {int(counts.max())} of {int(length)} tokens ({100.0 * float(counts.max()) / max(int(length), 1):.1f}%), "
                      f"empty {int((counts == 0).sum())}", flush=True)
            page_align = int(os.environ.get("SEMBLO_FA3_ALIGN", page_align))
            if page_align > 1:
                order, indptr, inv_row = _page_align_order(
                    order, counts, positions, page_align)
                inverse_rows.append(inv_row)
            else:
                indptr = torch.cat([
                    torch.zeros(1, dtype=torch.int32,
                                device=positions.device),
                    counts.cumsum(0).to(torch.int32),
                ])
            # The representative is stored split in halves because the kernel
            # scores q0.c0 + q1.c1, which is the plain dot product q.c.  How the
            # halves are normalised must match the query side (the trivial router
            # in model.py):
            #   whole  the whole vector unit-norm, then split (default)
            #   half   each half unit-norm
            #   none   the representative as it is: a pooled key is not a point
            #          on a sphere, and normalising it would undo the pooling that
            #          defines positional blocks.
            norm_mode = os.environ.get("SEMBLO_ROUTING_NORMALIZE", "whole") or "whole"
            if norm_mode == "none":
                unit = cents.float()
            else:
                # routing_space reads the same variable, so half, whole and
                # split:<n> all come out of one definition.
                unit = routing_space(cents)
            desc = torch.stack(list(unit.chunk(2, dim=-1)), dim=0).to(
                product_keys.dtype)
            codebooks.append(desc)
            if descriptor_weight is not None:
                # g_c = sum over the cluster's members of psi(z_i), where z_i
                # is the SAME half-normalised vector the partition was fitted
                # in.  Accumulated in float32: at 512K a cluster holds ~512
                # members and an exp activation makes the terms large.
                feats = _descriptor_features(
                    space, descriptor_weight[kv_head],
                    descriptor_bias[kv_head], descriptor_activation)
                sums = torch.zeros(num_clusters, feats.shape[-1],
                                   dtype=torch.float32, device=feats.device)
                sums.index_add_(0, assign, feats)
                descriptor_rows.append(sums)
            posting_rows.append(positions[order].to(torch.int32).contiguous())
            indptr_rows.append(indptr.contiguous())
            size_rows.append(counts.contiguous())

        sizes = torch.stack(size_rows, dim=0)
        indptr = torch.stack(indptr_rows, dim=0)
        if inverse_rows:
            # heads pad by different amounts (each has its own bucket sizes),
            # but the posting axis has to be one width; extend the short ones
            # with row 0, which lies inside no bucket's [start, start + size)
            width = max(int(r.numel()) for r in posting_rows)
            posting_rows = [
                r if r.numel() == width else torch.cat(
                    [r, r[:1].expand(width - r.numel())])
                for r in posting_rows
            ]
        slots = int((sizes > 0).sum(dim=-1).max().item())
        bucket_slot, slot_start, slot_bucket = _bucket_slot_table(
            sizes, indptr, slots)
        if report:
            torch.cuda.synchronize()
            print(f"INDEXBUILD flat C={num_clusters} "
                  f"sample={fit_sample} iters={fit_iters} metric={flat_metric} "
                  f"seconds={_time.perf_counter() - _t0:.3f}", flush=True)
        _report_index_sizes("flat", sizes)
        return cls(
            cluster_descriptors=(
                torch.stack(descriptor_rows, dim=0).contiguous()
                if descriptor_rows else None
            ),
            codebooks=torch.stack(codebooks, dim=0).contiguous(),
            posting_positions=torch.stack(posting_rows, dim=0).contiguous(),
            bucket_indptr=indptr.contiguous(),
            bucket_sizes=sizes.contiguous(),
            prompt_length=length,
            max_position=int(positions[-1].item()),
            num_non_empty_buckets=slots,
            bucket_slot=bucket_slot,
            slot_start=slot_start,
            slot_bucket=slot_bucket,
            flat_clusters=num_clusters,
            group_descriptors=(torch.stack(group_rows, dim=0).contiguous()
                               if group_rows else None),
            group_radii=(torch.stack(radius_rows, dim=0).contiguous()
                         if radius_rows else None),
            hier_span=hier_span if group_rows else 0,
            page_align=page_align if inverse_rows else 0,
            _inverse_positions=(torch.stack(inverse_rows, dim=0).contiguous()
                                if inverse_rows else None),
        )

    def pack_prompt_kv(
        self, key: torch.Tensor, value: torch.Tensor, *, pool=None
    ) -> torch.Tensor:
        """Store post-RoPE K/V in bucket order so a bucket is one interval.

        The kernel that reads the paged cache must translate every selected
        position through the block table, because a bucket's tokens sit in as
        many blocks as it has tokens.  In bucket order the same selection is a
        handful of long runs, which a kernel can read contiguously.

        These are the same values vLLM writes to its paged cache, so this is a
        second copy of K/V for the layers that build an index, not a different
        representation of them.
        """

        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        if key.ndim != 3 or key.shape[0] != self.prompt_length:
            raise ValueError("key and value must have shape [L,Hkv,D]")
        if key.shape[1] != self.num_kv_heads:
            raise ValueError("key and value KV-head count differs from the index")
        # Gather straight into the layout the body reads, [2, L, Hkv, D]:
        # indexing the posting list transposed avoids transposing a buffer
        # that is 2.15 GB per layer at 512K.
        #
        # The FA3 body views this buffer as [L/page, page, Hkv, D], which is
        # only a legal view if the token axis is a whole number of pages, so
        # it is allocated padded to the largest page size we might pick.  The
        # extra rows are inside no bucket and are zeroed.
        order = self.posting_positions.to(torch.int64).t().contiguous()
        packed_len, head_dim = order.shape[0], key.shape[-1]
        pad = (-packed_len) % _PACK_ALIGN
        shape = (2, packed_len + pad, order.shape[1], head_dim)
        if pool is None:
            packed = torch.empty(
                *shape, dtype=key.dtype, device=key.device)
        else:
            # The captured decode step replays this buffer's address, so
            # taking the previous request's saves re-capturing.  Written whole
            # below, so nothing of that request survives in it.
            packed = pool(shape, key.dtype, key.device)
        if pad:
            packed[:, packed_len:].zero_()
        idx = order.unsqueeze(-1).expand(-1, -1, head_dim)
        torch.gather(key, 0, idx, out=packed[0, :packed_len])
        torch.gather(value, 0, idx, out=packed[1, :packed_len])
        self.packed_kv = packed
        return packed

    def inverse_positions(self) -> torch.Tensor:
        """[Hkv, L] int32: original position -> row in the packed buffer.

        The FA3 body attends the chunk's own tokens with a dense, exactly
        causal call over their ORIGINAL order, which needs to undo the
        bucket-order permutation for that slice.
        """
        if self._inverse_positions is None:
            order = self.posting_positions.to(torch.int64)
            inv = torch.empty_like(order)
            ar = torch.arange(order.shape[1], device=order.device)
            inv.scatter_(1, order, ar.expand_as(order))
            self._inverse_positions = inv.to(torch.int32).contiguous()
        return self._inverse_positions

    def record_generated_positions(
        self, positions: torch.Tensor, *, check: bool = True
    ) -> None:
        """Record the generated suffix range without assigning it to buckets.

        The generated suffix is read exactly and is not assigned to buckets, so
        no bucket IDs are computed for decode tokens.

        ``check=False`` advances the host counter and does nothing else.  The
        check is about ``positions``, which every sparse layer in a step is
        handed identically, so the runtime state runs it once per step and
        tells the other layers to skip it -- see
        ``SembloRuntimeState.record_step_positions``.
        """

        if positions.ndim != 1:
            raise ValueError("decode positions must be one-dimensional")
        if positions.numel() == 0:
            return
        expected_start = self.max_position + 1
        if DEFERRED_CHECKS.off or not check:
            # The suffix is contiguous by construction; with checks off the
            # host counter advances and nothing touches the device.
            self.max_position = expected_start + positions.numel() - 1
            return
        if DEFERRED_CHECKS.active:
            # Four host reads per layer per decode step; the suffix is
            # contiguous by construction, so the check is what costs, not the
            # arithmetic.  Advance the host counter by the token count and let
            # the device confirm the shape.
            # The same run of positions this call is about to append, which
            # generated_positions already holds.
            self.max_position = expected_start + positions.numel() - 1
            expected = self.generated_positions(device=positions.device)[
                -positions.numel():
            ].to(positions.dtype)
            DEFERRED_CHECKS.record_violations(
                SLOT_DECODE_ORDER,
                positions != expected,
                message=(
                    "the generated suffix must append contiguously after the "
                    "indexed prefix"
                ),
            )
            return
        if int(positions[0].item()) <= self.max_position:
            raise ValueError("decode positions must append after the indexed prefix")
        if positions.numel() > 1 and bool((positions[1:] <= positions[:-1]).any()):
            raise ValueError("decode positions must be strictly increasing")
        if int(positions[0].item()) != expected_start:
            raise ValueError(
                f"generated suffix must be contiguous: expected {expected_start}, "
                f"got {int(positions[0].item())}"
            )
        self.max_position = int(positions[-1].item())

    def generated_suffix_tiles(self, *, num_kv_heads: int, tile_size: int):
        """Tiles over the generated suffix, built once and never rewritten.

        Nothing in them changes as the suffix grows, because the attention
        kernel already masks every token by position: it reads the token's
        logical position and keeps it only when it is at or before the query's
        own (``token_valid &= logical_pos <= q_position``, before the block
        table is read, so a tile past the end of the suffix touches nothing).
        A tile can therefore cover reserved-but-not-yet-generated ground from
        the start, with a full length and a fixed count, and the growing part
        of the suffix takes care of itself.

        This keeps the per-step host work of decode constant and leaves no
        per-step host value that would stop the decode step from being captured
        in a CUDA graph.

        The cost is that the kernel walks the reserved slots instead of the
        live ones.  The reservation is the next power of two above the suffix
        length with a floor of 64, so it walks fewer than twice as many tiles
        over a suffix that is at most a few hundred tokens.
        """

        length = self.max_position + 1 - self.prompt_length
        if length <= 0:
            raise ValueError("there is no generated suffix yet")
        buffer = self._suffix_position_buffer(length)
        capacity = int(buffer.numel())
        slots = -(-capacity // int(tile_size))
        cached = self.suffix_tiles
        if (
            cached is None
            or int(cached[0].shape[-1]) != slots
            or int(cached[4]) != int(tile_size)
            or int(cached[0].shape[1]) != int(num_kv_heads)
        ):
            device = buffer.device
            starts = (
                torch.arange(slots, device=device, dtype=torch.int32)
                * int(tile_size)
            ).view(1, 1, -1).expand(1, num_kv_heads, -1).contiguous()
            lengths = torch.full(
                (1, num_kv_heads, slots), int(tile_size),
                device=device, dtype=torch.int32,
            )
            counts = torch.full(
                (1, num_kv_heads), slots, dtype=torch.int32, device=device
            )
            # Only the deferred checks read this one; the kernel does not.
            selected = torch.full_like(counts, capacity)
            # Every KV head reads the same positions, so broadcast rather
            # than copy.
            posting = buffer.unsqueeze(0).expand(num_kv_heads, -1)
            cached = (starts, lengths, counts, selected, int(tile_size), posting)
            self.suffix_tiles = cached
        starts, lengths, counts, selected, _, posting = cached
        return posting, starts, lengths, counts, selected

    def prompt_bucket_sizes(self) -> torch.Tensor:
        """Return the immutable prompt bucket sizes used by sparse prompt lookup."""

        return self.bucket_sizes

    def generated_positions(self, *, device: torch.device | None = None) -> torch.Tensor:
        """Return the exact generated suffix positions.

        The suffix is attended densely and merged with the sparse prompt
        attention by LSE.  It is not added to the prompt index, so decode append
        is O(1) and the prompt index is never rebuilt.

        The positions run on from the prompt, so they are laid down once and
        sliced.
        """

        length = self.max_position + 1 - self.prompt_length
        if length <= 0:
            result = self.posting_positions.new_empty((0,), dtype=torch.int32)
            return result if device is None else result.to(device=device)
        home = self.posting_positions.device
        result = self._suffix_position_buffer(length)[:length]
        return result if device is None or device == home else result.to(device=device)

    def _suffix_position_buffer(self, length: int) -> torch.Tensor:
        """The reserved suffix positions, at least ``length`` of them.

        Reserved rather than exact: the tiles built over it never change as
        the suffix grows, so growing the reservation is the only event that
        rebuilds anything.  The next power of two above the length, floored at
        64, makes that happen a logarithmic number of times per request.
        """

        cached = self.suffix_positions
        if cached is None or int(cached.numel()) < length:
            capacity = max(64, 1 << (max(length, 1) - 1).bit_length())
            cached = torch.arange(
                self.prompt_length,
                self.prompt_length + capacity,
                dtype=torch.int32,
                device=self.posting_positions.device,
            )
            self.suffix_positions = cached
        return cached


_ENGAGE_ANNOUNCED: dict[str, bool] = {}


@dataclass
class SembloRuntimeState:
    """One-request state shared by all custom global-attention layers."""

    # Own store for the generated suffix (own_prompt_cache).
    #
    # With the sparse layers declaring no KV cache, vLLM allocates nothing for
    # them and the generated tokens have nowhere to go.  The prompt is served
    # from the cluster-ordered copy; only the suffix needs a home, and it is
    # small: at most max_model_len - prompt_length tokens.
    #
    # The reader addresses the cache by ABSOLUTE position -- it computes
    # position // block_size and looks that up in a block table -- so the
    # buffer is not indexed from zero.  It covers positions from the block
    # containing prompt_length onwards, and the block table maps that range
    # onto it while every earlier entry points at block 0, which the suffix
    # posting list never names.
    suffix_kv: dict[int, Any] = field(default_factory=dict)
    suffix_block_table: Any = None
    suffix_base_block: int = 0
    suffix_block_size: int = 256
    # vLLM's max_model_len, recorded when the layers are wrapped.  The
    # suffix store is sized from it, so it must not be guessed.
    max_model_len: int = 0

    source_indexes: dict[int, ProductPostingIndex] = field(default_factory=dict)
    source_modules: dict[int, Any] = field(default_factory=dict)
    reset_layer_idx: int | None = None
    prompt_epoch: int = 0
    # The first position of the step being processed.  Every layer needs it
    # to tell prefill from decode; keeping it on the host avoids a
    # synchronisation per layer.
    step_first_position: int = -1
    # Where the next forward is expected to start.  A batch-1 decode step is
    # always one token after the last, so once a prefill has anchored this the
    # decode path never has to read the positions tensor on the host; a
    # deferred device-side check still verifies the assumption each step.
    expected_next_position: int = -1
    # True for the whole of a request whose prompt is no longer than the
    # engage threshold: every sparse layer then runs the stock attention and
    # builds no index (see maybe_begin_prompt).
    dense_request: bool = False
    # host clock at the last prompt begin, for SEMBLO_PREFILL_WALL_LOG
    prompt_began: float = 0.0
    # The generated suffix belongs to the step, not to a layer: every sparse
    # layer is handed the same positions and derives the same length, tiles
    # and contiguity check from them, so they are built once per step and
    # shared.  Decode is bound by host operations.
    step_suffix_key: tuple | None = None
    step_suffix_tiles: Any = None
    step_checked_key: tuple | None = None
    # Buffers the memory-profiling pass allocates so that vLLM's KV-pool
    # sizing sees the packed prompt K/V a served request will need.  See
    # ``_reserve_packed_kv_for_profiling``.  Held until a request starts,
    # because the profiled peak has to have every sparse layer in it at once.
    profiling_reservation: dict[int, torch.Tensor] = field(default_factory=dict)
    # Engine-lifetime buffers for the parts of a request's index that the
    # captured decode step reads.  A capture replays the addresses it was
    # taken with, so reusing the previous request's buffers keeps the
    # capture.  Keyed by shape: a request of a different length gets its
    # own buffer and its own capture.  Not cleared by reset().
    index_buffers: dict = field(default_factory=dict)

    def ensure_suffix_store(self, layer_idx, *, prompt_length, max_model_len,
                            num_kv_heads, head_dim, dtype, device):
        """Allocate this layer's suffix store, sized by what can still be generated.

        Capacity is not a constant to pick: a request can generate at most
        ``max_model_len - prompt_length`` tokens, and both are known once the
        prompt has been seen.  Sizing it that way also keeps the total this
        method holds -- cluster-ordered copy plus suffix -- at one
        max_model_len worth of K/V, which is what vLLM would have allocated.
        """

        import torch

        block = int(self.suffix_block_size)
        base = int(prompt_length) // block
        n_blocks = max(1, (int(max_model_len) + block - 1) // block - base)
        if self.suffix_block_table is None or self.suffix_base_block != base:
            total = base + n_blocks
            table = torch.zeros(1, total, dtype=torch.int32, device=device)
            table[0, base:] = torch.arange(n_blocks, dtype=torch.int32,
                                           device=device)
            self.suffix_block_table = table
            self.suffix_base_block = base
            self.suffix_kv.clear()
        buf = self.suffix_kv.get(int(layer_idx))
        if buf is None:
            buf = torch.zeros(2, n_blocks, block, num_kv_heads, head_dim,
                              dtype=dtype, device=device)
            self.suffix_kv[int(layer_idx)] = buf
        return buf

    def write_suffix_kv(self, layer_idx, positions, key, value) -> None:
        """Place one step's K/V at its own absolute position."""

        buf = self.suffix_kv[int(layer_idx)]
        block = int(self.suffix_block_size)
        rel = positions.to(torch.int64) - self.suffix_base_block * block
        b, off = rel // block, rel % block
        buf[0, b, off] = key.to(buf.dtype)
        buf[1, b, off] = value.to(buf.dtype)

    def pooled_like(self, key: str, like: torch.Tensor) -> torch.Tensor:
        """A buffer with ``like``'s shape and dtype, the same one each time."""

        return self.pooled_buffer(key, tuple(like.shape), like.dtype,
                                  like.device)

    def pooled_buffer(self, key, shape, dtype, device) -> torch.Tensor:
        """The engine-lifetime buffer for ``key``, made on first use.

        A shape or dtype change makes a new one, which is correct -- the old
        capture could not have used it either -- and is why a workload of
        varying prompt lengths gets no benefit from this.
        """

        signature = (tuple(shape), dtype, device)
        buffer = self.index_buffers.get(key)
        if buffer is None or self.index_buffers.get((key, "sig")) != signature:
            buffer = torch.empty(*shape, dtype=dtype, device=device)
            self.index_buffers[key] = buffer
            self.index_buffers[(key, "sig")] = signature
        return buffer

    def adopt_index_buffers(self, source_layer_idx: int, index) -> None:
        """Move an index's small arrays into engine-lifetime buffers.

        The packed K/V is not here: it is gigabytes and is written straight
        into its pooled buffer by ``pack_prompt_kv``.  These are the rest of
        what the captured decode step reads -- tens of megabytes at most -- so
        copying them once a request is cheaper than the capture it saves.
        """

        for name in ("posting_positions", "bucket_sizes", "bucket_indptr",
                     "codebooks"):
            current = getattr(index, name, None)
            if not isinstance(current, torch.Tensor):
                continue
            buffer = self.pooled_like(f"{source_layer_idx}:{name}", current)
            if buffer.data_ptr() == current.data_ptr():
                continue
            buffer.copy_(current)
            setattr(index, name, buffer)

    def reset(self) -> None:
        self.dense_request = False
        self.source_indexes.clear()
        self.profiling_reservation.clear()
        self.step_suffix_key = None
        self.step_suffix_tiles = None
        self.step_checked_key = None
        self.prompt_epoch += 1
        self.prompt_began = time.perf_counter()
        if os.environ.get("SEMBLO_MEM_LOG") == "1" and torch.cuda.is_available():
            # One line per prompt with the allocator's allocated and reserved
            # memory when a new prompt begins.
            import sys
            gib = float(1 << 30)
            free, total = torch.cuda.mem_get_info()
            print(f"[mem] prompt={self.prompt_epoch} "
                  f"allocated={torch.cuda.memory_allocated() / gib:.2f} "
                  f"reserved={torch.cuda.memory_reserved() / gib:.2f} "
                  f"max_allocated={torch.cuda.max_memory_allocated() / gib:.2f} "
                  f"max_reserved={torch.cuda.max_memory_reserved() / gib:.2f} "
                  f"device_free={free / gib:.2f}", file=sys.stderr, flush=True)
        if os.environ.get("SEMBLO_EMPTY_CACHE_AT_RESET", "1") == "1" and torch.cuda.is_available():
            # Hand the cached-but-free blocks back to the driver between
            # prompts.  Prompts of varying length otherwise leave `reserved`
            # growing while `allocated` stays flat.  Costs one synchronisation
            # per prompt; SEMBLO_EMPTY_CACHE_AT_RESET=0 keeps the blocks.
            torch.cuda.empty_cache()

    def record_step_positions(
        self, index: ProductPostingIndex, positions: torch.Tensor
    ) -> None:
        """Append this step's positions to ``index``, checking them once.

        The check verifies a property of ``positions`` -- that the suffix
        appends contiguously after the indexed prefix -- and every layer in a
        step receives the same tensor, so running it for the first layer
        verifies exactly what running it for all eight would.
        """

        key = (self.prompt_epoch, self.step_first_position)
        index.record_generated_positions(
            positions, check=self.step_checked_key != key
        )
        self.step_checked_key = key

    def suffix_tiles_for_step(
        self, index: ProductPostingIndex, *, num_kv_heads: int,
        tile_size: int, pool: bool = False,
    ):
        """The generated-suffix tiles of the current step, built once.

        The tiles depend on the suffix length, the head count and the tile
        size, none of which differ between the sparse layers of a step.  The
        buffers belong to whichever index asked first and the rest only read
        them.
        """

        key = (self.prompt_epoch, self.step_first_position, num_kv_heads, tile_size)
        if self.step_suffix_key != key:
            tiles = index.generated_suffix_tiles(
                num_kv_heads=num_kv_heads, tile_size=tile_size
            )
            if pool:
                # The tiles live on the index, so a new request rebuilds them
                # at new addresses and the captured decode step -- which reads
                # them -- has to be taken again.  They are kilobytes, so
                # copying them into engine-lifetime buffers is free next to
                # the capture it saves.
                names = ("suffix_posting", "suffix_starts", "suffix_lengths",
                         "suffix_counts", "suffix_selected")
                moved = []
                for name, tensor in zip(names, tiles):
                    buffer = self.pooled_like(name, tensor)
                    if buffer.data_ptr() != tensor.data_ptr():
                        buffer.copy_(tensor)
                    moved.append(buffer)
                tiles = tuple(moved)
            self.step_suffix_tiles = tiles
            self.step_suffix_key = key
        return self.step_suffix_tiles

    def register_source_module(self, source_layer_idx: int, module: Any) -> None:
        source_layer_idx = int(source_layer_idx)
        if source_layer_idx in self.source_modules:
            raise ValueError(f"duplicate semblo source layer {source_layer_idx}")
        self.source_modules[source_layer_idx] = module

    def get_source_module(self, source_layer_idx: int) -> Any:
        try:
            return self.source_modules[int(source_layer_idx)]
        except KeyError as exc:
            raise RuntimeError(
                f"semblo source module {source_layer_idx} is unavailable"
            ) from exc

    def maybe_begin_prompt(self, layer_idx: int, positions: torch.Tensor,
                           engage_above_tokens: int = 0) -> None:
        """Reset request-local indexes exactly once at a new batch-1 prompt.

        Only the earliest configured semblo layer is allowed to reset
        state.  Every later source layer then builds its own request-local
        index during the same forward pass.

        ``engage_above_tokens``: a prompt of at most this many tokens is
        served densely for the whole request (``dense_request``); 0 engages
        the sparse path at every length.
        """

        if self.reset_layer_idx is None or int(layer_idx) != int(self.reset_layer_idx):
            return
        if positions.ndim != 1 or positions.numel() == 0:
            raise ValueError("positions must be a non-empty one-dimensional tensor")
        is_tracked_decode = (
            positions.numel() == 1 and self.expected_next_position > 0
        )
        if is_tracked_decode:
            # A batch-1 decode step follows the previous one by exactly one
            # position, so the host counter already knows where it starts.
            # Reading positions[0] here was one synchronisation per token,
            # and after it the host had to re-dispatch the whole step with
            # the device idle.  Contiguity stays verified on the device: the
            # suffix append (record_generated_positions) folds a positions-
            # equal-expected check into SLOT_DECODE_ORDER every layer.
            self.step_first_position = self.expected_next_position
        else:
            self.step_first_position = int(positions[0].item())
        self.expected_next_position = self.step_first_position + positions.numel()
        # This runs once per forward pass, which is where the search path's
        # deferred checks are read.  Decode reads them without a stall (the
        # asynchronous copy surfaces a failure a few steps late); everything
        # else keeps the synchronous read.
        if positions.is_cuda and not DEFERRED_CHECKS.off:
            if not DEFERRED_CHECKS.active:
                DEFERRED_CHECKS.enable(positions.device)
            elif is_tracked_decode:
                DEFERRED_CHECKS.check_async()
            else:
                DEFERRED_CHECKS.check_and_reset()
        if self.step_first_position == 0:
            expected = torch.arange(
                positions.numel(), device=positions.device, dtype=positions.dtype
            )
            if not torch.equal(positions, expected):
                raise RuntimeError(
                    "the initial vLLM integration requires the full prompt in one "
                    "contiguous prefill call"
                )
            self.reset()
            length = positions.numel()
            self.dense_request = 0 < engage_above_tokens and length <= engage_above_tokens
            if self.dense_request != _ENGAGE_ANNOUNCED.get("dense"):
                # logged once per process per direction
                _ENGAGE_ANNOUNCED["dense"] = self.dense_request
                state = (f"not engaged (stock attention; threshold {engage_above_tokens})"
                         if self.dense_request else "engaged")
                print(f"[semblo] prompt of {length} tokens: sparse path {state}", flush=True)

    def put(self, source_layer_idx: int, index: ProductPostingIndex) -> None:
        source_layer_idx = int(source_layer_idx)
        if source_layer_idx in self.source_indexes:
            raise RuntimeError(
                f"semblo prompt index for source layer {source_layer_idx} "
                "was built more than once"
            )
        self.source_indexes[source_layer_idx] = index

    def get(self, source_layer_idx: int) -> ProductPostingIndex:
        try:
            return self.source_indexes[int(source_layer_idx)]
        except KeyError as exc:
            raise RuntimeError(
                f"semblo index for source layer {source_layer_idx} is unavailable"
            ) from exc
