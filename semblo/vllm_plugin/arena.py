"""Fixed-address decode state for vLLM's FULL cudagraph (``SEMBLO_FULL_GRAPH_DECODE=1``).

vLLM captures the decode step of a batch size once, at engine warm-up, with
dummy inputs, and replays that graph for every later request.  Everything the
sparse layer's decode body reads must therefore sit at addresses that do not
change between requests: the index (cluster descriptors, posting lists, the
cluster-ordered K/V copy), the generated-suffix positions and tiles, and the
workspaces.  This module keeps one such arena per source layer, shaped for the
configured token capacity, and copies a request's freshly built index into it
at the end of the request's prefill.  The decode body then runs over the arena
whether the step is being captured, replayed, or run eagerly.

Consequences:

* The kernels see the arena's cluster count and prompt capacity, not the
  request's.  Clusters beyond the request's own C have size 0, an empty
  indptr range and a zero descriptor, so they contribute nothing to the union
  and do not affect the token budget.
* The captured graph is the sparse decode step.  A request below the engage
  threshold cannot fall back to stock attention inside it, so this mode
  engages the index at every length.  The configuration check enforces that.
* Under ``own_prompt_cache`` the generated suffix has no vLLM cache either,
  and the per-request suffix store is allocated at prefill, after vLLM
  captures.  The arena therefore owns a fixed-address suffix store the size of
  the reservation and a block table that maps absolute positions onto it.  The
  table is rewritten on the device, outside the graph, when a request's index
  is adopted, and the decode step's K/V write goes through it.
"""
from __future__ import annotations

import os

import torch

from semblo.vllm_plugin.state import ProductPostingIndex, _PACK_ALIGN


def full_graph_decode_enabled() -> bool:
    return os.environ.get("SEMBLO_FULL_GRAPH_DECODE") == "1"


def arena_cluster_count(capacity: int, unit_tokens: int, min_clusters_log2: int) -> int:
    """The flat index's cluster rule evaluated at the capacity (state._build_flat)."""
    from semblo.inference.flat_index import flat_cluster_count
    raw_c = flat_cluster_count(int(capacity), int(unit_tokens))
    num_clusters = 1 << max(int(min_clusters_log2), (max(raw_c, 1) - 1).bit_length())
    return int(min(num_clusters, 1 << (int(capacity).bit_length() - 1)))


def suffix_reserve(room: int | None = None) -> int:
    """Positions reserved for the generated suffix; the tiles over it are static.

    SEMBLO_SUFFIX_RESERVE sets it outright.  Otherwise it is the room the model
    has for generated tokens, max_model_len minus the prompt capacity, capped
    at 65,536.  A reserve past that room can never be reached and only costs
    memory.
    """
    env = os.environ.get("SEMBLO_SUFFIX_RESERVE")
    if env is not None:
        return int(env)
    if room is not None:
        return max(1, min(65536, int(room)))
    return 65536


class DecodeArena:
    """Capacity-shaped buffers for one source layer, and an index object over them."""

    def __init__(self, state, layer: int, *, capacity: int, num_kv_heads: int,
                 head_dim: int, index_dim: int, num_clusters: int, page_align: int,
                 kv_dtype: torch.dtype, desc_dtype: torch.dtype, device: torch.device,
                 suffix_block: int = 256, packed_capacity: int | None = None,
                 table_positions: int | None = None, keep_kpre: bool = False,
                 reserve: int | None = None) -> None:
        self.state = state
        self.layer = int(layer)
        self.capacity = int(capacity)
        self.num_clusters = int(num_clusters)
        self.reserve = int(reserve) if reserve is not None else suffix_reserve()
        self.suffix_block = int(suffix_block)
        # The flat build pads every cluster's posting list to the FA3 page
        # alignment (state._page_align_order), so the posting lists are up to
        # C x (align - 1) longer than the prompt.  With the suffix flush
        # (suffix_flush.py) the packed store also receives the generated
        # tokens moved into the clusters, so it is sized for every position
        # the model can hold (packed_capacity).
        self.packed_capacity = int(packed_capacity if packed_capacity is not None else capacity)
        self.page_align = max(int(page_align), 1)
        width = self.packed_capacity + self.num_clusters * self.page_align
        self.width = width + (-width) % _PACK_ALIGN
        key = lambda name: f"arena:{self.layer}:{name}"  # noqa: E731
        pb = state.pooled_buffer
        self.posting = pb(key("posting_positions"), (num_kv_heads, self.width), torch.int32, device)
        self.sizes = pb(key("bucket_sizes"), (num_kv_heads, self.num_clusters), torch.int32, device)
        self.indptr = pb(key("bucket_indptr"), (num_kv_heads, self.num_clusters + 1), torch.int32, device)
        self.codebooks = pb(key("codebooks"), (num_kv_heads, 2, self.num_clusters, index_dim // 2), desc_dtype, device)
        self.packed_kv = pb(key("packed_kv"), (2, self.width, num_kv_heads, head_dim), kv_dtype, device)
        self.suffix_positions = pb(key("suffix_positions"), (self.reserve,), torch.int32, device)
        if not state.index_buffers.get(key("ready")):
            # A one-token index: cluster 0 of every head holds position 0 and
            # the other clusters are empty.  Warm-up and capture then run
            # every kernel over non-empty work and read finite (zero) K/V.
            # The content is replaced at the end of a request's prefill.
            for t in (self.posting, self.sizes, self.indptr, self.codebooks, self.packed_kv):
                t.zero_()
            self.sizes[:, 0] = 1
            self.indptr[:, 1:] = 1
            self.suffix_positions.copy_(torch.arange(self.capacity, self.capacity + self.reserve,
                                                     dtype=torch.int32, device=device))
            state.index_buffers[key("ready")] = True
        self.index = ProductPostingIndex(
            codebooks=self.codebooks, posting_positions=self.posting,
            bucket_indptr=self.indptr, bucket_sizes=self.sizes,
            prompt_length=self.capacity, max_position=self.capacity,
            num_non_empty_buckets=self.num_clusters, flat_clusters=self.num_clusters,
        )
        self.index.packed_kv = self.packed_kv
        self.index.suffix_positions = self.suffix_positions
        # the prompt length as a device scalar: the decode step derives the
        # live suffix-tile count from it without a host read
        self.prompt_length_t = pb(key("prompt_length"), (1,), torch.int32, device)
        if not state.index_buffers.get(key("plen_ready")):
            self.prompt_length_t.fill_(self.capacity)
            state.index_buffers[key("plen_ready")] = True
        # The generated suffix's own store, read only under own_prompt_cache
        # (about 135 MB a layer at the default reservation).  The suffix
        # kernel addresses a cache by absolute position through a block
        # table, so the store holds whole blocks starting from the one that
        # contains the prompt's end, one block more than the reservation.
        # One table serves every layer, since they share the prompt length.
        self.suffix_blocks = self.reserve // self.suffix_block + 1
        self.suffix_kv = pb(key("suffix_kv"),
                            (2, self.suffix_blocks, self.suffix_block, num_kv_heads, head_dim),
                            kv_dtype, device)
        span = int(table_positions if table_positions is not None else self.capacity + self.reserve)
        table_len = -(-span // self.suffix_block) + 1
        self.suffix_table = pb("arena:suffix_block_table", (1, table_len), torch.int32, device)
        # pre-RoPE keys of the generated tokens, kept only for the suffix
        # flush, which assigns them to clusters
        self.suffix_kpre = (pb(key("suffix_kpre"), (self.suffix_blocks, self.suffix_block, num_kv_heads, head_dim),
                               kv_dtype, device) if keep_kpre else None)
        if not state.index_buffers.get(key("suffix_ready")):
            self.suffix_kv.zero_()
            if self.suffix_kpre is not None:
                self.suffix_kpre.zero_()
            state.index_buffers[key("suffix_ready")] = True
        self.adopted_epoch = -1
        self.install_suffix_store(self.capacity)

    def packed_rows(self, rows: int, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """The first ``rows`` rows of the arena's packed K/V, for a request to pack into.

        A view, so the prefill's gather writes the arena directly and the
        adoption copies nothing.  Every reader takes the strides explicitly
        or indexes ``[0]`` / ``[1]`` (each a contiguous slab), so the view's
        leading stride (the arena's width) does not matter."""
        expect = (2, int(rows)) + tuple(self.packed_kv.shape[2:])
        if tuple(shape) != expect or dtype != self.packed_kv.dtype:
            raise RuntimeError(f"the packed K/V asked for {tuple(shape)} {dtype}; the arena for layer "
                               f"{self.layer} holds {tuple(self.packed_kv.shape)} {self.packed_kv.dtype}")
        if int(rows) > self.width:
            raise RuntimeError(f"packed K/V of {rows} rows does not fit the arena's {self.width}")
        return self.packed_kv[:, :int(rows)]

    def install_suffix_store(self, length: int) -> None:
        """Map positions from ``length`` on onto this layer's suffix store.

        Sets the state's suffix store fields to the arena's buffers, so the
        reader (``_source_paged_context``) finds them at warm-up, at prefill
        and at every decode step; the block table's contents follow the
        prompt length, its address never moves."""
        state = self.state
        base = int(length) // self.suffix_block
        if state.index_buffers.get("arena:suffix_base") != base:
            table = self.suffix_table
            table.zero_()
            n = max(0, min(self.suffix_blocks, int(table.shape[1]) - base))
            table[0, base:base + n] = torch.arange(n, dtype=torch.int32, device=table.device)
            state.index_buffers["arena:suffix_base"] = base
            state.index_buffers["arena:suffix_table_filled"] = base + n
        state.suffix_block_table = self.suffix_table
        state.suffix_base_block = base
        state.suffix_kv[self.layer] = self.suffix_kv

    def extend_suffix_table(self, upto_position: int) -> None:
        """Map every block up to ``upto_position`` onto the suffix store as a ring.

        Block b of positions maps to physical block (b - base) mod NBLK, which
        the initial install (``table[base + i] = i``) already satisfies.  The
        suffix flush frees the oldest blocks and this hands them to the blocks
        that follow.  Valid while the exact suffix fits in the reserve."""
        state = self.state
        base = int(state.index_buffers.get("arena:suffix_base", 0))
        filled = int(state.index_buffers.get("arena:suffix_table_filled", base))
        last = int(upto_position) // self.suffix_block
        table = self.suffix_table
        if last >= int(table.shape[1]):
            raise RuntimeError(f"position {upto_position} is beyond the suffix block table ({int(table.shape[1])} blocks)")
        if last >= filled:
            blocks = torch.arange(filled, last + 1, device=table.device)
            table[0, filled:last + 1] = ((blocks - base) % self.suffix_blocks).to(torch.int32)
            state.index_buffers["arena:suffix_table_filled"] = last + 1

    def write_suffix(self, positions: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> None:
        """One generated token's K/V into the store, through the block table.

        Device arithmetic only (``state.write_suffix_kv`` subtracts a host
        integer, which a captured graph would freeze)."""
        block = self.suffix_block
        pos = positions.to(torch.int64)
        physical = self.suffix_table[0, pos // block].to(torch.int64)
        offset = pos % block
        self.suffix_kv[0, physical, offset] = key.to(self.suffix_kv.dtype)
        self.suffix_kv[1, physical, offset] = value.to(self.suffix_kv.dtype)

    def adopt(self, index: ProductPostingIndex, epoch: int) -> ProductPostingIndex:
        """Copy a request's index into the arena; the arena index is what decode reads."""
        if self.adopted_epoch == epoch:
            return self.index
        length = int(index.prompt_length)
        clusters = int(index.num_buckets)
        width = int(index.posting_positions.shape[1])          # prompt plus the page-alignment padding
        if length > self.capacity or clusters > self.num_clusters or width > self.width:
            raise RuntimeError(
                f"full-graph decode arena for layer {self.layer} holds {self.capacity} tokens "
                f"({self.width} posting slots) and {self.num_clusters} clusters; the request has "
                f"{length} tokens ({width} slots) and {clusters}")
        self.posting[:, :width].copy_(index.posting_positions)
        self.sizes.zero_()
        self.sizes[:, :clusters].copy_(index.bucket_sizes)
        self.indptr[:, :clusters + 1].copy_(index.bucket_indptr)
        if clusters < self.num_clusters:
            # empty tail clusters: every indptr past the last real one is the end
            self.indptr[:, clusters + 1:].copy_(index.bucket_indptr[:, -1:].expand(-1, self.num_clusters - clusters))
        self.codebooks.zero_()
        self.codebooks[:, :, :clusters].copy_(index.codebooks)
        if index.packed_kv is None:
            raise RuntimeError("full-graph decode needs the cluster-ordered K/V copy (pack_prompt_kv)")
        packed_len = int(index.packed_kv.shape[1])
        if packed_len > self.width:
            raise RuntimeError(f"packed K/V of {packed_len} rows does not fit the arena's {self.width}")
        if index.packed_kv.data_ptr() != self.packed_kv.data_ptr():
            # the request packed into its own buffer rather than the arena's rows
            self.packed_kv[:, :packed_len].copy_(index.packed_kv)
        self.suffix_positions.copy_(torch.arange(length, length + self.reserve, dtype=torch.int32,
                                                 device=self.suffix_positions.device))
        self.index.prompt_length = length
        self.prompt_length_t.fill_(length)
        self.index.max_position = length          # one generated position: the tiles want a suffix
        # suffix_tiles stay: they were built over the reservation, which does not change
        self.install_suffix_store(length)
        self.adopted_epoch = epoch
        return self.index


def arena_for(state, layer: int, *, capacity: int, unit_tokens: int, min_clusters_log2: int,
              num_kv_heads: int, head_dim: int, index_dim: int, page_align: int,
              kv_dtype: torch.dtype, desc_dtype: torch.dtype, device: torch.device,
              packed_capacity: int | None = None, table_positions: int | None = None,
              keep_kpre: bool = False, reserve: int | None = None) -> DecodeArena:
    """The layer's arena, made on first use (before vLLM captures)."""
    key = ("arena", int(layer))
    arena = state.index_buffers.get(key)
    if arena is None:
        arena = DecodeArena(
            state, layer, capacity=capacity, num_kv_heads=num_kv_heads, head_dim=head_dim,
            index_dim=index_dim,
            num_clusters=arena_cluster_count(capacity, unit_tokens, min_clusters_log2),
            page_align=page_align, kv_dtype=kv_dtype, desc_dtype=desc_dtype, device=device,
            suffix_block=int(getattr(state, "suffix_block_size", 256)),
            packed_capacity=packed_capacity, table_positions=table_positions, keep_kpre=keep_kpre,
            reserve=reserve)
        state.index_buffers[key] = arena
    return arena
