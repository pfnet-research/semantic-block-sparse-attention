"""Token-order streamed prefill with blocks built from the start of the prompt.

The layer-wise streamed prefill (streamed_prefill.py) keeps the residual stream
of the whole prompt in memory, because each layer's blocks are built from all
of its keys. This schedule instead processes the prompt like ordinary chunked
prefill, one slice through every layer before the next slice, so activations
are slice-sized. In each sparse layer:

  dense      while a slice starts inside the densely read start of the prompt
             (chunk start <= x * K), it attends exactly over K/V staged in
             prompt order;
  fit        at the first slice past the fit point, the centroids are fitted
             on the staged keys, those keys are assigned, and the block-ordered
             store is laid out with a reservation per block sized from the
             observed proportions, so every block stays one contiguous run;
  append     every later slice assigns its keys to the fixed centroids, appends
             them to their blocks' runs, and attends through the sparse body,
             with its own tokens read through the exact causal window;
  finish     the slot tables are refreshed and the index moves into the decode
             arena as after a layer-wise prefill.

A block that outgrows its reservation triggers an in-place repack from the
current proportions. Enabled by SEMBLO_PREFILL_ORDER=token with
SEMBLO_PREFILL_CHUNK <= the query chunk.
"""
from __future__ import annotations

import os

import torch

from semblo.vllm_plugin import arena as _arena
from semblo.vllm_plugin.state import _bucket_slot_table
from semblo.vllm_plugin.suffix_flush import _nearest_cluster

_SAID: dict[str, bool] = {}


def reserved_layout(sizes_ref: torch.Tensor, n_ref: int, total_tokens: int,
                    width: int, align: int) -> torch.Tensor:
    """Rows reserved per block, [Hkv, C] int64, multiples of ``align``, summing to at most ``width``.

    Each block gets its share of ``total_tokens`` in the proportion it had
    among the ``n_ref`` reference keys, rounded up to whole pages and at least
    one page, since an empty block may still receive keys. The remaining rows
    are spread in the same proportion.
    """
    align = max(int(align), 1)
    sizes = sizes_ref.to(torch.float64)
    hkv, c = sizes.shape
    want = sizes * (float(total_tokens) / float(max(int(n_ref), 1)))
    base = torch.ceil(want / align).to(torch.int64).clamp_min(1) * align
    for h in range(hkv):
        spare = int(width) - int(base[h].sum())
        if spare < 0:
            raise RuntimeError(
                f"the reserved layout needs {int(base[h].sum())} rows for KV head {h}; "
                f"the store holds {width}")
        if spare >= align:
            share = want[h] / max(float(want[h].sum()), 1.0)
            extra = torch.floor(share * (spare // align)).to(torch.int64) * align
            base[h] += extra
    return base


def append_rows(ids: torch.Tensor, indptr: torch.Tensor, sizes: torch.Tensor):
    """Destination rows [Hkv, n] of a slice's keys, and per-block counts [Hkv, C].

    ``ids`` [Hkv, n] is the block of each new key in position order,
    ``indptr`` [Hkv, C+1] the start of each run and ``sizes`` [Hkv, C] the
    keys already in each run. New keys follow the run's last key in position
    order.
    """
    hkv, n = ids.shape
    c = int(sizes.shape[1])
    ids = ids.to(torch.int64)
    counts = torch.zeros(hkv, c, dtype=torch.int64, device=ids.device)
    counts.scatter_add_(1, ids, torch.ones_like(ids))
    order = torch.argsort(ids, dim=1, stable=True)
    sorted_ids = torch.gather(ids, 1, order)
    starts = torch.cumsum(counts, 1) - counts
    rank_sorted = torch.arange(n, device=ids.device)[None, :] - torch.gather(starts, 1, sorted_ids)
    rank = torch.empty_like(rank_sorted)
    rank.scatter_(1, order, rank_sorted)
    rows = (torch.gather(indptr[:, :-1].to(torch.int64), 1, ids)
            + torch.gather(sizes.to(torch.int64), 1, ids) + rank)
    return rows, counts


class TokenOuterSnapshot:
    """The snapshot the fused chunk body reads under this schedule.

    ``sizes`` are the causally visible rows of every block at the start of
    the slice being attended, i.e. the keys appended by earlier slices. The
    slice's own keys are already in the store but not counted here. The body
    attends them through its exact causal window.
    """

    def __init__(self, index) -> None:
        self.length = int(index.prompt_length)
        self.b_dim = int(index.num_buckets)
        self.sizes = torch.zeros_like(index.bucket_sizes)
        self.chunk_start = 0
        self.next_pos = 0
        self.packed_slot_bucket = None
        self.orig_bucket = None
        self.bitmask = None
        self.union_idx = None
        self.slot_rowmask = None
        self.qmask = None
        self.b2u = None
        self.num_units = 0
        self.kv8 = None
        self.kv_scales = None

    def advance(self, last_pos: int) -> torch.Tensor:
        if int(last_pos) + 1 != self.chunk_start:
            raise RuntimeError(
                f"the token-outer snapshot holds the counts before position {self.chunk_start}; "
                f"the body asked for position {last_pos}")
        return self.sizes

    def block_sizes(self, chunk_start: int, chunk_end: int, g_union: int) -> torch.Tensor:
        raise RuntimeError("the token-outer prefill serves the FA3 cluster-outer body only")


class TokenOuterPrefillSession:
    """One sparse layer's share of a token-outer prefill."""

    def __init__(self, layer, positions: torch.Tensor, chunk: int) -> None:
        self.layer = layer
        cfg = layer.runtime_config
        self.rope_positions, self.flat = layer._positions_for_runtime(positions)
        self.tokens = int(self.flat.numel())
        self.chunk = int(chunk)
        if self.chunk > int(cfg.query_chunk_size):
            raise RuntimeError(
                f"SEMBLO_PREFILL_CHUNK={self.chunk} must not exceed the query chunk "
                f"({cfg.query_chunk_size}): a slice's keys are appended once, before its queries")
        if int(cfg.dense_tail_tokens):
            raise RuntimeError("the token-outer prefill does not serve a dense tail")
        if os.environ.get("SEMBLO_ROUTING_CENTER") == "1":
            raise RuntimeError("the token-outer prefill does not centre keys")
        if (os.environ.get("SEMBLO_ROUTING_NORMALIZE", "whole") or "whole") != "whole":
            raise RuntimeError("the token-outer prefill assigns in the whole-vector routing space")
        if os.environ.get("SEMBLO_PREFIX_INDEX_DOUBLING", "0") == "1":
            raise RuntimeError("the token-outer prefill has its own prefix index")
        if os.environ.get("SEMBLO_KV_OUTER", "1") != "1" or os.environ.get("SEMBLO_FA3_BODY", "16") == "0":
            raise RuntimeError("the token-outer prefill serves the FA3 cluster-outer body only")
        self.profiling = layer._is_memory_profiling_run()
        x = int(os.environ.get("SEMBLO_FA3_DENSE_PREFIX_X", "4"))
        cap = x * int(layer._target_for_step(False))
        # Slices whose start is <= cap are read densely. The fit happens
        # earlier, at the first slice past SEMBLO_TOKEN_OUTER_FIT_TOKENS
        # (default 65,536) positions. Chunks between the fit and the cap are
        # still read densely by the fused body through the inverse
        # permutation. Staging memory therefore scales with the fit point,
        # not with K.
        fit_tokens = int(os.environ.get("SEMBLO_TOKEN_OUTER_FIT_TOKENS", "65536"))
        fit_tokens = min(fit_tokens, cap) if fit_tokens > 0 else cap
        self.fit_at = min(self.tokens, ((fit_tokens // self.chunk) + 1) * self.chunk)
        self.stage_k = self.stage_v = self.stage_kpre = None
        self.index = None
        self.snapshot = None
        self.reserve = None
        self.repacks = 0
        if not self.profiling:
            # the runtime wants the flat row, not the M-RoPE grid
            layer.runtime_state.maybe_begin_prompt(layer.layer_idx, self.flat)

    # -- per slice ------------------------------------------------------------

    def slice(self, hidden_states: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        layer = self.layer
        rope = layer._slice_positions(self.rope_positions, start, stop)
        q_post, k_post, v_states, q_pre, k_pre, gate = layer._project_qkv(rope, hidden_states)
        if k_post is None or v_states is None or k_pre is None:
            raise RuntimeError("the token-outer prefill needs a layer with its own K/V")
        if self.profiling:
            if start == 0:
                layer._reserve_packed_kv_for_profiling(k_post)
            attn = torch.zeros_like(q_post)
            return layer._finish_output(attn.view(hidden_states.shape[0], -1), gate)
        hkv, d = layer.num_kv_heads, layer.head_dim
        k3, v3, kp3 = (k_post.view(-1, hkv, d), v_states.view(-1, hkv, d), k_pre.view(-1, hkv, d))
        with torch.profiler.record_function("semblo::attention"):
            if self.index is None and start >= self.fit_at:
                self._fit_and_pack(k_post)
            if self.index is None:
                self._stage(k3, v3, kp3, start, stop)
                attn = self._dense(q_post.view(-1, layer.num_heads, d), start, stop)
            else:
                self._append(k3, v3, kp3, start, stop)
                attn = layer._run_product_attention(
                    q_post, q_pre, self.flat[start:stop], query_offset=start)
        return layer._finish_output(attn.view(hidden_states.shape[0], -1), gate)

    def finish(self) -> None:
        layer = self.layer
        if self.profiling:
            layer._release_profiling_reservation()
            return
        if self.index is None:
            # The prompt ended before the fit point: fit on all of it.
            like = self.stage_k.new_empty(0)
            self.fit_at = self.tokens
            self._fit_and_pack(like)
        self._refresh_slots()
        if _arena.full_graph_decode_enabled():
            layer._adopt_into_arena()

    # -- dense start -------------------------------------------------------------

    def _stage(self, k3, v3, kp3, start, stop) -> None:
        if self.stage_k is None:
            n = self.fit_at
            if _arena.full_graph_decode_enabled():
                # The arena's packed rows are unused until the fit, so the
                # K/V are staged there in prompt order and moved into the
                # reserved layout at the fit, avoiding a separate buffer.
                arena = self.layer._arena(k3)
                self.stage_k = arena.packed_kv[0, :n]
                self.stage_v = arena.packed_kv[1, :n]
                self._staged_in_arena = True
            else:
                self.stage_k = k3.new_empty((n,) + tuple(k3.shape[1:]))
                self.stage_v = v3.new_empty((n,) + tuple(v3.shape[1:]))
                self._staged_in_arena = False
            self.stage_kpre = kp3.new_empty((n,) + tuple(kp3.shape[1:]))
        self.stage_k[start:stop] = k3
        self.stage_v[start:stop] = v3
        self.stage_kpre[start:stop] = kp3

    def _dense(self, q: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """Exact causal attention of the slice over the staged prompt-order K/V."""
        from vllm.vllm_flash_attn import flash_attn_varlen_func as varlen
        from semblo.vllm_plugin.fused_prefill import _attend_part
        n, num_q_heads, d = q.shape
        hkv = self.stage_k.shape[1]
        q_per_kv = num_q_heads // hkv
        device = q.device
        cu_q = torch.tensor([0, n], device=device, dtype=torch.int32)
        cu_k = torch.tensor([0, stop], device=device, dtype=torch.int32)
        out = torch.empty_like(q)
        for h in range(hkv):
            sl = slice(h * q_per_kv, (h + 1) * q_per_kv)
            with _attend_part("semblo::attend_dense_prefix"):
                out[:, sl, :] = varlen(
                    q[:, sl, :], self.stage_k[:stop, h:h + 1, :], self.stage_v[:stop, h:h + 1, :],
                    max_seqlen_q=n, cu_seqlens_q=cu_q, max_seqlen_k=stop, cu_seqlens_k=cu_k,
                    softmax_scale=self.layer.scaling, causal=True,
                    fa_version=int(os.environ.get("SEMBLO_FA_VERSION", "3")))
        return out

    # -- fit --------------------------------------------------------------------

    def _product(self, kp3: torch.Tensor) -> torch.Tensor:
        return self.layer.semblo_key_router(kp3.unsqueeze(0))[0]

    def _fit_and_pack(self, like: torch.Tensor) -> None:
        layer = self.layer
        cfg = layer.runtime_config
        p = self.fit_at
        device = self.stage_k.device
        unit = int(os.environ.get("SEMBLO_FLAT_UNIT_TOKENS", cfg.flat_unit_tokens))
        min_log2 = max(3, int(os.environ.get("SEMBLO_FLAT_MIN_CLUSTERS_LOG2", cfg.flat_min_clusters_log2)))
        clusters = _arena.arena_cluster_count(self.tokens, unit, min_log2)
        product = self._product(self.stage_kpre[:p])
        os.environ["SEMBLO_FLAT_CLUSTERS"] = str(clusters)      # the count the whole prompt would get
        try:
            index = layer._fit_index(product, self.flat[:p])
        finally:
            del os.environ["SEMBLO_FLAT_CLUSTERS"]
        del product
        hkv, d = self.stage_k.shape[1], self.stage_k.shape[2]
        align = max(int(index.page_align), 1)
        pool = layer._index_buffer_pool()
        if _arena.full_graph_decode_enabled():
            width = int(layer._arena(like).width)
        else:
            width = self.tokens + int(index.num_buckets) * align
            width += (-width) % 256
        reserve = reserved_layout(index.bucket_sizes, p, self.tokens, width, align)
        indptr = torch.zeros(hkv, int(index.num_buckets) + 1, dtype=torch.int32, device=device)
        indptr[:, 1:] = torch.cumsum(reserve, 1).to(torch.int32)
        # move every fitted key from its block's packed run to the reserved one
        inv_old = index.inverse_positions().to(torch.int64)                   # [Hkv, p]
        old_ip = index.bucket_indptr.to(torch.int64)
        inv_new = torch.empty_like(inv_old)
        for h in range(hkv):
            bucket = (torch.searchsorted(old_ip[h].contiguous(), inv_old[h].contiguous(), right=True) - 1)
            inv_new[h] = inv_old[h] - old_ip[h][bucket] + indptr[h].to(torch.int64)[bucket]
        shape = (2, width, hkv, d)
        packed = pool(shape, self.stage_k.dtype, device) if pool is not None \
            else torch.zeros(shape, dtype=self.stage_k.dtype, device=device)
        posting = torch.zeros(hkv, width, dtype=torch.int32, device=device)
        inverse = torch.zeros(hkv, self.tokens, dtype=torch.int32, device=device)
        pos = self.flat[:p].to(torch.int32)
        staged_in_arena = getattr(self, "_staged_in_arena", False) and pool is not None
        for h in range(hkv):
            if staged_in_arena:
                # Source and destination are the same buffer, so copy one
                # head's rows through a temporary (p x D, 64 MiB at 131K).
                k_h = self.stage_k[:p, h].clone()
                v_h = self.stage_v[:p, h].clone()
            else:
                k_h, v_h = self.stage_k[:p, h], self.stage_v[:p, h]
            packed[0][:, h].index_copy_(0, inv_new[h], k_h)
            packed[1][:, h].index_copy_(0, inv_new[h], v_h)
            del k_h, v_h
            posting[h].index_copy_(0, inv_new[h], pos)
            inverse[h, :p] = inv_new[h].to(torch.int32)
        index.packed_kv = packed
        index.posting_positions = posting
        index.bucket_indptr = indptr
        index._inverse_positions = inverse
        index.prompt_length = self.tokens
        index.max_position = self.tokens - 1
        self.reserve = reserve
        self.index = index
        self.snapshot = TokenOuterSnapshot(index)
        index._token_outer_snapshot = self.snapshot
        self._refresh_slots()
        layer.runtime_state.put(layer.source_layer_idx, index)
        layer._adopt_index_buffers(index)
        if cfg.own_prompt_cache and _arena.full_graph_decode_enabled():
            layer._arena(like).install_suffix_store(self.tokens)
        elif cfg.own_prompt_cache:
            layer._ensure_own_suffix_store(index, like)
        if not _SAID.get("fit"):
            _SAID["fit"] = True
            print(f"[semblo] token-outer prefill: centroids fitted on the first {p:,} of {self.tokens:,} "
                  f"positions ({clusters} clusters), the store laid out with reservations "
                  f"({width:,} rows)", flush=True)
        self.stage_k = self.stage_v = self.stage_kpre = None

    # -- append -----------------------------------------------------------------

    def _append(self, k3, v3, kp3, start, stop) -> None:
        index = self.index
        hkv = k3.shape[1]
        metric = os.environ.get("SEMBLO_FLAT_METRIC", "cosine") or "cosine"
        product = self._product(kp3)                                           # [n, Hkv, I]
        from semblo.inference.flat_index import routing_space
        ids = torch.stack([
            _nearest_cluster(routing_space(product[:, h]), index.codebooks[h], metric)
            for h in range(hkv)])                                                # [Hkv, n]
        del product
        sizes = index.bucket_sizes
        rows, counts = append_rows(ids, index.bucket_indptr, sizes)
        if bool((sizes.to(torch.int64) + counts > self.reserve).any()):
            self._repack(start)
            rows, counts = append_rows(ids, index.bucket_indptr, sizes)
        pos = self.flat[start:stop].to(torch.int32)
        for h in range(hkv):
            index.packed_kv[0][:, h].index_copy_(0, rows[h], k3[:, h])
            index.packed_kv[1][:, h].index_copy_(0, rows[h], v3[:, h])
            index.posting_positions[h].index_copy_(0, rows[h], pos)
            index._inverse_positions[h, start:stop] = rows[h].to(torch.int32)
        # The body sees the counts before this slice. The slice's own keys go
        # through the exact window.
        self.snapshot.sizes = sizes.clone()
        self.snapshot.chunk_start = start
        sizes += counts.to(sizes.dtype)

    def _repack(self, n_so_far: int) -> None:
        """Lay the runs out again from the current proportions, in place."""
        index = self.index
        hkv = index.packed_kv.shape[2]
        align = max(int(index.page_align), 1)
        width = int(index.packed_kv.shape[1])
        reserve = reserved_layout(index.bucket_sizes, n_so_far, self.tokens, width, align)
        new_ip = torch.zeros_like(index.bucket_indptr)
        new_ip[:, 1:] = torch.cumsum(reserve, 1).to(torch.int32)
        old_ip = index.bucket_indptr.to(torch.int64)
        for h in range(hkv):
            rows_old = index._inverse_positions[h, :n_so_far].to(torch.int64)
            bucket = torch.searchsorted(old_ip[h].contiguous(), rows_old.contiguous(), right=True) - 1
            rows_new = rows_old - old_ip[h][bucket] + new_ip[h].to(torch.int64)[bucket]
            kv = index.packed_kv[:, :, h, :].clone()                            # [2, width, D]
            index.packed_kv[:, rows_new, h, :] = kv[:, rows_old]
            del kv
            posting = index.posting_positions[h].clone()
            index.posting_positions[h, rows_new] = posting[rows_old]
            index._inverse_positions[h, :n_so_far] = rows_new.to(torch.int32)
        index.bucket_indptr = new_ip
        index.__dict__.pop("_kv_outer_tables", None)
        self.reserve = reserve
        self.repacks += 1
        print(f"[semblo] token-outer prefill layer {getattr(self.layer, 'layer_idx', '?')}: a cluster outgrew "
              f"its reservation at position {n_so_far:,}; the store was laid out again "
              f"(repack {self.repacks})", flush=True)

    def _refresh_slots(self) -> None:
        index = self.index
        slots = int((index.bucket_sizes > 0).sum(dim=-1).max().item())
        bucket_slot, slot_start, slot_bucket = _bucket_slot_table(
            index.bucket_sizes, index.bucket_indptr, slots)
        index.num_non_empty_buckets = slots
        index.bucket_slot, index.slot_start, index.slot_bucket = bucket_slot, slot_start, slot_bucket
