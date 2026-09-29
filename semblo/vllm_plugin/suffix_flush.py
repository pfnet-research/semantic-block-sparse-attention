"""Move generated tokens from the exact suffix into the clustered store.

During decoding every generated token is read exactly from the arena's suffix
store (a reserve of 65,536 positions).  With ``SEMBLO_SUFFIX_FLUSH_EVERY=F`` the
newest ``SEMBLO_SUFFIX_KEEP_EXACT=W`` generated tokens stay exact, and whenever
the exact suffix reaches W + F tokens the oldest F are moved into the clusters.
Each token is assigned to the nearest centroid (the centroids do not move), and
the cluster-ordered K/V store is re-packed with the new tokens appended to
their clusters.  This applies the prefill's rule (a query attends its own chunk
exactly and everything older through the index) to decoding, and bounds the
exact suffix at W + F tokens however long the generation runs.

Everything the captured decode step reads (``packed_kv``, ``posting``,
``sizes``, ``indptr``, ``prompt_length_t``, the suffix block table) is
rewritten in place, so the CUDA graph stays valid.  The flush itself runs
eagerly, from a hook around vLLM's ``execute_model`` (``install_hook``).

The re-pack is the device-agnostic function pair ``repack_plan`` /
``repack_apply``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def flush_every() -> int:
    return int(os.environ.get("SEMBLO_SUFFIX_FLUSH_EVERY", "0") or 0)


def keep_exact() -> int:
    return int(os.environ.get("SEMBLO_SUFFIX_KEEP_EXACT", "1024") or 1024)


def enabled() -> bool:
    return flush_every() > 0


@dataclass
class RepackPlan:
    """One KV head's new packed layout.

    kind: 0 = a row of the old packed store, 1 = a flushed token, 2 = padding
    (repeats row 0, as ``state._page_align_order`` does; it is never read).
    """
    new_sizes: torch.Tensor      # [C] int64, true (unpadded) sizes
    new_indptr: torch.Tensor     # [C + 1] int32, page-aligned starts
    kind: torch.Tensor           # [total] int8
    old_src: torch.Tensor        # [total] int64 row of the old packed store (kind 0)
    new_src: torch.Tensor        # [total] int64 index into the flushed tokens (kind 1)
    total: int


def repack_plan(old_sizes: torch.Tensor, old_indptr: torch.Tensor,
                new_ids: torch.Tensor, align: int) -> RepackPlan:
    """Append the flushed tokens (cluster ids ``new_ids``, in position order) to their clusters.

    ``old_sizes`` [C] are the true sizes, ``old_indptr`` [C + 1] the page-aligned
    starts of the old store (rows [indptr[c], indptr[c] + sizes[c]) are real).
    Within a cluster the old rows keep their order and the newcomers follow in
    position order, so a cluster stays one interval.
    """
    device = old_sizes.device
    C = int(old_sizes.numel())
    old_sizes = old_sizes.to(torch.int64)
    old_indptr = old_indptr.to(torch.int64)
    new_ids = new_ids.to(torch.int64)
    added = torch.bincount(new_ids, minlength=C)[:C]
    new_sizes = old_sizes + added
    align = max(int(align), 1)
    padded = ((new_sizes + align - 1) // align) * align
    zero = new_sizes.new_zeros(1)
    new_ip = torch.cat([zero, padded.cumsum(0)])
    total = int(new_ip[-1])
    ar = torch.arange(total, device=device)
    bucket = torch.searchsorted(new_ip, ar, right=True) - 1
    bucket.clamp_(0, C - 1)
    off = ar - new_ip[bucket]
    is_old = off < old_sizes[bucket]
    is_new = (~is_old) & (off < new_sizes[bucket])
    kind = torch.full((total,), 2, dtype=torch.int8, device=device)
    kind[is_old] = 0
    kind[is_new] = 1
    old_src = torch.where(is_old, old_indptr[bucket] + off, torch.zeros_like(off))
    # New tokens grouped by cluster, in position order within a cluster.
    order_new = torch.argsort(new_ids, stable=True)
    added_ip = torch.cat([zero, added.cumsum(0)])
    within = (off - old_sizes[bucket]).clamp_min(0)
    slot = (added_ip[bucket] + within).clamp_max(max(int(new_ids.numel()) - 1, 0))
    new_src = torch.where(is_new, order_new[slot] if new_ids.numel() else torch.zeros_like(slot), torch.zeros_like(slot))
    return RepackPlan(new_sizes=new_sizes, new_indptr=new_ip.to(torch.int32), kind=kind,
                      old_src=old_src, new_src=new_src, total=total)


def repack_apply(packed_kv: torch.Tensor, posting: torch.Tensor, sizes: torch.Tensor,
                 indptr: torch.Tensor, flushed_kv: torch.Tensor, flushed_pos: torch.Tensor,
                 assign: torch.Tensor, align: int, tmp: torch.Tensor) -> int:
    """Re-pack every KV head in place.  Returns the widest new packed length.

    packed_kv [2, W, Hkv, D]; posting [Hkv, W] int32 (row -> position); sizes
    [Hkv, C] int32; indptr [Hkv, C + 1] int32; flushed_kv [2, N, Hkv, D];
    flushed_pos [N] int32; assign [Hkv, N] cluster ids; tmp [2, W, D] scratch.
    """
    two, width, hkv, dim = packed_kv.shape
    C = int(sizes.shape[1])
    n_new = int(flushed_pos.numel())
    widest = 0
    for h in range(hkv):
        old_sizes = sizes[h, :C].to(torch.int64)
        old_ip = indptr[h, :C + 1].to(torch.int64)
        old_total = int(old_ip[-1])
        plan = repack_plan(old_sizes, old_ip, assign[h], align)
        total = plan.total
        if total > width:
            raise RuntimeError(f"the re-packed store needs {total} rows; the arena holds {width}")
        widest = max(widest, total)
        # Old rows through a gather (row 0 for the rest, overwritten below).
        idx = plan.old_src.view(1, total, 1).expand(2, total, dim)
        torch.gather(packed_kv[:, :max(old_total, 1), h, :], 1, idx, out=tmp[:, :total])
        new_rows = torch.nonzero(plan.kind == 1, as_tuple=True)[0]
        if new_rows.numel():
            tmp[:, new_rows] = flushed_kv[:, plan.new_src[new_rows], h, :]
        packed_kv[:, :total, h, :].copy_(tmp[:, :total])
        pos_old = posting[h, plan.old_src]
        pos_new = flushed_pos.to(posting.dtype)[plan.new_src] if n_new else pos_old
        row0 = posting[h, 0]
        posting[h, :total] = torch.where(plan.kind == 0, pos_old,
                                         torch.where(plan.kind == 1, pos_new, row0))
        sizes[h, :C] = plan.new_sizes.to(sizes.dtype)
        indptr[h, :C + 1] = plan.new_indptr
    return widest


# --------------------------------------------------------------------------
# The host side: when to flush, and the flush of one layer's arena.
# --------------------------------------------------------------------------

def flush_range(prompt_len: int, written: int, plen_cur: int, keep: int, every: int):
    """The positions to move now, or None.

    ``written`` generated tokens sit at positions [prompt_len, prompt_len +
    written); those below ``plen_cur`` were moved already.  When the exact
    suffix reaches keep + every tokens, everything but the newest ``keep`` is
    moved, so the exact suffix then stays within [keep, keep + every).
    """
    live = prompt_len + written - plen_cur
    if live >= keep + every:
        return plen_cur, prompt_len + written - keep
    return None


def _nearest_cluster(space: torch.Tensor, codebooks_h: torch.Tensor,
                     metric: str = "cosine") -> torch.Tensor:
    """[N, I] routing keys, [2, C, I/2] stored descriptors -> [N] cluster ids.

    Cosine against the unit descriptors, as in the prefill's assignment.
    Under "euclidean" (SEMBLO_FLAT_METRIC), the smallest squared distance to
    the raw means.  An all-zero descriptor is an unused cluster and is never
    chosen.
    """
    cents = torch.cat([codebooks_h[0], codebooks_h[1]], dim=-1).float()     # [C, I]
    if metric == "cosine":
        sim = torch.nn.functional.normalize(space.float(), dim=-1) @ cents.t()
    else:
        sim = space.float() @ cents.t() - 0.5 * cents.pow(2).sum(-1)[None]
    sim[:, cents.norm(dim=-1) == 0] = -float("inf")
    return sim.argmax(dim=-1)


def flush_layer(arena, module, frm: int, to: int, tmp: torch.Tensor) -> int:
    """Move positions [frm, to) of one layer's exact suffix into its clusters.

    Reads their pre-RoPE keys and K/V from the suffix stores through the block
    table, assigns each token to the nearest centroid in the routing space the
    prefill used (key router, then routing_space), re-packs the cluster-ordered
    store in place, and advances the prompt length the kernels read.  Returns
    the widest packed length after the move.
    """
    from semblo.inference.flat_index import routing_space
    n = int(to) - int(frm)
    if n <= 0:
        return 0
    device = arena.packed_kv.device
    bs = int(arena.suffix_block)
    pos = torch.arange(int(frm), int(to), device=device, dtype=torch.int64)
    blk = pos // bs
    phys = arena.suffix_table[0, blk].to(torch.int64)
    off = pos - blk * bs
    kpre = arena.suffix_kpre[phys, off]                     # [N, Hkv, D]
    kv = arena.suffix_kv[:, phys, off]                      # [2, N, Hkv, D]
    product = module.semblo_key_router(kpre.unsqueeze(0))[0]      # [N, Hkv, I]
    mode = os.environ.get("SEMBLO_ROUTING_NORMALIZE", "whole") or "whole"
    metric = os.environ.get("SEMBLO_FLAT_METRIC", "cosine") or "cosine"
    hkv = int(arena.codebooks.shape[0])
    assign = torch.empty(hkv, n, dtype=torch.int64, device=device)
    for h in range(hkv):
        keys_h = product[:, h]
        space = keys_h.float() if mode == "none" else routing_space(keys_h)
        assign[h] = _nearest_cluster(space, arena.codebooks[h], metric)
    widest = repack_apply(arena.packed_kv, arena.posting, arena.sizes, arena.indptr,
                          kv, pos.to(torch.int32), assign, arena.page_align, tmp)
    arena.index.prompt_length = int(to)
    arena.index.max_position = int(to)
    arena.prompt_length_t.fill_(int(to))
    arena.extend_suffix_table(int(to) + arena.reserve - 1)
    return widest


class SuffixFlusher:
    """Bookkeeping across decode steps; one instance per process."""

    def __init__(self) -> None:
        self.keep = keep_exact()
        self.every = flush_every()
        self.state = None
        self.epoch = None
        self.prompt_len = 0
        self.written = 0
        self.pending_reset = True
        self.flushes = 0
        self.log = os.environ.get("SEMBLO_SUFFIX_FLUSH_LOG", "0") == "1"

    def _find_state(self, runner):
        if self.state is not None:
            return self.state
        model = getattr(runner, "model", None)
        # vLLM may wrap the model (e.g. in a graph wrapper); unwrap it.
        for _ in range(4):
            if model is None or hasattr(model, "modules"):
                break
            model = getattr(model, "model", getattr(model, "runnable", None))
        if model is None or not hasattr(model, "modules"):
            return None
        for mod in model.modules():
            st = getattr(mod, "semblo_runtime_state", None)
            if st is not None:
                self.state = st
                return st
        return None

    def _arenas(self):
        out = []
        for key, val in self.state.index_buffers.items():
            if isinstance(key, tuple) and len(key) == 2 and key[0] == "arena":
                out.append((int(key[1]), val))
        return sorted(out)

    def before_step(self, runner, scheduler_output) -> None:
        total = int(getattr(scheduler_output, "total_num_scheduled_tokens", 0))
        per_req = getattr(scheduler_output, "num_scheduled_tokens", {}) or {}
        if total != 1 or len(per_req) != 1:
            # A prefill chunk, several requests, or nothing: the next
            # single-token step starts a fresh count.
            self.pending_reset = True
            return
        state = self._find_state(runner)
        if state is None:
            return
        arenas = self._arenas()
        if not arenas or any(getattr(a, "suffix_kpre", None) is None for _, a in arenas):
            return
        if self.pending_reset or self.epoch != state.prompt_epoch:
            self.pending_reset = False
            self.epoch = state.prompt_epoch
            self.prompt_len = int(arenas[0][1].index.prompt_length)
            self.written = 0
        plen_cur = int(arenas[0][1].index.prompt_length)
        rng = flush_range(self.prompt_len, self.written, plen_cur, self.keep, self.every)
        if rng is not None:
            frm, to = rng
            t0 = None
            if self.log:
                torch.cuda.synchronize(); import time; t0 = time.perf_counter()
            first = arenas[0][1]
            tmp = state.pooled_buffer("arena:flush_tmp", (2, int(first.width), int(first.packed_kv.shape[-1])),
                                      first.packed_kv.dtype, first.packed_kv.device)
            for layer, arena in arenas:
                module = state.source_modules.get(layer)
                if module is None:
                    raise RuntimeError(f"no source module registered for layer {layer}")
                flush_layer(arena, module, frm, to, tmp)
            self.flushes += 1
            if self.log:
                torch.cuda.synchronize()
                print(f"[semblo] suffix flush {self.flushes}: positions [{frm}, {to}) of {len(arenas)} layers moved into "
                      f"the clusters in {1000 * (time.perf_counter() - t0):.1f} ms; exact suffix now "
                      f"{self.prompt_len + self.written - to} tokens", flush=True)
        self.written += 1


_HOOKED = False


def install_hook() -> None:
    """Wrap vLLM's model runner so the flush runs on the host before a decode step.

    The captured decode graph runs no plugin Python, so this hook is where
    host code runs once per step, before the graph is replayed on the same
    stream.
    """
    global _HOOKED
    if _HOOKED or not enabled():
        return
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    original = GPUModelRunner.execute_model
    flusher = SuffixFlusher()

    def execute_model(self, scheduler_output, *args, **kwargs):
        flusher.before_step(self, scheduler_output)
        return original(self, scheduler_output, *args, **kwargs)

    GPUModelRunner.execute_model = execute_model
    GPUModelRunner._semblo_suffix_flusher = flusher
    _HOOKED = True
    print(f"[semblo] suffix flush installed: keep {flusher.keep} exact, move {flusher.every} at a time", flush=True)
