"""Union construction: per-block selected-bucket bitmask -> flat slot list.

The selection kernel ORs each query's selected buckets into a bitmask per
(query block, KV head) (fused_select.py, DO_BITMASK). The union attention body
(fused_body.py) needs a flat list of packed-KV slot indices per (block, head).
Two small kernels convert one into the other:

  K2a _union_compact_kernel  (grid [NB, Hkv]):
      bitmask (B/32 words) -> compacted per-bucket ranges
      (start in packed KV, causal length, exclusive token prefix) + counts.
  K2b _union_expand_kernel   (grid [NB, Hkv, ceil(U/TILE)]):
      CSR-style expansion: output slot o binary-searches the token prefix
      to find its bucket, then reads start + (o - prefix).

Cost is ~2 extra passes over 4 bytes/token (vs 1 KB/token of K/V the body
reads), i.e. <1%.
"""

from __future__ import annotations

import torch

try:  # pragma: no cover
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _union_compact_kernel(
        rstart_ptr,   # [NB, H, RMAX] int32: packed-KV start of each bucket
        rlen_ptr,     # [NB, H, RMAX] int32: causal length
        rcum_ptr,     # [NB, H, RMAX] int32: exclusive token prefix
        rcount_ptr,   # [NB, H] int32: kept union buckets (<= rmax)
        rtokens_ptr,  # [NB, H] int32: tokens of the KEPT buckets
        rfull_ptr,    # [NB, H] int32: true union bucket count (overflow
                      # detection: full > rmax means buckets were dropped)
        rrow_ptr,     # [NB, H, RMAX] int32 (ROWMASK): bit r set if row r of
                      # this block selected the bucket
        qmask_ptr,    # [Q, H, B/32] int32 (ROWMASK): per-query selection
        num_queries,
        bitmask_ptr,  # [NB, H, B/32] int32
        sizes_ptr,    # [NB, H, B] int32 per-block causal snapshot
        indptr_ptr,   # [H, B+1] int32
        rmax,
        stride_r0, stride_r1, stride_r2,
        stride_zb, stride_z0, stride_z1,
        stride_i0, stride_i1,
        B: tl.constexpr,
        G_Q: tl.constexpr,
        ROWMASK: tl.constexpr,
    ):
        block = tl.program_id(0)
        kv_head = tl.program_id(1)
        n_words = B // 32
        words = tl.load(
            bitmask_ptr
            + (block * tl.num_programs(1) + kv_head).to(tl.int64) * n_words
            + tl.arange(0, B // 32)
        )
        bits = (words[:, None] >> tl.arange(0, 32)[None, :]) & 1
        selected = tl.reshape(bits, [B])
        flat_ids = tl.arange(0, B)
        sizes = tl.load(
            sizes_ptr + block.to(tl.int64) * stride_zb
            + kv_head * stride_z0 + flat_ids * stride_z1
        )
        # empty buckets may be set by no query, but guard anyway
        selected = selected & (sizes > 0)
        sel_len = tl.where(selected > 0, sizes, 0)
        pos = tl.cumsum(selected, axis=0) - selected
        cum = tl.cumsum(sel_len, axis=0) - sel_len
        keep = (selected > 0) & (pos < rmax)
        bstart = tl.load(
            indptr_ptr + kv_head * stride_i0 + flat_ids * stride_i1,
            mask=keep, other=0,
        )
        out_base = block.to(tl.int64) * stride_r0 + kv_head * stride_r1
        if ROWMASK:
            # Which rows of this block selected each bucket, one bit per row.
            # A bucket is a run of keys, so this is computed once per bucket
            # instead of per key in the body's inner loop.
            rowmask = tl.zeros([B], tl.int32)
            for r in tl.static_range(G_Q):
                row = block * G_Q + r
                qw = tl.load(
                    qmask_ptr
                    + (row.to(tl.int64) * tl.num_programs(1) + kv_head)
                    * n_words + tl.arange(0, B // 32),
                    mask=row < num_queries, other=0,
                )
                qb = (qw[:, None] >> tl.arange(0, 32)[None, :]) & 1
                rowmask = rowmask | (tl.reshape(qb, [B]).to(tl.int32) << r)
            tl.store(rrow_ptr + out_base + pos * stride_r2, rowmask, mask=keep)
        tl.store(rstart_ptr + out_base + pos * stride_r2, bstart, mask=keep)
        tl.store(rlen_ptr + out_base + pos * stride_r2, sel_len, mask=keep)
        tl.store(rcum_ptr + out_base + pos * stride_r2, cum, mask=keep)
        flat_out = block * tl.num_programs(1) + kv_head
        tl.store(rcount_ptr + flat_out, tl.minimum(tl.sum(selected), rmax))
        # tokens of the kept buckets only, so the expansion bound is exact
        tl.store(rtokens_ptr + flat_out,
                 tl.sum(tl.where(keep, sel_len, 0)))
        tl.store(rfull_ptr + flat_out, tl.sum(selected))

    @triton.jit
    def _union_expand_kernel(
        idx_ptr,      # [NB, H, UMAX] int32 out: packed-KV slots
        rstart_ptr, rlen_ptr, rcum_ptr, rcount_ptr, rtokens_ptr,
        rrow_ptr,     # [NB, H, RMAX] int32 (ROWMASK) in
        xrow_ptr,     # [NB, H, UMAX] int32 (ROWMASK) out: per-slot row bits
        umax,
        stride_x0, stride_x1, stride_x2,
        stride_r0, stride_r1, stride_r2,
        TILE: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        TILE_PROGRAMS: tl.constexpr,
        ROWMASK: tl.constexpr,
    ):
        # Persistent over the token axis. umax is sized for the worst case,
        # so instead of a grid proportional to umax, TILE_PROGRAMS programs
        # per (block, head) stride over the actual total.
        block = tl.program_id(0)
        kv_head = tl.program_id(1)
        flat = block * tl.num_programs(1) + kv_head
        # Bound by the buffer size, so an overflow cannot write out of bounds
        # before the asynchronous overflow check reports it.
        total = tl.minimum(tl.load(rtokens_ptr + flat), umax)
        count = tl.load(rcount_ptr + flat)
        r_base = block.to(tl.int64) * stride_r0 + kv_head * stride_r1
        x_base = block.to(tl.int64) * stride_x0 + kv_head * stride_x1
        t = tl.program_id(2)
        while t * TILE < total:
            o = t * TILE + tl.arange(0, TILE)
            live = o < total
            # binary search: largest r with rcum[r] <= o
            lo = tl.zeros([TILE], tl.int32)
            hi = tl.full([TILE], 1, tl.int32) * count  # exclusive
            for _ in tl.static_range(SEARCH_STEPS):
                mid = (lo + hi) // 2
                probe_ok = live & (mid < count)
                c = tl.load(rcum_ptr + r_base + mid * stride_r2,
                            mask=probe_ok, other=0)
                go_right = probe_ok & (c <= o)
                lo = tl.where(go_right, mid + 1, lo)
                hi = tl.where(go_right, hi, mid)
            r = tl.maximum(lo - 1, 0)
            start = tl.load(rstart_ptr + r_base + r * stride_r2, mask=live,
                            other=0)
            cum = tl.load(rcum_ptr + r_base + r * stride_r2, mask=live,
                          other=0)
            tl.store(idx_ptr + x_base + o * stride_x2, start + (o - cum),
                     mask=live)
            if ROWMASK:
                # the binary search already found this slot's bucket
                rw = tl.load(rrow_ptr + r_base + r * stride_r2, mask=live,
                             other=0)
                tl.store(xrow_ptr + x_base + o * stride_x2, rw, mask=live)
            t += TILE_PROGRAMS


def build_union_slots(
    bitmask: torch.Tensor,       # [NB, H, B/32] int32
    causal_sizes: torch.Tensor,  # [NB, H, B] int32 per union block
    bucket_indptr: torch.Tensor,  # [H, B+1] int32
    *,
    rmax: int,
    umax: int,
    tile: int = 1024,
    return_ranges: bool = False,
    ranges_only: bool = False,
    out_union_idx: torch.Tensor | None = None,
    qmask: torch.Tensor | None = None,     # [Q, H, B/32] int32
    queries_per_block: int = 0,
    out_slot_rowmask: torch.Tensor | None = None,
):
    """Expand the selected buckets of each (block, head) into packed-KV slots.

    Returns ``(union_idx [NB, H, umax] int32, union_tokens [NB, H] int32,
    rfull [NB, H] int32)``, where ``rfull`` is the number of selected buckets
    before truncation to ``rmax``. With ``return_ranges`` the third element is
    the tuple of per-bucket ranges ``(rstart, rlen, rcum, rcount, rfull)``.
    With ``ranges_only`` only ``(union_tokens, ranges)`` is returned and no
    slot list is built.

    Passing ``qmask`` also emits, for every slot, a 32-bit word naming which
    rows of the slot's block selected it, appended as a fourth element when
    ``return_ranges`` is false. Since a bucket is a contiguous run of slots,
    this lets the body enforce per-query selection with one contiguous load
    instead of a per-(row, key) gather.
    """
    if triton is None:
        raise RuntimeError("build_union_slots needs triton")
    nb, num_kv_heads, n_words = bitmask.shape
    b_dim = n_words * 32
    device = bitmask.device
    want_rowmask = qmask is not None
    if want_rowmask:
        if not 0 < queries_per_block <= 32:
            raise ValueError(
                "the row bitmask packs one bit per row of a block, so "
                f"queries_per_block must be in [1, 32], got {queries_per_block}")
        if qmask.shape[1:] != (num_kv_heads, n_words):
            raise ValueError("qmask must have shape [Q, H, B/32]")
    rstart = torch.empty(nb, num_kv_heads, rmax, dtype=torch.int32,
                         device=device)
    rlen = torch.empty_like(rstart)
    rcum = torch.empty_like(rstart)
    rcount = torch.empty(nb, num_kv_heads, dtype=torch.int32, device=device)
    rtokens = torch.empty_like(rcount)
    rfull = torch.empty_like(rcount)
    rrow = torch.empty_like(rstart) if want_rowmask else rstart
    _union_compact_kernel[(nb, num_kv_heads)](
        rstart, rlen, rcum, rcount, rtokens, rfull,
        rrow, qmask if want_rowmask else bitmask,
        qmask.shape[0] if want_rowmask else 0,
        bitmask, causal_sizes, bucket_indptr,
        rmax,
        rstart.stride(0), rstart.stride(1), rstart.stride(2),
        causal_sizes.stride(0), causal_sizes.stride(1),
        causal_sizes.stride(2),
        bucket_indptr.stride(0), bucket_indptr.stride(1),
        B=b_dim,
        G_Q=max(int(queries_per_block), 1),
        ROWMASK=want_rowmask,
        num_warps=8,
    )
    if ranges_only:
        # The FA3 body builds pages directly from these ranges, so the
        # token-level expansion and its [NB, Hkv, umax] buffer (1.07 GB at
        # 256K) are not needed.
        return rtokens, (rstart, rlen, rcum, rcount, rfull)
    if out_union_idx is not None:
        assert out_union_idx.shape == (nb, num_kv_heads, umax)
        union_idx = out_union_idx
    else:
        union_idx = torch.empty(nb, num_kv_heads, umax, dtype=torch.int32,
                                device=device)
    if want_rowmask:
        if out_slot_rowmask is not None:
            assert out_slot_rowmask.shape == (nb, num_kv_heads, umax)
            slot_rowmask = out_slot_rowmask
        else:
            slot_rowmask = torch.empty_like(union_idx)
    else:
        slot_rowmask = union_idx
    search_steps = max(1, rmax.bit_length())
    tile_programs = 16
    _union_expand_kernel[(nb, num_kv_heads, tile_programs)](
        union_idx,
        rstart, rlen, rcum, rcount, rtokens,
        rrow, slot_rowmask,
        umax,
        union_idx.stride(0), union_idx.stride(1), union_idx.stride(2),
        rstart.stride(0), rstart.stride(1), rstart.stride(2),
        TILE=tile, SEARCH_STEPS=search_steps,
        TILE_PROGRAMS=tile_programs,
        ROWMASK=want_rowmask,
        num_warps=4,
    )
    if return_ranges:
        return union_idx, rtokens, (rstart, rlen, rcum, rcount, rfull)
    if want_rowmask:
        return union_idx, rtokens, rfull, slot_rowmask
    return union_idx, rtokens, rfull
