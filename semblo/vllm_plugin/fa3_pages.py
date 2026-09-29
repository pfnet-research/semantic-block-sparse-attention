"""Union construction for FA3: selected-bucket bitmask -> page list.

`flash_attn_varlen_func` takes a `block_table`, so FA3 gathers KV at page
granularity from an arbitrary page list.  The packed KV is bucket-ordered, so
a bucket is a contiguous run of packed rows and a union is a set of runs,
which maps to a page list.  Each (query block, KV head) becomes one varlen
sequence.

This is the same two-kernel structure as fused_union.py, over pages instead
of tokens, so it emits 1/page_size as many entries:

  P1 _page_count_kernel   (grid [NB, Hkv]):
      compacted bucket ranges -> pages per bucket + exclusive page prefix
  P2 _page_expand_kernel   (grid [NB, Hkv, TILE_PROGRAMS]):
      output page slot o binary-searches the prefix for its bucket, then
      reads first_page + (o - prefix)

Causality.  A page is all-or-nothing, and packed order is bucket-id order,
so a page straddling a bucket boundary would include neighbours at arbitrary
positions, including tokens after the query.  Two rules make every emitted
page a subset of the bucket's own causal prefix:

  * bucket starts are page-aligned in the packed layout (the index pads each
    bucket to a multiple of page_size), so the first page holds no
    predecessor;
  * the causal prefix's final partial page is dropped, so the last page
    holds no successor.

The second rule drops up to page_size-1 of a bucket's most recent causal
tokens.  This under-inclusion is safe: the result is a subset of the union,
never a superset.  The number of dropped tokens is returned.
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
    def _page_count_kernel(
        pfirst_ptr,   # [NB, H, RMAX] int32 out: first page of each bucket
        pnum_ptr,     # [NB, H, RMAX] int32 out: pages kept for each bucket
        pcum_ptr,     # [NB, H, RMAX] int32 out: exclusive page prefix
        ptotal_ptr,   # [NB, H] int32 out: pages of this (block, head)
        pdrop_ptr,    # [NB, H] int32 out: causal tokens lost to rule 2
        rstart_ptr, rlen_ptr, rcount_ptr,   # from _union_compact_kernel
        stride_r0, stride_r1, stride_r2,
        RMAX: tl.constexpr,
        PAGE_LOG2: tl.constexpr,
        ALIGNED: tl.constexpr,
    ):
        block = tl.program_id(0)
        kv_head = tl.program_id(1)
        base = block.to(tl.int64) * stride_r0 + kv_head * stride_r1
        r = tl.arange(0, RMAX)
        count = tl.load(rcount_ptr + block * tl.num_programs(1) + kv_head)
        live = r < count
        start = tl.load(rstart_ptr + base + r * stride_r2, mask=live, other=0)
        ln = tl.load(rlen_ptr + base + r * stride_r2, mask=live, other=0)

        if ALIGNED:
            # Bucket starts are page multiples, so the whole pages of the
            # causal prefix are exactly floor(len / page).
            first = start >> PAGE_LOG2
            npage = ln >> PAGE_LOG2
        else:
            # Unaligned layout: also skip the straddling first page, so the
            # emitted pages hold no foreign tokens.  The serving path aligns
            # buckets when the index is built.
            page = 1 << PAGE_LOG2
            head_pad = (-start) & (page - 1)
            first = (start + head_pad) >> PAGE_LOG2
            npage = tl.maximum(ln - head_pad, 0) >> PAGE_LOG2
        npage = tl.where(live, npage, 0)
        dropped = tl.where(live, ln - (npage << PAGE_LOG2), 0)

        cum = tl.cumsum(npage, axis=0) - npage
        tl.store(pfirst_ptr + base + r * stride_r2, first, mask=live)
        tl.store(pnum_ptr + base + r * stride_r2, npage, mask=live)
        tl.store(pcum_ptr + base + r * stride_r2, cum, mask=live)
        flat = block * tl.num_programs(1) + kv_head
        tl.store(ptotal_ptr + flat, tl.sum(npage))
        tl.store(pdrop_ptr + flat, tl.sum(dropped))

    @triton.jit
    def _page_expand_kernel(
        table_ptr,    # [NB*H, PMAX] int32 out
        pfirst_ptr, pcum_ptr, pnum_ptr, ptotal_ptr, rcount_ptr,
        pmax,
        stride_t0, stride_t1,
        stride_r0, stride_r1, stride_r2,
        TILE: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        TILE_PROGRAMS: tl.constexpr,
    ):
        block = tl.program_id(0)
        kv_head = tl.program_id(1)
        heads = tl.num_programs(1)
        flat = block * heads + kv_head
        total = tl.minimum(tl.load(ptotal_ptr + flat), pmax)
        count = tl.load(rcount_ptr + flat)
        r_base = block.to(tl.int64) * stride_r0 + kv_head * stride_r1
        # Head-major rows: there is one FA3 call per KV head, so that call's
        # sequences must be contiguous.
        seq = kv_head * tl.num_programs(0) + block
        t_base = seq.to(tl.int64) * stride_t0
        t = tl.program_id(2)
        while t * TILE < total:
            o = t * TILE + tl.arange(0, TILE)
            liv = o < total
            # Largest r with pcum[r] <= o.  Buckets with zero kept pages
            # repeat a prefix value.  The `c <= o` test takes the last such r,
            # which is never a zero-page bucket because o < total.
            lo = tl.zeros([TILE], tl.int32)
            hi = tl.full([TILE], 1, tl.int32) * count
            for _ in tl.static_range(SEARCH_STEPS):
                mid = (lo + hi) // 2
                ok = liv & (mid < count)
                c = tl.load(pcum_ptr + r_base + mid * stride_r2, mask=ok,
                            other=0)
                go = ok & (c <= o)
                lo = tl.where(go, mid + 1, lo)
                hi = tl.where(go, hi, mid)
            r = tl.maximum(lo - 1, 0)
            first = tl.load(pfirst_ptr + r_base + r * stride_r2, mask=liv,
                            other=0)
            cum = tl.load(pcum_ptr + r_base + r * stride_r2, mask=liv,
                          other=0)
            tl.store(table_ptr + t_base + o * stride_t1, first + (o - cum),
                     mask=liv)
            t += TILE_PROGRAMS


def build_union_pages(
    rstart: torch.Tensor,   # [NB, H, RMAX] int32, from build_union_slots
    rlen: torch.Tensor,     # [NB, H, RMAX] int32
    rcount: torch.Tensor,   # [NB, H] int32
    *,
    page_size: int,
    pmax: int,
    aligned: bool,
    tile: int = 256,
    out_table: torch.Tensor | None = None,
):
    """Returns (block_table [H*NB, pmax] int32, seqused_k [H*NB] int32,
    dropped [NB, H] int32).

    Rows are head-major (``head * NB + block``) because there is one FA3 call
    per KV head: each head packs its KV under its own permutation, while a
    varlen block table is per sequence and shared across heads.

    `seqused_k` is pages * page_size, which is what FA3 reads.  `dropped`
    counts the causal tokens dropped so that no page holds foreign tokens.
    """
    if triton is None:
        raise RuntimeError("build_union_pages needs triton")
    if page_size & (page_size - 1):
        raise ValueError(f"page_size must be a power of two, got {page_size}")
    nb, num_kv_heads, rmax = rstart.shape
    device = rstart.device
    pfirst = torch.empty_like(rstart)
    pnum = torch.empty_like(rstart)
    pcum = torch.empty_like(rstart)
    ptotal = torch.empty(nb, num_kv_heads, dtype=torch.int32, device=device)
    pdrop = torch.empty_like(ptotal)
    _page_count_kernel[(nb, num_kv_heads)](
        pfirst, pnum, pcum, ptotal, pdrop,
        rstart, rlen, rcount,
        rstart.stride(0), rstart.stride(1), rstart.stride(2),
        RMAX=rmax,
        PAGE_LOG2=page_size.bit_length() - 1,
        ALIGNED=bool(aligned),
        num_warps=4,
    )
    if out_table is not None:
        assert out_table.shape == (nb * num_kv_heads, pmax)
        table = out_table
    else:
        table = torch.zeros(nb * num_kv_heads, pmax, dtype=torch.int32,
                            device=device)
    tile_programs = 16
    _page_expand_kernel[(nb, num_kv_heads, tile_programs)](
        table,
        pfirst, pcum, pnum, ptotal, rcount,
        pmax,
        table.stride(0), table.stride(1),
        rstart.stride(0), rstart.stride(1), rstart.stride(2),
        TILE=tile,
        SEARCH_STEPS=max(1, rmax.bit_length()),
        TILE_PROGRAMS=tile_programs,
        num_warps=4,
    )
    # ptotal is [NB, Hkv] but the table's rows are head-major, so transpose
    # before flattening.
    seqused = (ptotal.transpose(0, 1).reshape(-1) * page_size).to(torch.int32)
    return table, seqused, pdrop
