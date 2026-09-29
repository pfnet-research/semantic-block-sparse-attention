"""The suffix flush's re-pack keeps every invariant the decode kernels rely on."""
import torch

from semblo.vllm_plugin.suffix_flush import repack_apply, repack_plan


def _store(L, C, hkv, dim, align, seed=0):
    """A cluster-ordered store like the arena's: random assignment, page-aligned."""
    g = torch.Generator().manual_seed(seed)
    assign = torch.randint(0, C, (hkv, L), generator=g)
    width = L + 256 + C * align
    packed = torch.zeros(2, width, hkv, dim)
    posting = torch.zeros(hkv, width, dtype=torch.int32)
    sizes = torch.zeros(hkv, C, dtype=torch.int32)
    indptr = torch.zeros(hkv, C + 1, dtype=torch.int32)
    kv_by_pos = torch.randn(2, L, hkv, dim, generator=g)
    for h in range(hkv):
        counts = torch.bincount(assign[h], minlength=C)
        padded = ((counts + align - 1) // align) * align
        ip = torch.cat([torch.zeros(1, dtype=torch.int64), padded.cumsum(0)])
        order = torch.argsort(assign[h], stable=True)
        row = 0
        rows = []
        for c in range(C):
            members = order[assign[h][order] == c]
            rows.extend(members.tolist())
            rows.extend([order[0].item()] * int(padded[c] - counts[c]))   # padding repeats row 0, as _page_align_order does
        rows = torch.tensor(rows, dtype=torch.int64)
        packed[:, :rows.numel(), h, :] = kv_by_pos[:, rows, h, :]
        posting[h, :rows.numel()] = rows.to(torch.int32)
        sizes[h] = counts.to(torch.int32)
        indptr[h] = ip.to(torch.int32)
    return packed, posting, sizes, indptr, assign, kv_by_pos


def _check(packed, posting, sizes, indptr, assign_all, kv_by_pos_all, L_total, align):
    hkv, C = sizes.shape
    for h in range(hkv):
        seen = []
        for c in range(C):
            s, n = int(indptr[h, c]), int(sizes[h, c])
            assert s % align == 0
            assert int(indptr[h, c + 1]) - s >= n and (int(indptr[h, c + 1]) - s) % align == 0
            rows = torch.arange(s, s + n)
            pos = posting[h, rows].long()
            seen.extend(pos.tolist())
            assert (assign_all[h, pos] == c).all(), (h, c)
            assert torch.equal(packed[:, rows, h, :], kv_by_pos_all[:, pos, h, :])
            pad = torch.arange(s + n, int(indptr[h, c + 1]))
            if pad.numel():
                assert (posting[h, pad] == posting[h, 0]).all()
        assert sorted(seen) == list(range(L_total)), (h, len(seen))


def test_repack_appends_the_flushed_tokens_to_their_clusters():
    L, N, C, hkv, dim, align = 700, 96, 8, 2, 16, 16
    packed, posting, sizes, indptr, assign, kv_by_pos = _store(L, C, hkv, dim, align)
    g = torch.Generator().manual_seed(1)
    new_assign = torch.randint(0, C, (hkv, N), generator=g)
    flushed_kv = torch.randn(2, N, hkv, dim, generator=g)
    flushed_pos = torch.arange(L, L + N, dtype=torch.int32)
    tmp = torch.zeros(2, packed.shape[1], dim)
    widest = repack_apply(packed, posting, sizes, indptr, flushed_kv, flushed_pos, new_assign, align, tmp)
    assert widest <= packed.shape[1]
    assign_all = torch.cat([assign, new_assign], dim=1)
    kv_all = torch.cat([kv_by_pos, flushed_kv], dim=1)
    _check(packed, posting, sizes, indptr, assign_all, kv_all, L + N, align)
    # a second flush on top of the first
    new2 = torch.randint(0, C, (hkv, 40), generator=g)
    kv2 = torch.randn(2, 40, hkv, dim, generator=g)
    pos2 = torch.arange(L + N, L + N + 40, dtype=torch.int32)
    repack_apply(packed, posting, sizes, indptr, kv2, pos2, new2, align, tmp)
    _check(packed, posting, sizes, indptr, torch.cat([assign_all, new2], 1), torch.cat([kv_all, kv2], 1), L + N + 40, align)


def test_repack_plan_keeps_order_inside_a_cluster():
    old_sizes = torch.tensor([3, 0, 2])
    old_indptr = torch.tensor([0, 16, 16, 32])
    new_ids = torch.tensor([2, 0, 2, 1])          # positions p0..p3
    plan = repack_plan(old_sizes, old_indptr, new_ids, align=16)
    assert plan.new_sizes.tolist() == [4, 1, 4]
    assert plan.new_indptr.tolist() == [0, 16, 32, 48]
    # cluster 0: old rows 0,1,2 then position p1
    assert plan.kind[:4].tolist() == [0, 0, 0, 1] and plan.old_src[:3].tolist() == [0, 1, 2] and plan.new_src[3] == 1
    # cluster 2: old rows 16,17 then p0, p2 in position order
    assert plan.old_src[32:34].tolist() == [16, 17] and plan.new_src[34:36].tolist() == [0, 2]
    assert (plan.kind[36:48] == 2).all()


def test_empty_flush_is_the_identity():
    L, C, hkv, dim, align = 300, 4, 1, 8, 16
    packed, posting, sizes, indptr, assign, kv = _store(L, C, hkv, dim, align)
    before = (packed.clone(), posting.clone(), sizes.clone(), indptr.clone())
    tmp = torch.zeros(2, packed.shape[1], dim)
    repack_apply(packed, posting, sizes, indptr, torch.zeros(2, 0, hkv, dim), torch.zeros(0, dtype=torch.int32),
                 torch.zeros(hkv, 0, dtype=torch.int64), align, tmp)
    for a, b in zip(before, (packed, posting, sizes, indptr)):
        assert torch.equal(a, b)


def test_flush_range_keeps_the_newest_and_moves_the_rest():
    from semblo.vllm_plugin.suffix_flush import flush_range
    L, keep, every = 1000, 100, 50
    plen = L
    moved = []
    for written in range(0, 400):
        r = flush_range(L, written, plen, keep, every)
        live_before = L + written - plen
        assert live_before < keep + every + 1
        if r is not None:
            frm, to = r
            assert frm == plen and to == L + written - keep and to - frm >= every
            moved.append((frm, to)); plen = to
    assert moved and moved[0] == (L, L + 50)
    # the exact suffix never exceeds keep + every - 1 after the move is applied
    assert all(b - a == every for a, b in moved)
