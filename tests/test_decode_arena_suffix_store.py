"""The full-graph decode arena's fixed-address suffix store (own_prompt_cache).

The suffix kernel addresses a cache by absolute position: ``position //
block`` through a block table, ``position % block`` inside the block.  The
per-request store (``SembloRuntimeState.ensure_suffix_store``) is
allocated at prefill, so it does not exist when vLLM captures the decode step
at warm-up; the arena keeps one store at a fixed address and remaps the table
when a request's index is adopted.  These tests check, on the CPU, that the
mapping matches the per-request store's position by position.
"""
from __future__ import annotations

import pytest
import torch

from semblo.vllm_plugin import arena as arena_mod
from semblo.vllm_plugin.state import SembloRuntimeState, ProductPostingIndex


def _arena(monkeypatch, *, capacity=4096, reserve=512, block=256):
    monkeypatch.setenv("SEMBLO_SUFFIX_RESERVE", str(reserve))
    state = SembloRuntimeState()
    state.suffix_block_size = block
    return state, arena_mod.arena_for(
        state, 3, capacity=capacity, unit_tokens=2048, min_clusters_log2=3,
        num_kv_heads=2, head_dim=8, index_dim=4, page_align=16,
        kv_dtype=torch.float32, desc_dtype=torch.float32, device=torch.device("cpu"))


def test_the_store_exists_before_any_request_and_is_mapped_from_the_capacity(monkeypatch):
    state, arena = _arena(monkeypatch)
    # what the reader (_source_paged_context) looks up, present at warm-up
    assert state.suffix_kv[3] is arena.suffix_kv
    assert state.suffix_block_table is arena.suffix_table
    assert state.suffix_base_block == 4096 // 256
    # one block more than the reservation: the first block is partial
    assert arena.suffix_kv.shape == (2, 512 // 256 + 1, 256, 2, 8)
    table = arena.suffix_table[0]
    assert table[16:19].tolist() == [0, 1, 2]
    assert int(table[:16].abs().sum()) == 0


def test_the_write_lands_where_the_per_request_store_would_put_it(monkeypatch):
    state, arena = _arena(monkeypatch)
    arena.install_suffix_store(1000)          # a request of 1,000 tokens
    key = torch.randn(1, 2, 8)
    value = torch.randn(1, 2, 8)
    for pos in (1000, 1023, 1024, 1300):
        arena.write_suffix(torch.tensor([pos]), key + pos, value - pos)
    # the kernel's arithmetic, on the arena's table and store
    table = arena.suffix_table[0]
    for pos in (1000, 1023, 1024, 1300):
        physical = int(table[pos // 256])
        offset = pos % 256
        assert torch.equal(arena.suffix_kv[0, physical, offset], (key + pos)[0])
        assert torch.equal(arena.suffix_kv[1, physical, offset], (value - pos)[0])
    # and the same physical location the per-request store uses
    other = SembloRuntimeState()
    other.suffix_block_size = 256
    other.ensure_suffix_store(3, prompt_length=1000, max_model_len=1000 + 512,
                              num_kv_heads=2, head_dim=8, dtype=torch.float32,
                              device=torch.device("cpu"))
    for pos in (1000, 1023, 1024, 1300):
        other.write_suffix_kv(3, torch.tensor([pos]), key + pos, value - pos)
        physical_ours = int(arena.suffix_table[0, pos // 256])
        physical_theirs = int(other.suffix_block_table[0, pos // 256])
        assert physical_ours == physical_theirs
        assert torch.equal(other.suffix_kv[3][0, physical_theirs, pos % 256],
                           arena.suffix_kv[0, physical_ours, pos % 256])


def test_the_table_follows_the_adopted_prompt_and_its_address_does_not_move(monkeypatch):
    state, arena = _arena(monkeypatch)
    table_ptr = arena.suffix_table.data_ptr()
    store_ptr = arena.suffix_kv.data_ptr()
    hkv, length, clusters = 2, 3000, 4
    posting = torch.arange(length, dtype=torch.int32).repeat(hkv, 1)
    sizes = torch.full((hkv, clusters), length // clusters, dtype=torch.int32)
    indptr = torch.zeros(hkv, clusters + 1, dtype=torch.int32)
    indptr[:, 1:] = torch.cumsum(sizes, dim=1)
    index = ProductPostingIndex(
        codebooks=torch.zeros(hkv, 2, clusters, 2), posting_positions=posting,
        bucket_indptr=indptr, bucket_sizes=sizes, prompt_length=length,
        max_position=length - 1, num_non_empty_buckets=clusters, flat_clusters=clusters)
    index.packed_kv = torch.zeros(2, length + (-length) % 256, hkv, 8)
    arena.adopt(index, epoch=1)
    assert arena.suffix_table.data_ptr() == table_ptr
    assert arena.suffix_kv.data_ptr() == store_ptr
    assert state.suffix_base_block == 3000 // 256
    assert arena.suffix_table[0, 3000 // 256: 3000 // 256 + 3].tolist() == [0, 1, 2]
    assert int(arena.suffix_table[0, :3000 // 256].abs().sum()) == 0
    assert int(arena.prompt_length_t) == 3000
    # the store the reader will use is still the arena's
    assert state.suffix_kv[3] is arena.suffix_kv


def test_a_second_layer_shares_the_table(monkeypatch):
    state, arena3 = _arena(monkeypatch)
    arena7 = arena_mod.arena_for(
        state, 7, capacity=4096, unit_tokens=2048, min_clusters_log2=3,
        num_kv_heads=2, head_dim=8, index_dim=4, page_align=16,
        kv_dtype=torch.float32, desc_dtype=torch.float32, device=torch.device("cpu"))
    assert arena7.suffix_table is arena3.suffix_table
    assert arena7.suffix_kv is not arena3.suffix_kv
    assert state.suffix_kv[7] is arena7.suffix_kv and state.suffix_kv[3] is arena3.suffix_kv


def test_a_request_packs_straight_into_the_arena_and_adoption_copies_nothing(monkeypatch):
    """The pool hook hands the index the arena's first rows; the gathered copy
    lands in the arena, and adopt() leaves it in place."""
    state, arena = _arena(monkeypatch)
    hkv, length, clusters = 2, 700, 4
    order = torch.randperm(length, dtype=torch.int32)
    posting = torch.stack([order, order.flip(0)])
    sizes = torch.full((hkv, clusters), length // clusters, dtype=torch.int32)
    indptr = torch.zeros(hkv, clusters + 1, dtype=torch.int32)
    indptr[:, 1:] = torch.cumsum(sizes, dim=1)
    index = ProductPostingIndex(
        codebooks=torch.zeros(hkv, 2, clusters, 2), posting_positions=posting,
        bucket_indptr=indptr, bucket_sizes=sizes, prompt_length=length,
        max_position=length - 1, num_non_empty_buckets=clusters, flat_clusters=clusters)
    key, value = torch.randn(length, hkv, 8), torch.randn(length, hkv, 8)
    reference = ProductPostingIndex(
        codebooks=index.codebooks, posting_positions=posting, bucket_indptr=indptr,
        bucket_sizes=sizes, prompt_length=length, max_position=length - 1,
        num_non_empty_buckets=clusters, flat_clusters=clusters)
    reference.pack_prompt_kv(key, value)                       # its own buffer
    index.pack_prompt_kv(key, value, pool=lambda shape, dtype, device: arena.packed_rows(shape[1], shape, dtype))
    assert index.packed_kv.data_ptr() == arena.packed_kv.data_ptr()
    assert index.packed_kv.shape == reference.packed_kv.shape
    assert torch.equal(index.packed_kv, reference.packed_kv)
    assert index.packed_kv[0].is_contiguous() and index.packed_kv[1].is_contiguous()
    stamp = arena.packed_kv[:, : index.packed_kv.shape[1]].clone()
    arena.adopt(index, epoch=2)
    assert torch.equal(arena.packed_kv[:, : index.packed_kv.shape[1]], stamp)
    assert arena.index.packed_kv is arena.packed_kv
    with pytest.raises(RuntimeError):
        arena.packed_rows(10, (2, 10, 3, 8), torch.float32)   # wrong head count
