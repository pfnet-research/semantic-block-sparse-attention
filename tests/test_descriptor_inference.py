"""The cluster descriptor term in the selector.  Needs a GPU; the selector is Triton.

With gamma at zero the descriptor must leave the selection unchanged, and with
gamma non-zero it must change it.
"""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _inputs(device, *, q=64, hq=4, hkv=2, r=2, df=64, c=64, dm=6):
    # df=64 because the selection kernel calls tl.dot, which needs
    # K >= 16; the real index_dim is 128, i.e. two halves of 64.
    torch.manual_seed(0)
    v = torch.nn.functional.normalize(
        torch.randn(q, hq, r, 2, df, device=device), dim=-1)
    vectors = v.flatten(-2).contiguous()
    logmix = torch.full((q, hq, r), -0.6931, device=device)
    cb = torch.nn.functional.normalize(
        torch.randn(hkv, 2, c, df, device=device), dim=-1).contiguous()
    # One row per query: _streaming_select_kernel indexes causal_sizes with
    # the query id and a real stride(0).
    sizes = torch.full((q, hkv, c), 32, dtype=torch.int32, device=device)
    # Spread across clusters: a descriptor that is the same everywhere adds a
    # constant to every score and cannot change a ranking.
    g = torch.rand(hkv, c, dm, device=device).pow(4.0) * 1000.0
    phi = torch.rand(q, hq, r, dm, device=device)
    return vectors, logmix, cb, sizes, g, phi


def _select(vectors, logmix, cb, sizes, *, g=None, phi=None, gamma=None):
    """Select on the torch scoring path.

    The torch and kernel paths differ in the last bits, so both sides of a
    comparison use the same path.
    """

    import os

    from semblo.vllm_plugin.fused_select import theta_select_bitmask_flat

    # set for this call only, so that later tests are not affected
    prev = os.environ.get("SEMBLO_FLAT_TORCH_SCORES")
    os.environ["SEMBLO_FLAT_TORCH_SCORES"] = "1"
    try:
        hkv, c = cb.shape[0], cb.shape[2]
        # One row per selection row, as fused_prefill builds them: with
        # g_union=1 that is one per query.
        bitmask = torch.zeros(vectors.shape[0], hkv, c // 32, dtype=torch.int32,
                              device=vectors.device)
        counts, tokens = theta_select_bitmask_flat(
            vectors, logmix, cb, sizes, bitmask,
            cluster_descriptors=g, query_descriptors=phi, descriptor_gamma=gamma,
            temperature=0.05, beta=0.0, target_tokens=256, g_union=1,
        )
    finally:
        if prev is None:
            os.environ.pop("SEMBLO_FLAT_TORCH_SCORES", None)
        else:
            os.environ["SEMBLO_FLAT_TORCH_SCORES"] = prev
    return bitmask.clone(), counts.clone(), tokens.clone()


@cuda
def test_gamma_zero_selects_exactly_what_the_centroid_selects():
    device = "cuda"
    vectors, logmix, cb, sizes, g, phi = _inputs(device)
    gamma = torch.zeros(vectors.shape[1], device=device)
    plain = _select(vectors, logmix, cb, sizes)
    zero = _select(vectors, logmix, cb, sizes, g=g, phi=phi, gamma=gamma)
    # Every selection row is written, so all three outputs are compared whole.
    for a, b, name in zip(plain, zero, ("bitmask", "counts", "tokens")):
        assert torch.equal(a, b), f"{name} moved with gamma at zero"


@cuda
def test_a_nonzero_gamma_changes_the_selection():
    device = "cuda"
    vectors, logmix, cb, sizes, g, phi = _inputs(device)
    # the centroid term enters as (q.c)/tau with tau=0.05, so it spans about
    # +-40; gamma * log<phi, g> has to be comparable to move a ranking
    gamma = torch.full((vectors.shape[1],), 5.0, device=device)
    plain = _select(vectors, logmix, cb, sizes)
    moved = _select(vectors, logmix, cb, sizes, g=g, phi=phi, gamma=gamma)
    assert not torch.equal(plain[0], moved[0]), "the descriptor did nothing"


@cuda
def test_the_three_descriptor_arguments_go_together():
    from semblo.vllm_plugin.fused_select import theta_select_bitmask_flat

    device = "cuda"
    vectors, logmix, cb, sizes, g, phi = _inputs(device)
    bitmask = torch.zeros(1, cb.shape[0], cb.shape[2] // 32,
                          dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="all three"):
        theta_select_bitmask_flat(
            vectors, logmix, cb, sizes, bitmask, cluster_descriptors=g,
            temperature=0.05, beta=0.0, target_tokens=256, g_union=1)


# --- the descriptor inside the tensor-core kernel ---------------------------
#
# gamma=0 must change nothing, and with gamma non-zero the kernel must select
# what the torch form selects.
#
# DM = 64 here because tl.dot needs K >= 16.

def _select_on(path, vectors, logmix, cb, sizes, *, g=None, phi=None, gamma=None):
    import os

    from semblo.vllm_plugin.fused_select import theta_select_bitmask_flat

    prev = os.environ.get("SEMBLO_FLAT_TORCH_SCORES")
    os.environ["SEMBLO_FLAT_TORCH_SCORES"] = "1" if path == "torch" else "0"
    try:
        hkv, c = cb.shape[0], cb.shape[2]
        bitmask = torch.zeros(vectors.shape[0], hkv, c // 32,
                              dtype=torch.int32, device=vectors.device)
        counts, tokens = theta_select_bitmask_flat(
            vectors, logmix, cb, sizes, bitmask,
            cluster_descriptors=g, query_descriptors=phi,
            descriptor_gamma=gamma,
            temperature=0.05, beta=0.0, target_tokens=256, g_union=1,
        )
        return bitmask.clone(), counts.clone(), tokens.clone()
    finally:
        if prev is None:
            os.environ.pop("SEMBLO_FLAT_TORCH_SCORES", None)
        else:
            os.environ["SEMBLO_FLAT_TORCH_SCORES"] = prev


@cuda
def test_the_kernel_with_gamma_zero_is_the_kernel_without_a_descriptor():
    dev = "cuda"
    # c=256: the kernel stores a whole TILE_C of columns at a time, so C must
    # not be below the tile width.
    vectors, logmix, cb, sizes, g, phi = _inputs(dev, dm=64, c=256)
    gamma = torch.zeros(vectors.shape[1], device=dev)
    plain = _select_on("kernel", vectors, logmix, cb, sizes)
    zeroed = _select_on("kernel", vectors, logmix, cb, sizes,
                        g=g, phi=phi, gamma=gamma)
    for a, b in zip(plain, zeroed):
        assert torch.equal(a, b), (
            "gamma=0 changed the kernel's selection; the descriptor term is "
            "not exactly zero when it should be")


@cuda
def test_the_kernel_selects_what_the_torch_form_selects():
    dev = "cuda"
    vectors, logmix, cb, sizes, g, phi = _inputs(dev, dm=64, c=256)
    gamma = torch.full((vectors.shape[1],), 0.55, device=dev)
    ref = _select_on("torch", vectors, logmix, cb, sizes,
                     g=g, phi=phi, gamma=gamma)
    got = _select_on("kernel", vectors, logmix, cb, sizes,
                     g=g, phi=phi, gamma=gamma)

    # The descriptor must change the selection for the comparison to mean
    # anything.
    off = _select_on("kernel", vectors, logmix, cb, sizes)
    assert not torch.equal(off[0], got[0]), "the descriptor changed nothing"

    # Not bit-identical: the kernel's dots run on bf16 tensor cores while the
    # torch form casts to fp32, so marginal clusters can flip.  The selected
    # sets must agree up to a few bits.
    same = (ref[0] == got[0]).float().mean().item()
    total = ref[0].numel()
    assert same >= 1.0 - 4.0 / total, (
        f"kernel and torch selections agree on only {same:.6f} of the bitmask")


@cuda
def test_the_descriptor_refuses_the_approximated_normaliser():
    import os

    from semblo.vllm_plugin.fused_select import flat_mixture_scores_qtiled

    dev = "cuda"
    vectors, logmix, cb, sizes, g, phi = _inputs(dev, dm=64, c=256)
    gamma = torch.full((vectors.shape[1],), 0.55, device=dev)
    prev = os.environ.get("SEMBLO_SCORE_GNORM")
    os.environ["SEMBLO_SCORE_GNORM"] = "1"
    try:
        with pytest.raises(ValueError, match="GNORM"):
            flat_mixture_scores_qtiled(
                vectors, logmix, cb, temperature=0.05,
                group_desc=cb[:, :, :8].contiguous(),
                cluster_descriptors=g, query_descriptors=phi,
                descriptor_gamma=gamma,
            )
    finally:
        if prev is None:
            os.environ.pop("SEMBLO_SCORE_GNORM", None)
        else:
            os.environ["SEMBLO_SCORE_GNORM"] = prev
