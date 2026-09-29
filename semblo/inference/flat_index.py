"""Flat key partition: spherical k-means over the prompt's routing keys.

The partition is produced in the layout the runtime uses for every index
(a codebook of cluster descriptors, a cluster id per key, sizes and indptr),
so the selector and the attention body are independent of how it was built.

The cluster count follows the prompt length:
C = clamp(L // target_size, min_clusters, max_clusters).
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:  # pragma: no cover - exercised only without triton
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _segment_sum_kernel(
        out_ptr,      # [C, D] fp32 out
        x_ptr,        # [N, D] fp32
        order_ptr,    # [N] int64, ascending within each cluster
        indptr_ptr,   # [C + 1] int64
        stride_x0,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Sum each cluster's members, in index order.

        One program per cluster, walking its slice of the sorted order.  The
        loop order is fixed, so the result is reproducible, unlike an
        atomicAdd reduction.  The serial sum has a relative error of about
        1e-6 against float64, a few times that of the matmul form.
        """
        c = tl.program_id(0)
        lo = tl.load(indptr_ptr + c)
        hi = tl.load(indptr_ptr + c + 1)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        acc = tl.zeros([BLOCK_D], tl.float32)
        for i in tl.range(lo, hi):
            row = tl.load(order_ptr + i)
            acc += tl.load(x_ptr + row * stride_x0 + offs, mask=mask,
                           other=0.0)
        tl.store(out_ptr + c * D + offs, acc, mask=mask)


def _segment_sum_fused(x: torch.Tensor, assign: torch.Tensor,
                       num_clusters: int, counts: torch.Tensor) -> torch.Tensor:
    """Per-cluster sums by a sorted segmented reduction (Triton).

    Deterministic, and avoids the [chunk, C] one-hot of the matmul form.  It
    is faster per k-means iteration, but the index build is a small part of
    prefill, so it is not the default.  Selected by SEMBLO_FLAT_DET_KIND=fused.
    """
    if triton is None:
        raise RuntimeError("_segment_sum_fused needs triton")
    order = torch.argsort(assign, stable=True)
    indptr = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)]).to(
        torch.int64)
    out = torch.empty(num_clusters, x.shape[1], dtype=torch.float32,
                      device=x.device)
    block_d = 1 << max(0, (x.shape[1] - 1).bit_length())
    _segment_sum_kernel[(num_clusters,)](
        out, x.contiguous(), order.to(torch.int64), indptr, x.stride(0),
        D=x.shape[1], BLOCK_D=block_d,
        num_warps=int(os.environ.get("SEMBLO_SEGSUM_WARPS", "2")))
    return out


def flat_cluster_count(length: int, target_size: int, *, min_clusters: int = 64,
                       max_clusters: int = 16384) -> int:
    return int(min(max(length // max(target_size, 1), min_clusters),
                   max_clusters, length))


def _segment_sum_matmul(x: torch.Tensor, assign: torch.Tensor,
                        num_clusters: int, counts: torch.Tensor,
                        chunk: int = 8192) -> torch.Tensor:
    """Per-cluster sums as a chunked one-hot matmul (the default).

    A GEMM's reduction order is fixed by its algorithm, and chunks accumulate
    in loop order, so the result is reproducible run to run.  TF32 is
    disabled for the duration because its 10-bit mantissa would round the
    key vectors themselves.
    """
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        sums = torch.zeros(num_clusters, x.shape[1], dtype=torch.float32,
                           device=x.device)
        for lo in range(0, x.shape[0], chunk):
            hi = min(lo + chunk, x.shape[0])
            onehot = torch.zeros(hi - lo, num_clusters, dtype=torch.float32,
                                 device=x.device)
            onehot.scatter_(1, assign[lo:hi, None], 1.0)
            sums += onehot.T @ x[lo:hi].float()
        return sums
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


def _segment_sum(x: torch.Tensor, assign: torch.Tensor, num_clusters: int,
                 counts: torch.Tensor) -> torch.Tensor:
    """[C, D] per-cluster sums that do not depend on thread scheduling.

    ``index_add_`` lowers to float atomicAdd on CUDA, so its accumulation
    order, and with it the codebook, varies between identical runs.  Near a
    tie that changes an assignment and hence the model output.

    SEMBLO_FLAT_DET_KIND picks the implementation: ``matmul`` (default),
    ``fused`` (Triton segmented sum), or ``scan`` (differences of a prefix
    sum over the sorted sample, implemented below).  The scan form is much
    slower and slightly less accurate, and serves as an independent
    cross-check.
    """
    kind = os.environ.get("SEMBLO_FLAT_DET_KIND", "matmul")
    if kind == "fused":
        return _segment_sum_fused(x, assign, num_clusters, counts)
    if kind == "matmul":
        return _segment_sum_matmul(x, assign, num_clusters, counts)
    order = torch.argsort(assign, stable=True)
    csum = x[order].cumsum(dim=0)
    ends = counts.cumsum(dim=0) - 1                      # last row of each
    zero = torch.zeros(1, x.shape[1], dtype=csum.dtype, device=x.device)
    padded = torch.cat([zero, csum], dim=0)
    totals = padded[(ends + 1).clamp_min(0)]
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)[:-1]])
    heads = padded[starts]
    return torch.where(counts[:, None] > 0, totals - heads,
                       torch.zeros_like(totals))


def routing_space(key_index_vectors: torch.Tensor) -> torch.Tensor:
    """[.., I] -> [.., I] in the space the partition is fitted and scored in.

    ``SEMBLO_ROUTING_NORMALIZE`` picks the transform.  Keys and queries must
    use the same one (the query side goes through
    ``semblo.vllm_plugin.routing_map.routing_normalize``):

      unset / ``whole``  one unit vector over the whole vector (the default).
      ``half``          each half is unit-normalized separately, as for the
                        two factors of a product partition.
      ``none``          no normalization, for representatives that are plain
                        means of keys (the adjacent-block baseline).
      ``split:<n>``     the first n components and the rest are normalized
                        separately.  ``half`` is ``split:<d/2>``.

    Anything that fits a codebook for this runtime outside the index build
    should call this function, so that it uses the same transform.
    """
    mode = os.environ.get("SEMBLO_ROUTING_NORMALIZE", "whole") or "whole"
    if mode == "none":
        return key_index_vectors.float()
    if mode == "whole":
        return F.normalize(key_index_vectors.float(), dim=-1)
    dim = key_index_vectors.shape[-1]
    if mode == "half":
        cut = dim // 2
    elif mode.startswith("split:"):
        cut = int(mode.split(":", 1)[1])
        if not 0 < cut < dim:
            raise RuntimeError(
                f"SEMBLO_ROUTING_NORMALIZE={mode!r}: the split has to fall inside "
                f"the {dim} components")
    else:
        raise RuntimeError(
            f"SEMBLO_ROUTING_NORMALIZE={mode!r}: expected half, whole, none or "
            "split:<n>")
    x = key_index_vectors.float()
    return torch.cat(
        [F.normalize(x[..., :cut], dim=-1),
         F.normalize(x[..., cut:], dim=-1)], dim=-1)


def half_normalised_space(key_index_vectors: torch.Tensor) -> torch.Tensor:
    """The ``half`` transform, regardless of ``SEMBLO_ROUTING_NORMALIZE``."""
    halves = key_index_vectors.chunk(2, dim=-1)
    return torch.cat(
        [F.normalize(h.float(), dim=-1) for h in halves], dim=-1)


def fit_flat_codebook(
    key_index_vectors: torch.Tensor,   # [L, I]
    num_clusters: int,
    *,
    num_iters: int = 10,
    sample_size: int | None = 100000,
    generator: torch.Generator | None = None,
    chunk: int = 65536,
    init: str = "random",
    density_tau: float | None = None,
    density_pool: int = 65536,
    metric: str = "cosine",
) -> torch.Tensor:
    """k-means over the whole index vector ``key_index_vectors`` [L, I].

    ``metric``: "cosine" (spherical k-means, the default) unit-normalizes every
    vector, assigns by the largest cosine and re-normalizes the mean.
    "euclidean" keeps the vectors as they are, assigns by the smallest squared
    distance and uses the plain mean; queries must then be scored raw as well
    (SEMBLO_ROUTING_NORMALIZE=none).

    ``init``: "random" takes C sample points.  "kmeanspp" is D^2 sampling on
    the sample (each next centre drawn with probability proportional to its
    squared distance, 2 - 2 cos on the sphere, from the nearest centre so far).
    It helps when keys are concentrated around one direction, as in
    Nemotron 3.5, where random starts put several centres in one dense
    region.  "farthest" is greedy farthest-point seeding.

    ``density_tau`` draws the fit sample with weights 1 / (number of pool
    points within cosine ``density_tau``), from a pool of ``density_pool``
    uniform draws.

    Returns [C, I] centroids (unit-norm under "cosine").  The fit runs on a
    sample of ``sample_size`` points; the caller assigns the whole prompt
    with :func:`assign_flat_buckets`.
    """
    if key_index_vectors.ndim != 2:
        raise ValueError("key_index_vectors must have shape [L, I]")
    _check_metric(metric)
    x = _fit_space(key_index_vectors, metric)
    n = x.shape[0]
    fit_n = n if sample_size is None else min(n, int(sample_size))
    if fit_n < num_clusters:
        raise ValueError("sample must be at least as large as num_clusters")
    perm = torch.randperm(n, device=x.device, generator=generator)
    if density_tau is not None and metric != "cosine":
        raise ValueError("the density-weighted sample is defined by a cosine threshold")
    if density_tau is not None:
        # Density-weighted sample, drawn without replacement.  Heavily
        # repeated content then fills the sample in proportion to its variety
        # rather than its count.
        pool_n = min(n, max(int(density_pool), fit_n))
        pool = x[perm[:pool_n]]
        counts = torch.zeros(pool_n, device=x.device)
        for lo in range(0, pool_n, chunk):
            sims = pool[lo:lo + chunk] @ pool.T
            counts[lo:lo + chunk] = (sims > density_tau).sum(dim=1).float()
        weights = 1.0 / counts.clamp_min(1.0)
        pick = torch.multinomial(weights, fit_n, replacement=False, generator=generator)
        sample = pool[pick]
    else:
        sample = x[perm[:fit_n]]
    if init == "kmeanspp":
        centroids = _kmeanspp_init(sample, num_clusters, generator, metric)
    elif init == "farthest":
        centroids = _farthest_init(sample, num_clusters, generator, metric)
    elif init == "random":
        centroids = sample[torch.randperm(fit_n, device=x.device,
                                          generator=generator)[:num_clusters]] \
            .clone()
    else:
        raise ValueError(f"unknown flat k-means init {init!r}")
    # Deterministic sums by default (see _segment_sum).  They add about 2%
    # to a 256K prefill.  SEMBLO_FLAT_DETERMINISTIC=0 uses index_add_.
    deterministic = os.environ.get("SEMBLO_FLAT_DETERMINISTIC", "1") == "1"
    for _ in range(num_iters):
        assign = _argmax_sim_chunked(sample, centroids, chunk, metric)
        counts = torch.bincount(assign, minlength=num_clusters)
        if deterministic:
            sums = _segment_sum(sample, assign, num_clusters, counts)
        else:
            sums = torch.zeros(num_clusters, x.shape[1], dtype=torch.float32,
                               device=x.device)
            sums.index_add_(0, assign, sample)
        empty = counts == 0
        if bool(empty.any()):
            # Restart empty clusters on random points so that C cannot
            # shrink silently.
            repl = torch.randperm(fit_n, device=x.device,
                                  generator=generator)[: int(empty.sum())]
            sums[empty] = sample[repl]
            counts = counts.clone()
            counts[empty] = 1
        centroids = sums / counts[:, None].float()
        if metric == "cosine":
            centroids = F.normalize(centroids, dim=-1)
    return centroids.to(key_index_vectors.dtype)


METRICS = ("cosine", "euclidean")


def _check_metric(metric: str) -> None:
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")


def _fit_space(key_index_vectors: torch.Tensor, metric: str) -> torch.Tensor:
    x = key_index_vectors.float()
    return F.normalize(x, dim=-1) if metric == "cosine" else x


def _dist2_to_centre(sample: torch.Tensor, c: torch.Tensor, metric: str) -> torch.Tensor:
    """Squared distance from every row of ``sample`` [n, I] to one centre ``c`` [I]:
    the squared chord 2 - 2 cos on the sphere, the plain squared difference otherwise."""
    if metric == "cosine":
        return 2.0 - 2.0 * (sample @ c)
    return (sample - c).pow(2).sum(-1)


def _farthest_init(sample: torch.Tensor, num_clusters: int,
                   generator: torch.Generator | None, metric: str = "cosine") -> torch.Tensor:
    """Greedy farthest-point seeding on ``sample`` [n, I]: a random first
    centre, then repeatedly the point farthest from the centres chosen so far
    (on the sphere: the one whose largest cosine to them is smallest).
    Deterministic given the first point; a repeated key is never chosen twice."""
    n = sample.shape[0]
    dev = generator.device if generator is not None else sample.device
    first = int(torch.randint(n, (1,), device=dev, generator=generator))
    centres = [sample[first]]
    best = _dist2_to_centre(sample, sample[first], metric)    # distance to the nearest centre so far
    for _ in range(1, num_clusters):
        pick = int(torch.argmax(best))
        c = sample[pick]
        centres.append(c)
        best = torch.minimum(best, _dist2_to_centre(sample, c, metric))
    return torch.stack(centres).clone()


def _kmeanspp_init(sample: torch.Tensor, num_clusters: int,
                   generator: torch.Generator | None, metric: str = "cosine") -> torch.Tensor:
    """D^2 sampling of C centres from ``sample`` [n, I] (unit-norm under "cosine").

    The draws run on the generator's device (the flat build hands in a CUDA
    generator; the tests a CPU one)."""
    n = sample.shape[0]
    dev = generator.device if generator is not None else sample.device
    first = int(torch.randint(n, (1,), device=dev, generator=generator))
    centres = [sample[first]]
    best = _dist2_to_centre(sample, sample[first], metric)
    for _ in range(1, num_clusters):
        w = best.clamp_min(0.0).to(dev)
        total = float(w.sum())
        if total <= 0.0:
            pick = int(torch.randint(n, (1,), device=dev, generator=generator))
        else:
            pick = int(torch.multinomial(w / total, 1, generator=generator))
        c = sample[pick]
        centres.append(c)
        best = torch.minimum(best, _dist2_to_centre(sample, c, metric))
    return torch.stack(centres).clone()


def _argmax_sim_chunked(x: torch.Tensor, centroids: torch.Tensor,
                        chunk: int, metric: str = "cosine") -> torch.Tensor:
    """argmax_c <x_i, c> without materialising [N, C] for the whole set; under
    "euclidean" argmin_c |x_i - c|^2, i.e. argmax_c (<x_i, c> - |c|^2 / 2).

    The similarity GEMM dominates the flat build, so it runs in bf16 with
    fp32 accumulation.  An argmax is insensitive to the last mantissa bits,
    and fp32 costs about 10x more on tensor cores."""
    out = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    cb = centroids.to(torch.bfloat16)
    half_norm2 = None if metric == "cosine" else 0.5 * centroids.float().pow(2).sum(-1)
    for lo in range(0, x.shape[0], chunk):
        hi = min(lo + chunk, x.shape[0])
        sim = x[lo:hi].to(torch.bfloat16) @ cb.T
        if half_norm2 is not None:
            sim = sim.float() - half_norm2[None]
        out[lo:hi] = torch.argmax(sim, dim=1)
    return out


def assign_flat_buckets(key_index_vectors: torch.Tensor,
                        centroids: torch.Tensor,
                        *, chunk: int = 65536, metric: str = "cosine") -> torch.Tensor:
    """[L, I] keys -> [L] cluster ids (spherical nearest centroid; under
    "euclidean" the nearest by squared distance, vectors as they are)."""
    _check_metric(metric)
    return _argmax_sim_chunked(_fit_space(key_index_vectors, metric), centroids, chunk, metric)


# split_merge_sizes accumulates with index_add_ (float atomicAdd), so its
# centroids are not reproducible run to run.  It is only used when
# SEMBLO_FLAT_MIN_TOKENS or SEMBLO_FLAT_MAX_TOKENS is set.
def split_merge_sizes(
    assign: torch.Tensor,          # [L] cluster ids
    key_index_vectors: torch.Tensor,
    centroids: torch.Tensor,
    *,
    min_size: int,
    max_size: int,
    generator: torch.Generator | None = None,
    rounds: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bring cluster sizes into [min_size, max_size].

    Over-full clusters are split by a small k-means on their members, and
    under-full ones are merged into the nearest remaining centroid.
    Returns (assignment, centroids) with clusters renumbered densely."""
    x = F.normalize(key_index_vectors.float(), dim=-1)
    assign = assign.clone()
    n_clusters = int(centroids.shape[0])
    for _ in range(rounds):
        counts = torch.bincount(assign, minlength=n_clusters)
        over = torch.nonzero(counts > max_size).flatten()
        if over.numel() == 0:
            break
        for c in over.tolist():
            members = torch.nonzero(assign == c).flatten()
            parts = max(2, int((members.numel() + max_size - 1) // max_size))
            sub = fit_flat_codebook(x[members], parts, num_iters=5,
                                    sample_size=None, generator=generator)
            sub_a = assign_flat_buckets(x[members], sub)
            assign[members] = torch.where(
                sub_a == 0, torch.full_like(sub_a, c),
                sub_a - 1 + n_clusters)
            n_clusters += parts - 1
    for _ in range(rounds):
        counts = torch.bincount(assign, minlength=n_clusters)
        small = torch.nonzero((counts > 0) & (counts < min_size)).flatten()
        keep = torch.nonzero(counts >= min_size).flatten()
        if small.numel() == 0 or keep.numel() == 0:
            break
        cents = torch.zeros(n_clusters, x.shape[1], device=x.device)
        cents.index_add_(0, assign, x)
        cents = F.normalize(
            cents / counts.clamp_min(1)[:, None].float(), dim=-1)
        near = keep[torch.argmax(cents[small] @ cents[keep].T, dim=1)]
        remap = torch.arange(n_clusters, device=x.device)
        remap[small] = near
        assign = remap[assign]
    uniq, assign = torch.unique(assign, return_inverse=True)
    cents = torch.zeros(uniq.numel(), x.shape[1], device=x.device)
    cents.index_add_(0, assign, x)
    counts = torch.bincount(assign, minlength=uniq.numel())
    cents = F.normalize(cents / counts.clamp_min(1)[:, None].float(), dim=-1)
    return assign, cents.to(key_index_vectors.dtype)


def balanced_group_order(
    centroids: torch.Tensor,   # [C, I] cluster descriptors (fit space)
    span: int,
    *,
    iters: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group the C cluster descriptors into C/span groups of exactly span.

    The hierarchical scorer scores the groups first and then only the
    clusters inside the surviving groups, as one fixed-size tile per group.
    Equal group sizes avoid padding that tile.

    Balance comes from recursive halving.  At each level every node is split
    by 2-means, but the points are ranked by the difference of their
    similarities to the two centroids and the lower half goes left.  After
    log2(C/span) levels every leaf holds span descriptors.  All nodes at a
    level have the same size, so each level is one batched
    [nodes, size, I] step.

    Returns (order, group_descriptors):
      order [C]  -- old cluster ids in group-major order; cluster i of the
                    reordered index is old cluster order[i], and group g
                    owns the contiguous range [g*span, (g+1)*span)
      group_descriptors [C/span, I] -- unit-norm mean of each group
    """
    if centroids.ndim != 2:
        raise ValueError("centroids must have shape [C, I]")
    c_dim, i_dim = centroids.shape
    if span <= 0 or c_dim % span != 0:
        raise ValueError(f"span {span} must divide C {c_dim}")
    n_groups = c_dim // span
    if n_groups & (n_groups - 1):
        raise ValueError(f"C/span = {n_groups} must be a power of two")

    x = F.normalize(centroids.float(), dim=-1)
    order = torch.arange(c_dim, device=x.device)
    seg = c_dim
    while seg > span:
        nodes = c_dim // seg
        pts = x[order].view(nodes, seg, i_dim)
        # Deterministic 2-means init: the node's first point and the point
        # farthest from it.
        c0 = pts[:, 0]                                      # [nodes, I]
        far = torch.einsum("nsi,ni->ns", pts, c0).argmin(dim=1)
        c1 = pts[torch.arange(nodes, device=x.device), far]
        for _ in range(iters):
            # spherical: distance order == similarity order
            gap = (torch.einsum("nsi,ni->ns", pts, c1)
                   - torch.einsum("nsi,ni->ns", pts, c0))
            rank = torch.argsort(gap, dim=1, stable=True)
            half = seg // 2
            left, right = rank[:, :half], rank[:, half:]
            c0 = F.normalize(
                torch.gather(
                    pts, 1, left[:, :, None].expand(-1, -1, i_dim)
                ).mean(dim=1), dim=-1)
            c1 = F.normalize(
                torch.gather(
                    pts, 1, right[:, :, None].expand(-1, -1, i_dim)
                ).mean(dim=1), dim=-1)
        order = torch.gather(order.view(nodes, seg), 1, rank).reshape(-1)
        seg //= 2

    group_desc = F.normalize(
        x[order].view(n_groups, span, i_dim).mean(dim=1), dim=-1)
    return order, group_desc.to(centroids.dtype)


def group_half_radii(
    desc: torch.Tensor,        # [2, C, Df] unit-norm half descriptors
    group_desc: torch.Tensor,  # [2, G, Df] unit-norm half descriptors
) -> torch.Tensor:
    """[2, G] max distance from a group's half descriptor to its members'.

    The group score has to bound the best cluster in the group, which the
    group mean alone does not.  With r the group radius,
    q.c = q.mu + q.(c - mu) <= q.mu + ||q|| r, so q.mu + ||q|| r is an upper
    bound on any member's score, and a group is dropped only when even its
    best possible member loses.  In the kernel r is a per-group additive
    offset.
    """
    two, c_dim, df = desc.shape
    n_groups = group_desc.shape[1]
    span = c_dim // n_groups
    d = desc.float().view(two, n_groups, span, df) \
        - group_desc.float().view(two, n_groups, 1, df)
    return d.norm(dim=-1).amax(dim=-1)
