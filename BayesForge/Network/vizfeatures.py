"""Python-side helpers for the NetExplorer feature set.

Network metrics come **only** from :class:`BayesForge.Network.metrics.met`
(no networkx / scipy). Each is computed defensively — if one fails it is
simply omitted from the payload and the GUI won't offer it.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------- #
# palettes
# --------------------------------------------------------------------- #
PALETTES = {
    "default": {
        "categorical": ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"],
        "sequential": ["#e8eef5", "#1b3a5c"],
    },
    # Okabe-Ito categorical + viridis sequential — colour-blind safe
    "cb": {
        "categorical": ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9",
                        "#D55E00", "#F0E442", "#000000"],
        "sequential": ["#440154", "#3b528b", "#21918c", "#5ec962", "#fde725"],
    },
}


def palette(name: str) -> dict:
    return PALETTES.get(name, PALETTES["default"])


# --------------------------------------------------------------------- #
# metrics (met only)
# --------------------------------------------------------------------- #
def _met():
    from BayesForge.Network.metrics import met

    return met


def _density(n: int, edges: int, undirected: bool) -> float:
    """Fallback density matching ``met.density`` (m15): self-loops excluded,
    ``E / (n (n - 1))`` for a directed graph and ``E / (n (n - 1) / 2)``
    (``= 2 E / (n (n - 1))``) for an undirected one, where ``edges`` counts
    each undirected edge **once**. Returns 0.0 for ``n < 2``."""
    if n < 2:
        return 0.0
    return edges / (n * (n - 1) / (2 if undirected else 1))


def network_stats(adj) -> dict:
    """Global summary — n, edges, density, mean degree, components, diameter,
    global clustering.

    Self-loops (the diagonal) are **excluded** from ``edges``, ``mean_degree``
    and the density fallback (m15); the primary density path (``met.density``)
    ignores them too, so both paths agree on the same convention.
    """
    from BayesForge.Network.layouts import _bfs_all_pairs

    A = np.asarray(adj, dtype=float)
    n = int(A.shape[0])
    abin = A != 0
    np.fill_diagonal(abin, False)  # self-loops out of every count (m15)
    undirected = bool(np.allclose(A, A.T))
    edges = int(abin.sum() / (2 if undirected else 1))
    deg = abin.sum(1) + abin.sum(0) if not undirected else abin.sum(1)

    components = diameter = None
    if n <= 600:
        sym = ((abin | abin.T)).astype(float)
        D = _bfs_all_pairs(sym)
        lab = -np.ones(n, dtype=int)
        c = 0
        for i in range(n):
            if lab[i] >= 0:
                continue
            lab[np.isfinite(D[i])] = c
            c += 1
        components = int(c)
        fin = D[np.isfinite(D) & (D > 0)]
        diameter = int(fin.max()) if fin.size else 0

    m = _met()
    try:
        dens = float(np.asarray(m.density(adj)))
    except Exception:
        # same convention as met.density (m15): directed E/(n(n-1)),
        # undirected 2E/(n(n-1)), self-loops excluded
        dens = _density(n, edges, undirected)
    try:
        gcc = float(np.nanmean(np.asarray(m.cc(adj))))
    except Exception:
        gcc = float("nan")

    return {
        "nodes": n,
        "edges": edges,
        "directed": not undirected,
        "density": round(dens, 4),
        "mean_degree": round(float(deg.mean()), 2) if n else 0.0,
        "components": components,
        "diameter": diameter,
        "global_clustering": round(gcc, 4) if np.isfinite(gcc) else None,
    }


def centralities(adj, directed: bool = False) -> dict:
    """name -> per-node float array, for the Size-by / Colour-by dropdowns."""
    import os
    import warnings

    import jax.numpy as jnp

    m = _met()
    A = np.asarray(adj, dtype=float)
    n = int(A.shape[0])
    Aj = jnp.asarray(A)
    Abin = jnp.asarray((A != 0).astype(float))
    out: dict[str, np.ndarray] = {}

    def _try(name, fn):
        try:
            v = np.asarray(fn(), dtype=float)
            if v.shape == (n,) and np.isfinite(v).all():
                out[name] = v
        except Exception:
            pass

    _try("degree", lambda: m.degree(Aj, sym=not directed))
    _try("strength", lambda: m.strength(Aj, sym=not directed))
    if directed:
        _try("indegree", lambda: m.indegree(Aj))
        _try("outdegree", lambda: m.outdegree(Aj))
        _try("instrength", lambda: m.instrength(Aj))
        _try("outstrength", lambda: m.outstrength(Aj))
    _try("eigenvector", lambda: m.eigenvector(Aj))
    # betweenness vmaps a per-source routine that materialises (N, N, N):
    # ~2 GB at N=800, ~13 GB at N=1500. Skip it above a configurable limit,
    # the same way the _try failure path omits a metric.
    limit = int(os.environ.get("BF_BETWEENNESS_MAX_N", 500))
    if n > limit:
        warnings.warn(
            f"vis_net: betweenness skipped for N={n} > {limit} nodes "
            "(memory O(N^3)); set BF_BETWEENNESS_MAX_N to raise the limit.",
            UserWarning,
            stacklevel=2,
        )
    else:
        _try("betweenness", lambda: m.betweenness(Abin, n_nodes=n, directed=directed))
    _try("clustering", lambda: m.cc(Aj))
    return out


# --------------------------------------------------------------------- #
# posterior / uncertainty edges
# --------------------------------------------------------------------- #
def edge_posterior_summary(draws):
    """``(S, N, N)`` posterior edge-weight draws -> (mean, prob, lo, hi)
    matrices, where ``prob`` = P(weight > 0)."""
    d = np.asarray(draws, dtype=float)
    if d.ndim != 3 or d.shape[1] != d.shape[2]:
        raise ValueError("weight_posterior must be a (draws, N, N) array")
    mean = d.mean(0)
    prob = (d > 0).mean(0)
    lo = np.quantile(d, 0.05, axis=0)
    hi = np.quantile(d, 0.95, axis=0)
    np.fill_diagonal(mean, 0.0)
    return mean, prob, lo, hi
