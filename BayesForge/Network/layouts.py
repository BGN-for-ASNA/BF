"""Precomputed node layouts for the NetExplorer front-end.

The bundled d3 front-end computes Force / Circle / Linear / Multilayer live in
the browser. Layouts that need graph algorithms (eigdecomposition, shortest
paths, community detection, layering) are computed here in Python instead and
handed to the page as per-node ``lx`` / ``ly`` in ``[0, 1]`` — the front-end
just pins nodes to ``lx * width`` / ``ly * height``.

Every public function returns ``(lx, ly)`` as two float arrays in ``[0, 1]``
(``NaN`` for isolated nodes where a layout can't place them, which the page
leaves to the force sim). ``chord_matrix`` returns the group-by-group weight
matrix for the true d3.chord view.

``networkx`` and ``scipy`` are used when present but are not required; NumPy
fallbacks cover every layout except community detection, which then needs an
explicit group vector.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "PINNED",
    "radial",
    "arc",
    "geo",
    "spectral",
    "mds",
    "clustered",
    "layered",
    "chord_matrix",
    "compute",
]

# layouts whose coordinates come from here (page pins to lx/ly)
# Note: "clustered" is temporarily outcommented
PINNED = (# "clustered",
    # "mds",
    "spectral", "radial", "arc", "layered", "geo"
)



# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _unit(a: np.ndarray) -> np.ndarray:
    """Min-max each column of an (n, k) array into [0, 1]; flat columns -> 0.5."""
    a = np.asarray(a, dtype=float)
    lo = np.nanmin(a, axis=0)
    hi = np.nanmax(a, axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    out = (a - lo) / rng
    out[:, hi <= lo] = 0.5
    return out


def _sym_binary(adj: np.ndarray) -> np.ndarray:
    a = np.asarray(adj, dtype=float)
    a = ((a != 0) | (a.T != 0)).astype(float)
    np.fill_diagonal(a, 0.0)
    return a


def _bfs_all_pairs(adj_bin: np.ndarray) -> np.ndarray:
    """All-pairs shortest-path hop counts on an unweighted graph (BFS per
    source). ``inf`` between disconnected nodes."""
    n = adj_bin.shape[0]
    D = np.full((n, n), np.inf)
    for s in range(n):
        D[s, s] = 0.0
        frontier = np.array([s])
        d = 0
        seen = np.zeros(n, dtype=bool)
        seen[s] = True
        while frontier.size:
            d += 1
            nxt = np.unique(np.nonzero(adj_bin[frontier].sum(axis=0))[0])
            nxt = nxt[~seen[nxt]]
            if not nxt.size:
                break
            D[s, nxt] = d
            seen[nxt] = True
            frontier = nxt
    return D


def _spring(adj: np.ndarray, iters: int = 120, seed: int = 0) -> np.ndarray:
    """Tiny Fruchterman-Reingold, NumPy only. Returns (n, 2) in ~[-1, 1]."""
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-1, 1, (n, 2))
    if n < 2:
        return pos
    k = np.sqrt(1.0 / n)
    w = np.asarray(adj, dtype=float)
    w = np.maximum(w, w.T)
    for it in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta**2).sum(-1)) + 1e-9
        rep = (k * k / dist)[..., None] * (delta / dist[..., None])
        att = (dist / k)[..., None] * (delta / dist[..., None]) * w[..., None]
        disp = rep.sum(1) - att.sum(1)
        length = np.sqrt((disp**2).sum(-1)) + 1e-9
        step = 0.1 * (1.0 - it / iters) + 1e-3
        pos = pos + (disp / length[:, None]) * np.minimum(length, step)[:, None]
    return pos


# --------------------------------------------------------------------- #
# layouts
# --------------------------------------------------------------------- #
def radial(metric: np.ndarray, order: np.ndarray | None = None):
    """Concentric: radius from a nodal metric (rank -> [0.15, 1]), angle by
    ``order`` (default input order) around the circle."""
    metric = np.asarray(metric, dtype=float)
    n = len(metric)
    rank = np.argsort(np.argsort(metric))
    r = 0.15 + 0.85 * (rank / max(n - 1, 1))
    idx = np.arange(n) if order is None else np.argsort(np.asarray(order))
    ang = np.zeros(n)
    ang[idx] = 2 * np.pi * np.arange(n) / n
    lx = 0.5 + 0.5 * r * np.cos(ang)
    ly = 0.5 + 0.5 * r * np.sin(ang)
    return lx, ly


def arc(key: np.ndarray):
    """Nodes on a horizontal line, ordered by ``key`` (links drawn as arcs by
    the page)."""
    key = np.asarray(key)
    n = len(key)
    order = np.argsort(np.argsort(key, kind="stable"))
    lx = 0.05 + 0.9 * (order / max(n - 1, 1))
    ly = np.full(n, 0.5)
    return lx, ly


def geo(x: np.ndarray, y: np.ndarray):
    """Fixed coordinates from two columns (e.g. lon / lat), min-max scaled.
    ``y`` is flipped so north is up."""
    xy = _unit(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
    return xy[:, 0], 1.0 - xy[:, 1]


def spectral(adj: np.ndarray):
    """Laplacian eigenmaps: the two eigenvectors of the normalised graph
    Laplacian with smallest non-zero eigenvalue."""
    a = _sym_binary(adj)
    n = a.shape[0]
    deg = a.sum(1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    L = np.eye(n) - (dinv[:, None] * a * dinv[None, :])
    vals, vecs = np.linalg.eigh(L)
    keep = vecs[:, 1:3] if n > 2 else vecs[:, :2]
    xy = _unit(keep)
    return xy[:, 0], xy[:, 1]


def mds(adj: np.ndarray):
    """Classical MDS on shortest-path distances (2D)."""
    a = _sym_binary(adj)
    try:
        import networkx as nx

        D = nx.floyd_warshall_numpy(nx.from_numpy_array(a))
    except Exception:
        D = _bfs_all_pairs(a)
    D = np.asarray(D, dtype=float)
    finite = D[np.isfinite(D)]
    D[~np.isfinite(D)] = (finite.max() if finite.size else 1.0) * 2.0
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D**2) @ J
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1][:2]
    coords = vecs[:, order] * np.sqrt(np.maximum(vals[order], 0.0))
    xy = _unit(coords)
    return xy[:, 0], xy[:, 1]


def clustered(adj: np.ndarray, groups: np.ndarray | None = None, seed: int = 0):
    """Group-in-a-box: detect communities (or use ``groups``), lay each out with
    a spring, pack the community boxes on a grid."""
    a = _sym_binary(adj)
    n = a.shape[0]
    if groups is None:
        try:
            import networkx as nx

            g = nx.from_numpy_array(np.maximum(np.asarray(adj, float), np.asarray(adj, float).T))
            try:
                comms = nx.community.louvain_communities(g, seed=seed)
            except Exception:
                comms = nx.community.greedy_modularity_communities(g)
            lab = np.empty(n, dtype=int)
            for ci, nodes in enumerate(comms):
                for u in nodes:
                    lab[u] = ci
        except Exception:
            lab = np.zeros(n, dtype=int)
    else:
        lab = np.asarray(pd_factorize(groups))
    ncomm = int(lab.max()) + 1
    cols = int(np.ceil(np.sqrt(ncomm)))
    rows = int(np.ceil(ncomm / cols))
    lx = np.zeros(n)
    ly = np.zeros(n)
    for c in range(ncomm):
        idx = np.nonzero(lab == c)[0]
        if idx.size == 0:
            continue
        sub = _spring(a[np.ix_(idx, idx)], seed=seed + c)
        sub = _unit(sub) if idx.size > 1 else np.array([[0.5, 0.5]])
        gx, gy = c % cols, c // cols
        pad = 0.12
        lx[idx] = (gx + pad + (1 - 2 * pad) * sub[:, 0]) / cols
        ly[idx] = (gy + pad + (1 - 2 * pad) * sub[:, 1]) / rows
    return lx, ly


def layered(adj: np.ndarray, directed: bool = True, seed: int = 0):
    """Sugiyama-ish: assign each node to a layer (topological generation for a
    DAG, else BFS distance from the highest-degree node), spread within the
    layer. Layers run top (layer 0) to bottom."""
    a = np.asarray(adj, dtype=float)
    n = a.shape[0]
    layer = np.zeros(n, dtype=int)
    placed = False
    if directed:
        try:
            import networkx as nx

            g = nx.from_numpy_array((a != 0).astype(int), create_using=nx.DiGraph)
            if nx.is_directed_acyclic_graph(g):
                for gi, gen in enumerate(nx.topological_generations(g)):
                    for u in gen:
                        layer[u] = gi
                placed = True
        except Exception:
            placed = False
    if not placed:
        ab = _sym_binary(a)
        root = int(np.argmax(ab.sum(1)))
        D = _bfs_all_pairs(ab)[root]
        D[~np.isfinite(D)] = D[np.isfinite(D)].max() + 1 if np.isfinite(D).any() else 0
        layer = D.astype(int)
    K = int(layer.max()) + 1
    ly = layer / max(K - 1, 1)
    lx = np.zeros(n)
    rng = np.random.default_rng(seed)
    for k in range(K):
        idx = np.nonzero(layer == k)[0]
        m = idx.size
        offs = (np.arange(m) + 0.5) / m if m else np.array([])
        lx[idx] = 0.05 + 0.9 * offs[np.argsort(rng.random(m))]
    return lx, ly


def chord_matrix(groups: np.ndarray, adj: np.ndarray):
    """Aggregate edge weights into a ``G x G`` matrix for the d3.chord view.
    Returns ``(matrix, labels)`` with ``labels`` the distinct groups in order."""
    codes, labels = _factorize_labels(groups)
    G = len(labels)
    M = np.zeros((G, G))
    a = np.asarray(adj, dtype=float)
    n = a.shape[0]
    for i in range(n):
        for j in range(n):
            if a[i, j]:
                M[codes[i], codes[j]] += a[i, j]
    return M, list(labels)


# --------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------- #
def compute(name: str, adj, *, metric=None, key=None, x=None, y=None, groups=None,
            directed: bool = True):
    """Return ``(lx, ly)`` for a PINNED layout ``name``."""
    name = name.lower()
    if name == "radial":
        if metric is None:
            metric = _sym_binary(adj).sum(1)  # degree
        return radial(np.asarray(metric, float), order=groups)
    if name == "arc":
        if key is None:
            key = _sym_binary(adj).sum(1)
        return arc(np.asarray(key))
    if name == "geo":
        if x is None or y is None:
            raise ValueError("geo layout needs x and y (layout_x / layout_y columns)")
        return geo(x, y)
    if name == "spectral":
        return spectral(adj)
    if name == "mds":
        return mds(adj)
    if name == "clustered":
        return clustered(adj, groups=groups)
    if name == "layered":
        return layered(adj, directed=directed)
    raise ValueError(f"unknown pinned layout {name!r}; pick from {PINNED}")


# --------------------------------------------------------------------- #
# small factorize helpers (avoid a hard pandas dep here)
# --------------------------------------------------------------------- #
def _factorize_labels(values):
    vals = list(values)
    labels = sorted(set(vals), key=str)
    index = {v: i for i, v in enumerate(labels)}
    codes = np.array([index[v] for v in vals], dtype=int)
    return codes, labels


def pd_factorize(values):
    return _factorize_labels(values)[0]
