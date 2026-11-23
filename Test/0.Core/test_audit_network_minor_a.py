"""Regression tests for RELEASE_AUDIT.md, module Network, minors m1-m5."""
import inspect

import numpy as np
import jax.numpy as jnp
import pytest


# --------------------------------------------------------------------------- #
# m1 — betweenness: asymmetric matrix rejected when directed=False,
#      n_nodes validated, path-count accumulators follow jax_enable_x64
# --------------------------------------------------------------------------- #
def _asymmetric_4x4():
    # 0 -> 1 -> 2 -> 3 plus 0 -> 2: paths exist in one direction only.
    return np.array(
        [[0.0, 1.0, 1.0, 0.0],
         [0.0, 0.0, 1.0, 1.0],
         [0.0, 0.0, 0.0, 1.0],
         [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )


def _symmetric_4x4():
    # Triangle (0, 1, 2) plus a pendant node 3 hanging off node 1.
    A = np.array(
        [[0.0, 1.0, 1.0, 0.0],
         [1.0, 0.0, 1.0, 1.0],
         [1.0, 1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    return A


def test_m1_asymmetric_matrix_rejected_when_undirected():
    from BayesForge.Network.metrics import met

    A = jnp.asarray(_asymmetric_4x4())
    with pytest.raises(ValueError, match="not symmetric"):
        met.betweenness(A, 4)  # directed=False is the default


def test_m1_asymmetric_matrix_accepted_when_directed():
    from BayesForge.Network.metrics import met

    A = jnp.asarray(_asymmetric_4x4())
    b = np.asarray(met.betweenness(A, 4, directed=True))
    assert b.shape == (4,)
    assert np.all(np.isfinite(b))
    # 1 is the only interior node of directed paths in this graph.
    assert b[1] > 0.0


def test_m1_n_nodes_mismatch_raises():
    from BayesForge.Network.metrics import met

    A = jnp.asarray(_symmetric_4x4())
    with pytest.raises(ValueError, match=r"n_nodes=3"):
        met.betweenness(A, 3)


def test_m1_non_square_matrix_raises():
    from BayesForge.Network.metrics import met

    A = jnp.zeros((2, 3), dtype=jnp.float32)
    with pytest.raises(ValueError, match="square"):
        met.betweenness(A, 2)


def test_m1_symmetric_undirected_values_stay_finite():
    from BayesForge.Network.metrics import met

    A = jnp.asarray(_symmetric_4x4())
    b = np.asarray(met.betweenness(A, 4))
    assert b.shape == (4,)
    assert np.all(np.isfinite(b))
    # Node 1 is the only cut node: it carries all the betweenness.
    assert b[1] > 0.0
    assert np.all(b[[0, 2, 3]] == pytest.approx(0.0, abs=1e-6))


def test_m1_symmetric_undirected_matches_networkx():
    nx = pytest.importorskip("networkx")
    from BayesForge.Network.metrics import met

    A = _symmetric_4x4()
    got = np.asarray(met.betweenness(jnp.asarray(A), 4))
    G = nx.from_numpy_array(A)
    expected = nx.betweenness_centrality(G, normalized=True)
    want = np.array([expected[i] for i in range(A.shape[0])])
    np.testing.assert_allclose(got, want, atol=1e-5)


def test_m1_path_count_dtype_follows_x64_config():
    # The accumulators must not be hardcoded float32 any more (RELEASE_AUDIT
    # Network m1): the helper picks float64 exactly when x64 is enabled.
    import jax

    from BayesForge.Network.metrics import _path_count_dtype

    expected = np.float64 if jax.config.jax_enable_x64 else np.float32
    assert jnp.zeros(3, dtype=_path_count_dtype()).dtype == expected


# --------------------------------------------------------------------------- #
# m2 — onnela_clustering_coefficient: diagonal zeroed, negative weights raise
# --------------------------------------------------------------------------- #
def test_m2_self_loop_no_longer_deflates_a_complete_triangle():
    from BayesForge.Network.metrics import met

    # Complete triangle 0-1-2 with a self-loop of weight 5 on the hub 0:
    # every node is still in a complete triangle, so every coefficient is 1.
    # Before the fix the hub measured 0.46 (the self-loop inflated max_w and
    # entered the einsum).
    A = np.array(
        [[5.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    cc = np.asarray(met.onnela_clustering_coefficient(jnp.asarray(A)))
    assert cc == pytest.approx(1.0, abs=1e-6)


def test_m2_plain_triangle_is_one_without_self_loops():
    from BayesForge.Network.metrics import met

    A = np.ones((3, 3), dtype=np.float32) - np.eye(3, dtype=np.float32)
    cc = np.asarray(met.onnela_clustering_coefficient(jnp.asarray(A)))
    assert cc == pytest.approx(1.0, abs=1e-6)


def test_m2_negative_weight_raises():
    from BayesForge.Network.metrics import met

    A = np.array(
        [[0.0, 1.0, -1.0],
         [1.0, 0.0, 1.0],
         [-1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    with pytest.raises(ValueError, match="negative weights"):
        met.onnela_clustering_coefficient(jnp.asarray(A))


# --------------------------------------------------------------------------- #
# m3 — no JIT warm-up inside met.__init__
# --------------------------------------------------------------------------- #
def test_m3_init_does_not_warm_up_betweenness():
    from BayesForge.Network.metrics import met

    src = inspect.getsource(met.__init__)
    assert "betweenness(" not in src
    assert "jnp.zeros" not in src


# --------------------------------------------------------------------------- #
# m4 — normalize / clustering_coefficient callable on an instance
# --------------------------------------------------------------------------- #
def test_m4_normalize_works_on_an_instance():
    from BayesForge.Network.metrics import met

    x = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    m = jnp.asarray(_symmetric_4x4())
    instance = met()
    got = np.asarray(instance.normalize(x, m))  # was TypeError before the fix
    want = np.asarray(met.normalize(x, m))
    np.testing.assert_allclose(got, want)
    # x / (N - 1)
    np.testing.assert_allclose(got, np.asarray(x) / 3.0, rtol=1e-6)


def test_m4_clustering_coefficient_works_on_an_instance():
    from BayesForge.Network.metrics import met

    A = jnp.asarray(_symmetric_4x4())
    instance = met()
    got = np.asarray(instance.clustering_coefficient(A))
    want = np.asarray(met.clustering_coefficient(A))
    np.testing.assert_allclose(got, want)


# --------------------------------------------------------------------------- #
# m5 — the three Dijkstra variants document their (different) semantics
# --------------------------------------------------------------------------- #
def test_m5_dijkstra_variants_document_their_semantics():
    from BayesForge.Network.metrics import met

    d1 = met.dijkstra_jax.__doc__
    assert "lengths" in d1 and "no edge" in d1

    d2 = met.single_source_dijkstra.__doc__
    assert "lengths" in d2 and "no edge" in d2

    d3 = met._dijkstra_betweenness.__doc__
    # The old one-line "Breadth-First Search (BFS)" docstring was wrong:
    # this is Dijkstra with path counts, not BFS.
    assert "Breadth-First" not in d3
    assert "presence" in d3 and "weight_matrix" in d3 and "sigma" in d3

    # The variants cross-reference each other so the next reader can tell
    # strengths from lengths without reading the code.
    assert "single_source_dijkstra" in d1
    assert "dijkstra_jax" in d2 and "dijkstra_jax" in d3
