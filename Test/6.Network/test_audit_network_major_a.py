"""Regression tests for RELEASE_AUDIT.md, module Network, majors M1-M6."""
import warnings

import numpy as np
import jax.numpy as jnp
import pytest

nx = pytest.importorskip("networkx")

from BayesForge.Network.metrics import met


# --------------------------------------------------------------------------- #
# M1 — public dijkstra(m, source) no longer shadowed by the betweenness helper
# --------------------------------------------------------------------------- #
def test_m1_public_dijkstra_wrapper_matches_networkx():
    W = np.array(
        [[0.0, 2.0, 0.0, 0.0],
         [2.0, 0.0, 1.0, 0.0],
         [0.0, 1.0, 0.0, 3.0],
         [0.0, 0.0, 3.0, 0.0]],
        dtype=np.float32,
    )
    got = np.asarray(met.dijkstra(jnp.asarray(W), 0))  # was TypeError before the fix
    G = nx.from_numpy_array(W)
    expected = nx.single_source_dijkstra_path_length(G, 0, weight="weight")
    want = np.array([expected[i] for i in range(W.shape[0])])
    np.testing.assert_allclose(got, want, atol=1e-5)


# --------------------------------------------------------------------------- #
# M2 — eigenvector: component_aware implemented, directed graphs use in-edges
# --------------------------------------------------------------------------- #
def _two_triangles_plus_edge():
    A = np.zeros((8, 8), dtype=np.float32)
    for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (6, 7)]:
        A[i, j] = A[j, i] = 1.0
    return A  # two triangles + one isolated edge


def test_m2_component_aware_true_differs_from_false():
    A = _two_triangles_plus_edge()
    x_global = np.asarray(met.eigenvector(jnp.asarray(A)))
    x_aware = np.asarray(met.eigenvector(jnp.asarray(A), component_aware=True))
    assert not np.allclose(x_global, x_aware)
    # component-aware: each component keeps unit L2 norm within itself
    np.testing.assert_allclose(x_aware[:3], 1.0 / np.sqrt(3.0), atol=1e-4)
    np.testing.assert_allclose(x_aware[3:6], 1.0 / np.sqrt(3.0), atol=1e-4)
    np.testing.assert_allclose(x_aware[6:8], 1.0 / np.sqrt(2.0), atol=1e-4)
    # global iteration (component_aware=False): the dominant component wins
    assert np.all(x_global[:3] > 0.3)
    assert np.all(x_global[6:8] < 1e-3)


def test_m2_directed_path_matches_networkx_receiver_ordering():
    A = np.zeros((3, 3), dtype=np.float32)
    A[0, 1] = A[1, 2] = 1.0  # 0 -> 1 -> 2
    got = np.asarray(met.eigenvector(jnp.asarray(A)))
    # NetworkX's eigenvector_centrality on a DiGraph uses in-edges: the
    # pure receiver (node 2) must rank highest, the pure source lowest.
    assert got[2] > got[1] > got[0]
    # networkx's eigenvector_centrality(_numpy) is undefined on this DAG (not
    # strongly connected): it raises AmbiguousSolution / does not converge, so
    # only the receiver ordering is checked.


# --------------------------------------------------------------------------- #
# M3 — clustering: self-loops ignored, asymmetric matrix warns, result in [0, 1]
# --------------------------------------------------------------------------- #
def test_m3_clustering_ignores_self_loop_and_stays_in_unit_interval():
    A = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # self-loop on node 0
    got = np.asarray(met.cc(jnp.asarray(A)))  # symmetric: no warning expected
    assert np.all((got >= 0.0) & (got <= 1.0))
    G = nx.Graph()
    G.add_edge(0, 1)  # the same graph without the self-loop
    clust = nx.clustering(G)
    want = np.array([clust[i] for i in range(2)])
    np.testing.assert_allclose(got, want, atol=1e-6)


def test_m3_asymmetric_matrix_uses_directed_clustering():
    # A directed 3-cycle: the directed (Fagiolo) coefficient, as nx.clustering on a DiGraph.
    A = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    got = np.asarray(met.cc(jnp.asarray(A)))
    assert np.all((got >= 0.0) & (got <= 1.0))
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    np.testing.assert_allclose(got, [nx.clustering(G)[i] for i in range(3)], atol=1e-6)


# --------------------------------------------------------------------------- #
# M4 — density: 0.0 for N <= 1, self-loops ignored, binary formula
# --------------------------------------------------------------------------- #
def test_m4_density_single_node_is_zero_not_nan():
    assert float(met.density(jnp.zeros((1, 1)))) == 0.0


def test_m4_density_ignores_self_loop_matches_networkx():
    A = np.array([[1.0, 1.0], [1.0, 0.0]], dtype=np.float32)  # self-loop on node 0
    got = float(met.density(jnp.asarray(A)))
    G = nx.Graph()
    G.add_edge(0, 1)  # the same graph without the self-loop
    assert got == pytest.approx(nx.density(G), abs=1e-6)


# --------------------------------------------------------------------------- #
# M5 — degree/strength: symmetric matrix with sym=False warns; normalise fixes
# --------------------------------------------------------------------------- #
def _triangle():
    return np.array(
        [[0.0, 1.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )


def test_m5_symmetric_degree_with_sym_false_warns_sym_true():
    A = _triangle()
    with pytest.warns(UserWarning, match="sym=True"):
        got = np.asarray(met.degree(jnp.asarray(A)))
    np.testing.assert_allclose(got, 2.0 * np.array([2.0, 2.0, 2.0]))  # double-counted


def test_m5_symmetric_strength_with_sym_false_warns_sym_true():
    A = _triangle()
    with pytest.warns(UserWarning, match="sym=True"):
        got = np.asarray(met.strength(jnp.asarray(A)))
    np.testing.assert_allclose(got, 2.0 * np.array([2.0, 2.0, 2.0]))


def test_m5_degree_sym_true_matches_networkx_and_does_not_warn():
    A = _triangle()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # sym=True must not warn
        got = np.asarray(met.degree(jnp.asarray(A), sym=True))
    G = nx.from_numpy_array(A)
    want = np.array([G.degree(i) for i in range(3)], dtype=float)
    np.testing.assert_allclose(got, want)


def test_m5_normalize_single_node_is_zero_not_nan():
    A = np.zeros((1, 1), dtype=np.float32)
    got = np.asarray(met.degree(jnp.asarray(A), normalize=True))
    assert np.all(np.isfinite(got))
    assert float(got[0]) == 0.0


def test_m5_normalize_directed_total_divides_by_two_n_minus_two():
    A = np.zeros((3, 3), dtype=np.float32)
    A[0, 1] = A[1, 0] = A[0, 2] = 1.0  # directed: node 0 has out 2 + in 1 = 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # asymmetric: no warning expected
        got = np.asarray(met.degree(jnp.asarray(A), normalize=True))
    np.testing.assert_allclose(got[0], 3.0 / (2.0 * (3.0 - 1.0)))  # 0.75, never > 1


# --------------------------------------------------------------------------- #
# M6 — diameter raises on disconnected graphs instead of returning inf
# --------------------------------------------------------------------------- #
def test_m6_diameter_raises_value_error_on_disconnected_graph():
    A = np.zeros((4, 4), dtype=np.float32)
    A[0, 1] = A[1, 0] = 1.0
    A[2, 3] = A[3, 2] = 1.0  # two components
    with pytest.raises(ValueError, match="disconnected"):
        met.diameter(jnp.asarray(A))


def test_m6_diameter_connected_path_still_works():
    A = np.zeros((3, 3), dtype=np.float32)
    A[0, 1] = A[1, 0] = A[1, 2] = A[2, 1] = 1.0  # path 0 - 1 - 2
    assert float(met.diameter(jnp.asarray(A))) == pytest.approx(2.0)


def test_m3_clustering_asymmetric_matches_networkx_digraph():
    A = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    got = np.asarray(met.cc(jnp.asarray(A)))
    assert np.all((got >= 0.0) & (got <= 1.0))
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    np.testing.assert_allclose(got, [nx.clustering(G)[i] for i in range(3)], atol=1e-6)
