"""Regression tests for RELEASE_AUDIT.md, module 09 (Network), critical C1-C4 (vs networkx)."""
import numpy as np
import jax.numpy as jnp
import numpyro
import pytest

nx = pytest.importorskip("networkx")

from BayesForge.Network.metrics import met


def _star_plus_isolated():
    A = np.zeros((5, 5), dtype=np.float32)
    for i in (0, 1, 2):
        A[4, i] = A[i, 4] = 1.0
    return A  # node 3 isolated


def _nx_bc(A, **kw):
    G = nx.from_numpy_array(A)
    bc = nx.betweenness_centrality(G, normalized=True, **kw)
    return np.array([bc[i] for i in range(A.shape[0])])


def test_c1_weighted_betweenness_on_disconnected_graph_matches_networkx():
    A = _star_plus_isolated()
    got = np.asarray(met.betweenness(jnp.asarray(A), 5, weight_matrix=jnp.asarray(A)))
    np.testing.assert_allclose(got, _nx_bc(A, weight="weight"), atol=1e-5)


@pytest.mark.parametrize("weighted", [False, True])
def test_c2_endpoints_true_runs_and_matches_networkx(weighted):
    A = _star_plus_isolated()
    A[0, 1] = A[1, 0] = 1.0
    W = jnp.asarray(A) if weighted else None
    got = np.asarray(met.betweenness(jnp.asarray(A), 5, weight_matrix=W, endpoints=True))
    ref = _nx_bc(A, endpoints=True, **({"weight": "weight"} if weighted else {}))
    np.testing.assert_allclose(got, ref, atol=1e-5)


def test_c3_sender_receiver_with_receiver_only():
    from BayesForge.Network.model_effects import Neteffect
    ne = Neteffect()
    R = jnp.ones((4, 1))
    with numpyro.handlers.seed(rng_seed=0):
        out = ne.sender_receiver(sender_predictors=None, receiver_predictors=R)
    assert out is not None


def test_c4_single_source_dijkstra_on_binary_adjacency():
    B = np.zeros((4, 4), dtype=np.float32)
    for i in range(3):
        B[i, i + 1] = B[i + 1, i] = 1.0
    got = np.asarray(met.single_source_dijkstra(jnp.asarray(B), 0))
    np.testing.assert_allclose(got, [0, 1, 2, 3])
