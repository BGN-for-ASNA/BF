"""Regressions found by running the documentation pages.

1. Value checks on concrete inputs (empty block groups, 1-based codes,
   self-ties, negative weights, quantile range) must not use jnp: under
   jax >= 0.4.36 a jnp operation evaluated while a model is traced (jit,
   shard_map in m.fit) returns a tracer even when its input is concrete, so
   int()/bool()/flatnonzero on it raise ConcretizationTypeError.
2. plot_gmm must pair the per-draw responsibilities (subsampled to at most
   300 draws by predict_gmm) with the same draws of mu/cov.
"""
import matplotlib
matplotlib.use("Agg")
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from BayesForge.Network.model_effects import Neteffect
from BayesForge.Network.util import array_manip
from BayesForge.Network.metrics import met


def _in_trace(fn):
    """Run fn() under jit and under shard_map (as m.fit does); its inputs are concrete."""
    from jax.sharding import Mesh, PartitionSpec as P

    def body(x):
        fn()
        return x
    jax.jit(body)(0.0)
    mesh = Mesh(np.array(jax.devices()[:1]), ("d",))
    jax.jit(jax.shard_map(body, mesh=mesh, in_specs=P(), out_specs=P()))(jnp.zeros(1))


def test_block_build_mu_ij_inside_trace():
    _in_trace(lambda: Neteffect.block_build_mu_ij(jnp.array([3, 4]), 2))


def test_block_build_mu_ij_still_rejects_empty_group():
    with pytest.raises(ValueError, match="no member"):
        Neteffect.block_build_mu_ij(jnp.array([3, 0]), 2)


def test_mat_to_edgl_inside_trace():
    mat = jnp.ones((3, 3)) - jnp.eye(3)
    _in_trace(lambda: array_manip.mat_to_edgl(mat))


def test_onnela_inside_trace():
    mat = jnp.ones((3, 3)) - jnp.eye(3)
    _in_trace(lambda: met.onnela_clustering_coefficient(mat))


def test_betweenness_inside_trace():
    mat = jnp.ones((3, 3)) - jnp.eye(3)
    _in_trace(lambda: met.betweenness(mat, 3))


def test_nodal_random_effects_inside_trace():
    import numpyro
    design = jnp.ones((4, 2))
    codes = jnp.array([1, 2])

    def run():
        with numpyro.handlers.seed(rng_seed=0):
            Neteffect.nodal_random_effects(design, codes, 2, 2)
    _in_trace(run)


def test_tfp_asymmetric_laplace_quantile_inside_trace():
    pytest.importorskip("tensorflow_probability")
    from BayesForge.Distributions.tfp_dists import UnifiedDist
    _in_trace(lambda: UnifiedDist.asymmetric_laplace_quantile(0.0, 1.0, 0.3))


class _FakeSampler:
    def __init__(self, S=320, K=2, D=2, seed=0):
        rng = np.random.default_rng(seed)
        centers = np.array([[-3.0, 0.0], [3.0, 0.0]])
        self._s = {
            "weights": jnp.asarray(np.full((S, K), 1.0 / K)),
            "mu": jnp.asarray(centers[None] + 0.05 * rng.normal(size=(S, K, D))),
            "sigma": jnp.ones((S, K, D)),
            "Lcorr": jnp.broadcast_to(jnp.eye(D), (S, K, D, D)),
        }

    def get_samples(self, *a, **k):
        return self._s


def test_plot_gmm_with_more_than_300_draws():
    from BayesForge.Models.GMM import plot_gmm
    rng = np.random.default_rng(1)
    data = np.vstack([rng.normal([-3, 0], 0.5, size=(20, 2)),
                      rng.normal([3, 0], 0.5, size=(20, 2))])
    plot_gmm(data, _FakeSampler(S=320), show_density=False)
    plt.close("all")
