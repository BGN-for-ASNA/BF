"""RELEASE_AUDIT integration minors W13, W14, X7, X15."""
import inspect
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import BayesForge.Main.main as main_mod
from BayesForge import bf
from BayesForge.BNN.bnn import bnn
from BayesForge.Diagnostic.ppc import get_yrep
from BayesForge.Parallel.simulate import _worker_env


# --- W13 ----------------------------------------------------------------------
# Ensure TFP branch binds BNN to BF seed
def test_w13_tfp_branch_binds_bnn_to_bf_seed():
    src = inspect.getsource(main_mod.bf.__init__)
    tfp = src[src.index('elif backend == "tfp"'):]
    assert "self.bnn.dist._rng = self._rng" in tfp
    assert "self.bnn.dist.seed = self.seed" in tfp

# Ensure BNN docstring documents layer state
def test_w13_bnn_docstring_documents_layer_state():
    assert "State" in bnn.__doc__ and "LAST layer" in bnn.__doc__


# --- W14 ----------------------------------------------------------------------
# Ensure PPC warns when event dims are flattened
def test_w14_ppc_warns_when_event_dims_are_flattened():
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
    y = jnp.asarray(np.random.default_rng(0).normal(size=(10, 2)))

    def model(y):
        mu = m.dist.normal(0, 1, shape=(2,), name="mu")
        m.dist.multivariate_normal(mu, jnp.eye(2), obs=y)

    m.fit(model, obs={"y": y}, num_warmup=50, num_samples=50, num_chains=1,
          progress_bar=False)
    with pytest.warns(UserWarning, match="event dimensions"):
        yrep = get_yrep(m, seed=0)
    assert yrep.shape == (50, 20)


# --- X7 -----------------------------------------------------------------------
# Ensure worker env carries parent x64
def test_x7_worker_env_carries_parent_x64():
    before = bool(jax.config.jax_enable_x64)
    try:
        for flag in (False, True):
            jax.config.update("jax_enable_x64", flag)
            assert _worker_env(1)["JAX_ENABLE_X64"] == ("1" if flag else "0")
    finally:
        jax.config.update("jax_enable_x64", before)


# --- X15 ----------------------------------------------------------------------
def test_x15_graph_does_not_advance_the_rng_stream():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = bf(platform="cpu", rand_seed=42, print_devices_found=False)
        m2 = bf(platform="cpu", rand_seed=42, print_devices_found=False)

    y = jnp.zeros(5)

    def model(y):
        noise = m.dist.normal(0, 1, sample=True, shape=(5,))
        mu = m.dist.normal(0, 1, name="mu")
        m.dist.normal(mu + 0 * noise, 1, obs=y)

    m.data_on_model = {"y": y}
    m.graph(model, data={"y": y}, draw=False)
    a = np.asarray(m.dist.normal(0, 1, sample=True, shape=(3,)))
    b = np.asarray(m2.dist.normal(0, 1, sample=True, shape=(3,)))
    np.testing.assert_array_equal(a, b)
