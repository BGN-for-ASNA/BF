"""Regression tests for RELEASE_AUDIT.md, Main M-06 and Integration X4."""

import jax
import jax.numpy as jnp
import numpy as np

from BayesForge import bf


def _m(seed=0):
    return bf(platform="cpu", print_devices_found=False, rand_seed=seed)


def _normal_model(m):
    def model(y):
        mu = m.dist.normal(0, 5, name="mu")
        m.dist.normal(mu, 1, obs=y, name="y")
    return model


def _mean_stat(y):
    return jnp.mean(y, keepdims=True)


# Names attached to the diag *instance* by bind_diag_to_model (patch_diag.py);
# the raw Diag2 class defines none of them (X4: AttributeError after abc/svi).
_BOUND_NAMES = ("ppc_density", "summary", "density", "posterior", "model_checks")

# Ensure diag surface is bound to the model after inference
def _assert_bound_diag_surface(m):
    assert not hasattr(type(m.diag), "ppc_density")
    for name in _BOUND_NAMES:
        assert hasattr(m.diag, name), name


def _svi_normal():
    m = _m()
    m.data_on_model = {"y": jnp.zeros(5)}
    m.svi(_normal_model(m), num_steps=20, num_samples=8, seed=0)
    return m


# M-06 ------------------------------------------------------------------------------
def test_m06_svi_sampler_declares_one_chain():
    m = _svi_normal()
    assert m.sampler.num_chains == 1
    grouped = m.sampler.get_samples(group_by_chain=True)
    assert "mu" in grouped
    for site, draws in grouped.items():
        assert np.asarray(draws).shape[0] == 1, site


# X4 --------------------------------------------------------------------------------
def test_x4_svi_binds_diag_surface():
    _assert_bound_diag_surface(_svi_normal())


def test_x4_sbi_binds_diag_surface():
    y = 1.5 + jax.random.normal(jax.random.PRNGKey(1), (20,))
    m = _m()
    m.abc(_normal_model(m), data={"y": y}, summary_stats=_mean_stat,
          num_simulations=20_000, quantile=0.05, progress_bar=False, seed=0)
    _assert_bound_diag_surface(m)
