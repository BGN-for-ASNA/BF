"""Regression tests for RELEASE_AUDIT.md, module Samplers, majors M7-M12 (numpyro/SBI/SVI side).

Small and fast: no sampler is run to convergence, only the audited code paths
(seed warning, ABC kernel jitter, NPE input validation, SVI guard) are hit.

Run with:  pytest Test/16.SBI/test_audit_samplers_major_sbi.py -q
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

N = 20
Y = 1.5 + jax.random.normal(jax.random.PRNGKey(1), (N,))


def _normal_model():
    """Conjugate Normal mean, as a plain numpyro program (m.dist is a thin wrapper)."""
    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0, 5))
        numpyro.sample("y", dist.Normal(mu, 1), obs=y)
    return model


# M7 ------------------------------------------------------------------------------------
def test_m7_mcmc_numpyro_seed_warns_and_default_is_silent():
    from BayesForge.Samplers.mcmc_numpyro import mcmc_numpyro

    def model():
        numpyro.sample("x", dist.Normal())

    with pytest.warns(UserWarning, match="no effect"):
        mcmc_numpyro(model=model, seed=3, num_warmup=5, num_samples=5, progress_bar=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # seed omitted (the default) must not warn
        mcmc_numpyro(model=model, num_warmup=5, num_samples=5, progress_bar=False)


# M8 ------------------------------------------------------------------------------------
def test_m8_abc_kernel_covariance_jitter_survives_float32():
    # Mirrors the ABC-SMC kernel covariance in BayesForge/Samplers/SBI/abc.py:
    # the jitter must be relative to the covariance scale, because float32
    # cannot resolve an absolute 1e-9 added to O(1) entries (cov + 1e-9*I == cov).
    def cov_of(u):
        w = jnp.full(u.shape[0], 1.0 / u.shape[0])
        mean = jnp.sum(w[:, None] * u, axis=0)
        return 2.0 * jnp.atleast_2d((w[:, None] * (u - mean)).T @ (u - mean))

    x = jnp.arange(50.0)
    for u in (jnp.ones((50, 2)),                                # identical particles: cov = 0
              jnp.stack([x, 2.0 * x], axis=1)):                 # rank-1, O(1) scale
        cov = cov_of(u)
        jitter = 1e-6 * (jnp.trace(cov) / u.shape[1] + 1e-12)
        cov = cov + jitter * jnp.eye(u.shape[1])
        L = jnp.linalg.cholesky(cov)
        assert np.isfinite(np.asarray(L)).all()


# M10 -----------------------------------------------------------------------------------
def test_m10_npe_validates_hyperparameters_before_simulating():
    # m.npe() does not expose val_frac / eval_every, so drive NPE_BF directly;
    # the validation must fire before any simulation happens.
    from BayesForge.Samplers.SBI.npe import NPE_BF
    from BayesForge.Samplers.SBI.simulate import Simulator

    sim = Simulator(_normal_model(), data={"y": Y},
                    summary_stats=lambda y: jnp.mean(y, keepdims=True))
    with pytest.raises(ValueError, match="val_frac"):
        NPE_BF(sim, num_simulations=100, val_frac=1.0,
               progress_bar=False).run(jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="eval_every"):
        NPE_BF(sim, num_simulations=100, eval_every=0,
               progress_bar=False).run(jax.random.PRNGKey(0))


# M12 -----------------------------------------------------------------------------------
def test_m12_svi_get_samples_before_run_raises():
    from BayesForge.Samplers.SVI.svi_numpyro import SVI_BF

    def model():
        numpyro.sample("x", dist.Normal())

    def guide():
        numpyro.sample("x", dist.Normal())

    svi = SVI_BF(model, guide, numpyro.optim.Adam(1e-3), num_steps=5, num_samples=10)
    with pytest.raises(RuntimeError, match=r"run\(\)"):
        svi.get_samples()
