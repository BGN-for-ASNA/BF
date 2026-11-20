"""Regression tests for RELEASE_AUDIT.md, module 07 (Utils), the 5 critical findings."""
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as nd
import pytest

from BayesForge.Utils.Effects import effects
from BayesForge.Utils.Gaussian import gaussian
from BayesForge.Utils.link import link
from BayesForge.Utils.SampledData import SampledData


def test_varying_slope_non_centered_runs():
    cat = jnp.array([0, 1, 0, 1])
    grp = jnp.array([0, 0, 1, 1])
    with numpyro.handlers.seed(rng_seed=0):
        out = effects.varying_slope_non_centered(cat, grp, 2, 2, sample=True)
    assert out is not None


def test_varying_intercept_slope_uses_the_correlation_matrix():
    L = jnp.linalg.cholesky(jnp.array([[1.0, 0.6], [0.6, 1.0]]))
    sd = jnp.array([2.0, 0.5])
    tr = numpyro.handlers.trace(numpyro.handlers.seed(
        lambda: effects.varying_intercept_slope(
            N_group=3, group=jnp.array([0, 1, 2]), global_intercept=0.0, global_slope=0.0,
            group_std=sd, L_corr=L), rng_seed=0)).get_trace()
    site = [s for s in tr.values() if s["type"] == "sample" and s["name"].endswith("_params")][0]
    fn = site["fn"]
    while not hasattr(fn, "covariance_matrix"):
        fn = fn.base_dist
    cov = np.asarray(fn.covariance_matrix)
    cov = cov.reshape(-1, 2, 2)[0]
    expected = np.outer(sd, sd) * np.array([[1.0, 0.6], [0.6, 1.0]])
    np.testing.assert_allclose(cov, expected, rtol=1e-5)


def test_gaussian_process2_periodic_runs():
    D = jnp.abs(jnp.arange(4.0)[:, None] - jnp.arange(4.0)[None, :])
    K = gaussian.gaussian_process2(D, kernel_type="periodic", sigma=1.0)
    assert K.shape == (4, 4)


@pytest.mark.parametrize("eta", [0.5, 1.0, 4.0])
def test_lkjcor_matches_numpyro_density(eta):
    C = jnp.array([[1.0, 0.5], [0.5, 1.0]])
    ref = float(jnp.exp(nd.LKJ(2, eta).log_prob(C)))
    assert np.isclose(float(link.LKJcor(C, eta)), ref, rtol=1e-5)


def test_hdi_2d_is_per_column_and_nan_rejected():
    # 2-D draws used to be sorted along the wrong axis (silently wrong); each
    # column now gets the interval its own 1-D draws give.
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.normal(size=(200, 3)) * jnp.array([1.0, 2.0, 0.5]))
    lo, hi = SampledData(x).hdi(cred_mass=0.9)
    assert lo.shape == (3,) and hi.shape == (3,)
    for j in range(3):
        lo_j, hi_j = SampledData(x[:, j]).hdi(cred_mass=0.9)
        assert float(lo[j]) == float(lo_j) and float(hi[j]) == float(hi_j)
    with pytest.raises(ValueError):
        SampledData(jnp.array(1.0)).hdi()
    with pytest.raises(ValueError):
        SampledData(jnp.array([0.1, 0.2, jnp.nan, 0.4])).hdi()


def test_hdi_1d_still_correct():
    x = jnp.arange(100.0)
    lo, hi = SampledData(x).hdi(0.9)
    assert float(hi - lo) == 90.0
