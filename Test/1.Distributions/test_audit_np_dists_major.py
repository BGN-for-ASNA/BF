"""Regression tests for RELEASE_AUDIT.md, module 01 (np_dists), majors M1-M5."""

import warnings
import numpy as np
import jax.numpy as jnp
import pytest

from BayesForge import bf
import BayesForge.Distributions.np_dists as npd


def _m():
    return bf(platform='cpu', print_devices_found=False, rand_seed=0)


def test_m1_discrete_masked_draws_warn_and_are_float_nan():
    m = _m()
    keep = jnp.array([True, False, True, False])
    with pytest.warns(UserWarning, match="discrete"):
        d = m.dist.poisson(5.0, sample=True, shape=(4,), mask=keep)
    d = np.asarray(d)
    assert np.isnan(d[1]) and np.isnan(d[3]) and not np.isnan(d[0])


def test_m1_continuous_masked_draws_no_warning():
    m = _m()
    keep = jnp.array([True, False])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        d = np.asarray(m.dist.normal(0, 1, sample=True, shape=(2,), mask=keep))
    assert np.isnan(d[1])


def test_m3_sine_bivariate_vonmises_honours_mask_and_to_jax():
    m = _m()
    keep = jnp.array([True, False, True])
    d = m.dist.sine_bivariate_vonmises(0.0, 0.0, 1.0, 1.0, correlation=0.0,
                                       sample=True, shape=(3,), mask=keep)
    assert np.isnan(np.asarray(d)[1]).all()
    s = m.dist.sine_bivariate_vonmises(0.0, 0.0, 1.0, 1.0, correlation=0.0,
                                       sample=True, shape=(3,), to_jax=False)
    assert type(s).__name__ == "SampledData"


def test_m4_improper_uniform_sample_raises_clearly():
    from numpyro.distributions import constraints
    m = _m()
    with pytest.raises(NotImplementedError, match="improper"):
        m.dist.improper_uniform(constraints.real, (), (), sample=True)


def test_m5_obs_name_cache_is_bounded():
    c = npd._BoundedCache(maxsize=3)
    for i in range(10):
        c[i] = i
    assert list(c) == [7, 8, 9]
    assert isinstance(npd._OBS_NAMES, npd._BoundedCache)
