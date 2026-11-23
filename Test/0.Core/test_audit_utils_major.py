"""Regression tests for RELEASE_AUDIT.md, module Utils, majors (_parent_ids, kernels, link functions)."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge.Utils.Effects import _parent_ids
from BayesForge.Utils.Gaussian import gaussian
from BayesForge.Utils.link import link


# _parent_ids validation -------------------------------------------------------
def test_parent_ids_crossed_raises():
    # Child unit 1 observed under parent 0 and parent 1: crossed, not nested.
    with pytest.raises(ValueError, match="more than one parent"):
        _parent_ids(jnp.array([0, 1, 1]), jnp.array([0, 0, 1]), 2)


def test_parent_ids_missing_child_raises():
    # Child unit 1 never observed: its parent would silently default to 0.
    with pytest.raises(ValueError, match="no observation"):
        _parent_ids(jnp.array([0, 0]), jnp.array([1, 1]), 2)


def test_parent_ids_ok():
    out = _parent_ids(jnp.array([0, 1, 1, 2]), jnp.array([0, 0, 0, 1]), 3)
    assert np.array_equal(np.asarray(out), np.array([0, 0, 1]))


# kernel_sq_exp jitter ----------------------------------------------------------
def test_kernel_sq_exp_duplicate_points_finite():
    # m is the matrix of absolute distances; the kernel squares it internally.
    # Points 0 and 1 coincide, so without the default jitter K is singular
    # and cholesky returns NaN silently.
    x = jnp.array([0.0, 0.0, 1.0])
    m = jnp.abs(x[:, None] - x[None, :])
    z = jnp.zeros(3)
    K, L_SIGMA, k = gaussian.kernel_sq_exp(m, z)
    assert jnp.all(jnp.isfinite(K))
    assert jnp.all(jnp.isfinite(L_SIGMA))
    assert jnp.all(jnp.isfinite(k))


# kernel_exp float64 cast --------------------------------------------------------
def test_kernel_exp_no_truncation_warning():
    # With jax_enable_x64 off, the old astype(float64) silently truncated and
    # emitted a dtype warning; the kernel must now run in the input dtype.
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        m = jnp.array([[0.0, 1.0], [1.0, 0.0]])
        z = jnp.zeros(2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            K, L_SIGMA, k = gaussian.kernel_exp(m, z)
        assert jnp.all(jnp.isfinite(K))
        assert jnp.all(jnp.isfinite(L_SIGMA))
        assert jnp.all(jnp.isfinite(k))
    finally:
        jax.config.update("jax_enable_x64", prev)


# link functions: bounds and eps --------------------------------------------------
def test_logit_bounds():
    out = link.logit(jnp.array([0.0, 1.0]))
    assert float(out[0]) == -np.inf
    assert float(out[1]) == np.inf

    clipped = link.logit(jnp.array([0.0, 1.0]), eps=1e-6)
    assert jnp.all(jnp.isfinite(clipped))

    # Near 1 in float32: log(x) - log1p(-x) stays finite and accurate.
    x = jnp.array(1.0 - 1e-7)
    near_one = link.logit(x)
    assert bool(jnp.isfinite(near_one))
    expected = jnp.log(x) - jnp.log1p(-x)
    assert jnp.allclose(near_one, expected)

    assert jnp.all(jnp.isfinite(link.probit(jnp.array([0.0, 1.0]), eps=1e-6)))
    assert jnp.all(jnp.isfinite(link.cloglog(jnp.array([0.0, 1.0]), eps=1e-6)))
