"""Regression tests for RELEASE_AUDIT.md, module BNN, majors MAJ-1..MAJ-8.

Run with:  pytest "Test/10.BNN/test_audit_bnn_major.py" -q
"""
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

from BayesForge.BNN.bnn import bnn, _prior_scale


# --- MAJ-1: softplus overflow ---------------------------------------------

def test_maj1_softplus_finite_and_grad_finite_at_extremes():
    x = jnp.array([-100.0, -1.0, 0.0, 1.0, 100.0, 1000.0])
    y = bnn.softplus(x)
    assert jnp.all(jnp.isfinite(y))
    # log(1 + exp(x)) ~ x for large x
    np.testing.assert_allclose(np.asarray(y[4]), 100.0, rtol=1e-5)
    g = jax.grad(lambda t: jnp.sum(bnn.softplus(t)))(jnp.array([100.0, -100.0]))
    assert jnp.all(jnp.isfinite(g))


# --- MAJ-2: exp overflow ----------------------------------------------------

def test_maj2_exp_clipped_at_80():
    y = bnn.exp(jnp.array([0.0, 80.0, 1000.0]))
    assert jnp.all(jnp.isfinite(y))
    np.testing.assert_allclose(np.asarray(y[2]), np.exp(80.0), rtol=1e-4)


# --- MAJ-3: attention scores double scaling ---------------------------------

def test_maj3_attention_scores_scale_by_sqrt_d_model():
    src = inspect.getsource(bnn.layer_attention)
    assert "jnp.sqrt(self.d_model)" in src
    assert "** 0.5)" not in src


# --- MAJ-4: compound symmetry uses sigma, not sigma^2 ------------------------

def test_maj4_compound_symmetry_uses_sigma_squared():
    src = inspect.getsource(bnn.layer_compound_symmetry)
    assert "sigma**2" in src or "sigma ** 2" in src


# --- MAJ-5: opt-in fan_in prior scale ----------------------------------------

@pytest.mark.parametrize("fan_in,ps,expected", [
    (16, None, 1.0),
    (16, "fan_in", 0.25),
    (9, "fan_in", 1.0 / 3.0),
    (16, 2.0, 2.0),
])
def test_maj5_prior_scale_helper(fan_in, ps, expected):
    np.testing.assert_allclose(_prior_scale(fan_in, ps), expected, rtol=1e-6)



def _site_scale(fn):
    """Scale of a traced Normal site, unwrapping Expanded/Independent wrappers."""
    while hasattr(fn, "base_dist"):
        fn = fn.base_dist
    return float(jnp.ravel(fn.scale)[0])


def test_maj5_cov_prior_scale_traced_scale():
    net = bnn(rand_seed=0)
    N, hidden = 6, 4
    a = jnp.arange(N, dtype=jnp.float32)
    b = -jnp.arange(N, dtype=jnp.float32)
    with numpyro.handlers.seed(rng_seed=0):
        tr_default = numpyro.handlers.trace(
            lambda: net.cov(hidden, N, a, b)).get_trace()
    assert _site_scale(tr_default['W1']['fn']) == pytest.approx(1.0)
    net2 = bnn(rand_seed=0)
    with numpyro.handlers.seed(rng_seed=0):
        tr_fan = numpyro.handlers.trace(
            lambda: net2.cov(hidden, N, a, b, prior_scale="fan_in")).get_trace()
    assert _site_scale(tr_fan['W1']['fn']) == pytest.approx(N ** -0.5, rel=1e-6)
    assert _site_scale(tr_fan['W2']['fn']) == pytest.approx(hidden ** -0.5, rel=1e-6)


def test_maj5_signatures_default_to_none():
    for fn in (bnn.cov, bnn.layer_attention):
        sig = inspect.signature(fn)
        assert "prior_scale" in sig.parameters
        assert sig.parameters["prior_scale"].default is None
    from BayesForge.Models.BNNC import bnnc
    for fn in (bnnc.model, bnnc.__call__):
        sig = inspect.signature(fn)
        assert "prior_scale" in sig.parameters
        assert sig.parameters["prior_scale"].default is None


# --- MAJ-6: get_rho silent NaN -----------------------------------------------

def test_maj6_get_rho_valid_shapes():
    net = bnn(rand_seed=0)
    rng = np.random.default_rng(0)
    post3 = jnp.asarray(rng.normal(size=(7, 6, 2)), dtype=jnp.float32)
    rho = net.get_rho(post3)
    assert rho.shape == (2, 2)
    assert jnp.all(jnp.isfinite(rho))
    np.testing.assert_allclose(np.asarray(jnp.diagonal(rho)), 1.0, rtol=1e-4)
    rho2 = net.get_rho(post3[0])  # (N, 2) accepted too
    assert rho2.shape == (2, 2)


def test_maj6_get_rho_n_lt_2_raises():
    net = bnn(rand_seed=0)
    with pytest.raises(ValueError, match="get_rho"):
        net.get_rho(jnp.ones((5, 1, 2)))


def test_maj6_get_rho_zero_variance_raises():
    net = bnn(rand_seed=0)
    post = jnp.stack([jnp.ones((6,)), jnp.arange(6.0)], axis=-1)  # a column constant
    with pytest.raises(ValueError, match="zero variance"):
        net.get_rho(post)


def test_maj6_get_rho_bad_ndim_raises():
    net = bnn(rand_seed=0)
    with pytest.raises(ValueError, match="get_rho"):
        net.get_rho(jnp.ones((6,)))


# --- MAJ-7: npe module-level bnn singleton not rebound ------------------------

def test_maj7_npe_bnn_rebound_by_bf_seed():
    try:
        from BayesForge import bf
        import BayesForge.Samplers.SBI.npe as npe
    except ImportError:
        pytest.skip("SBI dependencies not installed")
    m = bf(platform="cpu", rand_seed=3, print_devices_found=False)
    assert npe._BNN.dist.seed == 3
    assert npe._BNN.dist._rng is m._rng
    m2 = bf(platform="cpu", rand_seed=7, print_devices_found=False)
    assert npe._BNN.dist.seed == 7
    assert npe._BNN.dist._rng is m2._rng


# --- MAJ-8: layer_linear activation lookup ------------------------------------

def test_maj8_swish_activation_resolves():
    net = bnn(rand_seed=0)
    X = jnp.ones((2, 4))
    W = jnp.asarray(np.arange(12, dtype=np.float32).reshape(4, 3)) - 5.0
    out = net.layer_linear(X, W, activation="swish", verbose=False)
    assert out.shape == (2, 3)
    assert jnp.all(jnp.isfinite(out))
    expected = bnn.silu(jnp.matmul(jnp.ones((2, 4)), W))
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)


def test_maj8_unknown_activation_raises_value_error():
    net = bnn(rand_seed=0)
    X = jnp.ones((2, 4))
    W = jnp.ones((4, 3))
    with pytest.raises(ValueError, match="Unknown activation"):
        net.layer_linear(X, W, activation="cov", verbose=False)


def test_maj8_none_activation_is_identity():
    net = bnn(rand_seed=0)
    X = jnp.ones((2, 4))
    W = jnp.ones((4, 3))
    out = net.layer_linear(X, W, activation=None, verbose=False)
    np.testing.assert_allclose(np.asarray(out), np.asarray(jnp.matmul(X, W)), rtol=1e-6)
