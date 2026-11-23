"""Regression tests for RELEASE_AUDIT.md, module 02 (tfp_dists.py), majors M1-M4."""
import numpy as np
import pytest

pytest.importorskip("tensorflow_probability")

import jax.numpy as jnp
import numpyro.distributions as nd

from BayesForge.Distributions.tfp_dists import UnifiedDist as T


def test_m1_mask_model_mode_zeroes_masked_log_prob():
    y = jnp.array([0.5, -1.0, 2.0])
    keep = jnp.array([True, False, True])
    d = T.normal(0.0, 1.0, obs=y, mask=keep, wrap=False)
    ref = nd.Normal(0.0, 1.0).log_prob(y)
    assert np.isclose(float(d.log_prob(y)), float(ref[0] + ref[2]), atol=1e-5)


def test_m1_mask_sample_mode_nan():
    keep = jnp.array([True, False, True, False])
    d = np.asarray(T.normal(0.0, 1.0, sample=True, shape=(4,), mask=keep, seed=1))
    assert np.isnan(d[1]) and np.isnan(d[3]) and not np.isnan(d[0])
    with pytest.warns(UserWarning, match="discrete"):
        c = np.asarray(T.discrete_uniform(0, 5, sample=True, shape=(4,), mask=keep, seed=1))
    assert np.isnan(c[1])

# Vérifie que les paramètres scalaires avec des observations vectorielles produisent une somme de log-probabilités.
def test_m2_scalar_params_vector_obs():
    y = jnp.array([0.1, 0.2, -0.3, 1.0, 0.0])
    d = T.normal(0.0, 1.0, obs=y, wrap=False)
    assert d.log_prob(y).shape == ()
    assert np.isclose(float(d.log_prob(y)), float(nd.Normal(0.0, 1.0).log_prob(y).sum()), atol=1e-5)


def test_m2_multivariate_obs_keeps_event_dim():
    Y = jnp.zeros((4, 2))
    d = T.multivariate_normal(loc=jnp.zeros(2), scale_tril=jnp.eye(2), obs=Y, wrap=False)
    assert tuple(d.event_shape) == (4, 2) and tuple(d.batch_shape) == ()
    ref = nd.MultivariateNormal(jnp.zeros(2), scale_tril=jnp.eye(2)).log_prob(Y).sum()
    assert np.isclose(float(d.log_prob(Y)), float(ref), atol=1e-5)


def test_m2_sample_with_obs_warns():
    with pytest.warns(UserWarning, match="obs is ignored"):
        T.normal(0.0, 1.0, obs=jnp.zeros(3), sample=True, seed=0)


def test_m3_outer_wrapper_carries_user_name():
    d = T.normal(0.0, 1.0, obs=jnp.zeros(3), name='y', wrap=False)
    assert d.name == 'y'
    s = T.normal(0.0, 1.0, shape=(3,), name='mu', wrap=False)
    assert s.name == 'mu'


def test_m4_create_obj_returns_bare_distribution():
    import tensorflow_probability.substrates.jax.distributions as tfd
    d = T.normal(0.0, 1.0, obs=jnp.zeros(3), create_obj=True)
    assert type(d) is tfd.Normal


# M5-M11 ---------------------------------------------------------------------

def test_m5_no_silent_clamping():
    d = T.gamma(-1.0, 1.0, wrap=False)
    assert float(d.concentration) == -1.0


def test_m6_precision_matrix_and_positional_covariance():
    P = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    C = jnp.linalg.inv(P)
    x = jnp.array([0.4, -0.2])
    ref = float(nd.MultivariateNormal(jnp.zeros(2), covariance_matrix=C).log_prob(x))
    assert np.isclose(float(T.multivariate_normal(jnp.zeros(2), precision_matrix=P, wrap=False).log_prob(x)), ref, atol=1e-4)
    assert np.isclose(float(T.multivariate_normal(jnp.zeros(2), C, wrap=False).log_prob(x)), ref, atol=1e-4)
    with pytest.raises(ValueError):
        T.multivariate_normal(jnp.zeros(2), wrap=False)


def test_m6_scale_identity_multiplier():
    d = T.multivariate_normal(jnp.zeros(3), scale_identity_multiplier=2.0, wrap=False)
    ref = nd.MultivariateNormal(jnp.zeros(3), covariance_matrix=4.0 * jnp.eye(3)).log_prob(jnp.ones(3))
    assert np.isclose(float(d.log_prob(jnp.ones(3))), float(ref), atol=1e-4)


@pytest.mark.parametrize("fn,ref", [
    (lambda: T.bernoulli(0.3, wrap=False), nd.Bernoulli(probs=0.3)),
    (lambda: T.binomial(5, 0.3, wrap=False), nd.Binomial(5, probs=0.3)),
])
def test_m7_positional_probs(fn, ref):
    assert np.isclose(float(fn().log_prob(1.0)), float(ref.log_prob(1)), atol=1e-5)


def test_m8_gaussian_random_walk_matches_numpyro():
    x = jnp.array([0.1, 0.5, -0.2, 0.3])
    d = T.gaussian_random_walk(1.5, 4, wrap=False)
    ref = nd.GaussianRandomWalk(scale=1.5, num_steps=4).log_prob(x)
    assert np.isclose(float(d.log_prob(x)), float(ref), atol=1e-4)


def test_m9_zero_inflated_poisson_matches_numpyro():
    y = jnp.array([0.0, 1.0, 3.0])
    d = T.zero_inflated_poisson(0.3, 2.0, wrap=False)
    ref = nd.ZeroInflatedPoisson(gate=0.3, rate=2.0).log_prob(y)
    assert np.allclose(np.asarray(d.log_prob(y)), np.asarray(ref), atol=1e-5)
    assert np.isfinite(float(T.zero_inflated_poisson(0.3, 1e-8, wrap=False).log_prob(0.0)))


def test_m10_wishart_numpyro_keywords():
    S = jnp.array([[2.0, 0.3], [0.3, 1.0]])
    X = jnp.array([[3.0, 0.5], [0.5, 2.0]])
    ref = float(nd.Wishart(concentration=4.0, scale_matrix=S).log_prob(X))
    assert np.isclose(float(T.wishart(concentration=4.0, scale_matrix=S, wrap=False).log_prob(X)), ref, atol=1e-3)
    assert np.isclose(float(T.wishart(df=4.0, scale_tril=jnp.linalg.cholesky(S), wrap=False).log_prob(X)), ref, atol=1e-3)


def test_m11_gamma_default_rate():
    d = T.gamma(2.0, wrap=False)
    assert np.isclose(float(d.log_prob(1.5)), float(nd.Gamma(2.0).log_prob(1.5)), atol=1e-5)
