"""Regression tests for RELEASE_AUDIT.md, module 02 (tfp_dists.py), critical C1-C8.

Each TFP wrapper is checked against the numpyro distribution it is meant to
mirror (np_dists.py backend), on log_prob or moments.
"""
import numpy as np
import pytest

pytest.importorskip("tensorflow_probability")

import jax
import jax.numpy as jnp
import numpyro.distributions as nd

from BayesForge.Distributions.tfp_dists import UnifiedDist as T


def _lp(d, x):
    return float(d.log_prob(jnp.asarray(x)))


# C1 -------------------------------------------------------------------------
def test_c1_zero_inflated_negative_binomial_builds_and_matches_mixture():
    d = T.zero_inflated_negative_binomial(
        total_count=10.0, probs=0.5, inflated_loc_probs=0.2, wrap=False)
    nb = nd.NegativeBinomialProbs(10.0, 0.5)  # failures before 10 successes
    # TFP NB(total_count, probs) counts successes; mean = n p / (1 - p) = 10
    assert np.isclose(float(d.mean()), 0.8 * 10.0, rtol=1e-4)
    x = T.zero_inflated_negative_binomial(
        total_count=10.0, probs=0.5, inflated_loc_probs=0.2,
        sample=True, shape=(4,), seed=1)
    assert x.shape == (4,)
    del nb


# C2 -------------------------------------------------------------------------
@pytest.mark.parametrize("y", [0, 1, 3, 7])
def test_c2_gamma_poisson_matches_numpyro(y):
    d = T.gamma_poisson(concentration=3.0, rate=2.0, wrap=False)
    ref = nd.GammaPoisson(3.0, 2.0)
    assert np.isclose(_lp(d, float(y)), float(ref.log_prob(y)), atol=1e-4)


def test_c2_gamma_poisson_scale_is_inverse_rate():
    a = T.gamma_poisson(concentration=3.0, scale=0.5, wrap=False)
    b = T.gamma_poisson(concentration=3.0, rate=2.0, wrap=False)
    assert np.isclose(_lp(a, 2.0), _lp(b, 2.0), atol=1e-6)


# C3 -------------------------------------------------------------------------
@pytest.mark.parametrize("y", [1, 5, 10])
def test_c3_discrete_uniform_is_a_pmf(y):
    d = T.discrete_uniform(low=1, high=10, wrap=False)
    ref = nd.DiscreteUniform(1, 10)
    assert np.isclose(_lp(d, y), float(ref.log_prob(y)), atol=1e-5)


def test_c3_discrete_uniform_zero_mass_off_support():
    d = T.discrete_uniform(low=1, high=10, wrap=False)
    assert _lp(d, 0) == -np.inf and _lp(d, 11) == -np.inf


def test_c3_discrete_uniform_samples_are_integers_in_range():
    x = np.asarray(T.discrete_uniform(low=1, high=10, sample=True, shape=(500,), seed=3))
    assert x.min() >= 1 and x.max() <= 10
    assert np.all(x == np.round(x))


# C4 -------------------------------------------------------------------------
def test_c4_unit_is_a_log_factor_like_numpyro():
    d = T.unit(log_factor=-1.5, wrap=False)
    ref = nd.Unit(jnp.array(-1.5))
    assert d.event_shape == ref.event_shape == (0,)
    assert np.isclose(_lp(d, jnp.zeros((0,))), float(ref.log_prob(jnp.zeros((0,)))))


# C5 -------------------------------------------------------------------------
def test_c5_inverse_gamma_rate_matches_numpyro():
    d = T.inverse_gamma(concentration=2.0, rate=4.0, wrap=False)
    ref = nd.InverseGamma(2.0, 4.0)
    for x in (0.5, 1.0, 3.0):
        assert np.isclose(_lp(d, x), float(ref.log_prob(x)), atol=1e-5)


def test_c5_inverse_gamma_positional_second_arg_is_rate():
    d = T.inverse_gamma(2.0, 4.0, wrap=False)
    assert np.isclose(float(d.mean()), float(nd.InverseGamma(2.0, 4.0).mean), rtol=1e-5)


# C6 -------------------------------------------------------------------------
def test_c6_pareto_positional_order_matches_numpyro():
    d = T.pareto(2.0, 3.0, wrap=False)  # numpyro: scale=2, alpha=3
    ref = nd.Pareto(2.0, 3.0)
    for x in (2.5, 4.0):
        assert np.isclose(_lp(d, x), float(ref.log_prob(x)), atol=1e-5)


def test_c6_pareto_without_alpha_raises():
    with pytest.raises((TypeError, ValueError)):
        T.pareto(scale=2.0, wrap=False)


# C7 -------------------------------------------------------------------------
@pytest.mark.parametrize("q", [0.5, 0.2, 0.9])
def test_c7_asymmetric_laplace_quantile_matches_numpyro(q):
    d = T.asymmetric_laplace_quantile(loc=0.0, scale=1.0, quantile=q, wrap=False)
    ref = nd.AsymmetricLaplaceQuantile(0.0, 1.0, q)
    for x in (-1.0, 0.3, 2.0):
        assert np.isclose(_lp(d, x), float(ref.log_prob(x)), atol=1e-5)


def test_c7_asymmetric_laplace_quantile_rejects_q_outside_open_interval():
    with pytest.raises(ValueError):
        T.asymmetric_laplace_quantile(loc=0.0, scale=1.0, quantile=1.0, wrap=False)


# C8 -------------------------------------------------------------------------
def test_c8_default_seed_draws_differ():
    a = np.asarray(T.normal(0.0, 1.0, shape=(5,), sample=True))
    b = np.asarray(T.normal(0.0, 1.0, shape=(5,), sample=True))
    assert not np.allclose(a, b)


def test_c8_seed_none_and_out_of_range_ints_do_not_crash():
    T.normal(0.0, 1.0, shape=(2,), sample=True, seed=None)
    T.normal(0.0, 1.0, shape=(2,), sample=True, seed=-1)
    T.normal(0.0, 1.0, shape=(2,), sample=True, seed=2 ** 32)


def test_c8_explicit_seed_reproducible_and_session_stream_reproducible():
    from BayesForge.Utils.rng import RNG
    np.testing.assert_array_equal(
        np.asarray(T.normal(0.0, 1.0, shape=(3,), sample=True, seed=7)),
        np.asarray(T.normal(0.0, 1.0, shape=(3,), sample=True, seed=7)))
    old = T._rng
    try:
        T._rng = RNG(11)
        s1 = [np.asarray(T.normal(0.0, 1.0, shape=(3,), sample=True)) for _ in range(2)]
        T._rng = RNG(11)
        s2 = [np.asarray(T.normal(0.0, 1.0, shape=(3,), sample=True)) for _ in range(2)]
    finally:
        T._rng = old
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a, b)
    assert not np.allclose(s1[0], s1[1])
