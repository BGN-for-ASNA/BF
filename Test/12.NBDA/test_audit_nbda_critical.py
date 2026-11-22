"""Regression tests for RELEASE_AUDIT.md, module 11 (NBDA), critical C1-C7."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

from BayesForge.NBDA.NBDA import NBDA


def _nbda(n=4, t=3):
    rng = np.random.default_rng(0)
    net = jnp.asarray(rng.integers(0, 2, size=(n, n)).astype(float))
    status = jnp.asarray(np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=float)[:n, :t])
    return NBDA(net, status)


def _base_dist(fn):
    """Walk the .base_dist chain down to the distribution holding the parameters."""
    while hasattr(fn, 'base_dist') and not hasattr(fn, 'loc'):
        fn = fn.base_dist
    return fn


def test_c1_priors_use_requested_distributions():
    nb = _nbda()
    nb.import_covNF(np.arange(4.0).reshape(4, 1), names=['age'])
    nb.get_cov()
    trace = numpyro.handlers.trace(numpyro.handlers.seed(nb.model, rng_seed=0)).get_trace(CES=True)

    fn = _base_dist(trace['social_learning']['fn'])
    assert np.isclose(float(fn.loc), 0.0)
    assert np.isclose(float(fn.scale), 2.0)

    fn = _base_dist(trace['ces_alpha']['fn'])
    assert np.isclose(float(fn.low), 0.0001)
    assert np.isclose(float(fn.high), 0.9999)


def test_c2_covNF_learner_on_axis0():
    nb = _nbda()
    df = np.array([[10.0], [20.0], [30.0], [40.0]])
    ci, cj = nb.covNF_dims(jnp.asarray(df), 4, 3, 1)
    # axis 0 = learner i, axis 1 = demonstrator j
    expected_i = np.broadcast_to(df[:, None, None, :], (4, 4, 3, 1))
    expected_j = np.broadcast_to(df[None, :, None, :], (4, 4, 3, 1))
    np.testing.assert_allclose(np.asarray(ci), expected_i)
    np.testing.assert_allclose(np.asarray(cj), expected_j)


def test_c3_covNV_shape_and_axes():
    nb = _nbda()
    covV = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    ci, cj = nb.convert_covNV(jnp.asarray(covV))
    assert ci.shape == (4, 4, 3, 2)
    c = np.transpose(covV, (1, 2, 0))  # (n, t, num_variables)
    np.testing.assert_allclose(np.asarray(ci), np.broadcast_to(c[:, None, :, :], (4, 4, 3, 2)))
    np.testing.assert_allclose(np.asarray(cj), np.broadcast_to(c[None, :, :, :], (4, 4, 3, 2)))


def test_c4_social_weight_is_probability():
    nb = _nbda()
    status = jnp.asarray(np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=float)[:, :, None])
    lk = nb.compute_probs(
        D_asocial=jnp.ones((4, 3, 1)),
        asoc=jnp.array([-3.0]),  # theta ~ 0.05, so a negative weight gives lk < 0
        D_social=jnp.zeros((4, 4, 3, 1)),
        soc=jnp.zeros(1),
        network=-5 * jnp.ones((4, 4, 3, 1)),
        status=status,
        N=4,
        T=3,
    )
    lk = np.asarray(lk)
    finite = lk[~np.isnan(lk)]
    assert finite.size > 0
    assert np.all(finite >= 0.0)
    assert np.all(finite <= 1.0)


def test_c5_link_gradients_finite_at_extremes():
    from BayesForge.Utils.link import link

    for x in (-200.0, 200.0):
        assert np.isfinite(jax.grad(link.inv_logit)(x))
        assert np.isfinite(jax.grad(link.inv_logit_scale)(x))
    assert np.isclose(link.inv_logit_scale(0.3), 2 / (1 + np.exp(-0.3)) - 1)


def test_c6_print_model_runs(monkeypatch):
    mod = importlib.import_module('BayesForge.NBDA.NBDA')
    monkeypatch.setattr(mod, 'display', lambda *a, **k: None)
    nb = _nbda()
    nb.import_covNF(np.arange(4.0).reshape(4, 1), names=['age'])
    nb.get_cov()
    nb.print_model()


def test_c7_fit_without_model_raises_and_models_nbda_is_class():
    from BayesForge import bf
    from BayesForge.Models.models import models as _models

    m = bf(platform='cpu', print_devices_found=False)
    with pytest.raises(TypeError):
        m.fit(None, num_samples=5, num_warmup=5, num_chains=1)
    assert _models(m).nbda is NBDA
