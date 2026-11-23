"""Regression test for RELEASE_AUDIT.md, Volet 2 interface critical X1."""

import warnings
import numpy as np
import jax.numpy as jnp
import numpyro
import pytest

from BayesForge import bf


def test_x1_sample_true_inside_traced_model_warns():
    m = bf(platform='cpu', print_devices_found=False, rand_seed=0)
    y = jnp.asarray(np.random.default_rng(0).normal(size=20).astype(np.float32))
    def model(y):
        z = m.dist.normal(0, 1, sample=True)
        m.dist.normal(z, 1.0, obs=y)
    with pytest.warns(UserWarning, match="sample=True"):
        numpyro.handlers.trace(numpyro.handlers.seed(model, rng_seed=0)).get_trace(y)


def test_x1_plain_dgp_draw_does_not_warn():
    m = bf(platform='cpu', print_devices_found=False, rand_seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        x = m.dist.normal(0, 1, sample=True, shape=(5,))
        with numpyro.handlers.seed(rng_seed=0):
            m.dist.normal(0, 1, sample=True, shape=(3,))
    assert np.asarray(x).shape == (5,)
