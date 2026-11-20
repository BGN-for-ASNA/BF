"""Regression tests for RELEASE_AUDIT.md, module 05 (Main/main.py), critical M-01..M-03."""
import numpy as np
import jax.numpy as jnp
import numpyro.distributions as nd
import pytest

from BayesForge import bf


def test_m01_log_prob_conditions_on_the_data():
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)

    def model(y):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y)

    y = jnp.array([1.0, 2.0, 3.0])
    init_params, potential_fn, _, _ = m.log_prob(model, y=y)
    mu0 = float(init_params.z["mu"])
    expected = float(nd.Normal(0.0, 1.0).log_prob(mu0) + nd.Normal(mu0, 1.0).log_prob(y).sum())
    assert np.isclose(-float(potential_fn(init_params.z)), expected, rtol=1e-5)

# Ensure fit do not create an alias of the user's data dict
def test_m02_fit_does_not_alias_the_users_data_dict():
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)

    def model(y):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y)

    d = {"y": jnp.array([0.1, -0.2, 0.3])}
    m.fit(model, obs=d, num_samples=20, num_warmup=20, num_chains=1, progress_bar=False)
    assert m.data_on_model is not d
    assert set(d) == {"y"}


@pytest.mark.parametrize("backend", ["stan", "TFP", "numpyro "])
def test_m03_unknown_backend_is_rejected(backend):
    with pytest.raises(ValueError):
        bf(platform="cpu", backend=backend, print_devices_found=False)


def test_m03_svi_on_tfp_backend_raises():
    pytest.importorskip("tensorflow_probability")
    m = bf(platform="cpu", backend="tfp", print_devices_found=False)

    def model(y):
        yield m.dist.normal(0.0, 1.0, name="mu")

    m.data_on_model = {"y": jnp.zeros(3)}
    with pytest.raises(NotImplementedError):
        m.svi(model, num_steps=2)
