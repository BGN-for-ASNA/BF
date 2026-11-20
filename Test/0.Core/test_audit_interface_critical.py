"""Regression tests for RELEASE_AUDIT.md, Volet 2 interface criticals V1-V3, W1."""

import numpy as np
import jax.numpy as jnp
import pytest

from BayesForge import bf


def _m(seed=0):
    return bf(platform='cpu', print_devices_found=False, rand_seed=seed)


def _data(n=40):
    rng = np.random.default_rng(0)
    x = rng.normal(size=n).astype(np.float32)
    y = (1.0 + 0.5 * x + rng.normal(scale=0.3, size=n)).astype(np.float32)
    return dict(x=jnp.asarray(x), y=jnp.asarray(y))


def test_v1_v2_svi_drops_previous_mcmc_state():
    m = _m()
    def model_a(x, y):
        a = m.dist.normal(0, 1, name='a')
        b = m.dist.normal(0, 1, name='b')
        m.dist.normal(a + b * x, 0.5, obs=y)
    m.fit(model_a, obs=_data(), num_samples=50, num_warmup=50, num_chains=1, progress_bar=False)
    assert m.extra_fields_by_chain is not None
    m.svi(model_a, num_steps=200, num_samples=60)
    assert m.extra_fields_by_chain is None
    assert np.asarray(m.posteriors_full['b']).shape[0] == 60
    m.filter_posteriors(include='b')
    assert np.asarray(m.posteriors['b']).shape[0] == 60


def test_v3_data_from_df_is_rebuilt_for_a_new_model():
    import pandas as pd
    m = _m()
    d = _data()
    m.data(pd.DataFrame({k: np.asarray(v) for k, v in d.items()}))
    def model_xy(x, y):
        a = m.dist.normal(0, 1, name='a')
        m.dist.normal(a * x, 0.5, obs=y)
    def model_y(y):
        mu = m.dist.normal(0, 1, name='mu')
        m.dist.normal(mu, 0.5, obs=y)
    m.fit(model_xy, num_samples=20, num_warmup=20, num_chains=1, progress_bar=False)
    m.fit(model_y, num_samples=20, num_warmup=20, num_chains=1, progress_bar=False)
    assert set(m.data_on_model) == {'y'}


def test_v3_stale_user_data_raises_clearly():
    m = _m()
    def model_xy(x, y):
        a = m.dist.normal(0, 1, name='a')
        m.dist.normal(a * x, 0.5, obs=y)
    def model_z(z):
        mu = m.dist.normal(0, 1, name='mu')
        m.dist.normal(mu, 0.5, obs=z)
    m.fit(model_xy, obs=_data(), num_samples=20, num_warmup=20, num_chains=1, progress_bar=False)
    with pytest.raises(ValueError, match="previous fit"):
        m.fit(model_z, num_samples=20, num_warmup=20, num_chains=1, progress_bar=False)


def test_w1_singletons_follow_the_instance_that_reseeds():
    import BayesForge.Models.GMM as GMM
    m1 = _m(1)
    m2 = _m(2)
    m1.set_seed(11)
    assert GMM.dist._rng is m1._rng
    m2.set_seed(5)
    assert GMM.dist._rng is m2._rng
