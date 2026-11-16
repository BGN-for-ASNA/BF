"""bf.sample() semantics (posterior / samples / data / remove_obs /
infer_discrete / seed).

``sample()`` now covers three modes through ``posterior``: True draws the
posterior predictive from the fitted samples (RuntimeError if unfitted),
False draws the prior predictive (1000 draws by default), and a dict of
fixed parameter values predicts at those values (1 draw by default).
``samples`` subsamples the posterior without replacement (ValueError when
it exceeds the number of posterior draws), repeats a posterior dict, or
counts prior draws. ``remove_obs=False`` drops the observed sites from the
result, ``infer_discrete=True`` requires posterior=True, an observed site
left with the default name 'x' warns, and an explicit ``seed`` makes the
draws identical across calls."""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge import bf

X = jnp.linspace(-1, 1, 30)
Y = 1.0 + 2.0 * X + 0.1 * np.asarray(
    np.random.default_rng(0).normal(size=30), dtype=X.dtype)
X_NEW = jnp.linspace(-2, 2, 7)


def _linear_model(m):
    def model(x, y):
        a = m.dist.normal(0, 5, name="a")
        b = m.dist.normal(0, 5, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y, name="y")
    return model


@pytest.fixture(scope="module")
def fitted():
    """One bf instance fitted once on (X, Y); all posterior tests reuse it."""
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    m.fit(_linear_model(m), obs={"x": X, "y": Y},
          num_warmup=100, num_samples=100, num_chains=1, progress_bar=False)
    return m


def test_default_posterior_predictive(fitted):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fitted.sample(seed=0)
    assert np.asarray(out["y"]).shape == (100, 30)
    assert not [w for w in caught if "batch dimension" in str(w.message)]


def test_samples_subsample_and_overflow(fitted):
    out = fitted.sample(samples=10, seed=0)
    assert np.asarray(out["y"]).shape == (10, 30)
    with pytest.raises(ValueError):
        fitted.sample(samples=1000, seed=0)


def test_new_data_predictions(fitted):
    out = fitted.sample(data={"x": X_NEW}, seed=0)
    assert np.asarray(out["y"]).shape == (100, 7)


def test_prior_predictive(fitted):
    out = fitted.sample(posterior=False, seed=0)
    assert np.asarray(out["y"]).shape[0] == 1000
    assert "a" in out
    out = fitted.sample(posterior=False, samples=5, seed=0)
    assert np.asarray(out["y"]).shape[0] == 5


def test_posterior_dict_fixed_values(fitted):
    out = fitted.sample(posterior=dict(a=1.0, b=2.0, s=0.1), seed=0)
    y = np.asarray(out["y"])
    assert y.shape == (1, 30)
    assert np.allclose(y[0], 1 + 2 * np.asarray(X), atol=0.5)
    out = fitted.sample(posterior=dict(a=1.0, b=2.0, s=0.1), samples=3, seed=0)
    assert np.asarray(out["y"]).shape == (3, 30)


def test_posterior_draws_uses_each_draw(fitted):
    # Two distinct draws on axis 0: each one is used, not repeated as one fixed draw.
    draws = dict(a=jnp.array([0.0, 10.0]), b=jnp.array([0.0, 0.0]), s=jnp.array([0.01, 0.01]))
    y = np.asarray(fitted.sample(posterior_draws=draws, seed=0)["y"])
    assert y.shape == (2, 30)
    assert np.allclose(y[0], 0.0, atol=0.1) and np.allclose(y[1], 10.0, atol=0.1)
    # samples= subsamples them, as with the fitted posterior
    assert np.asarray(fitted.sample(posterior_draws=draws, samples=1, seed=0)["y"]).shape == (1, 30)
    with pytest.raises(ValueError):
        fitted.sample(posterior_draws=draws, samples=3, seed=0)
    with pytest.raises(ValueError):          # draw counts must agree
        fitted.sample(posterior_draws=dict(a=jnp.zeros(2), b=jnp.zeros(3), s=jnp.ones(2)))
    with pytest.raises(ValueError):          # not together with a posterior dict
        fitted.sample(posterior=dict(a=1.0, b=2.0, s=0.1), posterior_draws=draws)


def test_remove_obs_false_drops_observed_site(fitted):
    out = fitted.sample(remove_obs=False, seed=0)
    assert "y" not in out


def test_infer_discrete_requires_posterior(fitted):
    with pytest.raises(ValueError):
        fitted.sample(posterior=False, infer_discrete=True)


def test_unfitted_posterior_raises():
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    # what fit() would set before building the sampler
    m.model = _linear_model(m)
    m.data_on_model = {"x": X, "y": Y}
    if not hasattr(m, "sampler"):
        with pytest.raises(RuntimeError):
            m.sample(posterior=True)
    else:
        pytest.skip("this backend pre-creates a sampler: unfitted state unreachable")


def test_default_name_x_warns():
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)

    def model(xx, y):
        a = m.dist.normal(0, 5, name="a")
        b = m.dist.normal(0, 5, name="b")
        s = m.dist.exponential(1, name="s")
        # DistProxy auto-names ``obs=y`` sites "y", so the legacy default
        # name "x" must be passed explicitly to reach the warning path.
        m.dist.normal(a + b * xx, s, obs=y, name="x")

    m.fit(model, obs={"xx": X, "y": Y},
          num_warmup=100, num_samples=100, num_chains=1, progress_bar=False)
    with pytest.warns(UserWarning, match="default name 'x'"):
        m.sample(seed=0)


def test_explicit_seed_is_reproducible(fitted):
    out1 = fitted.sample(seed=3)
    out2 = fitted.sample(seed=3)
    assert np.array_equal(np.asarray(out1["y"]), np.asarray(out2["y"]))
