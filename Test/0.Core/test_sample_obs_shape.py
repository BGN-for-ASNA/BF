"""m.sample(remove_obs=True) simulates the observed sites at the shape of
the data the model was fitted on.

Before, with obs=None a likelihood whose parameters are scalars -- e.g.
poisson(lam, obs=y) with a scalar lam, or categorical(probs=p, obs=y) with
p of shape (K,) -- drew ONE value per posterior draw: get_yrep() returned
shape (n_draws,) instead of (n_draws, n_obs) and every ppc_* failed."""
import jax.numpy as jnp
import numpy as np

from BayesForge import bf

Y_POIS = jnp.array(np.random.default_rng(0).poisson(3, 40))
Y_CAT = jnp.array(np.random.default_rng(1).integers(0, 3, 40))
X = jnp.linspace(-1, 1, 40)
Y_LM = 1.0 + 2.0 * X + 0.1 * np.random.default_rng(2).normal(size=40)
X_NEW = jnp.linspace(-2, 2, 7)


def _fit(model, **data):
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    m.fit(model(m), obs=data, num_warmup=100, num_samples=100, num_chains=1, progress_bar=False)
    return m


def _poisson_model(m):
    def model(y):
        lam = m.dist.exponential(1, name="lam")
        m.dist.poisson(lam, obs=y, name="y")
    return model


def _categorical_model(m):
    def model(y):
        p = m.dist.dirichlet(jnp.ones(3), name="p")
        m.dist.categorical(probs=p, obs=y, name="y")
    return model


def _linear_model(m):
    def model(x, y):
        a = m.dist.normal(0, 5, name="a")
        b = m.dist.normal(0, 5, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y, name="y")
    return model


def test_scalar_rate_poisson_expands_to_obs_shape():
    m = _fit(_poisson_model, y=Y_POIS)
    draws = np.asarray(m.sample()["y"])
    assert draws.shape == (100, 40)

    yrep = np.asarray(m.diag.get_yrep())
    assert yrep.shape == (100, 40)
    # the 40 columns are independent draws, not one value repeated
    assert not np.all(yrep == yrep[:, :1])


def test_categorical_probs_vector_expands_to_obs_shape():
    m = _fit(_categorical_model, y=Y_CAT)
    yrep = np.asarray(m.diag.get_yrep())
    assert yrep.shape == (100, 40)
    assert np.isin(yrep, [0, 1, 2]).all()


def test_batched_likelihood_unchanged():
    m = _fit(_linear_model, x=X, y=Y_LM)
    # the observed distribution already has batch (40,): same shape as before
    assert np.asarray(m.sample(seed=3)["y"]).shape == (100, 40)
    # new-data path untouched: shape follows the new data
    assert np.asarray(m.sample(data={"x": X_NEW}, seed=0)["y"]).shape == (100, len(X_NEW))


def test_ppc_loo_pit_builds_figure():
    m = _fit(_poisson_model, y=Y_POIS)
    fig = m.diag.ppc_loo_pit()
    assert fig is not None
