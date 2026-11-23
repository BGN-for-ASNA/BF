"""2.7 -- the data arguments that feed observed sites are found by tracing
the model, not by parsing its source (build_model_with_Y_None /
find_obs_in_model).

Before, ``inspect.getsource`` + ``ast`` was used: for a model without
available source (defined with exec, in R / Julia, functools.partial, ...)
no observed argument was found, and predicting on new data without the
outcome failed with "missing a required argument: 'y'"."""
import functools
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge import bf

X = jnp.linspace(-1, 1, 30)
Y = 1.0 + 2.0 * X + 0.1 * jnp.cos(9 * X)
X_NEW = jnp.linspace(-2, 2, 7)


def _fit(model, **data):
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    m.fit(model(m), obs=data, num_warmup=100, num_samples=100, num_chains=1, progress_bar=False)
    return m


def _source_model(m):
    def model(x, y):
        a = m.dist.normal(0, 5, name="a")
        b = m.dist.normal(0, 5, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y, name="y")
    return model


def _exec_model(m):
    # No source available: inspect.getsource raises OSError.
    ns = {"m": m}
    exec("def model(x, y):\n"
         "    a = m.dist.normal(0, 5, name='a')\n"
         "    b = m.dist.normal(0, 5, name='b')\n"
         "    s = m.dist.exponential(1, name='s')\n"
         "    m.dist.normal(a + b * x, s, obs=y, name='y')\n", ns)
    return ns["model"]


def _partial_model(m):
    def model(x, y, scale):
        a = m.dist.normal(0, scale, name="a")
        b = m.dist.normal(0, scale, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y, name="y")
    return functools.partial(model, scale=5.0)


def test_obs_args_found_by_tracing():
    for build in (_source_model, _exec_model, _partial_model):
        m = _fit(build, x=X, y=Y)
        assert m.find_obs_in_model() == ["y"], build.__name__


def test_predict_on_new_data_without_the_outcome():
    for build in (_source_model, _exec_model, _partial_model):
        m = _fit(build, x=X, y=Y)
        pred = m.sample(data={"x": X_NEW}, remove_obs=True, posterior=True, seed=0)
        assert np.asarray(pred["y"]).shape == (100, X_NEW.shape[0]), build.__name__


def test_same_predictions_with_and_without_source():
    a = _fit(_source_model, x=X, y=Y)
    b = _fit(_exec_model, x=X, y=Y)
    np.testing.assert_array_equal(np.asarray(a.posteriors["b"]), np.asarray(b.posteriors["b"]))
    pa = a.sample(data={"x": X_NEW}, seed=3)["y"]
    pb = b.sample(data={"x": X_NEW}, seed=3)["y"]
    np.testing.assert_array_equal(np.asarray(pa), np.asarray(pb))


def test_predict_with_default_after_required_data():
    # With y observed it gets a default, so x becomes keyword-only -- every
    # later positional parameter must become keyword-only too (defaults
    # kept), or sig.replace raises "wrong parameter order".
    def _default_after_obs(m):
        def model(y, x, scale=5.0):
            a = m.dist.normal(0, scale, name="a")
            b = m.dist.normal(0, scale, name="b")
            s = m.dist.exponential(1, name="s")
            m.dist.normal(a + b * x, s, obs=y, name="y")
        return model

    def _required_after_obs(m):
        def model(y, x, z, w=None):
            a = m.dist.normal(0, 5, name="a")
            b = m.dist.normal(0, 5, name="b")
            s = m.dist.exponential(1, name="s")
            m.dist.normal(a + b * x + z, s, obs=y, name="y")
        return model

    m = _fit(_default_after_obs, x=X, y=Y)
    pred = m.sample(data={"x": X_NEW}, remove_obs=True, posterior=True, seed=0)
    assert np.asarray(pred["y"]).shape == (100, X_NEW.shape[0])

    m = _fit(_required_after_obs, x=X, z=X, y=Y)
    pred = m.sample(data={"x": X_NEW, "z": X_NEW}, remove_obs=True,
                    posterior=True, seed=0)
    assert np.asarray(pred["y"]).shape == (100, X_NEW.shape[0])


def _computed_obs_model(m):
    # obs=y - 1 is a computed value, not a data entry: the site cannot be
    # un-conditioned, so sample() must say so instead of passing the
    # conditioning value off as a replicate.
    def model(x, y):
        a = m.dist.normal(0, 5, name="a")
        b = m.dist.normal(0, 5, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y - 1, name="y")
    return model


def test_computed_obs_warns_and_is_not_replicated():
    m = _fit(_computed_obs_model, x=X, y=Y)
    assert m.find_obs_in_model() == []          # no data entry feeds the site
    with pytest.warns(UserWarning, match="not a data entry"):
        pred = m.sample(seed=0)
    # the site is returned as-is: the conditioning value, broadcast over the
    # draws -- identical rows, so not a replicate
    y_pred = np.asarray(pred["y"])
    assert y_pred.shape == (100, len(Y))
    np.testing.assert_allclose(y_pred[0], np.asarray(Y) - 1)
    assert np.all(y_pred == y_pred[0])


def test_data_obs_replicates_without_warning():
    m = _fit(_source_model, x=X, y=Y)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = m.sample(data={"x": X_NEW}, seed=0)
    assert np.asarray(pred["y"]).shape == (100, X_NEW.shape[0])
    assert not [w for w in caught if "data entry" in str(w.message)]
