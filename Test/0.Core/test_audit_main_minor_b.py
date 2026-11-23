"""Regression tests for RELEASE_AUDIT.md, module Main, minors M-32 to M-45."""
import functools
import random
import types
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge import bf


def _new():
    return bf(platform="cpu", rand_seed=0, print_devices_found=False)


def _tiny_model(m, name="tiny"):
    """Smallest conditioning model: one scalar site, ten observations."""

    def tiny(y):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y, name="y")

    tiny.__name__ = name
    return tiny


def _fit_tiny(m, **kw):
    m.fit(_tiny_model(m), obs={"y": jnp.zeros(10)},
          num_samples=50, num_warmup=50, num_chains=1,
          progress_bar=False, **kw)
    return m


# M-32 ----------------------------------------------------------------------
def test_m32_sample_count_validated():
    """samples=0 / negative / non-numeric raise ValueError; valid counts pass."""
    m = _new()
    m.model = _tiny_model(m)
    for bad in (0, -5, "4", 1.5, True):
        with pytest.raises(ValueError, match="samples"):
            m.sample(samples=bad)
    # a whole-number float (R interop) still works: 2 prior draws
    m.data_on_model = {"y": jnp.zeros(10)}
    draws = m.sample(posterior=False, samples=2.0, seed=1)
    assert jnp.asarray(draws["y"]).shape[0] == 2


# M-33 ----------------------------------------------------------------------
def test_m33_setstate_resyncs_n_devices_with_mesh(monkeypatch):
    """A rebuilt mesh on another machine must drive n_devices, not the
    stale pickled count (_auto_shard_data divides by it)."""
    import BayesForge.SetDevice.set as setmod

    m = _new()
    fake_mesh = types.SimpleNamespace(devices=[0, 1, 2])
    monkeypatch.setattr(setmod, "setup_multi_device",
                        lambda platform, cores=None: (fake_mesh, "d", "r"))
    m.__setstate__({"_platform": "cpu", "n_devices": 8})
    assert m.mesh is fake_mesh
    assert m.n_devices == 3  # follows the rebuilt mesh, not the stale 8


def test_m33_setstate_single_device_or_failure_clears_state(monkeypatch):
    """No mesh after the rebuild (single device, or any setup error — not
    only RuntimeError) leaves no stale device state behind."""
    import BayesForge.SetDevice.set as setmod

    m = _new()
    monkeypatch.setattr(setmod, "setup_multi_device",
                        lambda platform, cores=None: (None, None, None))
    m.__setstate__({"_platform": "cpu", "n_devices": 8})
    assert m.n_devices == 1 and m._data_sharding is None

    def boom(platform, cores=None):
        raise ValueError("platform no longer resolves")

    monkeypatch.setattr(setmod, "setup_multi_device", boom)
    m.__setstate__({"_platform": "gpu", "n_devices": 4})
    assert m.n_devices == 1
    assert m.mesh is None and m._data_sharding is None


# M-34 ----------------------------------------------------------------------
def test_m34_shard_limits_warning_is_the_experimental_one_once():
    """The dead 'only 1 device' branch is gone: even with no mesh (the old
    branch's condition) the method emits the EXPERIMENTAL limits warning,
    once per instance."""
    m = _new()
    m._data_sharding = None  # force the old dead-branch condition
    m._shard_warned = False
    with pytest.warns(UserWarning, match="EXPERIMENTAL"):
        m._maybe_warn_shard_limits()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m._maybe_warn_shard_limits()  # second call: silent


# M-35 (already fixed: seed_global_rngs=False default) -----------------------
def test_m35_constructor_leaves_global_rngs_untouched():
    np.random.seed(12345)
    random.seed(12345)
    before_np = np.random.get_state()[1][0]
    before_py = random.getstate()[1][0]
    _new()
    assert np.random.get_state()[1][0] == before_np
    assert random.getstate()[1][0] == before_py

# Ensure seed_global_rngs=True seed global rngs
def test_m35_seed_global_rngs_is_opt_in():
    bf(platform="cpu", rand_seed=42, print_devices_found=False,
       seed_global_rngs=True)
    assert np.random.get_state()[1][0] == \
        np.random.RandomState(42).get_state()[1][0]


# M-36 (already fixed: reused data is checked against the new model) ---------
def test_m36_stale_fit_data_mismatch_raises_clear_error():
    """fit(B) without obs after fit(A, obs=dA) names the mismatch instead
    of an obscure NumPyro error; the docstring now describes the reuse."""
    m = _new()
    _fit_tiny(m)

    def other(y, x):  # needs x, stored data has only y
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y, name="y")

    with pytest.raises(ValueError, match="does not match"):
        m.fit(other, num_samples=20, num_warmup=20, num_chains=1,
              progress_bar=False)


# M-37 ----------------------------------------------------------------------
def test_m37_summary_before_fit_raises_clear_error():
    """summary()/summary_old()/summary_jax() pre-fit say 'Run fit() first'
    instead of AttributeError: 'bf' object has no attribute 'diag'."""
    m = _new()
    for call in (lambda: m.summary(), lambda: m.summary_old(),
                 lambda: m.summary_jax()):
        with pytest.raises(RuntimeError, match=r"Run fit\(\) first"):
            call()
    # and it works after a fit
    _fit_tiny(m)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = m.summary(backend="jax")
    assert tab is not None and len(tab) > 0


# M-38 ----------------------------------------------------------------------
def test_m38_empty_posterior_dict_raises_value_error():
    """posterior={} fails with a ValueError, not StopIteration; the
    subsample path still works on a fitted model."""
    m = _new()
    m.model = _tiny_model(m)
    m.data_on_model = {"y": jnp.zeros(10)}
    with pytest.raises(ValueError, match="empty"):
        m.sample(posterior={}, samples=3)
    with pytest.raises(ValueError, match="empty"):
        m.sample(posterior={})
    # the subsample path still works on a fitted model
    _fit_tiny(m)
    draws = m.sample(samples=3, seed=0)
    assert jnp.asarray(draws["y"]).shape[0] == 3


# M-39 ----------------------------------------------------------------------
# Ensure fit() does not truncate observed data by rank
def test_m39_expand_observed_rank_mismatch_raises():
    """Stored data of a rank below the event shape fails loudly instead of
    slicing a truncated batch through the negative slice."""
    import numpyro.distributions as dist
    from BayesForge.Main.main import _ExpandObserved

    ex = _ExpandObserved(lambda: None, {"y": ()})
    msg = {"type": "sample", "name": "y", "is_observed": False,
           "fn": dist.Independent(dist.Normal(jnp.zeros(3), 1.0), 1)}
    with pytest.raises(ValueError, match="event rank"):
        ex.process_message(msg)
    # the normal expansion path is unaffected
    ex2 = _ExpandObserved(lambda: None, {"y": (5,)})
    msg2 = {"type": "sample", "name": "y", "is_observed": False,
            "fn": dist.Normal(0.0, 1.0)}
    ex2.process_message(msg2)
    assert tuple(msg2["fn"].batch_shape) == (5,)

# Ensure fit() does not pass truncated observed data silently
def test_m39_expand_failure_warns_instead_of_passing_silently():
    from BayesForge.Main.main import _ExpandObserved

    class _NoExpand:
        event_shape = ()
        batch_shape = ()

        def expand(self, shape):
            raise NotImplementedError("no expand here")

    ex = _ExpandObserved(lambda: None, {"y": (5,)})
    msg = {"type": "sample", "name": "y", "is_observed": False,
           "fn": _NoExpand()}
    with pytest.warns(UserWarning, match="could not expand"):
        ex.process_message(msg)
    assert isinstance(msg["fn"], _NoExpand)  # left as is, but announced


# M-40 (documented: m.load is the dataset loader, delegating paths to bf.load)
def test_m40_load_dual_semantics_documented_and_working(tmp_path):
    from BayesForge.Resources.datasets import load as dataset_loader

    m = _new()
    assert isinstance(m.load, dataset_loader)  # instance attribute shadows
    doc = bf.load.__doc__.lower()
    assert "dataset loader" in doc and "shadow" in doc
    # both entry points load a saved session
    _fit_tiny(m)
    p = m.save(str(tmp_path / "s.pkl"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert isinstance(bf.load(p), bf)
        assert isinstance(m.load(p), bf)


# M-41 (docstring warning; the feature itself is kept) ------------------------
def test_m41_load_docstring_warns_about_code_execution():
    doc = bf.load.__doc__.lower()
    assert "arbitrary code" in doc
    assert "trust" in doc


# M-42 ----------------------------------------------------------------------
def test_m42_pca_latent_dim_default_warns():
    """The degenerate latent_dim = data_dim default is kept (same results)
    but announced; an explicit latent_dim stays silent."""
    m = _new()
    X = jnp.asarray(np.random.RandomState(0).randn(8, 3))
    with pytest.warns(UserWarning, match="latent_dim"):
        m.fit(m.models.pca(type="classic"), obs={"X": X},
              num_samples=20, num_warmup=20, num_chains=1, progress_bar=False)
    assert int(m.data_on_model["latent_dim"]) == 3  # default kept

    m2 = _new()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m2.fit(m2.models.pca(type="classic"), obs={"X": X, "latent_dim": 2},
               num_samples=20, num_warmup=20, num_chains=1, progress_bar=False)
    assert not any("latent_dim" in str(w.message) for w in rec)


# M-43 (already fixed: __name__ read via getattr with a default) --------------
def test_m43_nested_partial_model_fits():
    """partial(partial(model, ...), ...) must not raise AttributeError on
    .func.__name__ (CPython flattens nested partials, and a callable without
    __name__ falls back to the type name)."""
    m = _new()

    def model(y, scale):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, scale, obs=y, name="y")

    nested = functools.partial(functools.partial(model, scale=1.0))
    m.fit(nested, obs={"y": jnp.zeros(10)},
          num_samples=20, num_warmup=20, num_chains=1, progress_bar=False)
    assert "mu" in m.posteriors
    # CPython flattens the nested partial: the underlying name survives
    assert m.run_model_name == "model"

    # a callable without __name__ wrapped in a partial: display name falls
    # back to the type name, no AttributeError anywhere
    class ModelObj:
        def __call__(self, y):
            mu = m.dist.normal(0.0, 1.0, name="mu")
            m.dist.normal(mu, 1.0, obs=y, name="y")

    m2 = _new()
    m2.fit(functools.partial(ModelObj()), obs={"y": jnp.zeros(10)},
           num_samples=20, num_warmup=20, num_chains=1, progress_bar=False)
    assert "mu" in m2.posteriors
    assert m2.run_model_name == "partial"


# M-44 (already fixed: TFP get_samples() pools chains, numpyro contract) ------
def test_m44_pca_posterior_is_pooled():
    """models.pca.posterior receives the pooled draws, never the
    chain-structured ones."""
    m = _new()
    X = jnp.asarray(np.random.RandomState(1).randn(8, 3))
    m.fit(m.models.pca(type="classic"), obs={"X": X, "latent_dim": 2},
          num_samples=20, num_warmup=20, num_chains=2, progress_bar=False)
    pooled = m.models.pca.posterior
    assert set(pooled) == set(m.posteriors)
    for name, arr in pooled.items():
        # pooled draws (2 chains x 20), not chain-structured (2, 20, ...)
        assert int(np.asarray(arr).shape[0]) == 40


# M-45 ----------------------------------------------------------------------
def test_m45_pooled_ebfmi_is_the_worst_chain():
    """The pooled row reports the per-chain E-BFMI (worst chain), never a
    value computed with np.diff across chain boundaries."""
    m = _new()
    m.fit(_tiny_model(m), obs={"y": jnp.zeros(10)},
          num_samples=50, num_warmup=50, num_chains=2, progress_bar=False)
    pooled = m.sampler_stats()
    per = m.sampler_stats(per_chain=True)
    assert len(per) == 2
    assert pooled["ebfmi"] == min(r["ebfmi"] for r in per)
