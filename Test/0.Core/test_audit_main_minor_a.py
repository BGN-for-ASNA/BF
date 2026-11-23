"""Regression tests for RELEASE_AUDIT.md, module Main, minors M-17 to M-31."""
import os

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
        m.dist.normal(mu, 1.0, obs=y)

    tiny.__name__ = name
    return tiny


def _fit_tiny(m, name="tiny", **kw):
    m.fit(_tiny_model(m, name), obs={"y": jnp.zeros(10)},
          num_samples=50, num_warmup=50, num_chains=1,
          progress_bar=False, **kw)
    return m


# M-17 ----------------------------------------------------------------------
def test_m17_save_default_name_follows_run_model_name(tmp_path, monkeypatch):
    """save() default used bf_model.pkl always (model_name is reset after
    fit): two successive fits must not overwrite each other's file."""
    monkeypatch.chdir(tmp_path)
    m = _new()
    _fit_tiny(m, name="first")
    p1 = m.save()
    assert os.path.basename(p1) == "first_bf.pkl"
    _fit_tiny(m, name="second")
    p2 = m.save()
    assert os.path.basename(p2) == "second_bf.pkl"
    assert (tmp_path / "first_bf.pkl").exists()
    assert (tmp_path / "second_bf.pkl").exists()


# M-18 (already fixed, V6) ---------------------------------------------------
def test_m18_sampler_stats_per_chain_without_num_steps():
    """sampler_stats(per_chain=True) must not crash when extra_fields
    lacked 'num_steps'."""
    m = _new()
    _fit_tiny(m, extra_fields=("diverging", "accept_prob"))
    rows = m.sampler_stats(per_chain=True)
    assert isinstance(rows, list) and len(rows) == 1
    assert "divergences" in rows[0]


# M-19 ----------------------------------------------------------------------
def test_m19_extra_fields_string_rejected():
    """extra_fields="diverging" (a bare string) must raise a clear error,
    not be exploded into single characters."""
    m = _new()
    with pytest.raises(ValueError, match="list or tuple"):
        m.fit(_tiny_model(m), obs={"y": jnp.zeros(10)},
              extra_fields="diverging", num_samples=20, num_warmup=20,
              num_chains=1, progress_bar=False)


# M-20 ----------------------------------------------------------------------
def test_m20_count_and_seed_validation():
    """Whole-float rand_seed (from R) accepted; strings and non-positive
    fit counters rejected early with a message naming the argument."""
    m = bf(platform="cpu", rand_seed=1000.0, print_devices_found=False)
    assert isinstance(m.seed, int) and m.seed == 1000
    assert m._rng.base == 1000
    with pytest.raises(ValueError, match="whole number"):
        bf(platform="cpu", rand_seed=2.5, print_devices_found=False)

    m = _new()
    model, obs = _tiny_model(m), {"y": jnp.zeros(10)}
    with pytest.raises(ValueError, match="num_chains"):
        m.fit(model, obs=obs, num_chains="4", num_samples=20,
              num_warmup=20, progress_bar=False)
    with pytest.raises(ValueError, match="num_samples"):
        m.fit(model, obs=obs, num_samples=0, num_warmup=20,
              progress_bar=False)
    with pytest.raises(ValueError, match="num_samples"):
        m.fit(model, obs=obs, num_samples=-5, num_warmup=20,
              progress_bar=False)
    with pytest.raises(ValueError, match="thinning"):
        m.fit(model, obs=obs, num_samples=20, num_warmup=20, thinning=0,
              progress_bar=False)


# M-21 ----------------------------------------------------------------------
def test_m21_npe_helpers_before_fit():
    """npe_posterior/sbc_coverage on a fresh numpyro instance must raise
    RuntimeError, not AttributeError on the absent self.sampler."""
    m = _new()
    with pytest.raises(RuntimeError, match="npe"):
        m.npe_posterior(jnp.zeros(3))
    with pytest.raises(RuntimeError, match="npe"):
        m.sbc_coverage(n_test=2, n_post=2, plot=False)


# M-22 ----------------------------------------------------------------------
def test_m22_sample_without_model_or_data():
    """sample() on a fresh instance fails with a clear message instead of
    AttributeError; data= must be a dict."""
    m = _new()
    with pytest.raises(RuntimeError, match="model"):
        m.sample(posterior=False)
    m.model = _tiny_model(m)
    with pytest.raises(RuntimeError, match="data"):
        m.sample(posterior=False)
    with pytest.raises(TypeError, match="dict"):
        m.sample(posterior=False, data=[1, 2])


# M-23 ----------------------------------------------------------------------
def test_m23_pca_missing_X_raises():
    """fit of the built-in pca without an X entry raises ValueError instead
    of printing and returning None on a half-mutated object."""
    m = _new()
    with pytest.raises(ValueError, match="X"):
        m.fit(m.models.pca(type="classic"), obs={"data": np.zeros((6, 2))},
              num_samples=20, num_warmup=20, num_chains=1,
              progress_bar=False)


# M-24 ----------------------------------------------------------------------
def test_m24_history_snapshots_dicts():
    """history must hold copies: mutating m.posteriors or m.data_on_model
    afterwards must not retro-edit the snapshot of the previous fit."""
    m = _new()
    _fit_tiny(m)
    hist = m.history
    assert hist["posteriors"] is not m.posteriors
    assert hist["data"] is not m.data_on_model
    m.posteriors["mu"] = 999.0
    m.data_on_model["extra"] = 1
    assert "extra" not in hist["data"]
    assert np.asarray(hist["posteriors"]["mu"]).shape == (50,)


# M-25 ----------------------------------------------------------------------
def test_m25_shard_report_honest_after_fallback():
    """When the shard-check falls back to replicated, shard_report() must
    not keep announcing the arrays as sharded."""
    m = _new()
    m._data_sharding = object()  # pretend a multi-device mesh

    def fake_auto(data):
        m._last_shard_report = {"n_devices": 2, "sharded": ["y"],
                                "replicated_not_divisible": [],
                                "replicated_by_caller": [], "passthrough": []}
        return dict(data)

    m._auto_shard_data = fake_auto
    m._shard_check = lambda model, s, r: False
    _fit_tiny(m, shard=True)
    rep = m.shard_report()
    assert rep["sharded"] == []
    assert rep["replicated_shard_check"] == ["y"]

    # control: when the check passes, the report keeps the sharded arrays
    m._shard_check = lambda model, s, r: True
    _fit_tiny(m, shard=True)
    assert m.shard_report()["sharded"] == ["y"]


# M-26 (already fixed, fail-closed) ------------------------------------------
def test_m26_shard_check_fail_closed(monkeypatch):
    """An internal failure of run_shard_check must fall back to REPLICATED
    (return False), never shard on an unchecked model."""
    import BayesForge.Diagnostic.sharding_safeguards as safeguards

    monkeypatch.delenv("BF_SHARD_CHECK", raising=False)
    m = _new()

    def boom(*args, **kwargs):
        raise RuntimeError("check exploded")

    monkeypatch.setattr(safeguards, "run_shard_check", boom)
    with pytest.warns(UserWarning, match="REPLICATED"):
        assert m._shard_check(lambda: None, {}, {}) is False


# M-27 ----------------------------------------------------------------------
def test_m27_plot_before_fit():
    """plot() on a fresh instance must not raise AttributeError on the
    absent self.sampler; an unsupported model warns instead of a silent
    no-op."""
    m = _new()
    with pytest.warns(UserWarning, match="plot"):
        assert m.plot(np.zeros((10, 2))) is None


# M-28 ----------------------------------------------------------------------
def test_m28_failed_save_keeps_previous_file(tmp_path):
    """A failed cloudpickle.dump must leave the previous valid save
    untouched (write to .tmp, then os.replace)."""
    m = _new()
    _fit_tiny(m)
    path = str(tmp_path / "obj.pkl")
    m.save(path)
    with open(path, "rb") as f:
        first = f.read()
    m._unpicklable = (i for i in range(3))  # generators cannot be pickled
    with pytest.raises(Exception, match="pickle"):
        m.save(path)
    with open(path, "rb") as f:
        assert f.read() == first
    assert not os.path.exists(path + ".tmp")


# M-29 ----------------------------------------------------------------------
def test_m29_get_samples_error_not_masked():
    """A real error from get_samples() must surface with its own message,
    not be converted into 'call m.fit() first'; the fit-first case keeps
    its message."""
    m = _new()
    _fit_tiny(m)

    def boom(*args, **kwargs):
        raise RuntimeError("device lost")

    m.sampler.get_samples = boom
    with pytest.raises(RuntimeError, match="device lost"):
        m.sample()

    m2 = _new()
    m2.model = _tiny_model(m2)
    m2.data_on_model = {"y": jnp.zeros(10)}
    with pytest.raises(RuntimeError, match=r"call m\.fit"):
        m2.sample()


# M-30 ----------------------------------------------------------------------
def test_m30_observed_site_matched_by_identity():
    """With two equal data entries, the observed site must map to the
    entry that IS the obs object, not to the first equal one."""
    m = _new()
    y = np.arange(10.0)

    def model(y, y_copy):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y)

    m.model = model
    m.data_on_model = {"y_copy": y.copy(), "y": y}
    sites = m._observed_sites()
    assert sites == {"y": "y"}


# M-31 ----------------------------------------------------------------------
def test_m31_untraceable_model_raises():
    """sample(remove_obs=False) with data the model cannot trace must raise
    naming the cause, not return {} silently."""
    m = _new()

    def model(y, scale):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, scale, obs=y)

    m.fit(model, obs={"y": jnp.zeros(10), "scale": 1.0},
          num_samples=50, num_warmup=50, num_chains=1, progress_bar=False)
    with pytest.raises(RuntimeError, match="traced"):
        m.sample(remove_obs=False, data={"y": jnp.zeros(10)})
