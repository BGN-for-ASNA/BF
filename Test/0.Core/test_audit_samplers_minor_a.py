"""Regression tests for RELEASE_AUDIT.md, module Samplers, minors 1-14.

Covers Model_handler.py (1-4), mcmc_numpyro.py (5-6), mcmc_tfp.py (7-11)
and SVI/svi_numpyro.py (12-14). TFP-dependent tests skip when
tensorflow_probability is missing; the rest only build objects (no MCMC
sampling) except the two tiny TFP runs in tests 8 and 10.
"""
import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from BayesForge.Samplers import mcmc_numpyro as mcmc_numpyro_mod
from BayesForge.Samplers.mcmc_numpyro import mcmc_numpyro
from BayesForge.Samplers.SVI.svi_numpyro import SVI_BF, svi_numpyro

SRC_DIR = Path(mcmc_numpyro_mod.__file__).parent


def _no_warnings(rec):
    return [w for w in rec if "sequential" in str(w.message)
            or "device" in str(w.message)]


# ---------------------------------------------------------------------------
# Model_handler.py
# ---------------------------------------------------------------------------

def test_1_bare_yield_and_no_call_lines_do_not_crash():
    """Minor 1: lines without '=' or without '(' are skipped, not fatal.

    Already fixed by the AST rewrite (19f93e5d): yield_sites walks the tree,
    an unassigned yield gets target=None, a yield of a non-Call is skipped.
    """
    pytest.importorskip("tensorflow_probability")
    from BayesForge.Samplers.Model_handler import model_handler

    def model(y):
        mu = yield normal(0.0, 1.0, name="mu")   # assigned: '=' and '('
        note = no_call_here                       # no '=' with yield, no '('
        yield normal(mu, 1.0, name="z")           # bare latent yield
        yield normal(mu, 1.0, obs=y)              # bare observed yield

    h = model_handler()
    sites = h.yield_sites(model)
    assert [s["target"] for s in sites] == ["mu", None, None]
    h.get_model_distributions(model)
    assert set(h.model_info) == {"mu", "z"}
    assert h.find_obs_in_model(model) == ["y"]


def test_2_dead_var_model_line_removed():
    """Minor 2: the list comprehension overwritten on the next line is gone."""
    src = (SRC_DIR / "Model_handler.py").read_text()
    assert "var_model = [item for item" not in src


def test_3_no_prints_warning_goes_through_warnings(capsys):
    """Minor 3: INFO prints dropped, the bijector WARNING is a UserWarning."""
    pytest.importorskip("tensorflow_probability")
    from BayesForge.Samplers.Model_handler import model_handler

    src = (SRC_DIR / "Model_handler.py").read_text()
    code_lines = [ln for ln in src.splitlines()
                  if not ln.lstrip().startswith("#")]
    assert not any("print(" in ln for ln in code_lines)

    h = model_handler()
    infos = {"L": {"distribution": "lkj"},
             "s": {"distribution": "half_normal"},
             "w": {"distribution": "weird_dist"}}
    init = [jnp.eye(2), jnp.ones(()), jnp.zeros(())]
    with pytest.warns(UserWarning, match="No specific bijector"):
        p2, b = h.initialise(infos, init)
    captured = capsys.readouterr()
    assert "INFO" not in captured.out and "WARNING" not in captured.out
    assert len(p2) == len(b) == 3


def test_4_init_params_length_raises_value_error():
    """Minor 4: clear ValueError instead of IndexError on length mismatch."""
    pytest.importorskip("tensorflow_probability")
    from BayesForge.Samplers.Model_handler import model_handler

    h = model_handler()
    infos = {"a": {"distribution": "normal"}, "b": {"distribution": "normal"}}
    with pytest.raises(ValueError, match="init_params"):
        h.initialise(infos, [jnp.zeros(())])
    with pytest.raises(ValueError, match="init_params"):
        h.initialise(infos, [jnp.zeros(()), jnp.zeros(()), jnp.zeros(())])
    # matching length still works (observed keys are excluded from the count)
    with pytest.warns(UserWarning, match="No specific bijector"):
        p2, b = h.initialise(infos, [jnp.zeros(()), jnp.zeros(())])
    assert len(p2) == len(b) == 2
    with pytest.warns(UserWarning, match="No specific bijector"):
        p2, b = h.initialise(infos, [jnp.zeros(())], obs_names=["b"])
    assert len(p2) == len(b) == 1


# ---------------------------------------------------------------------------
# mcmc_numpyro.py
# ---------------------------------------------------------------------------

def _tiny_numpyro_model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)


def test_5_manip_import_removed():
    """Minor 5: no coupling to BayesForge.Data left in mcmc_numpyro."""
    src = (SRC_DIR / "mcmc_numpyro.py").read_text()
    assert "manip" not in src


def test_6_mcmc_numpyro_validates_run_counts():
    """Minor 6: early validation of num_warmup/num_samples/num_chains."""
    from numpyro.infer import MCMC
    with pytest.raises(ValueError, match="num_warmup"):
        mcmc_numpyro(_tiny_numpyro_model, num_warmup=-1)
    with pytest.raises(ValueError, match="num_samples"):
        mcmc_numpyro(_tiny_numpyro_model, num_samples=0)
    with pytest.raises(ValueError, match="num_chains"):
        mcmc_numpyro(_tiny_numpyro_model, num_chains=0)
    with pytest.raises(TypeError, match="num_samples"):
        mcmc_numpyro(_tiny_numpyro_model, num_samples=10.5)
    with pytest.raises(TypeError, match="num_chains"):
        mcmc_numpyro(_tiny_numpyro_model, num_chains=True)
    # valid input: plain MCMC object, no device warning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m = mcmc_numpyro(_tiny_numpyro_model, num_warmup=10, num_samples=10,
                         num_chains=1, progress_bar=False)
    assert isinstance(m, MCMC)
    assert _no_warnings(rec) == []


def test_6b_parallel_chains_on_few_devices_warns(monkeypatch):
    """Minor 6: num_chains > devices with chain_method='parallel' warns."""
    # Patch both the module's jax.local_device_count and the reference
    # numpyro.infer.mcmc bound at import time (it does its own check).
    np_mcmc = mcmc_numpyro_mod.numpyro.infer.mcmc
    monkeypatch.setattr(mcmc_numpyro_mod.jax, "local_device_count",
                        lambda *a, **k: 1)
    monkeypatch.setattr(np_mcmc, "local_device_count", lambda *a, **k: 1)
    with pytest.warns(UserWarning, match="sequential"):
        mcmc_numpyro(_tiny_numpyro_model, num_chains=2,
                     chain_method="parallel", progress_bar=False)
    monkeypatch.setattr(mcmc_numpyro_mod.jax, "local_device_count",
                        lambda *a, **k: 4)
    monkeypatch.setattr(np_mcmc, "local_device_count", lambda *a, **k: 4)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        mcmc_numpyro(_tiny_numpyro_model, num_chains=2,
                     chain_method="parallel", progress_bar=False)
    assert _no_warnings(rec) == []


# ---------------------------------------------------------------------------
# mcmc_tfp.py
# ---------------------------------------------------------------------------

def _tfp_sampler():
    pytest.importorskip("tensorflow_probability")
    from BayesForge import bf
    m = bf(platform='cpu', backend='tfp', print_devices_found=False)
    y = np.random.default_rng(0).normal(1.0, 0.7, size=20).astype(np.float32)
    m.data_on_model = dict(y=jnp.asarray(y))
    return m


def _tfp_model(m):
    def model(y):
        mu = yield m.dist.normal(0.0, 5.0, name='mu')
        sigma = yield m.dist.half_normal(2.0, name='sigma')
        yield m.dist.normal(mu + jnp.zeros_like(y), sigma, obs=y)
    return model


def test_7_obs_names_initialised_in_constructor():
    """Minor 7: obs_names exists before run() (no AttributeError)."""
    pytest.importorskip("tensorflow_probability")
    from BayesForge.Samplers.mcmc_tfp import mcmc
    s = mcmc()
    assert hasattr(s, "obs_names")
    assert s.obs_names is None


def test_8_nuts_returns_self():
    """Minor 8: self.sampler / run() expose the sampler, not None."""
    m = _tfp_sampler()
    model = _tfp_model(m)
    s = m.sampler
    s.data_on_model = m.data_on_model
    s.model = model
    ret = s.run(seed=0, model=model, n_chains=1, num_results=5,
                num_burnin_steps=5)
    assert ret is s
    assert s.sampler is s


def test_9_nuts_model_obs_documented_as_ignored():
    """Minor 9: signature keeps model/obs, documented as deprecated."""
    pytest.importorskip("tensorflow_probability")
    from BayesForge.Samplers.mcmc_tfp import mcmc
    doc = inspect.getdoc(mcmc.NUTS) or ""
    assert "ignored" in doc and "deprecated" in doc
    params = list(inspect.signature(mcmc.NUTS).parameters)
    assert params[:3] == ["self", "model", "obs"]


def test_10_posterior_and_sample_stats_same_type_both_branches(monkeypatch):
    """Minor 10: lists in the single-run AND the multi-run branch."""
    pytest.importorskip("tensorflow_probability")
    from BayesForge.Samplers import mcmc_tfp as mcmc_tfp_mod
    m = _tfp_sampler()
    model = _tfp_model(m)
    s = m.sampler
    s.data_on_model = m.data_on_model
    s.model = model
    s.run(seed=0, model=model, n_chains=1, num_results=5, num_burnin_steps=5)
    assert isinstance(s.posterior, list)
    assert isinstance(s.sample_stats, list)
    # force the multi-run branch regardless of the server's device count
    monkeypatch.setattr(mcmc_tfp_mod.jax, "local_device_count",
                        lambda *a, **k: 1)
    with pytest.warns(UserWarning, match="sequential"):
        s.run(seed=1, model=model, n_chains=2, num_results=5,
              num_burnin_steps=5)
    assert isinstance(s.posterior, list)
    assert isinstance(s.sample_stats, list)
    assert len(s.posterior) == 2          # mu, sigma
    assert len(s.sample_stats) == 5       # the five trace_fn statistics


def test_11_n_chains_validated():
    """Minor 11: n_chains < 1 (or not an int) fails fast with ValueError."""
    pytest.importorskip("tensorflow_probability")
    from BayesForge.Samplers.mcmc_tfp import mcmc
    s = mcmc()
    for bad in (0, -3, 1.5, "2", True):
        with pytest.raises(ValueError, match="n_chains"):
            s.NUTS(None, None, n_chains=bad, target_log_prob_fn=lambda *p: 0.0)


# ---------------------------------------------------------------------------
# SVI/svi_numpyro.py
# ---------------------------------------------------------------------------

def _svi_model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(mu, 1.0), obs=y)


def _svi_guide(y=None):
    loc = numpyro.param("loc", 0.0)
    scale = numpyro.param("scale", 1.0, constraint=dist.constraints.positive)
    numpyro.sample("mu", dist.Normal(loc, scale))


def test_12_svi_run_num_steps_is_keyword_only():
    """Minor 12: run(key, y) passes y to the model, not to num_steps."""
    params = inspect.signature(SVI_BF.run).parameters
    assert params["num_steps"].kind is inspect.Parameter.KEYWORD_ONLY
    y = jnp.asarray(np.random.default_rng(0).normal(1.0, 1.0, size=16))
    s = SVI_BF(_svi_model, _svi_guide, numpyro.optim.Adam(0.1), num_steps=3)
    s.run(jax.random.PRNGKey(0), y, num_steps=4)
    assert len(s.losses) == 4
    # without num_steps the constructor value is used (y was NOT swallowed)
    s2 = SVI_BF(_svi_model, _svi_guide, numpyro.optim.Adam(0.1), num_steps=2)
    s2.run(jax.random.PRNGKey(1), y)
    assert len(s2.losses) == 2


def test_13_loss_default_is_not_a_shared_mutable():
    """Minor 13: loss=None builds a fresh Trace_ELBO per instance."""
    from numpyro.infer import Trace_ELBO
    assert inspect.signature(SVI_BF.__init__).parameters["loss"].default is None
    assert inspect.signature(svi_numpyro).parameters["loss"].default is None
    a = SVI_BF(_svi_model, _svi_guide, numpyro.optim.Adam(1e-3))
    b = SVI_BF(_svi_model, _svi_guide, numpyro.optim.Adam(1e-3))
    assert isinstance(a.svi.loss, Trace_ELBO)
    assert a.svi.loss is not b.svi.loss
    elbo = Trace_ELBO(num_particles=3)
    c = svi_numpyro(_svi_model, _svi_guide, numpyro.optim.Adam(1e-3), loss=elbo)
    assert c.svi.loss is elbo


def test_14_get_samples_seed_default_documented():
    """Minor 14: seed=0 kept but documented (bf.svi always passes one)."""
    doc = inspect.getdoc(SVI_BF.get_samples) or ""
    assert "seed" in doc and "0" in doc
    assert inspect.signature(SVI_BF.get_samples).parameters["seed"].default == 0
