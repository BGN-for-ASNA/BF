"""Regression tests for RELEASE_AUDIT.md, module Diagnostic, majors 3, 5, 6, 10, 13, 14, 24."""
import warnings

import numpy as np
import pytest

import numpyro
import numpyro.distributions as dist

from BayesForge import bf
from BayesForge.Diagnostic.Diag2 import diagWIP
from BayesForge.Diagnostic.jax_diagnostics import _ess_raw, compute_log_likelihood


# ---------------------------------------------------------------------------
# Helpers (copied from test_audit_diag_critical.py / API_Routing/test_diag_backend.py)
# ---------------------------------------------------------------------------

class _FakeSampler:
    """Minimal stand-in for a fitted sampler, as in test_audit_diag_critical.py."""
    num_chains = 1

    def __init__(self, samples):
        self._samples = samples

    def get_samples(self, group_by_chain=True):
        return self._samples


def _tiny_numpyro_model(y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 1))
    numpyro.sample("y", dist.Normal(mu, 1), obs=y)


def _fit_two_params():
    """Tiny two-parameter fit (pattern of API_Routing/test_diag_backend.py::_fit)."""
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    y = np.random.default_rng(0).normal(0.5, 1.0, size=12)

    def model(y):
        mu = m.dist.normal(0, 1, name="mu")
        sigma = m.dist.uniform(0.5, 5.0, name="sigma")
        m.dist.normal(mu, sigma, obs=y, shape=(y.shape[0],))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(model, obs={"y": y}, num_warmup=50, num_samples=100,
              num_chains=2, progress_bar=False)
    return m


# ---------------------------------------------------------------------------
# Bug 3 -- single-chain split R-hat warns and returns NaN (like ArviZ)
# ---------------------------------------------------------------------------

def test_bug3_single_chain_rhat_warns():
    draws = np.random.default_rng(0).normal(size=(1, 200))
    d = diagWIP(_FakeSampler({"mu": draws}))
    with pytest.warns(UserWarning, match="single chain"):
        val = d._rhat_manual(draws)
    # RELEASE_AUDIT bug 3: NaN like ArviZ (owner decision 2026-11-13)
    assert np.isnan(val)


# ---------------------------------------------------------------------------
# Bug 5 -- _ess_raw guards (too-few-draws -> nan, constant -> size)
# ---------------------------------------------------------------------------

def test_bug5_ess_raw_guards():
    assert np.isnan(_ess_raw(np.ones((2, 3))))
    assert _ess_raw(np.full((2, 50), 3.0)) == 100.0


# ---------------------------------------------------------------------------
# Bug 6 -- explicit has_chains silences the chain-axis heuristic
# ---------------------------------------------------------------------------

def test_bug6_explicit_has_chains_no_warning():
    y = np.random.default_rng(0).normal(size=12)
    post = {"mu": np.random.default_rng(1).normal(size=(4, 10))}
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ll = compute_log_likelihood(_tiny_numpyro_model, post, y=y,
                                    has_chains=True)
    assert np.asarray(ll).shape == (4, 10, 12)


def test_bug6_heuristic_warns():
    y = np.random.default_rng(0).normal(size=12)
    post = {"mu": np.random.default_rng(1).normal(size=(4, 10))}
    with pytest.warns(UserWarning, match="chain axis"):
        ll = compute_log_likelihood(_tiny_numpyro_model, post, y=y)
    assert np.asarray(ll).shape == (4, 10, 12)


# ---------------------------------------------------------------------------
# Bug 13 -- shard-check is fail-closed when the check itself crashes
# ---------------------------------------------------------------------------

def test_bug13_shard_check_fail_closed(monkeypatch):
    import BayesForge.Diagnostic.sharding_safeguards as safeguards

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated shard-check crash")

    monkeypatch.setattr(safeguards, "run_shard_check", _boom)
    monkeypatch.delenv("BF_SHARD_CHECK", raising=False)
    m = bf(platform="cpu", print_devices_found=False)
    with pytest.warns(UserWarning, match="could not run"):
        assert m._shard_check(lambda: None, {}, {}) is False


# ---------------------------------------------------------------------------
# Bug 24 -- m.diag.summary(exclude=...) is honoured on the arviz backend
# ---------------------------------------------------------------------------

def test_bug24_summary_exclude():
    m = _fit_two_params()
    # The fixed code is patch_diag._summary, bound to m.diag.summary; the
    # public m.summary(exclude=...) routes through filter_posteriors +
    # summary_old and never reaches it.
    df = m.diag.summary(exclude=["mu"])
    assert not any(str(i).startswith("mu") for i in df.index)
    assert any(str(i).startswith("sigma") for i in df.index)
