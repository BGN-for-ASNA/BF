"""Regression tests for RELEASE_AUDIT.md, module Diagnostic, majors 22-23 (ArviZ >= 1.2 plot API)."""
import matplotlib
matplotlib.use("Agg")

import warnings

import numpy as np
import pytest

import matplotlib.pyplot as plt

from BayesForge import bf
from BayesForge.Diagnostic.Diag import diag
from BayesForge.Diagnostic.Diag2 import diagWIP


# ---------------------------------------------------------------------------
# Helpers (fit pattern copied from test_audit_diag_major.py::_fit_two_params:
# 2 parameters, 2 chains, 100 draws)
# ---------------------------------------------------------------------------

def _fit(extra_param=False, seed=0):
    """Tiny two-parameter fit; optionally with one extra unused parameter."""
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    y = np.random.default_rng(seed).normal(0.5, 1.0, size=12)

    def model(y):
        mu = m.dist.normal(0, 1, name="mu")
        sigma = m.dist.uniform(0.5, 5.0, name="sigma")
        if extra_param:
            m.dist.normal(0, 1, name="eta")  # sampled from the prior, unused
        m.dist.normal(mu, sigma, obs=y, shape=(y.shape[0],))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(model, obs={"y": y}, num_warmup=50, num_samples=100,
              num_chains=2, progress_bar=False)
    return m


def _fit_one_param():
    """Same, but with a single parameter (the old plt.subplots(1, 1) crash)."""
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    y = np.random.default_rng(1).normal(0.0, 1.0, size=12)

    def model(y):
        mu = m.dist.normal(0, 1, name="mu")
        m.dist.normal(mu, 1.0, obs=y, shape=(y.shape[0],))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(model, obs={"y": y}, num_warmup=50, num_samples=100,
              num_chains=2, progress_bar=False)
    return m


@pytest.fixture(scope="module")
def legacy_diag():
    """Legacy diag class, built the way patch_diag does (Diag.py, not diagWIP)."""
    m = _fit()
    d = diag(sampler=m.sampler)
    d.to_az(backend="numpyro")
    return d


@pytest.fixture(scope="module")
def comp_df():
    """Compare table for two tiny models fitted on the same data."""
    m1 = _fit(seed=0)
    m2 = _fit(extra_param=True, seed=0)
    return diagWIP.compare({"m1": m1, "m2": m2})


# ---------------------------------------------------------------------------
# Bug 22 -- every legacy plot method must run on the ArviZ >= 1.2 API
# ---------------------------------------------------------------------------

_PLOT_CALLS = {
    "posterior": lambda d: d.posterior(),
    "density": lambda d: d.density(),
    "plot_trace": lambda d: d.plot_trace(),
    "rank": lambda d: d.rank(),
    "forest": lambda d: d.forest(),
    "pair": lambda d: d.pair(),
    "autocor": lambda d: d.autocor(),
    "plot_ess": lambda d: d.plot_ess(),
}


@pytest.mark.parametrize("name", sorted(_PLOT_CALLS))
def test_bug22_legacy_plots_run(legacy_diag, name):
    try:
        assert _PLOT_CALLS[name](legacy_diag) is not None
    finally:
        plt.close("all")


def test_bug22_single_parameter_model():
    m = _fit_one_param()
    d = diag(sampler=m.sampler)
    d.to_az(backend="numpyro")
    try:
        assert d.posterior() is not None
        assert d.rank() is not None
    finally:
        plt.close("all")


def test_bug22_ignored_option_warns(legacy_diag):
    # ess= is an ArviZ 0.x plot_forest option: accepted, ignored, warned.
    with pytest.warns(UserWarning, match="ArviZ 0.x"):
        legacy_diag.forest(ess=False)
    plt.close("all")


# ---------------------------------------------------------------------------
# Bug 23 -- plot_compare must forward only ArviZ >= 1.2 keywords
# ---------------------------------------------------------------------------

def test_bug23_plot_compare_runs(comp_df):
    try:
        assert diagWIP.plot_compare(comp_df) is not None
        # Diag.py keeps its own staticmethod with the same 1.x body.
        assert diag.plot_compare(comp_df) is not None
        with pytest.warns(UserWarning, match="ArviZ 0.x"):
            diagWIP.plot_compare(comp_df, insample_dev=True)
    finally:
        plt.close("all")
