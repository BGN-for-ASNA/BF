"""Deep test for doc page "15. Varying intercepts".

Runs the page's OWN python code (extracted with doc_blocks.blocks) once in a
shared namespace, then adds checks the page itself does not make: convergence
and agreement with the Statistical Rethinking reference (m13.2, reedfrogs:
posterior mean a_bar ~ 1.35, sigma ~ 1.62). If the page code changes, this
test automatically runs the new version.

The page must keep the object names this test relies on: `m` as the model
object, two `m.fit(...)` blocks, and params `a_bar` / `sigma` in the first
model's summary.

Discovered from the fitted objects (BF 2.x, Numpyro sampler):
- `m.summary()` returns a pandas DataFrame indexed by parameter name with
  columns ['mean', 'sd', 'eti89_lb', 'eti89_ub', 'ess_bulk', 'ess_tail',
  'r_hat', 'mcse_mean', 'mcse_sd'].
- Fit 0 (centered, explicit names): index 'a_bar', 'alpha[0]'..'alpha[47]',
  'sigma'.
- Fit 1 (built-in varying_intercept): hyper-parameters are named
  'global_intercept_tank' (the group mean, ~ a_bar) and 'sd_tank' (the group
  sd, ~ sigma); group intercepts are 'intercept_tank[0]'..'intercept_tank[47]'.
  We match the hyper-parameters by name (contain 'tank', not indexed like
  '...[n]') which is reliable here, so no fallback to comparing the mean over
  the 48 group intercepts is needed.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "3.Documentation"))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")

from doc_blocks import DOCS, blocks, copy_data

PAGE = DOCS / "15. Varying intercepts.qmd"


def run_page(tmp_path, monkeypatch):
    """Exec every python block of the page in ONE shared namespace.

    Returns the list of model objects `m` captured after each block that
    contains `m.fit(`; the second `bf(...)` call creates a new object, so the
    two references differ.
    """
    copy_data(tmp_path)
    monkeypatch.chdir(tmp_path)
    ns = {}
    fits = []
    for block in blocks(PAGE, "python"):
        exec(block, ns)
        if "m" in ns and "m.fit(" in block:
            fits.append(ns["m"])
    return fits


@pytest.fixture(scope="module")
def fits(tmp_path_factory):
    import _pytest.monkeypatch

    mp = _pytest.monkeypatch.MonkeyPatch()
    tmp = tmp_path_factory.mktemp("doc15")
    try:
        return run_page(tmp, mp)
    finally:
        mp.undo()


def mean_of(fit, name):
    return float(fit.summary().loc[name, "mean"])


@pytest.mark.pages
def test_page_runs(fits):
    assert len(fits) == 2


@pytest.mark.pages
def test_centered_matches_rethinking(fits):
    if os.environ.get("BF_DOC_FAST") == "1":
        pytest.skip("posterior estimates are not meaningful in BF_DOC_FAST mode")
    a_bar = mean_of(fits[0], "a_bar")
    sigma = mean_of(fits[0], "sigma")
    # Statistical Rethinking m13.2 reference values.
    assert abs(a_bar - 1.35) < 0.35, f"a_bar = {a_bar}"
    assert abs(sigma - 1.62) < 0.4, f"sigma = {sigma}"


@pytest.mark.pages
def test_converged(fits):
    if os.environ.get("BF_DOC_FAST") == "1":
        pytest.skip("r_hat is not meaningful with BF_DOC_FAST sampling")
    for i, fit in enumerate(fits):
        r_hat = fit.summary()["r_hat"]
        bad = r_hat[r_hat >= 1.05]
        assert bad.empty, f"fit {i} has r_hat >= 1.05:\n{bad}"


@pytest.mark.pages
def test_builtin_agrees_with_centered(fits):
    if os.environ.get("BF_DOC_FAST") == "1":
        pytest.skip("posterior estimates are not meaningful in BF_DOC_FAST mode")
    s1 = fits[1].summary()
    # Hyper-parameters of the built-in varying_intercept: names containing
    # 'tank' that are not one of the 48 indexed group intercepts.
    hyper = [n for n in s1.index if "tank" in n and not re.search(r"\[\d+\]$", n)]
    assert "global_intercept_tank" in hyper and "sd_tank" in hyper, hyper
    a_bar = mean_of(fits[0], "a_bar")
    sigma = mean_of(fits[0], "sigma")
    mu = float(s1.loc["global_intercept_tank", "mean"])
    sd = float(s1.loc["sd_tank", "mean"])
    assert abs(mu - a_bar) < 0.35, f"global_intercept_tank={mu} vs a_bar={a_bar}"
    assert abs(sd - sigma) < 0.35, f"sd_tank={sd} vs sigma={sigma}"
