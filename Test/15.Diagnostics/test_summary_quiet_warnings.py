"""Degenerate chains must not print float-division noise through m.diag.summary
(Contract from the review: silencier les artefacts, garder l'information).

All-equal draws inside a chain (or NaN draws) make ArviZ divide by
zero/NaN when computing ESS, mcse and r_hat. There the RuntimeWarnings are
known float artifacts, not model problems: diag.summary/rhat/ess silence
them via ``_quiet_float_summary`` while keeping the numeric truth (NaN
stays in the table).

Runs on ArviZ 0.x *and* 1.x: the public tests pin the contract without
knowing which internal op emits the warning; the helper unit tests pin
exactly what is swallowed and what passes through, and that the module
state is fully restored afterwards (no filter leaks).
"""
import warnings

import numpy as np
import pytest

from BayesForge.Diagnostic.Diag import diag, _quiet_float_summary

arviz = pytest.importorskip("arviz")
from BayesForge.Diagnostic.Diag2 import _az_from_dict  # noqa: E402  (after arviz)


def _diag_with(draws):
    """A diag whose trace is injected directly (no sampler, no fit)."""
    d = diag(object())
    d.trace = _az_from_dict({"posterior": {"x": np.asarray(draws, dtype=float)}})
    return d


def _record(fn):
    """Run fn() capturing every warning; return (result, formatted list)."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = fn()
    return out, [f"{w.category.__name__}: {w.message}" for w in rec]


# ================= _quiet_float_summary unit contract =================

def test_helper_swallows_invalid_and_zero_divide():
    """0/0 ('invalid') and 1/0 ('divide by zero') stay silent inside."""
    with _quiet_float_summary(), warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        bad = np.float64(0) / np.float64(0)          # invalid: matches the filter
        inf = np.float64(1) / np.float64(0)          # zero divide: errstate ignore
    assert np.isnan(bad) and np.isinf(inf)
    assert [f"{w.category.__name__}: {w.message}" for w in rec] == []


def test_helper_passes_unrelated_warnings_through():
    """The dark-zone is only float noise: everything else still surfaces."""
    with pytest.warns(UserWarning, match="unrelated"):
        with _quiet_float_summary():
            warnings.warn("an unrelated warning", UserWarning)


def test_helper_restores_state_afterwards():
    """np.errstate and the catch_warnings filters are functionally back:
    float ops warn again after exit, and warnings after exit still surface.
    (Comparing warnings.filters snapshots is not stable under pytest, whose
    plugin rewrites the global filter stack during a test.)"""
    err_before = np.geterr()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with _quiet_float_summary():
            np.float64(0) / np.float64(0)
        np.sqrt(-1.0)                                  # invalid: warns again
        np.float64(1) / np.float64(0)                  # divide: warns again
    rt = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert any("sqrt" in str(w.message) for w in rt)   # errstate restored
    assert any("divide by zero" in str(w.message) for w in rt)
    assert np.geterr() == err_before                   # division state restored
    with pytest.warns(UserWarning, match="post-helper"):
        with _quiet_float_summary():
            pass
        warnings.warn("post-helper emitted", UserWarning)  # filters restored


# ================= public API on degenerate traces =================

def test_summary_constant_chains_quiet_with_nan_kept():
    d = _diag_with(np.full((2, 50), 1.5))              # all-equal draws
    (_, labels) = _record(d.summary)
    runtime = [lab for lab in labels if "RuntimeWarning" in lab]
    assert runtime == [], runtime                      # no float-division noise
    assert "x" in d.tab_summary.index                  # the table is still built


def test_summary_nan_draws_quiet_and_nan_output():
    rng = np.random.default_rng(0)
    vals = rng.normal(0.0, 1.0, (2, 50))
    vals[:, -5:] = np.nan                              # NaN draws -> 0/0 in ess
    d = _diag_with(vals)
    (_, labels) = _record(d.summary)
    assert [lab for lab in labels if "RuntimeWarning" in lab] == []
    assert d.tab_summary.loc["x", "mean"] is np.nan or np.isnan(
        d.tab_summary.loc["x", "mean"])                # NaN kept, not hidden


def test_rhat_ess_constant_chains_quiet_and_nan_kept():
    d = _diag_with(np.full((2, 50), 1.5))
    (rhat, labels) = _record(d.rhat)
    assert [lab for lab in labels if "RuntimeWarning" in lab] == []
    (ess, ess_labels) = _record(d.ess)
    assert [lab for lab in ess_labels if "RuntimeWarning" in lab] == []
    assert np.isnan(float(rhat["x"]))                  # r_hat NaN on degeneracy


def test_summary_round_to_and_kind_still_keywords():
    """round_to/kind/hdi_prob are keyword-only now; positional extras go to az."""
    d = _diag_with(np.full((2, 50), 1.5))
    assert d.summary(round_to=3) is d.tab_summary      # keyword path unchanged
    with pytest.raises(TypeError):                     # positional extra -> az.summary
        d.summary(3)
