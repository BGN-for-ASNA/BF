"""Tests for survival.ppc_km_by_cov(): grouped KM PPCs from sample_times().

No model fit needed: the object is built directly, the posteriors dict is
filled with hand-made draws, and the parent's ``diag`` namespace is a stub
exposing the real ``ppc.ppc_km_overlay``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from BayesForge.Diagnostic import ppc
from BayesForge.Models.surv import survival


def _object():
    s = survival(SimpleNamespace(posteriors={}))
    time = np.arange(1, 21, dtype=float)
    event = np.ones(20)
    s.import_time_even(time, event, interval_length=1)
    s.import_covF(np.r_[np.zeros(10), np.ones(10)], ["x"])   # binary
    s.import_covF(np.linspace(0.0, 1.0, 20), ["z"])          # continuous
    s.parent.diag = SimpleNamespace(ppc_km_overlay=ppc.ppc_km_overlay)
    return s


def _posteriors(s):
    K = s.n_intervals
    s.parent.posteriors.update({
        "Baseline_rate": np.full((50, K), 0.2),
        "beta": np.column_stack([np.zeros(50), np.full(50, np.log(2.0))]),
    })


def _panel_titles(fig):
    return [a.text for a in fig.layout.annotations]


def test_one_figure_per_covariate_plus_risk():
    s = _object()
    _posteriors(s)
    # Default is single plot overlay (facet=False)
    figs = s.ppc_km_by_cov()
    assert set(figs) == {"x", "z", "risk"}
    for k, fig in figs.items():
        assert fig.layout.title.text == f"PPC: Kaplan-Meier overlay of time by {k}"
        assert len(fig.layout.annotations or []) == 0

    # With facet=True: separated panels
    figs_faceted = s.ppc_km_by_cov(facet=True)
    assert set(figs_faceted) == {"x", "z", "risk"}
    assert _panel_titles(figs_faceted["x"]) == ["x = 0", "x = 1"]
    assert _panel_titles(figs_faceted["z"]) == ["z = Q1", "z = Q2", "z = Q3"]
    assert _panel_titles(figs_faceted["risk"]) == ["risk = Q1", "risk = Q2", "risk = Q3"]
    for k, fig in figs_faceted.items():
        assert fig.layout.title.text == f"PPC: Kaplan-Meier overlay of time by {k}"


def test_subset_and_unknown_covariate():
    s = _object()
    _posteriors(s)
    assert set(s.ppc_km_by_cov(covs="z", risk=False)) == {"z"}
    with pytest.raises(ValueError, match="Unknown covariate"):
        s.ppc_km_by_cov(covs="nope")


def test_panels_hold_the_right_patients():
    s = _object()
    _posteriors(s)
    # Faceted version
    fig_fac = s.ppc_km_by_cov(covs="x", risk=False, facet=True)["x"]
    x = np.asarray(s.cov)[:, 0].astype(bool)
    obs_fac = [t for t in fig_fac.data if t.name == "time (observed)"]
    assert len(obs_fac) == 2                       # one per panel, level 0 then 1
    t, surv = ppc._km(np.asarray(s.time)[x], np.asarray(s.event)[x])
    np.testing.assert_allclose(np.asarray(obs_fac[1].x, dtype=float), t)
    np.testing.assert_allclose(np.asarray(obs_fac[1].y, dtype=float), surv)

    # Single-plot overlay version (default)
    fig_ov = s.ppc_km_by_cov(covs="x", risk=False)["x"]
    obs_ov = [t for t in fig_ov.data if "observed" in t.name]
    assert len(obs_ov) == 2
    np.testing.assert_allclose(np.asarray(obs_ov[1].x, dtype=float), t)
    np.testing.assert_allclose(np.asarray(obs_ov[1].y, dtype=float), surv)


def test_posterior_draws_hazard_rate_fallback():
    s = _object()
    s.parent.posteriors.update({
        "Baseline_rate": np.full((50, s.n_intervals), 0.2),
        "Hazard_rate_x": np.zeros((50, 1)),
        "Hazard_rate_z": np.full((50, 1), np.log(2.0)),
    })
    figs = s.ppc_km_by_cov()
    assert set(figs) == {"x", "z", "risk"}
