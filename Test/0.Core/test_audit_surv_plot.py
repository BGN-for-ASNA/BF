"""Regression test for RELEASE_AUDIT.md, module 04, critical C5 (surv.py plot_surv hazard)."""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np

from BayesForge.Models.surv import survival


def _object(beta_value):
    s = survival(SimpleNamespace(posteriors={}))
    time = np.arange(1, 21, dtype=float)
    s.import_time_even(time, np.ones(20), interval_length=1)
    s.import_covF(np.r_[np.zeros(10), np.ones(10)], ["treated"])
    K = s.n_intervals
    s.parent.posteriors.update({
        "Baseline_rate": np.full((50, K), 0.2),
        "Hazard_rate_treated": np.full((50, 1), beta_value),
    })
    return s


def _met_curves(fig_axes):
    _, (hazard_ax, surv_ax) = fig_axes
    return (np.asarray(hazard_ax.lines[1].get_ydata()),
            np.asarray(surv_ax.lines[1].get_ydata()))


def test_c5_covariate_hazard_is_lambda0_times_exp_beta():
    beta = -0.7
    s = _object(beta)
    cum_h, surv = _met_curves(s.plot_surv(beta="Hazard_rate_treated"))
    expected = np.asarray(s.cum_hazard(np.full((50, s.n_intervals), 0.2 * np.exp(beta)))).mean(0)[:len(cum_h)]
    np.testing.assert_allclose(cum_h, expected, rtol=1e-5)
    assert np.all(cum_h >= 0) and np.all(surv <= 1.0 + 1e-6)


def test_c5_default_beta_is_the_model_site():
    s = _object(0.3)
    s.plot_surv()  # must not raise KeyError('beta')
