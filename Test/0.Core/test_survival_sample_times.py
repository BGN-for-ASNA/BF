"""Tests for survival.sample_times(): posterior predictive event times.

No model fit needed: the object is built directly and the posteriors dict is
filled with hand-made draws.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import kstest

from BayesForge.Models.surv import survival


def _object():
    s = survival(SimpleNamespace(posteriors={}))
    time = np.arange(1, 21, dtype=float)
    event = np.ones(20)
    s.import_time_even(time, event, interval_length=1)
    s.import_covF(np.r_[np.zeros(10), np.ones(10)], ["x"])
    return s


def test_constant_hazard_is_exponential():
    s = _object()
    K = s.n_intervals
    s.parent.posteriors.update({
        "Baseline_rate": np.full((400, K), 0.2),
        "beta": np.zeros((400, 1)),
    })

    times = s.sample_times()
    assert times.shape == (400, 20)
    assert np.all(times > 0)

    flat = times.ravel()
    assert abs(flat.mean() - 5.0) / 5.0 < 0.05
    assert kstest(flat, "expon", args=(0, 5.0)).pvalue > 0.01


def test_covariate_halves_the_mean_time():
    s = _object()
    K = s.n_intervals
    s.parent.posteriors.update({
        "Baseline_rate": np.full((400, K), 0.2),
        "beta": np.full((400, 1), np.log(2.0)),
    })

    times = s.sample_times()
    x = np.asarray(s.cov)[:, 0].astype(bool)
    mean_unexposed = times[:, ~x].mean()
    mean_exposed = times[:, x].mean()

    assert abs(mean_exposed / mean_unexposed - 0.5) / 0.5 < 0.10


def test_zero_rate_first_interval_delays_every_event():
    s = _object()
    K = s.n_intervals
    rate = np.full((400, K), 1.0)
    rate[:, 0] = 0.0
    s.parent.posteriors.update({
        "Baseline_rate": rate,
        "beta": np.zeros((400, 1)),
    })

    times = s.sample_times()
    assert np.all(times >= 1.0)  # bound of interval 0 with interval_length=1


def test_zero_hazard_and_draws_and_seed():
    s = _object()
    K = s.n_intervals
    s.parent.posteriors.update({
        "Baseline_rate": np.zeros((400, K)),
        "beta": np.zeros((400, 1)),
    })

    times = s.sample_times()
    assert np.all(np.isinf(times))

    few = s.sample_times(n_draws=5)
    assert few.shape == (5, 20)
    assert np.array_equal(few, s.sample_times(n_draws=5))

    s.parent.posteriors["Baseline_rate"] = np.full((400, K), 0.2)
    assert np.array_equal(s.sample_times(seed=7), s.sample_times(seed=7))
    assert not np.array_equal(s.sample_times(seed=7), s.sample_times(seed=8))
