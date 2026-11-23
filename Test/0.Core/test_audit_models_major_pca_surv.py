"""Regression tests for RELEASE_AUDIT.md, module Models, PCA P1-P3 and survival V1-V4.

Run:
    pytest Test/0.Core/test_audit_models_major_pca_surv.py -q
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpyro.handlers
import pytest

from BayesForge import bf
from BayesForge.Models.surv import survival

PCA_PATH = Path(__file__).resolve().parents[2] / "BayesForge" / "Models" / "PCA.py"


def _surv_object():
    """Survival object with 20 individuals, times 1..20, all events (setup of
    test_survival_sample_times.py, without the covariate import)."""
    s = survival(SimpleNamespace(posteriors={}))
    time = np.arange(1, 21, dtype=float)
    event = np.ones(20)
    s.import_time_even(time, event, interval_length=1)
    return s


# ---------------------------------------------------------------------------
# V1: the observed() method is no longer shadowed by the all-ones attribute
# ---------------------------------------------------------------------------
def test_v1_observed_method_still_callable():
    s = _surv_object()
    # Before the fix, import_time_even replaced the method with a jnp array,
    # so s.observed(arr) raised TypeError: 'JaxArray' object is not callable.
    assert callable(s.observed)
    s.observed(np.ones((s.n_patients, s.n_intervals)))
    assert s.observed_mask.shape == (s.n_patients, s.n_intervals)


# ---------------------------------------------------------------------------
# V2: no covariate imported -> the model still defines an observed likelihood
# ---------------------------------------------------------------------------
def test_v2_no_covariate_model_has_likelihood():
    m = bf(platform="cpu", print_devices_found=False)
    s = m.models.survival
    time = np.arange(1, 21, dtype=float)
    event = np.ones(20)
    s.import_time_even(time, event, interval_length=1)

    data = {k: s.data_on_model[k] for k in ("death", "cov", "exposure")}
    trace = numpyro.handlers.trace(numpyro.handlers.seed(s.model, 0)).get_trace(**data)

    def _base(fn):
        # The likelihood is masked (observed_mask) and may be wrapped in
        # Independent: unwrap to the underlying distribution.
        while hasattr(fn, "base_dist"):
            fn = fn.base_dist
        return fn

    poisson_obs = [
        name
        for name, site in trace.items()
        if site["type"] == "sample"
        and site.get("is_observed", False)
        and type(_base(site["fn"])).__name__ == "Poisson"
    ]
    assert poisson_obs, f"no observed Poisson site in trace {sorted(trace)}"


# ---------------------------------------------------------------------------
# V3: import_time_even validates time and event
# ---------------------------------------------------------------------------
def test_v3_bad_times_raise():
    time = np.arange(1, 21, dtype=float)
    event = np.ones(20)
    # survival time must be strictly positive (> 0)
    zero = time.copy()
    zero[0] = 0.0
    with pytest.raises(ValueError):
        survival(SimpleNamespace(posteriors={})).import_time_even(zero, event, interval_length=1)
    # survival time t_i > 0
    negative = time.copy()
    negative[1] = -1.0
    with pytest.raises(ValueError):
        survival(SimpleNamespace(posteriors={})).import_time_even(negative, event, interval_length=1)
    # event indicator must be binary
    bad_event = event.copy()
    bad_event[2] = 2
    with pytest.raises(ValueError):
        survival(SimpleNamespace(posteriors={})).import_time_even(time, bad_event, interval_length=1)


# ---------------------------------------------------------------------------
# P1: fit no longer transposes the user's X in place; two fits in a row work
# ---------------------------------------------------------------------------
def test_p1_fit_twice_keeps_user_data():
    m = bf(platform="cpu", print_devices_found=False)
    X = np.random.default_rng(0).normal(size=(30, 4))
    m.data_on_model = dict(X=X)
    # API follows Test/8.PCA/PCA.py: m.fit(m.models.pca(type="classic"), ...)
    m.fit(m.models.pca(type="classic"), num_samples=20, num_warmup=20, num_chains=1)
    assert m.data_on_model["X"].shape == (30, 4)
    # z is the (N, K) score matrix: one row per observation.
    z = np.asarray(m.posteriors["z"])
    assert z.shape[-2] == 30
    # Second fit on the same dict: before the fix X was transposed twice.
    m.fit(m.models.pca(type="classic"), num_samples=20, num_warmup=20, num_chains=1)
    assert m.data_on_model["X"].shape == (30, 4)


# ---------------------------------------------------------------------------
# P3: the plt.cScalarMappable typo branch (static check, plotting is heavy)
# ---------------------------------------------------------------------------
def test_p3_scalar_mappable_branch():
    source = PCA_PATH.read_text()
    assert "cScalarMappable" not in source
    assert "matplotlib.cm.ScalarMappable(cmap='viridis')" in source
