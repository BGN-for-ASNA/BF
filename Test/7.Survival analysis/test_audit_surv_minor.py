"""Regression tests for the minor audit points U1-U9 (surv.py) and G1-G2 (models.py).

Run:
    pytest "Test/7.Survival analysis/test_audit_surv_minor.py" -q

Every test fails on the pre-audit code and passes on the fixed code.
"""

import inspect
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import BayesForge.Models.GMM as GMM_mod
from BayesForge.Models.surv import survival, survival_old
from BayesForge.Models.models import models


def _surv_object():
    """Survival object with 20 individuals, times 1..20, all events (setup of
    test_audit_models_major_pca_surv.py)."""
    s = survival(SimpleNamespace(posteriors={}))
    time = np.arange(1, 21, dtype=float)
    event = np.ones(20)
    s.import_time_even(time, event, interval_length=1)
    return s


# ---------------------------------------------------------------------------
# U1: _posterior_draws raises an informative KeyError instead of silently
#     falling back to posteriors[beta] when neither beta nor the per-covariate
#     Hazard_rate_<name> keys are present.
# ---------------------------------------------------------------------------
def test_u1_posterior_draws_missing_beta_raises_informative_keyerror():
    s = _surv_object()
    s.import_covF(np.ones((20, 1)), ['trt'])
    parent = SimpleNamespace(posteriors={'Baseline_rate': np.ones((2, s.n_intervals))})
    s.parent = parent
    with pytest.raises(KeyError) as exc:
        s._posterior_draws('Baseline_rate', 'beta', None)
    # The pre-audit code raised KeyError('beta') from the dict lookup; the fix
    # must name both accepted key conventions and list the available keys.
    msg = str(exc.value)
    assert 'Hazard_rate_' in msg
    assert 'beta' in msg
    assert 'Baseline_rate' in msg


# ---------------------------------------------------------------------------
# U2: legacy survival_old.hazards honours its lambda0/beta arguments instead
#     of hard-coding posteriors["lambda0"] / posteriors["beta"].
# ---------------------------------------------------------------------------
def test_u2_legacy_hazards_uses_key_arguments():
    lam = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]])
    b = np.array([0.5, 0.0, -0.5])
    m = SimpleNamespace(posteriors={'lam_key': lam,
                                    'b_key': b,
                                    'b_zero': np.zeros(3)})
    s = survival_old(SimpleNamespace(df=None, posteriors={}))

    base, met = s.hazards(m, lambda0='lam_key', beta='b_key')
    np.testing.assert_allclose(base, lam, rtol=1e-6)
    np.testing.assert_allclose(met, lam * np.exp(b)[:, None], rtol=1e-6)

    # A different beta key must change the result: with beta == 0 the
    # covariate contributes nothing, so the met-hazard equals lambda0.
    _, met0 = s.hazards(m, lambda0='lam_key', beta='b_zero')
    np.testing.assert_allclose(met0, lam, rtol=1e-6)


# ---------------------------------------------------------------------------
# U3: legacy survival_old.plot_censoring indexes the time column by the
#     ``time`` argument instead of the hard-coded 'time' label.
# ---------------------------------------------------------------------------
def test_u3_legacy_plot_censoring_uses_time_column_name():
    df = pd.DataFrame({'tt': [1.0, 2.0, 3.0, 4.0],
                       'event': [1, 0, 1, 0],
                       'metastasized': [0, 1, 0, 1]})
    m = survival_old(SimpleNamespace(df=df, posteriors={}))
    m.plot_censoring(event='event', time='tt', cov='metastasized')
    plt.close('all')


# ---------------------------------------------------------------------------
# U4: constructing survival_old no longer assigns self.df = None, which went
#     through the df property and erased the parent's data.
# ---------------------------------------------------------------------------
def test_u4_survival_old_construction_keeps_parent_df():
    df = pd.DataFrame({'time': [1.0, 2.0], 'event': [1, 0]})
    parent = SimpleNamespace(df=df, posteriors={})
    m = survival_old(parent)
    assert parent.df is df
    assert m.df is df


# ---------------------------------------------------------------------------
# U5: the import_* / build_surv_covariates summaries are silent by default
#     and come back with self.verbose = True; the legacy development warning
#     print in surv_object is gone.
# ---------------------------------------------------------------------------
def test_u5_import_summaries_silent_by_default_and_restored_with_verbose(capsys):
    time = np.arange(1, 21, dtype=float)
    event = np.ones(20)

    s = survival(SimpleNamespace(posteriors={}))
    s.import_time_even(time, event, interval_length=1)
    s.import_covF(np.ones((20, 1)), ['trt'])
    out = capsys.readouterr().out
    assert out == "", f"import_* printed without verbose: {out!r}"

    s_verbose = survival(SimpleNamespace(posteriors={}))
    s_verbose.verbose = True
    s_verbose.import_time_even(time, event, interval_length=1)
    out_verbose = capsys.readouterr().out
    assert 'Survival concern' in out_verbose


# ---------------------------------------------------------------------------
# U6: survival.__init__ no longer writes parent.model_name (fit() owns it).
# ---------------------------------------------------------------------------
def test_u6_survival_construction_does_not_set_model_name():
    parent = SimpleNamespace(posteriors={})
    survival(parent)
    assert not hasattr(parent, 'model_name')


# ---------------------------------------------------------------------------
# U7: interval assignment uses ceil(time / interval_length) - 1, so an
#     observation exactly on an interval bound lands in the interval that
#     ends at it; the old floor((time - eps) / interval_length) collapsed
#     small times onto interval 0.
# ---------------------------------------------------------------------------
def test_u7_interval_assignment_exact_multiples():
    s = survival(SimpleNamespace(posteriors={}))
    time = np.array([1e-8, 2e-8, 3e-8])
    event = np.ones(3)
    s.import_time_even(time, event, interval_length=1e-8)
    assert list(s.last_period) == [0, 1, 2]


# ---------------------------------------------------------------------------
# U8: import_covF / import_covV accept a single string name (it used to be
#     iterated character by character or rejected by the length check).
# ---------------------------------------------------------------------------
def test_u8_import_covF_accepts_single_string_name():
    s = _surv_object()
    s.import_covF(np.ones((20, 1)), 'trt')
    assert s.cov_names == ['trt']
    assert s.n_cov == 1


def test_u8_import_covV_accepts_single_string_name():
    s = _surv_object()
    s.import_covV(np.ones((20, s.n_intervals)), 'tv')
    assert s.cov_v_names == ['tv']
    assert s.n_cov_v == 1


# ---------------------------------------------------------------------------
# U9: n_cov / n_cov_v exist from construction (0) and stay in sync with the
#     imported covariates.
# ---------------------------------------------------------------------------
def test_u9_n_cov_attributes_initialised_and_synced():
    s = survival(SimpleNamespace(posteriors={}))
    assert s.n_cov == 0
    assert s.n_cov_v == 0
    time = np.arange(1, 21, dtype=float)
    event = np.ones(20)
    s.import_time_even(time, event, interval_length=1)
    s.import_covF(np.ones((20, 1)), ['a'])
    assert s.n_cov == 1
    s.import_covV(np.ones((20, s.n_intervals)), ['b'])
    assert s.n_cov_v == 1


# ---------------------------------------------------------------------------
# G1: models().gmm carries .predict / .plot shortcuts, and attaching them no
#     longer mutates the module-level gmm function.
# ---------------------------------------------------------------------------
def test_g1_models_gmm_exposes_predict_and_plot_without_leaking():
    mod = models(SimpleNamespace(posteriors={}))
    assert callable(mod.gmm)
    assert callable(mod.gmm.predict)
    assert callable(mod.gmm.plot)
    assert mod.gmm.predict is GMM_mod.predict_gmm
    assert mod.gmm.plot is GMM_mod.plot_gmm
    # The module-level function must stay clean for every other user.
    assert not hasattr(GMM_mod.gmm, 'predict')
    assert not hasattr(GMM_mod.gmm, 'plot')


# ---------------------------------------------------------------------------
# G2: models stores the parent it receives (audit point models.py:13-17), and
#     the gmm wrapper keeps the identity main.py checks for (fit dispatches on
#     __name__ == 'gmm' and __module__ == 'BayesForge.Models.GMM', and
#     _check_data_matches_model reads inspect.signature).
# ---------------------------------------------------------------------------
def test_g2_models_stores_parent_and_wrapper_keeps_identity():
    parent = SimpleNamespace(posteriors={})
    mod = models(parent)
    # Pre-audit code received parent but never stored it.
    assert mod.parent is parent
    func = mod.gmm
    # Same condition as BayesForge/Main/main.py (GMM dispatch branch).
    assert func.__name__ == 'gmm'
    assert func.__module__ == 'BayesForge.Models.GMM'
    assert list(inspect.signature(func).parameters) == ['data', 'K', 'initial_means']
    # It is still the wrapper carrying the shortcuts, not the raw function.
    assert callable(func.predict)
