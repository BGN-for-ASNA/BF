"""diagWIP.summary/diagnose delegate to ArviZ.

Owner decision 2026-11-13 (RELEASE_AUDIT): ``diagWIP.summary`` computes its
table with ``az.summary`` (HDI columns renamed to the BayesForge convention,
``median`` re-inserted after ``mean``, old column order) and the ESS / R-hat
sections of ``diagnose`` call ``az.ess(method="bulk")`` / ``az.rhat`` once.
Single-chain R-hat is NaN, as in ArviZ, here and in ``jax_diagnostics``.
"""
# BayesForge before arviz: its __init__ patches jax.interpreters.pxla, and
# arviz pulling jax in first can leave that submodule in a state that breaks
# the patch (see test_diag_parity.py).
from BayesForge.Diagnostic.Diag2 import diagWIP, _az_from_dict
from BayesForge.Diagnostic import jax_diagnostics as jd

import numpy as np
import pytest

import arviz as az

# 4 chains x 500 draws: one scalar parameter and one (2, 3) parameter.
RNG = np.random.default_rng(7)
POST = {
    "mu": RNG.normal(0.0, 1.0, size=(4, 500)),
    "x": RNG.normal(2.0, 0.5, size=(4, 500, 2, 3)),
}
LABELS = ["mu", "x[0, 0]", "x[0, 1]", "x[0, 2]", "x[1, 0]", "x[1, 1]", "x[1, 2]"]
HDIP = 0.89
BF_LOWER = f"hdi_{HDIP * 100}%_lower"
BF_UPPER = f"hdi_{HDIP * 100}%_upper"
AZ_LOWER = f"hdi{int(HDIP * 100)}_lb"
AZ_UPPER = f"hdi{int(HDIP * 100)}_ub"


def _az_ref(kind):
    """The reference az.summary, raw values (round_to="none")."""
    idata = _az_from_dict({"posterior": POST})
    return az.summary(idata, kind=kind, ci_prob=HDIP, ci_kind="hdi",
                      round_to="none")


# ================= summary == az.summary (both kinds) =================

@pytest.mark.parametrize("kind", ["all", "stats"])
def test_summary_matches_az_summary_values(kind):
    """Every ArviZ column matches az.summary to rel=1e-9 before rounding."""
    d = diagWIP(dict(POST))
    got = d.summary(round_to=12, hdi_prob=HDIP, kind=kind)  # round_to large
    ref = _az_ref(kind)

    # Row labels: "name" for scalars, "name[i, j]" for elements.
    assert list(got.index) == LABELS
    assert list(ref.index) == LABELS

    # Column names and order of the old manual table.
    columns = ["mean", "median", "sd", BF_LOWER, BF_UPPER]
    if kind == "all":
        columns += ["mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat"]
    assert list(got.columns) == columns

    # Every az column (HDI under its BayesForge name), same values.
    az_to_bf = {"mean": "mean", "sd": "sd",
                AZ_LOWER: BF_LOWER, AZ_UPPER: BF_UPPER}
    if kind == "all":
        az_to_bf.update({"mcse_mean": "mcse_mean", "mcse_sd": "mcse_sd",
                         "ess_bulk": "ess_bulk", "ess_tail": "ess_tail",
                         "r_hat": "r_hat"})
    for az_col, bf_col in az_to_bf.items():
        for label in LABELS:
            assert float(got.loc[label, bf_col]) == pytest.approx(
                float(ref.loc[label, az_col]), rel=1e-9), f"{label}.{bf_col}"


def test_summary_median_column_present_and_correct():
    """median sits right after mean and is np.median over chains and draws."""
    d = diagWIP(dict(POST))
    got = d.summary(round_to=12, hdi_prob=HDIP)
    assert list(got.columns)[0:2] == ["mean", "median"]
    for label in LABELS:
        if "[" in label:
            idx = tuple(int(i) for i in
                        label[label.index("[") + 1:-1].split(", "))
            draws = POST["x"][:, :, idx[0], idx[1]]
        else:
            draws = POST["mu"]
        assert float(got.loc[label, "median"]) == pytest.approx(
            float(np.median(draws)), rel=1e-9)


# ================= var_names / exclude_vars filtering =================

def test_summary_var_names_filtering():
    d = diagWIP(dict(POST))
    assert list(d.summary(var_names=["mu"]).index) == ["mu"]
    x_labels = [lab for lab in LABELS if lab != "mu"]
    assert list(d.summary(var_names=["x"]).index) == x_labels
    # single string accepted (R path), same semantics as a list
    assert list(d.summary(var_names="mu").index) == ["mu"]


def test_summary_exclude_vars_filtering():
    d = diagWIP(dict(POST))
    x_labels = [lab for lab in LABELS if lab != "mu"]
    assert list(d.summary(exclude_vars=["mu"]).index) == x_labels
    assert list(d.summary(exclude_vars="x").index) == ["mu"]


# ================= diagnose: single chain R-hat problem line =================

def test_diagnose_single_chain_lists_rhat_not_computable():
    rng = np.random.default_rng(11)
    d = diagWIP({"mu": rng.normal(size=(1, 1000))})
    report = d.diagnose()
    assert ("R-hat could not be computed for: mu (fewer than 2 chains or "
            "too few draws); convergence between chains is not checked."
            in report)
    # diagnose must no longer report success on a single chain.
    assert "no problems detected" not in report
    assert "1 problem(s) detected" in report


def test_diagnose_two_chains_reports_no_rhat_problem():
    rng = np.random.default_rng(12)
    d = diagWIP({"mu": rng.normal(size=(2, 1000))})
    report = d.diagnose()
    assert "R-hat could not be computed" not in report
    assert "Rank-normalized split R-hat values satisfactory" in report
    assert "no problems detected" in report


# ================= single-chain R-hat = NaN (ArviZ behaviour) =================

def test_jax_diagnostics_rhat_single_chain_nan():
    """RELEASE_AUDIT bug 3: NaN like ArviZ (owner decision 2026-11-13)."""
    x = np.random.default_rng(4).normal(size=(1, 500))
    with pytest.warns(UserWarning, match="single chain"):
        val = jd.rhat({"a": x})["a"]
    assert np.isnan(val)


def test_diag2_rhat_manual_single_chain_nan():
    """RELEASE_AUDIT bug 3: NaN like ArviZ (owner decision 2026-11-13)."""
    draws = np.random.default_rng(5).normal(size=(1, 200))
    d = diagWIP({"mu": draws})
    with pytest.warns(UserWarning, match="single chain"):
        val = d._rhat_manual(draws)
    assert np.isnan(val)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
