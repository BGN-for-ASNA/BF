"""Regression test for RELEASE_AUDIT.md, module Diagnostic, Bug 4 (short chains: ESS/R-hat parity with ArviZ)."""

import numpy as np
import pytest

# NOTE: import BayesForge *before* arviz. BayesForge.__init__ applies a
# jax.interpreters.pxla compatibility patch on import; importing arviz (which
# pulls jax) first breaks it. Same order as test_diag_parity.py.
from BayesForge.Diagnostic.Diag2 import diagWIP
import arviz as az


# ---------------------------------------------------------------------------
# Helpers (patterns copied from the sibling audit/parity tests, not imported)
# ---------------------------------------------------------------------------

class _FakeSampler:
    """Minimal stand-in for a fitted sampler (pattern of test_audit_diag_major.py)."""

    def __init__(self, samples):
        self._samples = samples

    def get_samples(self, group_by_chain=True):
        return self._samples


@pytest.fixture(scope="module")
def wip():
    """diagWIP over a dummy 2-chain trace; the manual helpers take raw (C, S) arrays."""
    return diagWIP(_FakeSampler({"mu": np.random.default_rng(0).normal(size=(2, 8))}))


def _az_float(res):
    """Scalar metric from an ArviZ result for ndarray input.

    az.ess / az.rhat on a raw (chain, draw) ndarray go through arviz_stats'
    array layer, which reduces both sample axes and returns a scalar (0-d);
    float(np.asarray(...)) is the same extraction test_diag_parity.py::_ds_val
    uses for Dataset results.
    """
    return float(np.asarray(res))


# ---------------------------------------------------------------------------
# Bug 4 -- min-draws guard must run on the UNSPLIT chains, like ArviZ
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_draw", [4, 5, 6, 7, 10])
def test_short_chain_ess_rhat_finite_and_match_arviz(wip, n_draw):
    """4..7 draws used to give NaN: the guard ran inside _ess_raw_manual on the
    already-SPLIT array (2..3 draws), while ArviZ's _not_valid(min_draws=4)
    checks the UNSPLIT chains before splitting. Diag2 must now return finite
    values equal to az.ess / az.rhat."""
    arr = np.random.default_rng(1).normal(size=(2, n_draw))

    bulk = wip._ess_bulk_manual(arr)
    tail = wip._ess_tail_manual(arr)
    rhat = wip._rhat_manual(arr)
    assert np.isfinite(bulk), f"bulk ESS not finite for n_draw={n_draw}"
    assert np.isfinite(tail), f"tail ESS not finite for n_draw={n_draw}"
    assert np.isfinite(rhat), f"R-hat not finite for n_draw={n_draw}"

    assert bulk == pytest.approx(_az_float(az.ess(arr, method="bulk")), rel=1e-6)
    assert rhat == pytest.approx(_az_float(az.rhat(arr)), rel=1e-6)

    if n_draw % 2 == 0:
        # Even S: the split array holds exactly the unsplit draws, so Diag2
        # (split -> quantile) and arviz_stats 1.3.x _ess_quantile (quantile on
        # the unsplit draws -> split the indicator) threshold the same set.
        assert tail == pytest.approx(
            _az_float(az.ess(arr, method="tail", prob=0.05)), rel=1e-6)
    # Odd S (5, 7): arviz_stats >= 1.3 thresholds BEFORE splitting, so the
    # dropped middle draw leaves the INDICATOR, while Diag2 drops it from the
    # draws before thresholding. That pre-existing ordering difference is out
    # of Bug 4's scope (the fix must not change any formula) and can shift the
    # tail ESS by ~1 draw; only finiteness -- the Bug 4 symptom -- is asserted
    # for odd n_draw above.


def test_three_draws_still_nan_like_arviz(wip):
    """3 unsplit draws: ArviZ's _not_valid(min_draws=4) returns NaN; Diag2 must
    match whatever ArviZ returns (NaN-aware)."""
    arr = np.random.default_rng(1).normal(size=(2, 3))
    for manual, reference in (
        (wip._ess_bulk_manual(arr), az.ess(arr, method="bulk")),
        (wip._ess_tail_manual(arr), az.ess(arr, method="tail", prob=0.05)),
        (wip._rhat_manual(arr), az.rhat(arr)),
    ):
        reference = _az_float(reference)
        if np.isnan(reference):
            assert np.isnan(manual)
        else:
            assert manual == pytest.approx(reference, rel=1e-6)
