# Run:
#   pytest -v "Test/Diag2 vs Diag/test_diag_parity.py"
#
# Purpose
# -------
# `BayesForge.Diagnostic.Diag.diag` is the original, ArviZ-backed diagnostics
# class. `BayesForge.Diagnostic.Diag2.diagWIP` is the newer, JAX/NumPy-native
# reimplementation that BayesForge now uses by default (main.py imports
# `diagWIP as diag`). This suite pins that the two return the *same* numbers on
# the *same* fitted sampler — with particular attention to the hard cases:
#
#   * R-hat and ESS via the public API (both delegate to ArviZ -> must be equal)
#   * the hand-rolled convergence measures in Diag2 that do NOT go through
#     ArviZ: rank-normalized split R-hat (`_rhat_manual`), bulk/tail ESS
#     (`_ess_bulk_manual` / `_ess_tail_manual`) and E-BFMI (`_ebfmi_manual`).
#     These are checked against ArviZ's reference implementations
#     (`az.rhat(method="rank")`, `az.ess(method="bulk"/"tail")`, `az.bfmi`).
#   * the summary table (mean / std / HDI).
#
# One model is fitted once (module-scoped fixture) and shared by every test.
#
# The reference values are read through the helpers below, which adapt only
# *how* ArviZ is asked for a statistic: 1.x (arviz-stats) takes
# `az.from_dict({"posterior": ...})` instead of `from_dict(posterior=...)`,
# returns DataTrees instead of Datasets, renamed `hdi_prob=` to `prob=` (hdi)
# and `ci_prob=` (summary), and removed `az.waic` (its numbers are recomputed here
# from the standard formulas). No tolerance, expectation or comparison changed.

# NOTE: import BayesForge *before* arviz. BayesForge.__init__ applies a
# jax.interpreters.pxla compatibility patch on import; importing arviz (which
# pulls in jax) first can leave that submodule in a state that breaks the patch.
from BayesForge import bf
from BayesForge.Diagnostic.Diag import diag as diag_ref      # ArviZ reference
from BayesForge.Diagnostic.Diag2 import (                    # JAX/NumPy reimpl.
    diagWIP as diag_wip,
    _az_from_dict,
)

import numpy as np
import pytest
import arviz as az
from scipy.special import logsumexp


# ================== Shared fitted model ==================

@pytest.fixture(scope="module")
def fitted():
    """Fit a small, well-identified model with several chains, once.

    Returns (ref, wip, params) where ref/wip are the two diagnostic objects
    built on the *same* sampler and params is the list of parameter names.
    """
    m = bf(platform="cpu")

    data_path = m.load.howell1(only_path=True)
    m.data(data_path, sep=";")
    m.df = m.df[m.df.age > 18]
    m.scale(["weight"])

    def model(weight, height):
        a = m.dist.normal(178, 20)
        b = m.dist.log_normal(0, 1)
        s = m.dist.uniform(0, 50)
        m.dist.normal(a + b * weight, s, obs=height)

    # 4 chains x 1500 draws -> stable ESS / R-hat estimates.
    m.fit(model, num_samples=1500, num_warmup=1000, num_chains=4, progress_bar=False)

    ref = diag_ref(sampler=m.sampler)
    ref.to_az()
    wip = diag_wip(sampler=m.sampler)
    wip.to_az()

    params = list(ref.priors_name)
    return ref, wip, params


# --- helpers -----------------------------------------------------------------

def _chains(wip, param):
    """Per-chain samples (C, S) for a scalar parameter, as a NumPy array."""
    return np.asarray(wip.posterior_samples[param])


def _ds_val(ds, param):
    """Scalar metric for `param` from an ArviZ rhat/ess result.

    ArviZ 1.x returns an ``xarray.DataTree`` rooted at the group it computed
    (0.x returned an ``xarray.Dataset``); the variable is at ``ds[param]``
    either way, with ``ds["posterior"][param]`` as the fallback for builds that
    keep the group as a child node.
    """
    try:
        obj = ds[param]
    except (KeyError, TypeError):
        obj = ds["posterior"][param]
    return float(np.asarray(obj.values))


def _elpd(res):
    """elpd from an ELPDData across arviz majors (0.x: elpd_loo, 1.x: elpd)."""
    for attr in ("elpd_loo", "elpd"):
        val = getattr(res, attr, None)
        if val is not None:
            return float(val)
    raise AttributeError(f"no elpd on {type(res).__name__}")


def _log_likelihood(idata):
    """The pointwise log-likelihood held by an InferenceData/DataTree."""
    node = idata["log_likelihood"]
    names = list(node.data_vars)
    if len(names) != 1:
        raise KeyError(f"expected exactly one log_likelihood group variable, got {names}")
    return np.asarray(node[names[0]].values)


def _waic_reference(log_likelihood):
    """(elpd_waic, p_waic, se) from the pointwise log-likelihood, no az.waic.

    ``az.waic`` was removed in ArviZ 1.x, so the reference is the standard
    estimator (Vehtari, Gelman & Gabry 2017) re-implemented here with scipy:

        lppd_i    = logsumexp_s(ll_si) - log(S)      (= log mean_s exp ll_si)
        p_waic_i  = var_s(ll_si)
        elpd_i    = lppd_i - p_waic_i
        elpd_waic = sum_i elpd_i,   p_waic = sum_i p_waic_i
        se        = sqrt(N * var_i(elpd_i))

    Both variances use ddof=0, the convention ArviZ itself had (xarray ``.var()``
    default) and the one BayesForge documents in ``Diag2._waic`` and
    ``jax_diagnostics.waic``. ddof=1 is a different statistic: it shifts p_waic
    by 1/(S-1), ~1e-3 relative at S=6000, far outside the 1e-6 tolerance.
    """
    ll = np.asarray(log_likelihood, dtype=np.float64)
    if ll.ndim == 3:
        n_chain, n_draw, n_obs = ll.shape
        ll = ll.reshape(n_chain * n_draw, n_obs)
    elif ll.ndim != 2:
        raise ValueError(f"expected a (S, N) or (C, S, N) log-likelihood, got {ll.shape}")
    n_samples, n_obs = ll.shape
    lppd_i = logsumexp(ll, axis=0) - np.log(n_samples)
    p_waic_i = np.var(ll, axis=0)
    elpd_i = lppd_i - p_waic_i
    return (float(np.sum(elpd_i)),
            float(np.sum(p_waic_i)),
            float(np.sqrt(n_obs * np.var(elpd_i))))


# ================== Parameter set parity ==================

def test_same_parameter_names(fitted):
    """Both classes discover the same posterior variables."""
    ref, wip, params = fitted
    assert set(ref.priors_name) == set(wip.priors_name)
    assert set(params) == {"a", "b", "s"}


# ================== Public API: R-hat and ESS (both go through ArviZ) ==================

def test_public_rhat_matches(fitted):
    """diagWIP.rhat() == diag.rhat() == az.rhat.

    Calls the METHODS. Calling az.rhat directly on both traces (the previous
    version) compared ArviZ to itself and asserted nothing about either class.
    """
    ref, wip, params = fitted
    rhat_ref = ref.rhat()
    rhat_wip = wip.rhat()
    expected = az.rhat(wip.trace)
    for p in params:
        assert _ds_val(rhat_ref, p) == pytest.approx(_ds_val(rhat_wip, p), rel=1e-6, abs=1e-6)
        assert _ds_val(rhat_wip, p) == pytest.approx(_ds_val(expected, p), rel=1e-6, abs=1e-6)


def test_public_ess_matches(fitted):
    """diagWIP.ess() == diag.ess() == az.ess (via the methods, not az directly)."""
    ref, wip, params = fitted
    ess_ref = ref.ess()
    ess_wip = wip.ess()
    expected = az.ess(wip.trace)
    for p in params:
        assert _ds_val(ess_ref, p) == pytest.approx(_ds_val(ess_wip, p), rel=1e-6, abs=1e-6)
        assert _ds_val(ess_wip, p) == pytest.approx(_ds_val(expected, p), rel=1e-6, abs=1e-6)


def test_metric_methods_are_reusable(fitted):
    """Calling a metric method twice must work.

    These methods used to assign their result over their own name
    (self.rhat = az.rhat(...)), so the second call raised
    "'Dataset' object is not callable".
    """
    ref, wip, params = fitted
    for obj in (ref, wip):
        for name in ("rhat", "ess"):
            getattr(obj, name)()
            getattr(obj, name)()      # must not raise
            assert callable(getattr(obj, name))


# ================== Manual (non-ArviZ) measures vs ArviZ reference ==================
# These are the "more complex" measures Diag2 computes by hand in JAX/NumPy.

def test_manual_rhat_matches_arviz_rank(fitted):
    """Diag2._rhat_manual == az.rhat(method='rank') (same estimator).

    Tolerance is rel=1e-6, not abs=1e-2. The bug this guards against (a missing
    folded-max half) produces a ~0.12 discrepancy, so the old tolerance was 12x
    too wide to see it -- and on a well-identified model like this fixture,
    rhat_bulk ~= rhat_tail ~= 1.00 anyway. test_manual_rhat_detects_variance_
    nonconvergence below is the case that actually discriminates.
    """
    ref, wip, params = fitted
    rhat_arviz = az.rhat(ref.trace, method="rank")
    for p in params:
        manual = wip._rhat_manual(_chains(wip, p))
        reference = _ds_val(rhat_arviz, p)
        assert manual == pytest.approx(reference, rel=1e-6), f"{p}: {manual} vs {reference}"


def test_manual_rhat_detects_variance_nonconvergence(fitted):
    """Chains differing only in VARIANCE must be flagged.

    Plain rank-normalized split R-hat (without the folded half) scores these
    ~1.00 and passes the 1.01 gate; the correct estimator scores them >1.05.
    """
    ref, wip, params = fitted
    rng = np.random.default_rng(0)
    bad = np.stack([rng.normal(0, 1 + 0.8 * c, 2000) for c in range(4)])
    assert wip._rhat_manual(bad) > 1.05
    idata = _az_from_dict({"posterior": {"p": bad}})
    assert wip._rhat_manual(bad) == pytest.approx(
        _ds_val(az.rhat(idata, method="rank"), "p"), rel=1e-6)


def test_manual_ess_bulk_matches_arviz(fitted):
    """Diag2._ess_bulk_manual ~= az.ess(method='bulk')."""
    ref, wip, params = fitted
    ess_bulk = az.ess(ref.trace, method="bulk")
    for p in params:
        manual = wip._ess_bulk_manual(_chains(wip, p))
        reference = _ds_val(ess_bulk, p)
        # Faithful port of arviz's estimator -> matches to floating-point noise.
        assert manual == pytest.approx(reference, rel=1e-3), f"{p}: {manual} vs {reference}"


def test_manual_ess_tail_matches_arviz(fitted):
    """Diag2._ess_tail_manual == az.ess(method='tail').

    After fixing _ess_raw_manual to use var_plus (within+between variance) and
    Geyer's initial-monotone sequence, the tail-ESS estimator matches arviz.
    """
    ref, wip, params = fitted
    # Diag2._ess_tail_manual uses prob=0.05 (Stan's 5%/95% tails); compare
    # against arviz with the same prob (arviz 1.x's *default* tail prob differs).
    ess_tail = az.ess(ref.trace, method="tail", prob=0.05)
    for p in params:
        manual = wip._ess_tail_manual(_chains(wip, p))
        reference = _ds_val(ess_tail, p)
        assert manual == pytest.approx(reference, rel=1e-3), f"{p}: {manual} vs {reference}"


def test_manual_ebfmi_matches_arviz_bfmi(fitted):
    """Diag2._ebfmi_manual ~= az.bfmi (per chain)."""
    ref, wip, params = fitted
    extra = wip.sampler.get_extra_fields(group_by_chain=True)
    if "energy" not in extra or extra["energy"] is None:
        pytest.skip("energy not collected in extra_fields; cannot check E-BFMI")
    energy = np.asarray(extra["energy"])          # (C, S)
    manual = np.asarray(wip._ebfmi_manual(energy))
    # Pass the energy array straight to az.bfmi (recent arviz returns a DataTree
    # from InferenceData input, which np.asarray can't consume).
    reference = np.asarray(az.bfmi(energy))       # (C,)
    assert manual.shape == reference.shape
    # Exact: _ebfmi_manual now uses mean(diff**2)/var(E, ddof=1), the Stan and
    # az.bfmi form. It previously used var(diff, ddof=1) in the numerator, and
    # this assertion carried rtol=0.1 to accommodate that -- documenting the
    # bug rather than failing on it.
    # rtol=1e-7 is float-accumulation headroom only (ArviZ sums in a different
    # order). It was 0.1, which existed to absorb the formula difference: the
    # numerator used var(diff, ddof=1) instead of mean(diff**2), a ~5e-4 bias.
    np.testing.assert_allclose(manual, reference, rtol=1e-7)


# ================== Summary table parity (mean / std / HDI) ==================

def _az_summary(idata, **kw):
    """az.summary across arviz majors (0.x takes hdi_prob=, 1.x takes ci_prob=)."""
    import inspect
    prob = kw.pop("prob")
    name = "ci_prob" if "ci_prob" in inspect.signature(az.summary).parameters else "hdi_prob"
    return az.summary(idata, **{name: prob}, **kw)


def test_summary_mean_sd_match(fitted):
    """Diag2.summary reproduces ArviZ's mean and sd exactly."""
    ref, wip, params = fitted

    # Diag.summary is just az.summary(kind="stats"). On this model az.summary
    # trips on the unnamed obs site ("x") during its internal re-conversion, so
    # build the same ArviZ summary on a clean InferenceData of the latents only
    # (identical samples) — that is the reference Diag2 must reproduce.
    post = {p: np.asarray(wip.posterior_samples[p]) for p in params}
    idata_ref = _az_from_dict({"posterior": post})   # 0.x keywords / 1.x mapping
    df_ref = _az_summary(idata_ref, kind="stats", prob=0.89, round_to=6)

    df_wip = wip.summary(round_to=6, hdi_prob=0.89)
    assert df_wip is not None, "summary() must return the table, not just store it"
    assert df_wip is wip.tab_summary

    for p in params:
        assert float(df_ref.loc[p, "mean"]) == pytest.approx(
            float(df_wip.loc[p, "mean"]), abs=1e-3)
        # Both ddof=1 now; the column is 'sd', matching ArviZ (it was 'std'
        # with ddof=0, and this assertion carried rel=1e-2 to absorb that).
        assert float(df_ref.loc[p, "sd"]) == pytest.approx(
            float(df_wip.loc[p, "sd"]), rel=1e-4)


def test_summary_uses_real_hdi_not_percentiles(fitted):
    """The hdi_* columns must be the narrowest interval, not equal-tailed."""
    ref, wip, params = fitted
    df = wip.summary(round_to=8, hdi_prob=0.89)
    for p in params:
        s = np.asarray(wip.posterior_samples[p]).flatten()
        lo, hi = float(df.loc[p, "hdi_89.0%_lower"]), float(df.loc[p, "hdi_89.0%_upper"])
        eq_lo, eq_hi = np.percentile(s, [5.5, 94.5])
        # An HDI is never wider than the equal-tailed interval at the same mass.
        assert (hi - lo) <= (eq_hi - eq_lo) + 1e-8


# ================== Information criteria (identical code paths) ==================

def test_loo_matches(fitted):
    """loo() is arviz-1.x compatible in both classes and gives equal elpd.

    ArviZ 1.x ELPDData exposes .elpd (not .elpd_loo); both classes build a
    NumPy-backed idata with log_likelihood from the same sampler.
    """
    ref, wip, params = fitted
    loo_ref = ref.loo()
    loo_wip = wip.loo()
    assert _elpd(loo_ref) == pytest.approx(_elpd(loo_wip), rel=1e-6, abs=1e-6)


def test_waic_matches(fitted):
    """WAIC() is native/arviz-free and matches both classes AND ArviZ's log-lik.

    az.waic was removed in arviz 1.x. WAIC is computed from the NumPyro pointwise
    log-likelihood (no arviz round-trip). We also verify it equals the WAIC one
    would get from arviz's *own* log_likelihood group — i.e. same result as
    arviz, just without depending on it: the reference is the standard estimator
    re-implemented in ``_waic_reference`` (formulas and ddof documented there)
    and applied to ``az.from_numpyro(..., log_likelihood=True)``, so a wrong
    BayesForge formula still fails.
    """
    ref, wip, params = fitted
    waic_ref = ref.WAIC()
    waic_wip = wip.WAIC()
    # cross-class parity
    assert float(waic_ref.elpd_waic) == pytest.approx(float(waic_wip.elpd_waic), rel=1e-6, abs=1e-6)
    assert float(waic_ref.p_waic) == pytest.approx(float(waic_wip.p_waic), rel=1e-6, abs=1e-6)

    # vs ARVIZ'S LOG-LIKELIHOOD. This used to compare _waic(ll_numpyro) against
    # _waic(ll_arviz) -- the same function on two nearly identical inputs, which
    # exercises the log-likelihood extraction and asserts nothing about the WAIC
    # formula. The SE bug (scaled by sqrt(|scale|) instead of |scale|) was
    # invisible to it.
    idata = az.from_numpyro(wip.sampler, log_likelihood=True)
    elpd_az, p_az, se_az = _waic_reference(_log_likelihood(idata))
    assert float(waic_wip.elpd_waic) == pytest.approx(elpd_az, rel=1e-6)
    assert float(waic_wip.p_waic) == pytest.approx(p_az, rel=1e-6)
    assert float(waic_wip.se) == pytest.approx(se_az, rel=1e-6)


@pytest.mark.parametrize("scale,mult", [("log", 1.0), ("negative_log", -1.0),
                                        ("deviance", -2.0)])
def test_waic_scale_transforms_elpd_and_se(fitted, scale, mult):
    """elpd scales by `mult`; the SE scales by |mult|, not sqrt(|mult|).

    The old sqrt form under-reported the deviance SE by a factor of 2
    (0.0793 where ArviZ gives 0.1112).
    """
    ref, wip, params = fitted
    base = wip.WAIC()
    got = wip.WAIC(scale=scale)
    assert float(got.elpd_waic) == pytest.approx(mult * float(base.elpd_waic), rel=1e-9)
    assert float(got.se) == pytest.approx(abs(mult) * float(base.se), rel=1e-9)


def test_loo_runs(fitted):
    """loo() must not raise.

    _numpy_backed_idata called az.from_dict POSITIONALLY, which collapsed every
    group into `posterior` under tuple names and left no log_likelihood group;
    az.loo then raised "log likelihood not found in inference data object".
    """
    ref, wip, params = fitted
    for obj in (ref, wip):
        res = obj.loo()
        val = getattr(res, "elpd_loo", None)
        if val is None:
            val = res.elpd
        assert np.isfinite(float(val))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
