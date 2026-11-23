"""ArviZ parity tests for BayesForge.Diagnostic.jax_diagnostics.

This is the module `main.py` actually uses: `m.summary()` calls
`jax_diagnostics.summary`, and `patch_diag.bind_diag_to_model` routes
`m.diag.rhat/ess/mcse/loo/WAIC` here. Before this file existed the only
accuracy test in the repo (`Test/Diag2 vs Diag/test_diag_parity.py`) tested
`Diag2`'s *manual* estimators and never touched this module at all, which is
how the following shipped:

  * r_hat was plain split R-hat, not the rank-normalized folded-max estimator
    the 1.01 threshold is calibrated for -> 10.75% under-report
  * mcse_sd used a normal approximation instead of ArviZ's delta-method form
  * mcse_mean divided by ess_bulk instead of ess_mean
  * filter_posterior_dict never matched keys carrying an index suffix
  * ess()/mcse() raised IndexError on the flat posteriors m.posteriors holds

ArviZ is the oracle throughout. Tolerances are tight on purpose: a tolerance
wide enough to absorb a wrong estimator is not a test.

The suite was written against ArviZ 0.x and is read here through the helpers in
the next section, which only adapt *how* a reference value is obtained: 1.x
(arviz-stats) renamed `hdi_prob=` to `prob=` (hdi) and `ci_prob=` (summary),
returns an `xarray.DataTree` where it used to return a Dataset, renamed the
summary HDI columns, dropped `az.waic`, and changed the default tail-ESS
probability. No tolerance, expectation or comparison was changed by that port.
"""
import numpy as np
import pytest

# BayesForge before arviz: its __init__ patches jax.interpreters.pxla, and
# arviz pulling jax in first can leave that submodule in a state that breaks
# the patch.
from BayesForge.Diagnostic import jax_diagnostics as jd
from BayesForge.Diagnostic.Diag2 import _az_from_dict

import arviz as az
from scipy.special import logsumexp


# ================== ArviZ 1.x reference-reading helpers ==================

def _val(res, param):
    """Values of `param` from an ArviZ rhat/ess/mcse result.

    ArviZ 1.x returns an ``xarray.DataTree`` rooted at the group it computed, so
    the variable sits at ``res[param]``; a plain dict of arrays still yields an
    ``xarray.Dataset``. Builds that keep the group as a child node are covered
    by the ``res["posterior"][param]`` fallback.
    """
    try:
        obj = res[param]
    except (KeyError, TypeError):
        obj = res["posterior"][param]
    return np.asarray(obj.values)


def _scalar(res, param):
    """Same, reduced to the scalar for a scalar parameter."""
    return float(_val(res, param))


def _az_hdi(x, prob):
    """az.hdi on both majors: ``hdi_prob=`` became ``prob=`` in arviz-stats 1.0."""
    import inspect
    key = "prob" if "prob" in inspect.signature(az.hdi).parameters else "hdi_prob"
    return az.hdi(x, **{key: prob})


def _elpd(res):
    """ELPD from an ELPDData across majors (0.x: ``elpd_loo``, 1.x: ``elpd``)."""
    for attr in ("elpd_loo", "elpd"):
        val = getattr(res, attr, None)
        if val is not None:
            return float(val)
    raise AttributeError(f"no elpd on {type(res).__name__}")


def _p_loo(res):
    """p_loo from an ELPDData across majors (0.x: ``p_loo``, 1.x: ``p``)."""
    for attr in ("p_loo", "p"):
        val = getattr(res, attr, None)
        if val is not None:
            return float(val)
    raise AttributeError(f"no p_loo on {type(res).__name__}")


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
    estimator (Vehtari, Gelman & Gabry 2017) applied here with scipy:

        lppd_i    = logsumexp_s(ll_si) - log(S)      (= log mean_s exp ll_si)
        p_waic_i  = var_s(ll_si)
        elpd_i    = lppd_i - p_waic_i
        elpd_waic = sum_i elpd_i,   p_waic = sum_i p_waic_i
        se        = sqrt(N * var_i(elpd_i))

    Both variances use ddof=0, the convention ArviZ itself had (xarray ``.var()``
    default) and the one BayesForge documents in ``Diag2._waic`` and
    ``jax_diagnostics.waic``. ddof=1 would shift p_waic by 1/(S-1) -- ~1e-3
    relative at S=8000, i.e. far outside the 1e-6 tolerance -- so it is a
    different statistic, not a rounding choice.
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


def _az_summary_ref(idata, prob=0.89, round_to=8):
    """az.summary expressed in ``jd.summary``'s column names.

    Three ArviZ 1.x changes are undone; none of them changes the statistic:

      * ``hdi_prob=`` is now ``ci_prob=``, and the interval kind must be asked
        for (``ci_kind="hdi"`` -- 1.x defaults to equal-tailed),
      * the interval columns are ``hdi89_lb``/``hdi89_ub`` instead of
        ``hdi_5.5%``/``hdi_94.5%``,
      * ``ess_tail`` is reported at ArviZ's own tail probability
        (``1 - ci_prob`` = 0.11) while BayesForge uses the 5%/95% tails of
        Vehtari et al. (2021), so it is taken from
        ``az.ess(method="tail", prob=0.05)`` -- the same estimator, the prob
        ArviZ 0.x used for it.
    """
    import inspect
    params = inspect.signature(az.summary).parameters
    kw = {"round_to": round_to}
    if "ci_prob" in params:
        kw["ci_prob"] = prob
        if "ci_kind" in params:
            kw["ci_kind"] = "hdi"
    else:                                       # arviz 0.x
        kw["hdi_prob"] = prob
    ref = az.summary(idata, **kw)

    lo_pct = (1 - prob) / 2 * 100               # 5.5 for prob=0.89
    hi_pct = 100 - lo_pct                       # 94.5
    pct = int(round(prob * 100))                # 89
    ref = ref.rename(columns={f"hdi{pct}_lb": f"hdi_{lo_pct:.1f}%",
                              f"hdi{pct}_ub": f"hdi_{hi_pct:.1f}%"})
    tail = az.ess(idata, method="tail", prob=0.05)
    ref["ess_tail"] = [_scalar(tail, name) for name in ref.index]
    return ref


# ============================== fixtures ==============================

@pytest.fixture(scope="module")
def chains():
    """Four chain sets, each stressing a different failure mode.

    Returns (dict_of_arrays, InferenceData) over the same draws.
    """
    rng = np.random.default_rng(0)

    def ar1(rho, n, sd=1.0):
        x = np.zeros(n)
        e = rng.normal(0, sd, n)
        for i in range(1, n):
            x[i] = rho * x[i - 1] + e[i]
        return x

    C, S = 4, 2000
    post = {
        # well mixed
        "well":  np.stack([ar1(0.6, S) for _ in range(C)]),
        # chain MEANS disagree -> caught by plain split R-hat too
        "bad":   np.stack([ar1(0.9, S) + 0.4 * c for c in range(C)]),
        # chain VARIANCES disagree -> ONLY the folded half of rank-normalized
        # R-hat sees this. Plain split R-hat reports 1.0006 where ArviZ says
        # 1.1211, i.e. "converged" for a model that is not.
        "scale": np.stack([ar1(0.5, S, sd=1 + 0.8 * c) for c in range(C)]),
        # heavy right skew -> separates rank-normalized from raw estimators
        "skew":  np.stack([np.exp(ar1(0.7, S)) for _ in range(C)]),
    }
    # _az_from_dict is BayesForge's shim over the from_dict signature change:
    # 0.x takes one keyword per group, 1.x takes the whole mapping positionally
    # and rejects the group keywords.
    return post, _az_from_dict({"posterior": post})


PARAMS = ["well", "bad", "scale", "skew"]


# ============================== R-hat ==============================

@pytest.mark.parametrize("param", PARAMS)
def test_rhat_matches_arviz_rank(chains, param):
    post, idata = chains
    expected = _scalar(az.rhat(idata, method="rank"), param)
    assert float(jd._rhat_1d(post[param])) == pytest.approx(expected, rel=1e-6)


def test_rhat_detects_variance_nonconvergence(chains):
    """Regression guard for the estimator swap, stated as a behaviour.

    The `scale` chains differ only in variance. Plain split R-hat scores them
    ~1.0006 (below the 1.01 threshold, i.e. a silent false negative); the
    correct estimator scores them well above it.
    """
    post, _ = chains
    assert float(jd._rhat_1d(post["scale"])) > 1.05


def test_rhat_dict_api(chains):
    post, idata = chains
    got = jd.rhat(post)
    rhat_az = az.rhat(idata, method="rank")
    for p in PARAMS:
        assert got[p] == pytest.approx(_scalar(rhat_az, p), rel=1e-6)


# ============================== ESS ==============================

@pytest.mark.parametrize("param", PARAMS)
@pytest.mark.parametrize("kind", ["bulk", "tail"])
def test_ess_matches_arviz(chains, param, kind):
    post, idata = chains
    fn = jd._ess_1d if kind == "bulk" else jd._ess_tail_1d
    # BayesForge's tail ESS uses the 5%/95% tails of Vehtari et al. (2021);
    # ArviZ 1.x defaults to prob = 1 - ci_prob = 0.11, so ask for 0.05 and keep
    # both sides on the same estimator. bulk has no prob argument.
    kw = {"prob": 0.05} if kind == "tail" else {}
    expected = _scalar(az.ess(idata, method=kind, **kw), param)
    assert fn(post[param]) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("param", PARAMS)
@pytest.mark.parametrize("kind", ["mean", "sd"])
def test_ess_mean_and_sd_match_arviz(chains, param, kind):
    post, idata = chains
    fn = jd._ess_mean_1d if kind == "mean" else jd._ess_sd_1d
    expected = _scalar(az.ess(idata, method=kind), param)
    assert fn(post[param]) == pytest.approx(expected, rel=1e-6)


# ============================== summary table ==============================

SUMMARY_COLS = ["mean", "sd", "mcse_mean", "mcse_sd",
                "ess_bulk", "ess_tail", "r_hat", "hdi_5.5%", "hdi_94.5%"]


@pytest.fixture(scope="module")
def az_summary(chains):
    """The ArviZ reference table, in ``jd.summary``'s column names."""
    _, idata = chains
    return _az_summary_ref(idata)


@pytest.mark.parametrize("col", SUMMARY_COLS)
@pytest.mark.parametrize("param", PARAMS)
def test_summary_column_matches_arviz(chains, az_summary, col, param):
    post, _ = chains
    got = jd.summary(post, round_to=8, hdi_prob=0.89, group_by_chain=True)
    assert float(got.loc[param, col]) == pytest.approx(
        float(az_summary.loc[param, col]), rel=1e-6)


def test_summary_reports_rhat_for_a_single_chain():
    """RELEASE_AUDIT bug 3: NaN like ArviZ (owner decision 2026-11-13).

    A single chain only compares its own two halves, so R-hat is NaN and the
    user is warned; it is no longer reported as a finite ~1.0 value.
    """
    x = np.random.default_rng(4).normal(size=(1, 2000))
    with pytest.warns(UserWarning, match="single chain"):
        df = jd.summary({"a": x}, group_by_chain=True, round_to=8)
    assert np.isnan(float(df.loc["a", "r_hat"]))


# ============================== HDI ==============================

@pytest.mark.parametrize("n,p", [(1000, 0.89), (997, 0.94), (333, 0.5),
                                 (500, 0.95), (1234, 0.8)])
def test_hdi_matches_arviz(n, p):
    """Fails if hdi() rounds the interval size with ceil instead of floor."""
    x = np.random.default_rng(1).normal(size=n)
    np.testing.assert_allclose(np.asarray(jd.hdi(x, hdi_prob=p)),
                               np.asarray(_az_hdi(x, p)), rtol=1e-12)


def test_hdi_returns_double_precision():
    x = np.random.default_rng(1).normal(size=500)
    assert np.asarray(jd.hdi(x, 0.89)).dtype == np.float64


# ============================== LOO / WAIC ==============================

@pytest.fixture(scope="module")
def loglik(chains):
    post, _ = chains
    rng = np.random.default_rng(2)
    ll = rng.normal(-2.0, 0.6, size=(4, 2000, 60))
    idata = _az_from_dict({"posterior": {"z": post["well"]},
                           "log_likelihood": {"y": ll}})
    return ll, idata


def test_waic_matches_arviz(loglik):
    """jd.waic against the WAIC formulas, computed here from ArviZ's log-lik.

    ``az.waic`` was removed in ArviZ 1.x (LOO is the recommended IC), so the
    reference is no longer an ArviZ call. It is the standard estimator
    (Vehtari, Gelman & Gabry 2017) re-implemented in ``_waic_reference`` and
    applied to the pointwise log-likelihood from the ``log_likelihood`` group
    of the InferenceData -- the same input az.waic used, so a wrong BayesForge
    formula still fails. See that helper for the formulas and the ddof choice.
    """
    ll, idata = loglik
    got = jd.waic(ll)
    elpd, p, se = _waic_reference(_log_likelihood(idata))
    assert got.elpd == pytest.approx(elpd, rel=1e-6)
    assert got.p == pytest.approx(p, rel=1e-6)
    assert got.se == pytest.approx(se, rel=1e-6)


def test_loo_matches_arviz(loglik):
    ll, idata = loglik
    got, ref = jd.loo(ll, pointwise=True), az.loo(idata, pointwise=True)
    assert got.elpd == pytest.approx(_elpd(ref), rel=1e-4)
    assert got.p == pytest.approx(_p_loo(ref), rel=1e-3)


def test_loo_pareto_k_matches_arviz_when_reff_is_supplied(chains, loglik):
    """PSIS sizes its tail as 3*sqrt(S/reff).

    reff comes from the POSTERIOR (that is what az.loo does), not from the
    log-likelihood. Hardcoding reff=1 left pareto_k off by up to 0.13 -- enough
    to move observations across the 0.7 threshold that
    sensitivity.influence_plot colours on.

    NOTE (open, not tolerated): ArviZ 1.x sizes the tail as
    ``floor(3*sqrt(S/reff))`` (arviz_stats ``_get_ps_tails``), while
    ``jax_diagnostics._psis_weights`` still uses the ArviZ 0.x rule
    ``ceil(min(S/5, 3*sqrt(S/reff)))``. One extra tail draw moves k_hat by up to
    ~1e-2, so this comparison fails on ArviZ 1.3 by ~9e-3 (max abs). The GPD fit
    and the cutoff convention are identical -- verified by feeding both tail
    sizes through arviz's own ``_gpdfit``. The tolerance is left as it is: the
    two implementations disagree, they do not merely round differently.
    """
    post, _ = chains
    ll, idata = loglik
    reff = jd.relative_eff({"z": post["well"]})
    got = jd.loo(ll, pointwise=True, reff=reff)
    ref = az.loo(idata, pointwise=True, reff=reff)
    np.testing.assert_allclose(np.asarray(got.pareto_k),
                               np.asarray(ref.pareto_k.values).ravel(), atol=1e-6)


def test_relative_eff_is_a_fraction(chains):
    post, _ = chains
    reff = jd.relative_eff(post)
    assert 0.0 < reff <= 1.5


@pytest.mark.parametrize("scale,mult", [("log", 1.0), ("negative_log", -1.0),
                                        ("deviance", -2.0)])
def test_waic_scales(loglik, scale, mult):
    ll, _ = loglik
    assert jd.waic(ll, scale=scale).elpd == pytest.approx(
        mult * jd.waic(ll).elpd, rel=1e-9)


# ============================== compare ==============================

def test_compare_elpd_diff_sign_matches_arviz():
    """elpd_diff must be 0 for the best model and POSITIVE for worse ones."""
    rng = np.random.default_rng(5)
    good = rng.normal(-1.0, 0.4, size=(2, 500, 40))
    bad = rng.normal(-3.0, 0.4, size=(2, 500, 40))
    df = jd.compare({"good": good, "bad": bad}, ic="waic")
    assert df.loc["good", "rank"] == 0
    assert df.loc["good", "elpd_diff"] == pytest.approx(0.0, abs=1e-9)
    assert df.loc["bad", "elpd_diff"] > 0


def test_compare_rejects_mismatched_observation_counts():
    rng = np.random.default_rng(6)
    a = rng.normal(size=(2, 200, 40))
    b = rng.normal(size=(2, 200, 30))
    with pytest.raises(ValueError, match="same observations"):
        jd.compare({"a": a, "b": b}, ic="waic")


# ============================== filtering / shapes ==============================

def test_filter_posterior_dict_matches_base_names():
    d = {"mu[0]": np.zeros(3), "mu[1]": np.zeros(3), "sigma": np.zeros(3)}
    assert set(jd.filter_posterior_dict(d, include="mu")) == {"mu[0]", "mu[1]"}
    assert set(jd.filter_posterior_dict(d, exclude="mu")) == {"sigma"}


def test_filter_posterior_dict_does_not_match_prefixes():
    d = {"var": np.zeros(3), "var_1": np.zeros(3)}
    assert set(jd.filter_posterior_dict(d, include="var")) == {"var"}


def test_ess_and_mcse_accept_flat_posteriors():
    """m.posteriors is flat (N,); these used to raise IndexError."""
    flat = {"a": np.random.default_rng(3).normal(size=2000)}
    assert np.isfinite(jd.ess(flat)["a"])
    assert np.isfinite(jd.mcse(flat)["a"])


def test_ess_and_mcse_accept_flat_vector_posteriors():
    """Flat (N, K) draws of a K-vector, declared via group_by_chain=False."""
    flat = {"b": np.random.default_rng(3).normal(size=(2000, 3))}
    assert jd.ess(flat, group_by_chain=False)["b"].shape == (3,)
    assert jd.mcse(flat, group_by_chain=False)["b"].shape == (3,)


@pytest.mark.parametrize("kind", ["mean", "sd"])
def test_mcse_matches_arviz(chains, kind):
    post, idata = chains
    got = jd.mcse(post, kind=kind)
    ref = az.mcse(idata, method=kind)
    for p in PARAMS:
        assert got[p] == pytest.approx(_scalar(ref, p), rel=1e-6)


def test_multidim_parameters_are_expanded(chains):
    post, _ = chains
    vec = {"v": np.stack([post["well"], post["bad"]], axis=-1)}  # (C, S, 2)
    assert jd.rhat(vec)["v"].shape == (2,)
    assert jd.ess(vec)["v"].shape == (2,)
    df = jd.summary(vec, group_by_chain=True)
    assert list(df.index) == ["v[0]", "v[1]"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
