"""Regression tests for the minor Diagnostic bugs in RELEASE_AUDIT.md.
Covers bugs 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21, 25, 26.
"""
import inspect
import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
import pytest
import BayesForge.Diagnostic as diagpkg
from BayesForge.Diagnostic import Diag2, jax_diagnostics as jd, patch_diag, sensitivity
from BayesForge.Diagnostic.Diag import diag
from BayesForge.Diagnostic.Diag2 import _DictPosterior, diagWIP


class _FakeTfpSampler:
    """Minimal stand-in for a fitted TFP sampler (canonical fixture)."""

    def __init__(self):
        self.model_info = {"mu": {}, "sigma": {}}
        self.posterior = [jnp.zeros((1, 6)), jnp.ones((1, 6))]
        self.sample_stats = [
            jnp.full((1, 6), -10.0),        # target_log_prob
            jnp.full((1, 6), 31.0),         # leapfrogs_taken
            jnp.zeros((1, 6), dtype=bool),  # has_divergence
            jnp.full((1, 6), 12.0),         # energy
            jnp.full((1, 6), -0.25),        # log_accept_ratio
        ]
        self.num_chains = 1

    def get_samples(self, group_by_chain=True):
        return dict(zip(self.model_info, self.posterior))


class _FakeTfpSampler2Chains(_FakeTfpSampler):
    """Same, but two chains so the chain axis is exercised."""

    def __init__(self):
        super().__init__()
        self.posterior = [jnp.zeros((2, 6)), jnp.ones((2, 6))]
        self.sample_stats = [s.repeat(2, axis=0) for s in self.sample_stats]
        self.num_chains = 2


class _FakeTfpSamplerVarMismatch(_FakeTfpSampler):
    """model_info lists 3 sites but the trace only has 2 posterior sites."""

    def __init__(self):
        super().__init__()
        self.model_info = {"mu": {}, "sigma": {}, "tau": {}}


def _make_diag(posteriors):
    """diagWIP over a plain {name: (chain, draw)} dict (bug 25 input)."""
    return diagWIP(posteriors)


def _make_sampler(kernel, extra):
    """Fake fitted sampler exposing a kernel and extra_fields."""

    class _S:
        def __init__(self):
            self.sampler = kernel

        def get_samples(self, group_by_chain=True):
            return {"mu": jnp.zeros((2, 100))}

        def get_extra_fields(self, group_by_chain=True):
            return self._extra

    s = _S()
    s._extra = extra
    return s


def _nuts_kernel():
    from numpyro.infer import NUTS
    return NUTS(potential_fn=lambda x: 0.5 * jnp.sum(x ** 2))


def _hmc_kernel():
    from numpyro.infer import HMC
    return HMC(potential_fn=lambda x: 0.5 * jnp.sum(x ** 2))


# ---------------------------------------------------------------------------
# Bug 20 — filter_posterior_dict: include/exclude semantics, no mutation
# ---------------------------------------------------------------------------
def test_filter_posterior_dict_include():
    d = {"mu": np.zeros(3), "sigma": np.ones(3), "mu_tau": np.ones(3)}
    out = jd.filter_posterior_dict(d, include=["mu"])
    assert set(out) == {"mu"}
    assert set(d) == {"mu", "sigma", "mu_tau"}  # input untouched


def test_filter_posterior_dict_exclude():
    d = {"mu": np.zeros(3), "sigma": np.ones(3)}
    out = jd.filter_posterior_dict(d, exclude=["sigma"])
    assert set(out) == {"mu"}


def test_filter_posterior_dict_unknown_name_returns_empty():
    # Unknown names filter to nothing (no raise, no mutation).
    d = {"mu": np.zeros(3)}
    out = jd.filter_posterior_dict(d, include=["nope"])
    assert out == {}
    assert set(d) == {"mu"}


# ---------------------------------------------------------------------------
# Bug 8 — ess(kind="tail", prob=...)
# ---------------------------------------------------------------------------
def _ess_data():
    rng = np.random.RandomState(7)
    return rng.randn(4, 400)


def test_ess_tail_prob_changes_result():
    arr = _ess_data()
    default = jd.ess({"mu": arr}, kind="tail")["mu"]
    wider = jd.ess({"mu": arr}, kind="tail", prob=(0.1, 0.9))["mu"]
    assert default != pytest.approx(wider)


def test_ess_tail_default_prob_matches_explicit():
    arr = _ess_data()
    a = jd.ess({"mu": arr}, kind="tail")["mu"]
    b = jd.ess({"mu": arr}, kind="tail", prob=(0.05, 0.95))["mu"]
    assert a == pytest.approx(b)


def test_ess_prob_ignored_for_other_kinds():
    arr = _ess_data()
    a = jd.ess({"mu": arr}, kind="bulk")["mu"]
    b = jd.ess({"mu": arr}, kind="bulk", prob=(0.1, 0.9))["mu"]
    assert a == pytest.approx(b)


def test_ess_tail_1d_accepts_prob():
    arr = _ess_data()
    v1 = jd._ess_tail_1d(arr)
    v2 = jd._ess_tail_1d(arr, prob=(0.25, 0.75))
    assert v1 != pytest.approx(v2)


# ---------------------------------------------------------------------------
# Bug 9 — loo(log_likelihood array, reff=None) == reff=1.0; relative_eff(dict)
# ---------------------------------------------------------------------------
def _loo_array():
    rng = np.random.RandomState(11)
    return jnp.array(rng.standard_normal((2, 100, 50)))


def test_loo_default_reff_equals_one():
    ll = _loo_array()
    out_default = jd.loo(ll, pointwise=True)
    out_one = jd.loo(ll, reff=1.0, pointwise=True)
    assert out_default.elpd == pytest.approx(out_one.elpd)
    assert np.allclose(np.asarray(out_default.pointwise_elpd),
                       np.asarray(out_one.pointwise_elpd))


def test_relative_eff_returns_finite_float():
    ll = _loo_array()
    reff = jd.relative_eff({"mu": ll})
    assert np.isscalar(float(reff))
    assert np.isfinite(reff)
    assert reff > 0


def test_loo_accepts_relative_eff_output():
    ll = _loo_array()
    out_default = jd.loo(ll)
    out_reff = jd.loo(ll, reff=jd.relative_eff({"mu": ll}))
    # iid log-lik: reff is close to 1, so the elpd stays in the same ballpark.
    assert out_reff.elpd == pytest.approx(out_default.elpd, rel=0.1)


# ---------------------------------------------------------------------------
# Bug 25 — diagWIP(dict), chain dimension, num_chains cross-check
# ---------------------------------------------------------------------------
def test_dict_posterior_num_chains_from_leading_dim():
    dp = _DictPosterior({"mu": np.zeros((3, 100))})
    assert dp.num_chains == 3


def test_dict_posterior_rejects_bad_input():
    with pytest.raises(ValueError):
        _DictPosterior("not a dict")
    with pytest.raises(ValueError):
        _DictPosterior({})
    with pytest.raises(ValueError):
        _DictPosterior({"mu": np.zeros(())})


def test_diagwip_dict_infers_num_chains():
    d = _make_diag({"mu": np.random.RandomState(0).randn(3, 100)})
    assert d.num_chains == 3


def test_diagwip_rejects_sampler_chain_mismatch():
    class _S:
        num_chains = 4  # lies about the leading dimension

        def get_samples(self, group_by_chain=True):
            return {"mu": np.zeros((2, 100))}

    with pytest.raises(ValueError, match="num_chains"):
        diagWIP(_S())


def test_to_az_dict_posterior_smoke():
    d = _make_diag({"mu": np.random.RandomState(1).randn(2, 100)})
    idata = d.to_az()
    assert "mu" in idata.posterior


def test_to_az_preserves_chain_dimension():
    d = _make_diag({"mu": np.random.RandomState(2).randn(3, 100)})
    idata = d.to_az()
    assert idata.posterior.sizes["chain"] == 3


def test_to_az_tfp_smoke():
    d = diagWIP(_FakeTfpSampler())
    idata = d.to_az(backend="tfp")
    assert "mu" in idata.posterior


def test_to_az_tfp_preserves_chain_dimension():
    d = diagWIP(_FakeTfpSampler2Chains())
    idata = d.to_az(backend="tfp")
    assert idata.posterior.sizes["chain"] == 2


# ---------------------------------------------------------------------------
# Bug 12 — to_az(backend=) validation
# ---------------------------------------------------------------------------
def test_to_az_unknown_backend_raises():
    d = _make_diag({"mu": np.zeros((2, 50))})
    with pytest.raises(ValueError, match="backend"):
        d.to_az(backend="stan")


def test_to_az_tfp_rejects_short_sample_stats_name():
    d = diagWIP(_FakeTfpSampler())
    with pytest.raises(ValueError, match="sample_stats_name"):
        d.to_az(backend="tfp", sample_stats_name=["energy"])


def test_to_az_tfp_rejects_var_posterior_mismatch():
    # var_names comes from sampler.model_info; 3 sites vs 2 trace sites.
    d = diagWIP(_FakeTfpSamplerVarMismatch())
    with pytest.raises(ValueError, match="var_names"):
        d.to_az(backend="tfp")


def test_to_az_tfp_accepts_matching_stat_names():
    d = diagWIP(_FakeTfpSampler())
    idata = d.to_az(
        backend="tfp",
        sample_stats_name=["target_log_prob", "leapfrogs_taken",
                           "has_divergence", "energy", "log_accept_ratio"],
    )
    assert "target_log_prob" in idata.sample_stats


# ---------------------------------------------------------------------------
# Bugs 15/16 — canonical _acf / _ebfmi_per_chain shared by all modules
# ---------------------------------------------------------------------------
def test_acf_constant_input_is_delta():
    out = jd._acf(np.full(100, 3.0), max_lag=10)
    assert isinstance(out, list)
    assert out[0] == pytest.approx(1.0)
    assert all(v == pytest.approx(0.0) for v in out[1:])


def test_acf_matches_manual_formula():
    x = np.random.RandomState(5).randn(200)
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    n = len(x)
    expected = [float(np.dot(xc[: n - k], xc[k:]) / denom) for k in range(5)]
    got = jd._acf(x, max_lag=5)
    assert len(got) == 5
    assert got == pytest.approx(expected)


def test_ebfmi_per_chain_iid_is_about_two():
    rng = np.random.RandomState(9)
    out = jd._ebfmi_per_chain(rng.randn(2, 500))
    assert np.all(np.asarray(out) > 1.3)  # no >1.3 clipping anywhere


def test_ebfmi_per_chain_zero_variance_is_nan():
    out = jd._ebfmi_per_chain(np.full((2, 50), 3.0))
    assert np.all(np.isnan(np.asarray(out)))


def test_ebfmi_per_chain_is_per_chain_not_pooled():
    rng = np.random.RandomState(3)

    def ar1(n, rho=0.95):
        x = np.zeros(n)
        e = rng.randn(n) * np.sqrt(1.0 - rho ** 2)
        for t in range(1, n):
            x[t] = rho * x[t - 1] + e[t]
        return x

    # Chain 0 iid (E-BFMI ~ 2), chain 1 strongly autocorrelated (~ 0.1).
    energy = np.vstack([rng.randn(500), ar1(500)])
    per_chain = np.asarray(jd._ebfmi_per_chain(energy))
    pooled = np.asarray(jd._ebfmi_per_chain(energy.reshape(1, -1)))
    assert per_chain[0] == pytest.approx(2.0, rel=0.15)
    assert per_chain[1] < 0.5
    # A pooled estimator would report one blended value, not the iid 2.0.
    assert pooled[0] < per_chain[0] - 0.5


def test_diag2_ebfmi_manual_delegates_to_canonical():
    rng = np.random.RandomState(4)
    energy = rng.randn(2, 300)
    d = _make_diag({"mu": rng.randn(2, 300)})
    manual = list(d._ebfmi_manual(energy))
    canonical = list(jd._ebfmi_per_chain(energy))
    assert manual == pytest.approx(canonical)


def test_patch_diag_autocor_uses_canonical_acf(monkeypatch):
    calls = []
    real = jd._acf

    def spy(arr, max_lag=40):
        calls.append(arr)
        return real(arr, max_lag)

    monkeypatch.setattr(patch_diag, "_acf", spy)
    d = _make_diag({"mu": np.random.RandomState(6).randn(2, 200)})
    fig = patch_diag._plot_autocor(d, None, None, None, 40, True)
    assert len(calls) == 2  # once per chain
    assert isinstance(fig, go.Figure)


def test_diag2_autocor_uses_canonical_acf(monkeypatch):
    calls = []
    real = jd._acf

    def spy(arr, max_lag=40):
        calls.append(arr)
        return real(arr, max_lag)

    monkeypatch.setattr(Diag2, "_acf", spy)
    d = _make_diag({"mu": np.random.RandomState(6).randn(2, 200)})
    fig = d.autocor()
    assert len(calls) == 2  # once per chain
    assert isinstance(fig, go.Figure)


def test_sensitivity_energy_plot_uses_canonical_ebfmi(monkeypatch):
    calls = []
    real = jd._ebfmi_per_chain

    def spy(energy):
        calls.append(energy)
        return real(energy)

    monkeypatch.setattr(jd, "_ebfmi_per_chain", spy)

    from types import SimpleNamespace

    rng = np.random.RandomState(8)

    class _S:
        def get_extra_fields(self, group_by_chain=True):
            return {
                "energy": rng.randn(2, 100),
                "diverging": np.zeros((2, 100), dtype=bool),
            }

    m = SimpleNamespace(sampler=_S())
    fig = sensitivity.divergence_energy_plot(m)
    assert len(calls) == 2  # once per chain
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Bug 17 — pair-plot subsample seed is a parameter
# ---------------------------------------------------------------------------
def _pair_model(n=1500):
    """Model-like fake for patch_diag._plot_pair: flat (pooled) posteriors."""
    from types import SimpleNamespace
    rng = np.random.RandomState(13)
    return SimpleNamespace(posteriors={
        "a": rng.randn(2 * n), "b": rng.randn(2 * n) + 1.0})


def _pair_diag(n=50):
    """diagWIP over a dict fake for the Diag2.pair smoke test."""
    rng = np.random.RandomState(13)
    return _make_diag({"a": rng.randn(2, n), "b": rng.randn(2, n) + 1.0})


def test_pair_seed_is_a_parameter():
    for fn in (patch_diag._plot_pair, diagWIP.pair):
        assert "seed" in inspect.signature(fn).parameters
    # pair_jax is installed by the decorator; it must forward the seed.
    src = inspect.getsource(patch_diag.patch_diag_class)
    assert "def pair_jax" in src and "seed=seed" in src


def _marker_xs(fig):
    # the scatter subsample (mode='markers', many points); the median marker
    # is a single-point markers trace and is excluded by the size filter.
    return [np.asarray(t.x) for t in fig.data
            if getattr(t, "mode", None) == "markers"
            and np.asarray(t.x).size > 1]


def test_plot_pair_same_seed_same_subsample():
    m = _pair_model()
    f1 = patch_diag._plot_pair(m, max_points=500, seed=42)
    f2 = patch_diag._plot_pair(m, max_points=500, seed=42)
    xs1, xs2 = _marker_xs(f1), _marker_xs(f2)
    assert len(xs1) == len(xs2) > 0
    for a, b in zip(xs1, xs2):
        assert len(a) == 500
        assert np.array_equal(a, b)


def test_plot_pair_different_seed_different_subsample():
    m = _pair_model()
    f1 = patch_diag._plot_pair(m, max_points=500, seed=42)
    f2 = patch_diag._plot_pair(m, max_points=500, seed=7)
    xs1, xs2 = _marker_xs(f1), _marker_xs(f2)
    assert not all(np.array_equal(a, b) for a, b in zip(xs1, xs2))


def test_pair_smoke_returns_figure():
    d = _pair_diag(n=50)
    assert isinstance(d.pair(), go.Figure)


def test_pair_jax_smoke_returns_figure():
    @patch_diag.patch_diag_class
    class _Patched:
        pass

    # the decorator installs pair_jax under the public name `pair`
    fig = _Patched().pair(_pair_model(n=50), max_points=40)
    assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Bug 7 — treedepth section only for NUTS kernels
# ---------------------------------------------------------------------------
def _diag_with_sampler(kernel, extra):
    return diagWIP(_make_sampler(kernel, extra))


def test_treedepth_reported_for_nuts():
    d = _diag_with_sampler(_nuts_kernel(),
                           {"num_steps": jnp.full((2, 100), 3.0)})
    report = d.diagnose()
    assert "Checking sampler transitions treedepth." in report
    assert "$num_max_treedepth" in report
    assert "Treedepth satisfactory for all transitions." in report


def test_treedepth_saturation_reported_for_nuts():
    d = _diag_with_sampler(_nuts_kernel(),
                           {"num_steps": jnp.full((2, 100), 1023.0)})
    report = d.diagnose()
    assert "maximum treedepth limit of 10" in report


def test_treedepth_skipped_for_plain_hmc():
    d = _diag_with_sampler(_hmc_kernel(),
                           {"num_steps": jnp.full((2, 100), 1023.0)})
    report = d.diagnose()
    assert "skipped (not a NUTS kernel)" in report
    assert "maximum treedepth limit" not in report


def test_treedepth_unavailable_without_extra_fields():
    d = _diag_with_sampler(_nuts_kernel(), {})
    report = d.diagnose()
    assert "Treedepth: extra_fields not collected (unavailable)." in report


# ---------------------------------------------------------------------------
# Bug 19 — diagnose(var_names=) filters the ESS / R-hat sections
# ---------------------------------------------------------------------------
def _diag_two_vars():
    rng = np.random.RandomState(19)
    return _make_diag({"mu": rng.randn(2, 100),
                       "sigma": np.abs(rng.randn(2, 100))})


def test_diagnose_var_names_filters_report():
    d = _diag_two_vars()
    report = d.diagnose(var_names=["mu"])
    assert "mu" in report
    assert "sigma" not in report


def test_diagnose_without_var_names_reports_all():
    report = _diag_two_vars().diagnose()
    assert "mu" in report and "sigma" in report


def test_diagnose_var_names_unknown_raises():
    d = _diag_two_vars()
    with pytest.raises(ValueError, match="nope"):
        d.diagnose(var_names=["nope"])


# ---------------------------------------------------------------------------
# Bugs 18, 21, 26 — documentation
# ---------------------------------------------------------------------------
def test_package_docstring_names_public_api():
    doc = (diagpkg.__doc__ or "").lower()
    for name in ("diagwip", "patch_diag", "summary", "ess", "rhat", "loo"):
        assert name in doc


def test_legacy_diag_class_marked_deprecated():
    doc = diag.__doc__ or ""
    assert "deprecated" in doc.lower()
    assert "diagWIP" in doc


def test_diag2_module_marks_patch_deprecated():
    doc = Diag2.__doc__ or ""
    assert "deprecated" in doc.lower()
    assert "patch_diag" in doc
