"""Regression tests for RELEASE_AUDIT.md, module Samplers/SBI, minors 15-34.

Small budgets: only the audited code paths are hit (guards, validation,
cache invalidation, chunking), no sampler is run to convergence.

Run with:  pytest Test/16.SBI/test_audit_sbi_minor.py -q
"""
import inspect
import pathlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest

from BayesForge.Samplers.SBI import abc, abc_rf, npe, simulate
from BayesForge.Samplers.SBI.abc import ABC_BF
from BayesForge.Samplers.SBI.abc_rf import ABCRF_BF
from BayesForge.Samplers.SBI.npe import NPE_BF
from BayesForge.Samplers.SBI.simulate import Simulator, SBIPosterior, weighted_resample
from BayesForge.Utils.dev_warning import DEV_WARNING_MSG, reset_dev_warnings, warn_dev

N = 20
Y = 1.5 + jax.random.normal(jax.random.PRNGKey(1), (N,))


def _normal_model():
    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0, 3))
        numpyro.sample("y", dist.Normal(mu, 1), obs=y)
    return model


def _mean_stat(y):
    return jnp.mean(y, keepdims=True)


def _sim(**kw):
    return Simulator(_normal_model(), data={"y": Y},
                     summary_stats=_mean_stat, **kw)


@pytest.fixture(autouse=True)
def _consume_dev_warning():
    """Spend the one-shot warn_dev("SBI") so it cannot pollute warning asserts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warn_dev("SBI")


# abc.py --------------------------------------------------------------------------------
def test_15_nan_summary_stats_raise_instead_of_silent_garbage():
    def nan_stats(y):
        return jnp.full((y.shape[0], 1), jnp.nan)

    sim = Simulator(_normal_model(), data={"y": Y}, summary_stats=nan_stats)
    with pytest.raises(ValueError, match="distances are not usable"):
        ABC_BF(sim, num_simulations=2000, quantile=0.05,
               num_samples=100, progress_bar=False).run(jax.random.PRNGKey(0))


def test_15_few_nan_summaries_warn_and_are_excluded():
    from BayesForge.Samplers.SBI.abc import _screen_distances
    d = jnp.array([0.1, jnp.nan, 0.3, 0.2])
    with pytest.warns(UserWarning, match="1 of 4 simulations"):
        out = _screen_distances(d)
    assert bool(jnp.isinf(out[1])) and float(out[0]) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="distances are not usable"):
        _screen_distances(jnp.array([jnp.nan, jnp.nan, jnp.nan, 0.1]))


def test_16_regression_adjust_survives_eps_zero():
    key = jax.random.PRNGKey(2)
    u = jax.random.normal(key, (200, 1))
    s = jax.random.normal(key, (200, 1))
    s_obs = jnp.zeros(1)
    # exact match (d == 0) and eps == 0: d / eps would be 0/0 or x/0.
    out = abc._regression_adjust(u, s, s_obs, jnp.zeros(200), 0.0)
    assert jnp.all(jnp.isfinite(out))
    out = abc._regression_adjust(u, s, s_obs, jnp.ones(200), 0.0)
    assert jnp.all(jnp.isfinite(out))


def test_17_smc_round0_stays_within_declared_budget():
    budget = 50_000
    res = ABC_BF(_sim(), method="smc", num_simulations=budget, num_rounds=4,
                 num_particles=200, num_samples=200, progress_bar=False
                 ).run(jax.random.PRNGKey(3))
    n0 = res.history[0]["n_simulations"]
    assert n0 <= max(budget // 4, 200), n0
    assert n0 <= budget, n0
    assert jnp.all(jnp.isfinite(res.posterior["mu"]))


def test_18_constructor_validates_quantile_epsilon_alpha_rounds():
    sim = _sim()
    with pytest.raises(ValueError, match=r"quantile must be in \(0, 1\]"):
        ABC_BF(sim, quantile=1.5)
    with pytest.raises(ValueError, match=r"quantile must be in \(0, 1\]"):
        ABC_BF(sim, quantile=0.0)
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        ABC_BF(sim, epsilon=0.0)
    with pytest.raises(ValueError, match=r"epsilon must be > 0"):
        ABC_BF(sim, epsilon=-1.0)
    with pytest.raises(ValueError, match=r"alpha must be in \(0, 1\)"):
        ABC_BF(sim, alpha=1.0)
    with pytest.raises(ValueError, match=r"num_rounds must be >= 1"):
        ABC_BF(sim, num_rounds=0)


def test_19_callable_distance_clear_errors():
    key = jax.random.PRNGKey(4)
    s_ref = jax.random.normal(key, (50, 1))
    s_obs = s_ref[0]

    good = abc._make_distance(lambda si, so: jnp.sum((si - so) ** 2), s_ref)
    d = good(s_ref, s_obs)
    assert d.shape == (50,) and bool(jnp.all(jnp.isfinite(d)))

    np_fn = abc._make_distance(lambda si, so: float(np.sum((si - so) ** 2)), s_ref)
    with pytest.raises(ValueError, match="jax-compatible"):
        np_fn(s_ref, s_obs)

    wide = abc._make_distance(lambda si, so: jnp.ones(3), s_ref)
    with pytest.raises(ValueError, match="scalar per row"):
        wide(s_ref, s_obs)


def test_20_mahalanobis_survives_degenerate_reference_stats():
    const = jnp.ones((100, 2))          # constant stats -> zero covariance
    d = abc._make_distance("mahalanobis", const)(const, const[0])
    assert jnp.all(jnp.isfinite(d))

    single = jnp.ones((1, 2))           # one row -> jnp.cov returns NaN
    d = abc._make_distance("mahalanobis", single)(single, single[0])
    assert jnp.all(jnp.isfinite(d))


# abc_rf.py ------------------------------------------------------------------------------
def _small_rf(sim, **kw):
    kw.setdefault("num_simulations", 4000)
    kw.setdefault("n_estimators", 50)
    kw.setdefault("max_features", 1.0)
    kw.setdefault("n_jobs", 1)
    kw.setdefault("num_samples", 200)
    kw.setdefault("progress_bar", False)
    return ABCRF_BF(sim, **kw)


@pytest.fixture(scope="module")
def fitted_rf():
    return _small_rf(_sim(), random_state=3).run(jax.random.PRNGKey(9))


def test_21_forest_seed_derives_from_run_key_random_state_warns():
    key = jax.random.PRNGKey(11)
    key_seed = int(jax.random.randint(key, (), 0, 2 ** 31 - 1))

    # random_state matching the key: silent (one RNG source).
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _small_rf(_sim(), random_state=key_seed).run(key)
    assert not any("random_state" in str(w.message) for w in rec)

    # random_state overriding the key: RuntimeWarning, and reproducible.
    with pytest.warns(RuntimeWarning, match="random_state"):
        r1 = _small_rf(_sim(), random_state=7).run(key)
    r2 = _small_rf(_sim(), random_state=7).run(key)
    assert np.allclose(r1.weights["mu"], r2.weights["mu"])

    # default (random_state=None): no warning, seed comes from the key.
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _small_rf(_sim()).run(key)
    assert not any("random_state" in str(w.message) for w in rec)


def test_22_non_finite_s_obs_rejected():
    sim = _sim()
    sim.s_obs = jnp.full((1, 1), jnp.nan)
    with pytest.raises(ValueError, match="non-finite summary statistics"):
        _small_rf(sim).run(jax.random.PRNGKey(12))


def test_23_reference_table_and_oob_consistent_constant_column_finite(fitted_rf):
    Theta, S = fitted_rf.reference_table
    assert Theta.shape[0] == S.shape[0] == fitted_rf.weights["mu"].shape[0]
    w = fitted_rf.weights["mu"]
    assert np.all(np.isfinite(w)) and np.isclose(w.sum(), 1.0)
    assert np.all(np.isfinite(fitted_rf.oob.to_numpy()))

    # constant parameter column: ss_tot == 0 must give 0.0, not inf/NaN.
    row = ABCRF_BF._oob_row(np.ones(50), np.ones(50))
    assert np.isfinite(row["oob_r2"]) and np.isfinite(row["oob_mae"])


def test_24_qrf_weights_processed_in_row_chunks(fitted_rf):
    forest = fitted_rf.forests["mu"]
    _, S = fitted_rf.reference_table
    s_obs = np.asarray(fitted_rf.sim.s_obs, dtype=np.float64).ravel()

    rows_per_call = []
    orig_apply = forest.apply
    counting = lambda X: (rows_per_call.append(np.asarray(X).shape[0]),
                          orig_apply(X))[1]
    forest.apply = counting
    try:
        w = abc_rf._qrf_weights(forest, S, s_obs, chunk_rows=1000)
    finally:
        forest.apply = orig_apply

    n = S.shape[0]
    assert len(rows_per_call) >= int(np.ceil(n / 1000)) + 1  # chunks + s_obs
    assert sum(rows_per_call) == n + 1                       # every row once
    assert w.shape == (n,) and np.all(np.isfinite(w))


# npe.py ---------------------------------------------------------------------------------
def _small_npe(sim, **kw):
    kw.setdefault("num_simulations", 2000)
    kw.setdefault("max_steps", 100)
    kw.setdefault("eval_every", 50)
    kw.setdefault("batch_size", 64)
    kw.setdefault("patience", 100)
    kw.setdefault("progress_bar", False)
    return NPE_BF(sim, **kw)


@pytest.fixture(scope="module")
def fitted_npe():
    return _small_npe(_sim()).run(jax.random.PRNGKey(21))


def test_25_run_splits_key_once_no_reuse():
    src = inspect.getsource(NPE_BF.run)
    assert "jax.random.split(rng_key, 6)" in src

    key = jax.random.PRNGKey(22)
    r1 = _small_npe(_sim()).run(key)
    r2 = _small_npe(_sim()).run(key)
    r3 = _small_npe(_sim()).run(jax.random.PRNGKey(23))
    assert np.allclose(np.asarray(r1.posterior["mu"]), np.asarray(r2.posterior["mu"]))
    assert not np.allclose(np.asarray(r1.posterior["mu"]), np.asarray(r3.posterior["mu"]))


def test_26_npe_constructor_validates_lr_batch_prior_scale():
    sim = _sim()
    with pytest.raises(ValueError, match=r"lr must be > 0"):
        NPE_BF(sim, lr=0.0)
    with pytest.raises(ValueError, match=r"batch_size must be >= 1"):
        NPE_BF(sim, batch_size=0)
    with pytest.raises(ValueError, match=r"prior_scale must be > 0"):
        NPE_BF(sim, prior_scale=0.0)
    with pytest.raises(ValueError, match=r"prior_scale must be > 0"):
        NPE_BF(sim, prior_scale=-1.0)


def test_27_log_q_u_shape_errors_are_explicit(fitted_npe):
    u = jnp.zeros((5, 1))
    with pytest.raises(ValueError, match="summary statistics have 1 dimensions"):
        fitted_npe.log_q_u(u, jnp.zeros(3))
    with pytest.raises(ValueError, match=r"expected \(5, 1\) to match u"):
        fitted_npe.log_q_u(u, jnp.zeros((4, 1)))
    # valid shapes must not raise
    out = fitted_npe.log_q_u(u, jnp.zeros((5, 1)))
    assert out.shape == (5,) and jnp.all(jnp.isfinite(out))


def test_28_sample_u_documents_fold_in_correlation():
    doc = NPE_BF.sample_u.__doc__ or ""
    assert "fold_in(key, 7)" in doc


# simulate.py ----------------------------------------------------------------------------
def test_29_sanitize_log_density_reports_plus_inf_inversion():
    reset_dev_warnings()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = simulate._sanitize_log_density(
            jnp.array([jnp.inf, 2.0, -jnp.inf, jnp.nan]))
    assert bool(jnp.isneginf(out[0])) and bool(jnp.isneginf(out[2]))
    assert bool(jnp.isneginf(out[3])) and float(out[1]) == 2.0
    # warn_dev() emits the standard development message (the key text only
    # de-duplicates), so match DEV_WARNING_MSG, not the key.
    assert any(isinstance(w.message, RuntimeWarning)
               and str(w.message) == DEV_WARNING_MSG for w in rec)


def test_30_mutating_data_invalidates_jit_caches():
    sim = _sim()
    key = jax.random.PRNGKey(30)
    sim.simulate_prior(key, 8)
    fn1 = sim._prior_fn[1]
    sim.simulate_prior(key, 8)
    assert sim._prior_fn[1] is fn1                      # cached while data unchanged

    sim.data = {"y": Y + 1.0}                           # dict replaced
    sim.simulate_prior(key, 8)
    assert sim._prior_fn[1] is not fn1                  # rebuilt on the new data

    sim.simulate_given(key, jnp.zeros((4, 1)))
    cond1 = dict(sim._cond_fn)
    sim.data["y"] = Y + 2.0                             # entry replaced in place
    sim.simulate_given(key, jnp.zeros((4, 1)))
    assert sim._cond_fn[4] is not cond1[4]


def test_31_zero_batch_raises_explicit_value_error():
    sim = _sim()
    key = jax.random.PRNGKey(31)
    with pytest.raises(ValueError, match=r"simulate_prior needs n >= 1, got 0"):
        sim.simulate_prior(key, 0)
    with pytest.raises(ValueError, match=r"at least one parameter row, got 0"):
        sim.simulate_given(key, jnp.zeros((0, 1)))


def test_32_dead_sbi_observed_flag_removed():
    src = pathlib.Path(simulate.__file__).read_text()
    assert "_sbi_observed" not in src


def test_33_weighted_resample_rejects_bad_weights():
    key = jax.random.PRNGKey(33)
    u = jnp.arange(10.0).reshape(10, 1)
    with pytest.raises(ValueError, match="positive sum"):
        weighted_resample(key, u, jnp.array([1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0]), 5)
    with pytest.raises(ValueError, match="positive sum"):
        weighted_resample(key, u, jnp.zeros(10), 5)
    with pytest.raises(ValueError, match="positive sum"):
        weighted_resample(key, u, jnp.full(10, jnp.nan), 5)
    out = weighted_resample(key, u, jnp.ones(10), 5)
    assert out.shape == (5, 1)


def test_34_sbi_posterior_pseudo_chains_and_dict_contract():
    post = SBIPosterior()
    post.posterior = {"mu": jnp.arange(999.0)}
    assert post.num_chains == 4                     # 999 draws: 4 chains, not 1
    grouped = post.get_samples(group_by_chain=True)
    assert grouped["mu"].shape == (4, 249)          # largest multiple of 4 kept
    assert post.get_samples()["mu"].shape == (999,)
    assert isinstance(post.get_samples(), dict)
    post.posterior = {"mu": jnp.arange(8.0)}
    assert post.num_chains == 1
