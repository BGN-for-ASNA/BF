"""Simulation-based inference: m.abc (rejection, SMC) and m.npe (MDN, flow, Bayesian).

Checked against the conjugate Normal-mean model, whose posterior is analytic:
    mu ~ Normal(0, 5),  y_i ~ Normal(mu, 1)
    => mu | y ~ Normal(sum(y) / (N + 1/25), 1 / sqrt(N + 1/25)).

Run with:  pytest "Test/16.SBI" -q
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge import bf

N = 20
Y = 1.5 + jax.random.normal(jax.random.PRNGKey(1), (N,))
PREC = N + 1 / 25
POST_MEAN = float(Y.sum() / PREC)
POST_SD = float(1 / np.sqrt(PREC))


def _bf():
    return bf(platform="cpu", print_devices_found=False)


def _normal_model(m):
    def model(y):
        mu = m.dist.normal(0, 5, name="mu")
        m.dist.normal(mu, 1, obs=y, name="y")
    return model


def _mean_stat(y):
    return jnp.mean(y, keepdims=True)


def _check(m, mean_tol, sd_rtol):
    draws = np.asarray(m.posteriors["mu"])
    assert abs(draws.mean() - POST_MEAN) < mean_tol, (draws.mean(), POST_MEAN)
    assert abs(draws.std() / POST_SD - 1) < sd_rtol, (draws.std(), POST_SD)


# ABC ------------------------------------------------------------------------------------
def test_abc_rejection_recovers_analytic_posterior():
    m = _bf()
    m.abc(_normal_model(m), data={"y": Y}, summary_stats=_mean_stat,
          num_simulations=200_000, quantile=0.005, progress_bar=False, seed=0)
    _check(m, mean_tol=0.05, sd_rtol=0.3)
    tab = m.summary(kind="stats")
    assert "mu" in tab.index


def test_abc_smc_recovers_analytic_posterior_and_respects_budget():
    m = _bf()
    m.abc(_normal_model(m), data={"y": Y}, method="smc", summary_stats=_mean_stat,
          num_simulations=50_000, progress_bar=False, seed=0)
    _check(m, mean_tol=0.05, sd_rtol=0.3)
    assert m.sampler.n_simulations <= 50_000 + 1000
    eps = [h["epsilon"] for h in m.sampler.history]
    assert all(e1 <= e0 for e0, e1 in zip(eps, eps[1:]))  # tolerance shrinks


def test_abc_with_tracked_output_and_nuisance_latents():
    """g-and-k: simulator output is a track() site; z is a nuisance latent."""
    m = _bf()

    def g_and_k(n):
        A = m.dist.uniform(0, 10, name="A")
        B = m.dist.uniform(0, 10, name="B")
        g = m.dist.uniform(0, 10, name="g")
        k = m.dist.uniform(0, 10, name="k")
        z = m.dist.normal(0, 1, shape=(n,), name="z")
        m.dist.track("y", A + B * (1 + 0.8 * jnp.tanh(g * z / 2)) * (1 + z**2) ** k * z)

    z = jax.random.normal(jax.random.PRNGKey(0), (500,))
    y = 3 + (1 + 0.8 * jnp.tanh(z)) * (1 + z**2) ** 0.5 * z
    m.abc(g_and_k, data={"n": 500}, observed={"y": y}, params=["A", "B", "g", "k"],
          summary_stats="robust", method="smc", num_simulations=100_000,
          progress_bar=False, seed=0)
    assert set(m.posteriors) == {"A", "B", "g", "k"}
    assert abs(float(m.posteriors["A"].mean()) - 3.0) < 0.6
    assert float(m.posteriors["k"].min()) >= 0.0


# NPE ------------------------------------------------------------------------------------
@pytest.mark.parametrize("estimator", ["mdn", "flow"])
def test_npe_recovers_analytic_posterior(estimator):
    m = _bf()
    m.npe(_normal_model(m), data={"y": Y}, estimator=estimator, summary_stats=_mean_stat,
          num_simulations=5_000, max_steps=6_000, progress_bar=False, seed=0)
    _check(m, mean_tol=0.08, sd_rtol=0.25)


def test_npe_amortised_and_constrained():
    m = _bf()

    def model(y):
        mu = m.dist.normal(0, 5, name="mu")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(mu, s, obs=y, name="y")

    stat = lambda y: jnp.stack([jnp.mean(y), jnp.log(jnp.std(y))])
    m.npe(model, data={"y": Y}, summary_stats=stat, num_simulations=5_000,
          max_steps=6_000, progress_bar=False, seed=0)
    assert float(m.posteriors["s"].min()) > 0
    shifted = m.npe_posterior(Y + 3.0, seed=1)  # no retraining
    assert abs(float(shifted["mu"].mean() - m.posteriors["mu"].mean()) - 3.0) < 0.3


def test_bayesian_npe_is_wider_than_npe_with_small_budget():
    sds = {}
    for bayesian in (False, True):
        m = _bf()
        m.npe(_normal_model(m), data={"y": Y}, summary_stats=_mean_stat, bayesian=bayesian,
              num_simulations=300, svi_steps=2_000, progress_bar=False, seed=0)
        sds[bayesian] = float(m.posteriors["mu"].std())
        cov = m.sbc_coverage(n_test=50, n_post=200, levels=[0.5, 0.9], plot=False)
        assert list(cov.columns) == ["nominal", "empirical"]
    assert sds[True] > sds[False]


# ABC random forest -------------------------------------------------------------------
@pytest.mark.parametrize("joint", [False, True])
def test_abc_rf_recovers_analytic_posterior_and_ignores_noise_stat(joint):
    m = _bf()

    # Second statistic is uninformative about mu (pure function of the noise scale).
    stats = lambda y: jnp.stack([jnp.mean(y), jnp.std(y - jnp.mean(y))])
    m.abc_rf(_normal_model(m), data={"y": Y}, summary_stats=stats,
             stat_names=["mean", "resid_sd"], num_simulations=10_000,
             n_estimators=200, max_features=1.0,  # few stats: consider all at each split
             joint=joint, progress_bar=False, seed=0)
    _check(m, mean_tol=0.1, sd_rtol=0.5)
    imp = m.sampler.importance.iloc[:, 0]
    assert imp["mean"] > imp["resid_sd"]
    assert m.sampler.oob.loc["mu", "oob_r2"] > 0.9
