"""Tier 1.7 verification: expose MixedHMC / DiscreteHMCGibbs / HMCGibbs.

Roadmap reference (feature_roadmap.md, Tier 1):
    `m.fit(sampler="nuts"|"mixed_hmc"|"discrete_gibbs"|"hmc_gibbs")`,
    mapped onto `numpyro.infer`.

Verification (per roadmap):
    * N-mixture with small N: posterior of λ and p with `discrete_gibbs` =
      posterior with exact enumeration (NUTS), within MC error.
    * DPMM `latent`: recovers 3 simulated clusters (ARI > 0.9).

Plus smoke coverage that every new sampler name runs end-to-end and that
bad sampler arguments fail fast with a clear error.

Agreement between two samplers is tested within Monte Carlo error: for
each parameter, |difference of means| < 3 x the combined Monte Carlo
standard error (MCSE = sd / sqrt(ESS)), as in Examples/compare_languages.py
of the manuscript.

Run:
    pytest Test/0.Core/test_discrete_samplers.py          (all, ~3 min)
    pytest Test/0.Core/test_discrete_samplers.py -m "not slow"
"""

import functools

import arviz as az
import jax.numpy as jnp
from jax.scipy.special import gammaln, logsumexp
import numpy as np
import numpyro
import pytest

from BayesForge import bf


def _z_mcse(a, b):
    """z-score of the difference of posterior means, from by-chain draws
    (chains, draws): (mean_b - mean_a) / sqrt(MCSE_a^2 + MCSE_b^2)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mcse = lambda x: x.std() / np.sqrt(float(az.ess(x)))
    return (b.mean() - a.mean()) / np.hypot(mcse(a), mcse(b))

# --------------------------------------------------------------------------
# N-mixture data: S sites x J visits, abundance N in 0..NMAX.
#
# The abundance prior is a Poisson(λ) truncated to 0..NMAX, expressed as a
# Categorical over that support: finite (so the discrete latents can be
# summed out exactly) yet parameterised by the literal Poisson rate λ, with
# J = 5 repeated visits making (λ, p) identifiable.
#
# Layout note: the J visits enter as the Binomial trial multiplicity --
# Y_i = Σ_j y_ij ~ Binomial(J·N_i, p), the sufficient statistic of the
# (S, J) visit layout -- under a single `sites` plate. A literal (S, J)
# nested-plate/event layout is not used because funsor cannot reconcile
# the plate dims with the enumeration dim there (broadcast of the plate
# dim against the enum dim); the statistical comparison performed here
# (Gibbs vs exact marginalisation of the same discrete latents) is
# unaffected by the sufficient-statistic reduction.
# --------------------------------------------------------------------------
_NMIX = dict(S=12, J=5, NMAX=8, LAM_TRUE=4.0, P_TRUE=0.6)
_rng_nmix = np.random.RandomState(1)
_N_TRUE = _rng_nmix.poisson(_NMIX["LAM_TRUE"], size=_NMIX["S"]).clip(0, _NMIX["NMAX"])
_Y_NMIX = jnp.asarray(
    np.stack(
        [_rng_nmix.binomial(n, _NMIX["P_TRUE"], size=_NMIX["J"]) for n in _N_TRUE]
    ).sum(axis=1)
)
# Per-site feasible abundance init (J·N >= total count at that site).
_N_INIT = ((np.asarray(_Y_NMIX) + _NMIX["J"] - 1) // _NMIX["J"]).clip(0, _NMIX["NMAX"])
_K_NMIX = jnp.arange(_NMIX["NMAX"] + 1)


def make_nmix(m, enumerate_N):
    """Binomial N-mixture with truncated-Poisson(λ) abundance. `enumerate_N`
    selects exact marginalisation of the discrete latents (reference, fit
    with NUTS) or plain discrete sites (challenger, fit with
    `discrete_gibbs`)."""

    def model(y):
        lam = m.dist.exponential(0.5, name="lam")
        p = m.dist.beta(2.0, 2.0, name="p")
        log_pi = _K_NMIX * jnp.log(lam) - lam - gammaln(_K_NMIX + 1.0)
        pi = jnp.exp(log_pi - logsumexp(log_pi))
        with m.dist.plate("sites", _NMIX["S"]):
            if enumerate_N:
                N = m.dist.categorical_probs(
                    pi, name="N", infer={"enumerate": "parallel"}
                )
            else:
                N = m.dist.categorical_probs(pi, name="N")
            m.dist.binomial_probs(p, total_count=_NMIX["J"] * N, obs=y)

    model.__name__ = "nmix"
    return model


def _new_session(seed=7):
    return bf(platform="cpu", rand_seed=seed, print_devices_found=False)


@pytest.mark.slow
def test_nmix_discrete_gibbs_matches_enumeration(verify):
    """`discrete_gibbs` posterior of λ and p = exact-enumeration posterior
    (NUTS), within Monte Carlo error (|z| < 3, MCSE from the ESS).

    Gibbs on the abundances mixes slowly (ESS ~2-3% of the draws), hence
    4 chains x 2000 draws: the MCSE then makes the test sensitive to a bias
    of ~0.3 posterior sd while keeping false alarms below 1%."""
    chains, warmup, draws = 4, 1000, 2000
    m_ref = _new_session()
    m_ref.fit(
        make_nmix(m_ref, enumerate_N=True),
        obs={"y": _Y_NMIX},
        # A uniform draw can land below an observed count, where the Binomial
        # log-density is -inf: start the abundances at a feasible point.
        init_strategy=numpyro.infer.init_to_value(values={"N": _N_INIT}),
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        progress_bar=False,
    )
    assert m_ref.sampler_name == "nuts"

    m_gibbs = _new_session(seed=8)
    m_gibbs.fit(
        make_nmix(m_gibbs, enumerate_N=False),
        obs={"y": _Y_NMIX},
        sampler="discrete_gibbs",
        init_strategy=numpyro.infer.init_to_value(values={"N": _N_INIT}),
        # Gibbs conditions each block's init on the other blocks' starting
        # values: start the abundances at a feasible point (N >= y), with one
        # leading entry per chain.
        # NumPyro needs every latent site in init_params (continuous ones in
        # unconstrained space: log for lam > 0, logit for p in (0, 1)).
        init_params={"N": np.stack([_N_INIT] * chains),
                     "lam": jnp.full(chains, jnp.log(4.0)),
                     "p": jnp.zeros(chains)},
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        progress_bar=False,
    )
    assert m_gibbs.sampler_name == "discrete_gibbs"
    # The challenger samples the discrete latents; the reference sums them out.
    assert "N" in m_gibbs.posteriors and "N" not in m_ref.posteriors

    for key in ("lam", "p"):
        a = m_ref.posteriors_by_chain[key]
        b = m_gibbs.posteriors_by_chain[key]
        assert float(az.rhat(np.asarray(b))) < 1.05, f"{key}: Gibbs chains did not mix"
        z = _z_mcse(a, b)
        verify.record(quantity=key, reference="NUTS + exact enumeration of N",
                      criterion="|z| < 3", value=z, passed=abs(z) < 3, ref=a, new=b,
                      ref_label="NUTS (enumerated)", new_label="discrete_gibbs")
        assert abs(z) < 3, (
            f"{key}: enum mean={np.mean(a):.3f}, gibbs mean={np.mean(b):.3f}, z={z:.2f}"
        )
        print(f"  {key}: enum mean={np.mean(a):.3f} | gibbs mean={np.mean(b):.3f} | z={z:.2f}")
    print("test_nmix_discrete_gibbs_matches_enumeration: OK")


# --------------------------------------------------------------------------
# DPMM latent: 3 well-separated 2-D clusters, truncation T = 3.
# --------------------------------------------------------------------------
_DPMM = dict(N_PER=20, T=3, SEED=11)
_rng_dpmm = np.random.RandomState(2)
_CENTERS = np.array([[-4.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
_X_DPMM = np.vstack(
    [_rng_dpmm.normal(c, 0.7, size=(_DPMM["N_PER"], 2)) for c in _CENTERS]
)
_Z_DPMM = np.repeat([0, 1, 2], _DPMM["N_PER"])


def _ari(true, pred):
    """Adjusted Rand Index (no sklearn dependency)."""
    from math import comb

    true = np.asarray(true)
    pred = np.asarray(pred)
    classes, _ = np.unique(true, return_inverse=True)
    clusters, _ = np.unique(pred, return_inverse=True)
    cont = np.zeros((classes.size, clusters.size), dtype=np.int64)
    np.add.at(cont, (np.searchsorted(np.unique(true), true),
                     np.searchsorted(np.unique(pred), pred)), 1)
    sum_comb = sum(comb(n, 2) for n in cont.ravel())
    sum_rows = sum(comb(n, 2) for n in cont.sum(axis=1))
    sum_cols = sum(comb(n, 2) for n in cont.sum(axis=0))
    n = true.size
    expected = sum_rows * sum_cols / comb(n, 2)
    denom = 0.5 * (sum_rows + sum_cols) - expected
    return (sum_comb - expected) / denom if denom else 0.0


@pytest.mark.slow
def test_dpmm_latent_recovers_clusters(verify):
    """DPMM `latent` fit with `discrete_gibbs` recovers 3 clusters (ARI > 0.9)."""
    m = _new_session(seed=_DPMM["SEED"])
    m.fit(
        functools.partial(m.models.dpmm, method="latent"),
        obs={"data": jnp.asarray(_X_DPMM), "T": _DPMM["T"]},
        sampler="discrete_gibbs",
        num_warmup=1000,  # 300 left one Gibbs chain in a merged-cluster mode under the data-scaled priors of 4a318ce5
        num_samples=300,
        num_chains=2,
        progress_bar=False,
    )
    z = np.asarray(m.posteriors["z"]).reshape(-1, _Z_DPMM.size)
    aris = [_ari(_Z_DPMM, zd) for zd in z]
    mean_ari = float(np.mean(aris))
    verify.record(quantity="mean ARI vs simulated clusters", reference="simulated labels",
                  criterion="> 0.9", value=mean_ari, passed=mean_ari > 0.9)
    assert mean_ari > 0.9, f"mean ARI={mean_ari:.3f} (per-draw: {np.round(aris, 3)})"
    print(f"test_dpmm_latent_recovers_clusters: OK (mean ARI={mean_ari:.3f})")


# --------------------------------------------------------------------------
# Smoke: mixed_hmc and hmc_gibbs run end-to-end; bad args fail fast.
# --------------------------------------------------------------------------
_rng_smoke = np.random.RandomState(3)
_Y_SMOKE = jnp.asarray(
    np.where(_rng_smoke.binomial(1, 0.5, size=40),
             _rng_smoke.normal(2.0, 1.0, size=40),
             _rng_smoke.normal(-2.0, 1.0, size=40))
)


def _mixture_stats(m):
    """Label-invariant summaries of the 2-component mixture, by chain:
    the lower and upper means and the weight of the upper component."""
    mu = np.asarray(m.posteriors_by_chain["mu"])
    p = np.asarray(m.posteriors_by_chain["p"])
    upper_is_1 = mu[..., 1] > mu[..., 0]
    return {"mu_low": mu.min(axis=-1), "mu_high": mu.max(axis=-1),
            "w_high": np.where(upper_is_1, p, 1 - p)}


@pytest.mark.slow
def test_mixed_hmc_matches_enumeration(verify):
    """MixedHMC (component labels sampled) = NUTS with the labels summed out,
    within Monte Carlo error, on label-invariant summaries."""

    def fit(sampler, enumerate_c, seed):
        m = _new_session(seed)

        def model(y):
            p = m.dist.beta(2.0, 2.0, name="p")
            with m.dist.plate("components", 2):
                mu = m.dist.normal(0.0, 5.0, name="mu")
            with m.dist.plate("N", y.shape[0]):
                c = m.dist.bernoulli_probs(
                    p, name="c", infer={"enumerate": "parallel"} if enumerate_c else None)
                m.dist.normal(mu[c], 1.0, obs=y)

        model.__name__ = f"mixture_{sampler}"
        m.fit(model, obs={"y": _Y_SMOKE}, sampler=sampler, num_warmup=1000,
              num_samples=2000, num_chains=4, progress_bar=False)
        return m

    ref, mixed = fit("nuts", True, 1), fit("mixed_hmc", False, 2)
    assert mixed.sampler_name == "mixed_hmc"
    assert "c" in mixed.posteriors and "c" not in ref.posteriors
    a, b = _mixture_stats(ref), _mixture_stats(mixed)
    for key in a:
        assert float(az.rhat(b[key])) < 1.05, f"{key}: MixedHMC chains did not mix"
        z = _z_mcse(a[key], b[key])
        verify.record(quantity=key, reference="NUTS + exact enumeration",
                      criterion="|z| < 3", value=z, passed=abs(z) < 3, ref=a[key], new=b[key],
                      ref_label="NUTS (enumerated)", new_label="mixed_hmc")
        assert abs(z) < 3, f"{key}: enum {a[key].mean():.3f}, mixed {b[key].mean():.3f}, z={z:.2f}"
        print(f"  {key}: enum {a[key].mean():.3f} | mixed {b[key].mean():.3f} | z={z:.2f}")


_Y_GIBBS = jnp.asarray(_rng_smoke.normal(1.5, 1.0, size=30))


def test_hmc_gibbs_matches_nuts(verify):
    m = _new_session()

    def model(y):
        x = m.dist.normal(0.0, 2.0, name="x")
        s = m.dist.exponential(1.0, name="s")
        # NOTE: the obs site must be named explicitly here: an unnamed
        # `dist.*(obs=...)` call defaults to name='x', which would collide
        # with the latent above.
        m.dist.normal(x, s, obs=y, name="obs")

    model.__name__ = "smoke_hmc_gibbs"

    n = _Y_GIBBS.shape[0]
    y_sum = float(_Y_GIBBS.sum())

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # x | y, s is Normal in closed form; s comes from the HMC block.
        import jax
        s = hmc_sites["s"]
        post_var = 1.0 / (0.25 + n / s ** 2)
        post_mean = post_var * y_sum / s ** 2
        return {"x": post_mean + jnp.sqrt(post_var) * jax.random.normal(rng_key)}

    m.fit(model, obs={"y": _Y_GIBBS}, sampler="hmc_gibbs",
          gibbs_fn=gibbs_fn, gibbs_sites=["x"],
          num_warmup=500, num_samples=1000, num_chains=2, progress_bar=False)
    assert m.sampler_name == "hmc_gibbs"
    # Same model with plain NUTS: the exact Gibbs step for x must give the
    # same posterior for x and s, within Monte Carlo error.
    ref = _new_session(seed=9)
    ref.fit(model, obs={"y": _Y_GIBBS}, num_warmup=500, num_samples=1000,
            num_chains=2, progress_bar=False)
    for key in ("x", "s"):
        a, b = ref.posteriors_by_chain[key], m.posteriors_by_chain[key]
        z = _z_mcse(a, b)
        verify.record(quantity=key, reference="plain NUTS", criterion="|z| < 3",
                      value=z, passed=abs(z) < 3, ref=a, new=b,
                      ref_label="NUTS", new_label="hmc_gibbs")
        assert abs(z) < 3, f"{key}: z={z:.2f}"
    x_mean = float(np.asarray(m.posteriors["x"]).mean())
    print(f"test_hmc_gibbs_matches_nuts: OK (x_mean={x_mean:.2f})")


def test_sampler_args_validated():
    m = _new_session()

    def model(y):
        s = m.dist.exponential(1.0, name="s")
        m.dist.normal(0.0, s, obs=y)

    model.__name__ = "smoke_valid"
    for kwargs, msg in (
        ({"sampler": "bogus"}, "unknown sampler"),
        ({"sampler": "hmc_gibbs"}, "hmc_gibbs without gibbs_fn"),
        ({"sampler": "hmc_gibbs", "gibbs_fn": lambda *a: {}}, "hmc_gibbs without sites"),
    ):
        try:
            m.fit(model, obs={"y": _Y_GIBBS}, progress_bar=False, **kwargs)
        except ValueError:
            print(f"test_sampler_args_validated: OK ({msg} -> ValueError)")
        else:
            raise AssertionError(f"{msg}: expected ValueError")


def test_non_numpyro_backend_rejects_other_samplers():
    """The TFP backend runs only its own NUTS: asking for another sampler
    must fail instead of being silently ignored."""
    m = _new_session()
    m.backend = "tfp"  # checked before any backend-specific work

    def model(y):
        s = m.dist.exponential(1.0, name="s")
        m.dist.normal(0.0, s, obs=y)

    with pytest.raises(ValueError, match="numpyro backend"):
        m.fit(model, obs={"y": _Y_GIBBS}, sampler="discrete_gibbs", progress_bar=False)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-s"]))
