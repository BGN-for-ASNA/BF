"""Tier 1.6 verification: explicit enumeration marking (`enum=True`).

Roadmap reference (feature_roadmap.md, Tier 1):
    `infer={"enumerate": "parallel"}` set by BF in the discrete `dist.*`
    when `enum=True`; turn off the implicit auto-enumeration.

Verification (per roadmap):
    ZIP / mixture / HMM models: same posterior before and after
    (fixed seed, |Δ| < 1e-10); test with `warnings.simplefilter("error")`
    for future NumPyro warnings.

Notes on model choice:
    * A 2-component Gaussian mixture with a `categorical` assignment latent
      exercises the enumeration path (this is the "mixture" case).
    * ZIP / HMM have no finite-support enumerable site at the `dist.*` level
      used here (Poisson support is infinite; BF has no NumPyro HMM dist --
      HMMs built from `categorical` + scan are covered by the same flag), so
      the mixture is the representative finite-support case.

Run:
    python Test/0.Core/test_enumeration.py
    (also pytest-collectable: `pytest Test/0.Core/test_enumeration.py`)
"""

import warnings

import jax.numpy as jnp
import numpy as np

from BayesForge import bf

# Fixed synthetic data: two well-separated 1-D components, n = 30.
_rng = np.random.RandomState(0)
_N = 30
_Z_TRUE = _rng.binomial(1, 0.4, size=_N)
_Y = jnp.asarray(
    np.where(_Z_TRUE == 0, _rng.normal(-2.0, 0.5, size=_N), _rng.normal(2.0, 0.5, size=_N))
)

WARMUP = 200
SAMPLES = 200
CHAINS = 2
SEED = 123


def make_model(m, mode):
    """Two-component mixture. `mode` selects how the assignment site is marked.

    * "default" -- status-quo call, relies on NumPyro's implicit handling.
    * "enum"    -- explicit `enum=True` (the new flag under test).
    * "infer"   -- explicit `infer={"enumerate": "parallel"}` (reference).
    """

    def model(y):
        w = m.dist.dirichlet(jnp.ones(2), name="w")
        # Component params live in their own plate: with explicit enumeration
        # NumPyro validates that every batch dim is plate-enclosed.
        with m.dist.plate("components", 2):
            mu = m.dist.normal(0.0, 5.0, name="mu")
        sigma = m.dist.exponential(1.0, name="sigma")
        with m.dist.plate("N", y.shape[0]):
            if mode == "infer":
                z = m.dist.categorical_probs(
                    w, name="z", infer={"enumerate": "parallel"}
                )
            elif mode == "enum":
                z = m.dist.categorical_probs(w, name="z", enum=True)
            else:
                z = m.dist.categorical_probs(w, name="z")
            m.dist.normal(mu[z], sigma, obs=y)

    model.__name__ = f"mixture_{mode}"
    return model


def fit_mixture(mode, strict_warnings=True):
    m = bf(platform="cpu", rand_seed=SEED, print_devices_found=False)
    with warnings.catch_warnings():
        if strict_warnings:
            # Future NumPyro warnings (e.g. about enumeration) must not go
            # unnoticed: fail loudly if one is raised. Scoped to
            # FutureWarning/DeprecationWarning so benign runtime notes
            # (e.g. the "not enough devices" UserWarning on single-CPU
            # machines) do not fail the test.
            warnings.simplefilter("error", FutureWarning)
            warnings.simplefilter("error", DeprecationWarning)
        m.fit(
            make_model(m, mode),
            obs={"y": _Y},
            num_warmup=WARMUP,
            num_samples=SAMPLES,
            num_chains=CHAINS,
            progress_bar=False,
        )
    return m


def _means(m):
    return {k: np.asarray(v).mean(axis=0) for k, v in m.posteriors.items()}


def test_enum_matches_default(verify):
    """Same posterior before (implicit) and after (explicit `enum=True`).

    Fixed seed, |Δ| < 1e-10 on every posterior draw of every shared site.

    The "default" fit relies on NumPyro's implicit auto-enumeration, which
    current NumPyro versions flag with a FutureWarning ("...enumerated sites
    need to be marked with `infer={'enumerate': 'parallel'}`") -- that warning
    is expected here and therefore not escalated. The `enum=True` fit must
    raise no such warning (it is run with `strict_warnings=True`).
    """
    m_default = fit_mixture("default", strict_warnings=False)
    m_enum = fit_mixture("enum")
    assert set(m_default.posteriors) == set(m_enum.posteriors), (
        f"posterior sites differ: {set(m_default.posteriors)} vs {set(m_enum.posteriors)}"
    )
    for k in m_enum.posteriors:
        a = np.asarray(m_default.posteriors[k], dtype=np.float64)
        b = np.asarray(m_enum.posteriors[k], dtype=np.float64)
        delta = np.max(np.abs(a - b))
        verify.record(quantity=k, reference="NumPyro, other path (consistency lock)",
                      criterion="max |Δ| < 1e-10", value=delta, passed=delta < 1e-10,
                      ref=a, new=b, ref_label="implicit", new_label="enum=True")
        assert delta < 1e-10, f"site {k!r}: max |Δ| = {delta}"
    print("test_enum_matches_default: OK "
          f"(sites={sorted(m_enum.posteriors)}, max |Δ| < 1e-10)")


def test_enum_is_exact_enumeration(verify):
    """`enum=True` is exactly `infer={"enumerate": "parallel"}`.

    Locks the flag-to-infer mapping, and proves the discrete latent is
    summed out (absent from the posteriors) rather than sampled.
    """
    m_enum = fit_mixture("enum")
    m_infer = fit_mixture("infer")
    assert "z" not in m_enum.posteriors, (
        f"assignment site should be enumerated away, got sites {sorted(m_enum.posteriors)}"
    )
    assert set(m_enum.posteriors) == set(m_infer.posteriors)
    for k in m_enum.posteriors:
        a = np.asarray(m_enum.posteriors[k], dtype=np.float64)
        b = np.asarray(m_infer.posteriors[k], dtype=np.float64)
        delta = np.max(np.abs(a - b))
        verify.record(quantity=k, reference='infer={"enumerate": "parallel"}',
                      criterion="max |Δ| = 0", value=delta, passed=delta == 0.0)
        assert delta == 0.0, f"site {k!r} differs from infer-dict path"
    # Sanity: the mixture recovers the two modes (labels can swap).
    mu = np.asarray(m_enum.posteriors["mu"]).reshape(-1, 2).mean(axis=0)
    lo, hi = sorted(float(x) for x in mu)
    assert abs(lo + 2.0) < 0.5 and abs(hi - 2.0) < 0.5, f"mu={mu}"
    print("test_enum_is_exact_enumeration: OK "
          f"(sites={sorted(m_enum.posteriors)}, mu≈({lo:.2f}, {hi:.2f}))")


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-s"]))
