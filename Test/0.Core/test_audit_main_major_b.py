"""Regression tests for RELEASE_AUDIT.md, module Main, majors M-08 and M-16."""
import jax.numpy as jnp
import pytest

from BayesForge import bf


def _fit_tiny(m):
    """Fit the two-parameter (a, b) model on three points (audit fit pattern)."""
    def model(y):
        a = m.dist.normal(0.0, 1.0, name="a")
        b = m.dist.normal(0.0, 1.0, name="b")
        m.dist.normal(a + b, 1.0, obs=y)

    y = jnp.array([0.1, -0.2, 0.3])
    m.fit(model, obs={"y": y}, num_samples=50, num_warmup=50, num_chains=2,
          progress_bar=False)

# Ensure PRNG JAX-state of the Dist singleton is updated after fitting 
def test_m08_second_instance_does_not_steal_rng():
    import BayesForge.Models.GMM as G
    if not hasattr(G, "dist"):
        pytest.skip("BayesForge.Models.GMM has no module-level dist singleton")

    m1 = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    m2 = bf(platform="cpu", rand_seed=2, print_devices_found=False)

    # m2 was constructed last, so the module singletons currently point at
    # it; fitting m1 must rebind them to m1's RNG (RELEASE_AUDIT M-08).
    _fit_tiny(m1)
    assert G.dist._rng is m1._rng

    _fit_tiny(m2)
    assert G.dist._rng is m2._rng


def test_m08_tfp_instances_have_own_rng():
    pytest.importorskip("tensorflow_probability")

    m1 = bf(platform="cpu", backend="tfp", rand_seed=1, print_devices_found=False)
    m2 = bf(platform="cpu", backend="tfp", rand_seed=2, print_devices_found=False)

    def _tfp_model(m):
        def model(y):
            a = yield m.dist.normal(0.0, 1.0, name="a")
            b = yield m.dist.normal(0.0, 1.0, name="b")
            yield m.dist.normal(a + b, 1.0, obs=y)
        return model

    y = jnp.array([0.1, -0.2, 0.3])
    # Rebind fallback (RELEASE_AUDIT M-08): tfp_dists._draw_key is a
    # @staticmethod reading the literal UnifiedDist._rng, so a per-instance
    # subclass would be ignored; the class attribute must follow the instance
    # that runs, hence the check after each fit rather than for both at once.
    m1.fit(_tfp_model(m1), obs={"y": y}, num_samples=50, num_warmup=50,
           num_chains=2, progress_bar=False)
    assert m1.dist._rng is m1._rng

    m2.fit(_tfp_model(m2), obs={"y": y}, num_samples=50, num_warmup=50,
           num_chains=2, progress_bar=False)
    assert m2.dist._rng is m2._rng


def test_m16_summary_does_not_mutate_posteriors():
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
    _fit_tiny(m)

    keys = set(m.posteriors)
    assert {"a", "b"} <= keys

    m.summary(include=["a"])
    assert set(m.posteriors) == keys

    m.summary(exclude=["a"])
    assert set(m.posteriors) == keys


def test_m16_filter_regex_arviz():
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
    _fit_tiny(m)

    tab = m.summary(filter_regex="^a", backend="arviz")
    assert len(tab) > 0
    assert all(str(i).startswith("a") for i in tab.index)


def test_m16_group_by_chain_arviz_warns():
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
    _fit_tiny(m)

    with pytest.warns(UserWarning, match="group_by_chain"):
        m.summary(group_by_chain=False, backend="arviz")
