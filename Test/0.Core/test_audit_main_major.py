"""Regression tests for RELEASE_AUDIT.md, module Main, majors M-07, M-11, M-13, M-14."""
import jax
import jax.numpy as jnp
import pytest

from BayesForge import bf


def _tiny_model(m):
    """Smallest conditioning model: one scalar site, ten observations."""

    def model(y):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y)

    return model


# M-14 ----------------------------------------------------------------------
# Ensure fit(seed) accepts jax PRNGKey
def test_m14_prngkey_seed_accepted():
    """fit(seed=jax.random.PRNGKey(...)) must run, not crash on int(key)."""
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
    model = _tiny_model(m)
    m.fit(model, obs={"y": jnp.zeros(10)}, seed=jax.random.PRNGKey(0),
          num_samples=50, num_warmup=50, num_chains=1, progress_bar=False)
    assert "mu" in m.posteriors
    key = m._resolve_key(jax.random.PRNGKey(3), "fit")
    assert jnp.array_equal(key, jax.random.PRNGKey(3))

# Ensure fit(seed) does not accept non-integer seeds
def test_m14_bad_seeds_raise():
    """bool and fractional-float seeds raise TypeError; whole floats pass."""
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
    with pytest.raises(TypeError):
        m._resolve_key(True, "fit")
    with pytest.raises(TypeError):
        m._resolve_key(42.9, "fit")
    assert jnp.array_equal(m._resolve_key(42.0, "fit"), jax.random.PRNGKey(42))


# M-11 ----------------------------------------------------------------------
# Ensure fit() does not mistake user function names for built-in ones
def test_m11_user_function_named_pca():
    """A user function named 'pca' must not hit the built-in PCA branch."""
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)

    def pca(y=None):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y)

    m.data_on_model = dict(y=jnp.zeros(10))
    m.fit(pca, num_samples=50, num_warmup=50, num_chains=1, progress_bar=False)
    assert "mu" in m.posteriors
    assert m.run_model_name == "pca"  # kept as the display/save name

# Ensure fit() does not mistake user function names for built-in ones
def test_m11_user_function_named_gmm():
    """A user function named 'gmm' must not trigger the KMEANS warm start."""
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)

    def gmm(y=None):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y)

    m.data_on_model = dict(y=jnp.zeros(10))
    m.fit(gmm, num_samples=50, num_warmup=50, num_chains=1, progress_bar=False)
    assert "mu" in m.posteriors
    assert m.run_model_name == "gmm"


# M-13 ----------------------------------------------------------------------
# Ensure fit() does not mistake data keys for run() keywords
def test_m13_data_key_collision_raises():
    """A model argument named like a run() keyword fails with a clear error."""
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)

    def model(y, extra_fields=None):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y)

    m.data_on_model = dict(y=jnp.zeros(10), extra_fields=jnp.zeros(3))
    with pytest.raises(TypeError, match="collide"):
        m.fit(model, num_samples=50, num_warmup=50, num_chains=1,
              progress_bar=False)


# M-07 ----------------------------------------------------------------------
def test_m07_tfp_float64_kept():
    """backend='tfp' + float_precision=64 warns and keeps x64 on."""
    pytest.importorskip("tensorflow_probability")
    prev = jax.config.jax_enable_x64
    try:
        with pytest.warns(UserWarning, match="float64"):
            bf(backend="tfp", float_precision=64, platform="cpu",
               print_devices_found=False)
        assert jax.config.jax_enable_x64
    finally:
        jax.config.update("jax_enable_x64", prev)
