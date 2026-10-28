"""m.summary() leaves out factor sites.

A likelihood written as a factor, m.dist.unit(log_factor=..., name=...), is
sampled as an empty array per draw. az.summary could not reduce it and
m.summary() raised "cannot reshape array of size 0" (the NBDA examples of the
documentation use such factors)."""
import jax.numpy as jnp
import numpy as np

from BayesForge import bf

Y = jnp.array(np.random.default_rng(0).normal(1.0, 1.0, 50))


def _factor_model(m):
    def model(y):
        mu = m.dist.normal(0, 5, name="mu")
        log_lik = jnp.sum(-0.5 * (y - mu) ** 2)
        m.dist.unit(log_factor=log_lik, name="y_lik")
    return model


def test_summary_skips_factor_sites():
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)
    m.fit(_factor_model(m), obs=dict(y=Y), num_warmup=100, num_samples=100,
          num_chains=1, progress_bar=False)
    summary = m.summary()
    assert list(summary.index) == ["mu"]
    assert abs(summary.loc["mu", "mean"] - float(Y.mean())) < 0.5
