import jax.interpreters.pxla  # noqa: F401  (pre-import avoids the TFP/JAX patch clash)

import numpy as np

from BayesForge import bf


def test_zip_gate_convention_zero_inflation_probability():
    """zero_inflated_poisson(gate=g) means P(y=0) = g + (1-g)*exp(-rate)."""
    m = bf(platform="cpu", print_devices_found=False)
    g, lam = 0.3, 2.0
    y = np.asarray(m.dist.zero_inflated_poisson(
        gate=g, rate=lam, sample=True, shape=(1000,), seed=0))
    p0 = g + (1 - g) * np.exp(-lam)
    assert abs(np.mean(y == 0) - p0) < 3 * np.sqrt(p0 * (1 - p0) / 1000)
