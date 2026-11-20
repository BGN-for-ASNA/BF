"""Regression tests for RELEASE_AUDIT.md, module 12 (BNN), critical CRIT-1/CRIT-2."""
import numpy as np
import jax.numpy as jnp
import numpyro
import pytest

from BayesForge.BNN.bnn import bnn


def test_crit1_call_before_attention_raises_clear_error():
    net = bnn(rand_seed=0)
    with pytest.raises(RuntimeError):
        net(jnp.ones((3, 8)))


def test_crit1_call_after_attention_runs():
    net = bnn(rand_seed=0)
    with numpyro.handlers.seed(rng_seed=0):
        net.layer_attention(b_kv=3, b_q=2, d_model=8, sample=False, name='t')
    out, w = net(jnp.ones((5, 8)))
    assert out.shape == (5, 8)
    assert w.shape == (5, 5)
    np.testing.assert_allclose(np.asarray(w.sum(-1)), 1.0, rtol=1e-5)


@pytest.mark.parametrize("N", [2, 5])
def test_crit2_cov_offsets_align_with_rows(N):
    net = bnn(rand_seed=0)
    a = jnp.arange(N, dtype=jnp.float32)
    b = -10.0 * jnp.arange(N, dtype=jnp.float32)
    with numpyro.handlers.seed(rng_seed=0):
        tr = numpyro.handlers.trace(lambda: net.cov(4, N, a, b)).get_trace()
    W1 = tr['W1']['value']; W2 = tr['W2']['value']
    delta = jnp.tanh(jnp.eye(N) @ W1) @ W2
    rf = tr['rf']['value']
    assert rf.shape == (N, 2)
    np.testing.assert_allclose(np.asarray(rf[:, 0] - delta[:, 0]), np.asarray(a), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(rf[:, 1] - delta[:, 1]), np.asarray(b), rtol=1e-5, atol=1e-5)


def test_crit2_cov_scalar_offsets_still_work():
    net = bnn(rand_seed=0)
    with numpyro.handlers.seed(rng_seed=0):
        tr = numpyro.handlers.trace(lambda: net.cov(4, 3, 3.5, -1.0)).get_trace()
    assert tr['rf']['value'].shape == (3, 2)
