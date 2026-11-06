"""BNN output layers for mixture density networks: layer_cholesky, layer_mixture_normal.

Run with:  pytest "Test/10.BNN/test_mixture_layers.py" -q
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as npd
import pytest

from BayesForge import bf


def _bf(seed=0):
    return bf(platform="cpu", rand_seed=seed, print_devices_found=False)


def test_layer_cholesky_batched_lower_triangular_positive_diagonal():
    m = _bf()
    X = jax.random.normal(jax.random.PRNGKey(0), (7, 5, 6))      # 6 = 3 * 4 / 2
    L = m.bnn.layer_cholesky(X, dim=3)
    assert L.shape == (7, 5, 3, 3)
    assert jnp.allclose(jnp.triu(L, k=1), 0.0)
    assert jnp.all(jnp.diagonal(L, axis1=-2, axis2=-1) > 0)
    # Off-diagonal entries are copied as is, in tril_indices order
    rows, cols = jnp.tril_indices(3)
    off = rows != cols
    assert jnp.allclose(L[..., rows[off], cols[off]], X[..., off])
    # Valid covariance: Cholesky of L L^T recovers L
    S = L @ jnp.swapaxes(L, -1, -2)
    assert jnp.allclose(jnp.linalg.cholesky(S), L, atol=1e-4)


def test_layer_cholesky_diag_offset_and_size_check():
    m = _bf()
    L = m.bnn.layer_cholesky(jnp.zeros(3), dim=2, diag_offset=0.5413, min_diag=0.0)
    assert jnp.allclose(jnp.diag(L), 1.0, atol=1e-3)               # softplus(0.5413) ~ 1
    with pytest.raises(ValueError):
        m.bnn.layer_cholesky(jnp.zeros(4), dim=2)


def test_layer_mixture_normal_matches_manual_mixture():
    m = _bf()
    K, D = 3, 2
    size = m.bnn.mixture_normal_size(K, D)
    assert size == K * (1 + D + D * (D + 1) // 2)
    X = jax.random.normal(jax.random.PRNGKey(1), (10, size))
    q = m.bnn.layer_mixture_normal(X, K, D)
    u = jax.random.normal(jax.random.PRNGKey(2), (10, D))

    logits = X[:, :K]
    loc = X[:, K:K + K * D].reshape(10, K, D)
    L = m.bnn.layer_cholesky(X[:, K + K * D:].reshape(10, K, 3), D)
    ref = npd.MixtureSameFamily(npd.Categorical(logits=logits),
                                npd.MultivariateNormal(loc, scale_tril=L))
    assert q.batch_shape == (10,) and q.event_shape == (D,)
    assert jnp.allclose(q.log_prob(u), ref.log_prob(u), atol=1e-5)
    with pytest.raises(ValueError):
        m.bnn.layer_mixture_normal(X[:, :-1], K, D)


def test_layer_mixture_normal_sampling_is_reproducible_with_model_seed():
    K, D = 2, 3
    X = jnp.zeros(_bf().bnn.mixture_normal_size(K, D))
    draws = [_bf(seed=0).bnn.layer_mixture_normal(X, K, D, sample=True, shape=(500,))
             for _ in range(2)]
    assert draws[0].shape == (500, D)
    assert jnp.allclose(draws[0], draws[1])
    other = _bf(seed=1).bnn.layer_mixture_normal(X, K, D, sample=True, shape=(500,))
    assert not jnp.allclose(draws[0], other)
    # zeros -> components N(0, softplus(0) + 1e-3); sd ~ 0.69
    assert abs(float(np.std(draws[0])) - (np.log(2) + 1e-3)) < 0.06


def test_layer_mixture_normal_as_likelihood_trained_with_svi_and_sampled():
    """A mixture density network written as a BF model: name=/obs= makes the
    output layer a likelihood; m.svi fits it; m.sample(posterior=dict) predicts."""
    from numpyro.infer import Trace_ELBO
    from numpyro.infer.autoguide import AutoDelta
    from numpyro.infer.initialization import init_to_sample
    import numpyro

    m = _bf()
    K, D = 2, 1
    c = m.dist.normal(0, 1, sample=True, shape=(500, 1))
    u = 2.0 * c + 0.1 * m.dist.normal(0, 1, sample=True, shape=(500, 1))   # u | c ~ N(2c, 0.1)

    def net(c, u=None, batch_size=None):
        W = m.dist.normal(0, 1, shape=(1, m.bnn.mixture_normal_size(K, D)), name="W")
        bias = m.dist.normal(0, 1, shape=(m.bnn.mixture_normal_size(K, D),), name="b")
        with m.dist.plate("rows", c.shape[0], subsample_size=batch_size) as idx:
            h = m.bnn.layer_linear(c[idx], W, bias=bias, verbose=False)
            m.bnn.layer_mixture_normal(h, K, D, name="u", obs=u if u is None else u[idx])

    m.data_on_model = {"c": c, "u": u, "batch_size": 100}
    m.svi(net, guide=AutoDelta(net, init_loc_fn=init_to_sample),
          optim=numpyro.optim.Adam(1e-2), loss=Trace_ELBO(), num_steps=3000, num_samples=200)
    # AutoDelta: every stored draw is the same point estimate; a posterior dict is
    # one fixed draw, repeated `samples` times.
    weights = {k: v[0] for k, v in m.posteriors.items()}
    draws = m.sample(posterior=weights, data={"c": jnp.array([[1.0]]), "u": None, "batch_size": None},
                     samples=200)["u"]
    assert draws.shape == (200, 1, 1)
    assert abs(float(draws.mean()) - 2.0) < 0.2
