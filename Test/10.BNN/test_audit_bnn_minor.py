"""Regression tests for RELEASE_AUDIT.md, module BNN, minors."""
import inspect
import re

import jax
import jax.numpy as jnp
import numpyro
import pytest

from BayesForge.BNN import bnn as bnn_pkg
from BayesForge.BNN.bnn import bnn


# ---------------------------------------------------------------- MIN-1
def test_min1_elu_stable_for_large_negative_input():
    from BayesForge.BNN.activations import activation

    x = jnp.array([-1000.0, -500.0, -20.0, 0.0, 3.0])
    out = activation.elu(x, alpha=1.0)
    assert jnp.all(jnp.isfinite(out))
    # elu(x) -> -alpha for x -> -inf; identity for x > 0.
    expected = jnp.array([-1.0, -1.0, -1.0, 0.0, 3.0])
    assert jnp.allclose(out, expected, atol=1e-6)
    # expm1 form is accurate near 0: the naive exp(x) - 1 rounds to exactly
    # 0.0 in float32 for tiny negative x, losing the negative branch.
    tiny = activation.elu(jnp.array([-1e-8]), alpha=1.0)
    assert float(tiny[0]) < 0.0
    assert jnp.allclose(tiny, jnp.array([-1e-8]), rtol=1e-3)


# ---------------------------------------------------------------- MIN-2
def test_min2_make_pd_escalates_jitter_on_non_pd_matrix():
    net = bnn()
    A = jnp.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric, eigenvalues 3 and -1
    with pytest.raises(ValueError, match="positive definite"):
        net.make_pd_and_cholesky_jax(A)
    # A well-conditioned PD matrix still works.
    B = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    L = net.make_pd_and_cholesky_jax(B)
    assert jnp.all(jnp.isfinite(L))


# ---------------------------------------------------------------- MIN-3
def test_min3_log_space_comments_match_log_variance_names():
    # The audit cites the "stored in log space" comments of the three
    # covariance-block layers; the parameters are raw draws mapped to
    # positive values by softplus, so the comment must not claim log space.
    for name, attr in (("layer_toeplitz", "self.log_sigma"),
                       ("layer_compound_symmetry", "self.log_sigma"),
                       ("layer_diagonal", "self.log_variances")):
        src = inspect.getsource(getattr(bnn, name))
        assert "log space" not in src, name
        # The attributes are still named log_* (no rename).
        assert attr in src, name


# ---------------------------------------------------------------- MIN-4
def test_min4_layer_linear_rejects_incompatible_shapes():
    net = bnn()
    X = jnp.zeros((5, 3))
    W = jnp.zeros((4, 2))
    with pytest.raises(ValueError, match="last dimension of X"):
        net.layer_linear(X, W, verbose=False)
    # Compatible shapes still work (identity path, activation=None).
    out = net.layer_linear(jnp.zeros((5, 4)), W, verbose=False)
    assert out.shape == (5, 2)


# ---------------------------------------------------------------- MIN-5
def test_min5_cov_has_no_eye_matrix_multiplication():
    src = inspect.getsource(bnn.cov)
    assert "jnp.eye" not in src
    assert "jnp.tanh(W1)" in src


# ---------------------------------------------------------------- MIN-6
def test_min6_layer_attention_keeps_batch_dim():
    net = bnn()
    b_kv, b_q, d_model, B = 3, 2, 8, 4
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(0), 4)
    emb_q = jax.random.normal(k1, (B, b_q, d_model))
    emb_kv = jax.random.normal(k2, (B, b_kv, d_model))
    proj = jax.random.normal(k3, (B, d_model, d_model, 3))
    out_w = jax.random.normal(k4, (B, d_model * b_q, b_q * b_kv))
    # Explicit weights: no draw inside the call, so vmap works (a draw
    # would resolve a tracer seed). The batch dimension must be kept.
    res = jax.vmap(
        lambda q, kv, p, o: net.layer_attention(b_kv, b_q, d_model=d_model,
                                                weights=(q, kv, p, o))
    )(emb_q, emb_kv, proj, out_w)
    assert res.shape == (B, b_q, b_kv)
    # The historical unbatched call shape is unchanged and matches the
    # corresponding slice of the batched result.
    res1 = net.layer_attention(b_kv, b_q, d_model=d_model,
                               weights=(emb_q[0], emb_kv[0], proj[0], out_w[0]))
    assert res1.shape == (b_q, b_kv)
    assert jnp.allclose(res[0], res1, atol=1e-4)


# ---------------------------------------------------------------- MIN-8
def test_min8_cov_prefix_avoids_plate_name_collisions():
    net = bnn()
    N, a, b = 3, 3.5, -1.0

    def model():
        net.cov(4, N, a, b)
        net.cov(4, N, a, b, prefix="gp")

    # Latent sites need a seeded handler; a bare trace tries to draw with
    # a None PRNG key.
    tr = numpyro.handlers.trace(
        numpyro.handlers.seed(model, jax.random.PRNGKey(0)))
    tr()
    names = {k for k, v in tr.trace.items() if v["type"] == "sample"}
    assert {"W1", "W2", "gp/W1", "gp/W2"} <= names


# ---------------------------------------------------------------- MIN-9
def test_min9_sample_flag_documented_in_layer_docstrings():
    for name in ("layer_attention", "layer_compound_symmetry", "layer_diagonal"):
        doc = inspect.getdoc(getattr(bnn, name)) or ""
        assert "sample=True" in doc, name
        assert "sample=False" in doc, name


# ---------------------------------------------------------------- MIN-10
def test_min10_layer_linear_rejects_bias_none():
    net = bnn()
    X = jnp.zeros((2, 4))
    W = jnp.zeros((4, 3))
    with pytest.raises(ValueError, match="bias=None"):
        net.layer_linear(X, W, bias=None, verbose=False)
    # An explicit array bias still works.
    out = net.layer_linear(X, W, bias=jnp.zeros(3), verbose=False)
    assert out.shape == (2, 3)


# ---------------------------------------------------------------- MIN-11
def test_min11_layer_linear_dist_documented_as_weights():
    # MIN-11: the weight parameter is named `dist` (kept for backward
    # compatibility); the docstring must say it is a weight matrix, not a
    # distribution object, and give the (in, out) convention.
    doc = inspect.getdoc(bnn.layer_linear) or ""
    assert "dist (jnp.ndarray)" in doc
    assert "not a distribution" in doc
    # The parameter name itself is unchanged (no public API rename).
    assert "dist" in inspect.signature(bnn.layer_linear).parameters


# ---------------------------------------------------------------- MIN-12
def test_min12_mixture_sample_with_name_raises():
    net = bnn()
    X = jnp.zeros(6)
    with pytest.raises(ValueError, match="name"):
        net.layer_mixture_normal(X, 2, 1, sample=True, name="t")


def test_min12_mixture_obs_is_never_silently_ignored():
    net = bnn()
    X = jnp.zeros(6)
    y = jnp.zeros(1)
    # obs with sample=True: no site would be declared.
    with pytest.raises(ValueError, match="sample=False"):
        net.layer_mixture_normal(X, 2, 1, sample=True, obs=y)
    # obs without a site name: no likelihood would be declared.
    with pytest.raises(ValueError, match="obs requires name"):
        net.layer_mixture_normal(X, 2, 1, obs=y)
    # The docstring states the contract.
    assert "require" in net.layer_mixture_normal.__doc__
    assert "sample=False" in net.layer_mixture_normal.__doc__


def test_min12_mixture_guard_runs_before_cholesky(monkeypatch):
    net = bnn()
    def boom(*a, **k):
        raise AssertionError("layer_cholesky called before the argument guard")
    monkeypatch.setattr(net, "layer_cholesky", boom)
    with pytest.raises(ValueError):
        net.layer_mixture_normal(jnp.zeros(6), 2, 1, sample=True, name="t")


# ---------------------------------------------------------------- MIN-13
def test_min13_layer_toeplitz_has_docstring():
    fn = bnn.layer_toeplitz
    assert fn.__doc__
    assert "block_size" in fn.__doc__
    src = inspect.getsource(fn)
    doc = src.index('"""')  # first string literal right after the def
    warn = src.index("warn_dev")
    assert doc < warn


# ---------------------------------------------------------------- MIN-14
def test_min14_get_rho_documented():
    doc = inspect.getdoc(bnn.get_rho) or ""
    assert "rho" in doc


# ---------------------------------------------------------------- MIN-15
def test_min15_package_exposes_bnn_class():
    from BayesForge.BNN import bnn as exported

    assert exported is not None
    assert callable(exported)
    assert exported is bnn


# ---------------------------------------------------------------- MIN-16
def test_min16_layer_diagonal_does_not_store_block_size():
    src = inspect.getsource(bnn.layer_diagonal)
    assert "self.b" not in src
    # layer_toeplitz still needs self.b.
    assert "self.b = block_size" in inspect.getsource(bnn.layer_toeplitz)


def test_min2_make_pd_cholesky_under_jit():
    """MIN-2 follow-up: under jit the escalation must not raise a false ValueError."""
    import jax
    import jax.numpy as jnp
    from BayesForge.BNN.bnn import bnn as _bnn_cls
    net = _bnn_cls()
    A = jnp.array([[1.0, 0.999999], [0.999999, 1.0]])
    L = jax.jit(net.make_pd_and_cholesky_jax)(A)
    assert bool(jnp.all(jnp.isfinite(L)))
