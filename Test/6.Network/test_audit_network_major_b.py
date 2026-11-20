"""Regression tests for RELEASE_AUDIT.md, module Network, majors M7-M9."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge.Network.model_effects import Neteffect
from BayesForge.Network.util import array_manip
from BayesForge.Network.vizfeatures import centralities


# --------------------------------------------------------------------------- #
# M7 — logit overflow + empty-group guard in block_build_mu_ij
# --------------------------------------------------------------------------- #
def test_m7_block_build_mu_ij_rejects_empty_group():
    with pytest.raises(ValueError, match="no member"):
        Neteffect.block_build_mu_ij(np.array([4, 0, 6]), 3)


def test_m7_logit_finite_near_one():
    # 1 - 1e-7 only differs from 1.0 in float64; force x64 for this check.
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        x = jnp.asarray([0.5, 1.0 - 1e-7])
        out = np.asarray(Neteffect.logit(x))
    finally:
        jax.config.update("jax_enable_x64", prev)
    assert np.all(np.isfinite(out))


def test_m7_block_prior_mean_finite_from_valid_sizes():
    mu = Neteffect.block_build_mu_ij(jnp.array([4.0, 6.0, 5.0]), 3)
    mu_ij = Neteffect.logit(jnp.clip(mu, 1e-6, 1 - 1e-6))
    assert mu.shape == (3, 3)
    assert bool(jnp.all(jnp.isfinite(mu_ij)))


# --------------------------------------------------------------------------- #
# M8 — mat_to_edgl validates shape and flags dropped self-ties
# --------------------------------------------------------------------------- #
def test_m8_mat_to_edgl_rejects_non_square():
    with pytest.raises(ValueError, match="square"):
        array_manip.mat_to_edgl(np.zeros((2, 3)))


def test_m8_mat_to_edgl_rejects_1d_input():
    with pytest.raises(ValueError, match="square"):
        array_manip.mat_to_edgl(np.arange(6.0))


def test_m8_mat_to_edgl_warns_on_nonzero_diagonal():
    A = np.array([[0.0, 1.0, 2.0], [3.0, 5.0, 4.0], [5.0, 6.0, 0.0]])
    with pytest.warns(UserWarning, match="diagonal"):
        out = array_manip.mat_to_edgl(A)
    assert out.shape == (3, 2)


def test_m8_mat_to_edgl_output_format_unchanged():
    A = np.array([[0.0, 1.0, 2.0], [3.0, 0.0, 4.0], [5.0, 6.0, 0.0]])
    out = np.asarray(array_manip.mat_to_edgl(A))
    expected = np.array([[1.0, 3.0], [2.0, 5.0], [4.0, 6.0]])  # 3 dyads
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, expected)


# --------------------------------------------------------------------------- #
# M9 — betweenness size guard in centralities (vis_net metrics=True)
# --------------------------------------------------------------------------- #
def test_m9_betweenness_skipped_above_limit(monkeypatch):
    A = np.zeros((5, 5))
    for i in range(4):
        A[i, i + 1] = A[i + 1, i] = 1.0
    monkeypatch.setenv("BF_BETWEENNESS_MAX_N", "3")
    with pytest.warns(UserWarning, match="betweenness skipped"):
        out = centralities(A)
    assert "betweenness" not in out
    assert "degree" in out
    assert out["degree"].shape == (5,)
