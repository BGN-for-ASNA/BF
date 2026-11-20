"""Regression tests for RELEASE_AUDIT.md, module 04 (Models/SRM.py), critical C1-C4."""
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import pytest

from BayesForge.Models.SRM import SRM


def _srm(n=4):
    net = jnp.asarray(np.random.default_rng(0).integers(0, 2, size=(n, n)))
    return SRM(net)


def _groups_covered(s, labels0):
    """For each dyad, the block parameters that fire; returns set of (i_group, j_group)."""
    covered = set()
    dat = np.asarray(s.dat_B[:, :3]).astype(int)
    for p, (v, gi, gj) in enumerate(dat):
        if np.asarray(s.Block_edgl[:, :, p]).sum() > 0:
            covered.add((gi, gj))
    return covered


@pytest.mark.parametrize("labels,index1", [
    (np.array([[1], [1], [2], [2]]), True),
    (np.array([[0], [0], [1], [1]]), False),
])
def test_c1_every_group_pair_is_modelled(labels, index1):
    s = _srm()
    s.import_block_predictors(jnp.asarray(labels), index1=index1)
    assert s.N_pars == 4  # 2 groups -> 2 x 2 block parameters
    assert _groups_covered(s, labels) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    np.testing.assert_array_equal(np.asarray(s.block_predictors), np.array([[0], [0], [1], [1]]))


def test_c2_block_predictors2_index1_keeps_intercept():
    s = _srm()
    raw = jnp.asarray(np.array([[1], [1], [2], [2]]))
    s.import_block_predictors2(raw, index1=True)
    bp = np.asarray(s.block_predictors)
    assert bp.shape == (4, 2)
    np.testing.assert_array_equal(bp[:, 0], np.ones(4))
    np.testing.assert_array_equal(bp[:, 1], np.array([0, 0, 1, 1]))


def _trace_bsrm(s, block_predictors, outcome_mode=1):
    n = 4
    ids = jnp.arange(n)
    S, R = jnp.meshgrid(ids, ids)
    n_dyads = int((S != R).sum())
    y = jnp.zeros(n_dyads)
    focal = jnp.ones((n, 1))
    with numpyro.handlers.seed(rng_seed=0):
        tr = numpyro.handlers.trace(s.numpyro_bsrm_model).get_trace(
            y, focal, focal, None, block_predictors, None, outcome_mode=outcome_mode)
    return tr


def test_c3_bsrm_model_supports_three_groups_and_two_block_variables():
    s = _srm()
    bp = jnp.asarray(np.array([[0, 1], [1, 0], [2, 1], [2, 0]]))
    tr = _trace_bsrm(s, bp)
    # var 0 has 3 groups, var 1 has 2 groups -> 5 block effects
    assert tr["block_effects"]["value"].shape == (5,)


@pytest.mark.parametrize("mode", [0, 5, None])
def test_c4_invalid_outcome_mode_raises(mode):
    s = _srm()
    with pytest.raises(ValueError):
        _trace_bsrm(s, None, outcome_mode=mode)


def test_c4_valid_outcome_mode_has_likelihood():
    s = _srm()
    tr = _trace_bsrm(s, None, outcome_mode=1)
    assert any(site["is_observed"] for site in tr.values() if site["type"] == "sample")


# Majors S1-S4 ---------------------------------------------------------------

def test_s3_bad_network_raises():
    with pytest.raises(ValueError):
        SRM(jnp.zeros((3, 3, 2)))


def test_s3_bad_exposure_raises():
    s = _srm(4)
    with pytest.raises(ValueError):
        s.import_exposure(jnp.ones((2, 2)))


def test_s2_square_dyadic_matrix_is_accepted():
    s = _srm(4)
    s.import_dyadic_predictors(jnp.ones((4, 4)))
    assert s.dyadic_predictors is not None
    assert s.dyadic_predictors.shape[:2] == (s.N_dyad, 2)


def test_s2_wrong_dyadic_shape_raises():
    s = _srm(4)
    with pytest.raises(ValueError):
        s.import_dyadic_predictors(jnp.ones((s.N_dyad, 2)))


def test_s1_block_padding_is_finite():
    s = _srm(4)
    s.blocks_mu_ij = [jnp.full((2, 2), 0.3), jnp.full((5, 5), 0.2)]
    s.prepare_block_effects()
    assert bool(jnp.isfinite(s.blocks_mu_ij_padded).all())
