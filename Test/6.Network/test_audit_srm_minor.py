"""Regression tests for RELEASE_AUDIT.md, module 04 (Models/SRM.py), minors S1-S7.

Each test fails on the pre-audit copy and passes after the fix; S1 additionally
keeps the pre-audit triple loop here as the reference for dat_B (the loop is
deliberately NOT kept in SRM.py).
"""
import ast
import inspect
import textwrap

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from BayesForge.Models.SRM import SRM, logit


def _srm(n=6):
    net = jnp.asarray(np.random.default_rng(0).integers(0, 2, size=(n, n)))
    return SRM(net)


def _dat_B_reference(block_predictors):
    """Pre-audit body of SRM.import_block_predictors (triple .at[].set loop)."""
    block_predictors = jnp.asarray(block_predictors)
    N_vars = block_predictors.shape[1]
    N_groups_per_var = (block_predictors.max(axis=0) + 1).astype(int)
    max_N_groups_per_var = max(N_groups_per_var)
    N_per_group = jnp.ones((N_vars, max_N_groups_per_var))
    for a in range(N_vars):
        N_per_group = N_per_group.at[a, :N_groups_per_var[a]].set(
            jnp.unique(block_predictors[:, a], return_counts=True)[1])
    N_pars = (N_groups_per_var * N_groups_per_var).sum()

    B_V = B_I = B_J = B_In = B_Base = B_SS = jnp.zeros(
        (N_vars, max_N_groups_per_var, max_N_groups_per_var))
    for v in range(N_vars):
        for i in range(N_groups_per_var[v]):
            for j in range(N_groups_per_var[v]):
                B_V = B_V.at[v, i, j].set(v)
                B_I = B_I.at[v, i, j].set(i)
                B_J = B_J.at[v, i, j].set(j)
                B_In = B_In.at[v, i, j].set(1)
                B_Base = B_Base.at[v, i, j].set(0.1 if i == j else 0.01)
                B_SS = B_SS.at[v, i, j].set(
                    jnp.sqrt(N_per_group[v, i] * 0.5 + N_per_group[v, j] * 0.5))

    dat_B = jnp.stack([B_V.flatten(), B_I.flatten(), B_J.flatten(), B_In.flatten(),
                       B_Base.flatten(), B_SS.flatten()], axis=-1)
    dat_B = dat_B[jnp.where(B_In.flatten() == 1)]
    Sigma = jnp.full((N_pars,), 2.5)
    Mu = logit(dat_B[:, -2] / dat_B[:, -1])
    return jnp.concatenate([dat_B, Mu[:, None], Sigma[:, None]], axis=1)


def _max_for_depth(node, depth=0):
    """Deepest nesting of ``For`` nodes below ``node`` (counting ``node``)."""
    best = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.For):
            best = max(best, _max_for_depth(child, depth + 1))
        else:
            best = max(best, _max_for_depth(child, depth))
    return best


def _tree(func):
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


# S1 ---------------------------------------------------------------------------

def test_s1_dat_B_is_bit_identical_to_the_reference_loop():
    bp = jnp.asarray(np.array([[0, 0], [1, 1], [2, 0], [0, 1], [1, 0], [2, 1]]))
    s = _srm(6)
    s.import_block_predictors(bp)
    np.testing.assert_array_equal(np.asarray(s.dat_B), np.asarray(_dat_B_reference(bp)))


def test_s1_import_block_predictors_has_no_triple_loop():
    assert _max_for_depth(_tree(SRM.import_block_predictors)) < 3


# S2 ---------------------------------------------------------------------------

def test_s2_block_prior_does_not_ask_for_float64():
    assert "jnp.float64" not in inspect.getsource(SRM.block_prior)


# S3 ---------------------------------------------------------------------------

def test_s3_core_model_has_no_dead_if_else():
    dead = [node.lineno for node in ast.walk(_tree(SRM._numpyro_bsrm_model_core))
            if isinstance(node, ast.If) and node.orelse
            and ast.dump(ast.Module(body=node.body, type_ignores=[]))
            == ast.dump(ast.Module(body=node.orelse, type_ignores=[]))]
    assert dead == []


# S4 ---------------------------------------------------------------------------

def test_s4_logit_is_finite_at_the_endpoints():
    assert np.isfinite(float(logit(0.0)))
    assert np.isfinite(float(logit(1.0)))
    assert np.isclose(float(logit(0.3)), float(np.log(0.3 / 0.7)), rtol=1e-5)


def test_s4_logit_interior_values_unchanged():
    """The endpoint clip must not move values away from 0 and 1."""
    x = jnp.asarray([0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999])
    assert jnp.allclose(logit(x), jnp.log(x / (1.0 - x)), atol=1e-6)


# S5 ---------------------------------------------------------------------------

@pytest.mark.parametrize("labels", [
    [1, 2, 2, 3, 1, 3],  # 1-based
    [0, 2, 2, 0, 2, 0],  # gap: label 1 missing
])
def test_s5_labels_must_be_zero_based_contiguous(labels):
    labels = np.asarray(labels)
    for importer in (SRM.import_block_predictors, SRM.import_block_predictors2):
        s = _srm(6)
        with pytest.raises(ValueError, match="0-based contiguous") as excinfo:
            importer(s, jnp.asarray(labels)[:, None])
        if labels.min() == 1:
            assert "index1=True" in str(excinfo.value)


def test_s5_valid_labels_still_import():
    s = _srm(6)
    s.import_block_predictors(jnp.asarray(np.array([[1], [1], [2], [2], [1], [2]])), index1=True)
    np.testing.assert_array_equal(np.asarray(s.block_predictors).ravel(),
                                  np.array([0, 0, 1, 1, 0, 1]))
    s2 = _srm(6)
    s2.import_block_predictors2(jnp.asarray(np.array([[1], [1], [2], [2], [1], [2]])), index1=True)
    np.testing.assert_array_equal(np.asarray(s2.block_predictors)[:, 1],
                                  np.array([0, 0, 1, 1, 0, 1]))


# S6 ---------------------------------------------------------------------------

def test_s6_signed_column_is_not_read_as_a_column_without_groups():
    s = _srm(6)
    s.block_predictors = jnp.array([[1, 0], [1, 1], [1, 0], [1, 1], [1, 0], [1, 1]])
    s.prepare_block_base_rate()
    assert s.blocks_base_rate[0].shape == (1, 1)  # constant column: one group
    assert s.blocks_base_rate[1].shape == (2, 2)

    s2 = _srm(6)
    s2.block_predictors = jnp.array([[1, -1], [1, 1], [1, -1], [1, 1], [1, -1], [1, 1]])
    s2.prepare_block_base_rate()
    assert s2.blocks_base_rate[0].shape == (1, 1)
    # (-1, +1) sums to 0 but holds two groups: on the pre-audit copy this is (1, 1)
    assert s2.blocks_base_rate[1].shape == (2, 2)


# S7 ---------------------------------------------------------------------------

def test_s7_block_importers_document_each_other():
    doc1 = SRM.import_block_predictors.__doc__ or ""
    doc2 = SRM.import_block_predictors2.__doc__ or ""
    assert "import_block_predictors2" in doc1
    assert "import_block_predictors" in doc2.replace("import_block_predictors2", "")
