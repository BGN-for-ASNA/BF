"""Regression tests for RELEASE_AUDIT.md, module Network, minors m6-m10, m16, m18."""
import ast
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest


def _source(module):
    """Source text of an already-imported module."""
    return Path(module.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# m6 — an orphaned second @staticmethod double-decorated block_model: it works
# on Python >= 3.10 (staticmethod is callable) but raises TypeError at import
# on <= 3.9, and Neteffect.block_model ends up wrapped twice.
# --------------------------------------------------------------------------- #
def test_m6_block_model_carries_a_single_staticmethod_decorator():
    import BayesForge.Network.model_effects as me

    tree = ast.parse(_source(me))
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "Neteffect")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "block_model")
    decorators = [d.id for d in fn.decorator_list if isinstance(d, ast.Name)]
    assert decorators.count("staticmethod") == 1, fn.decorator_list


def test_m6_block_model_is_a_plain_static_function():
    from BayesForge.Network.model_effects import Neteffect

    # A single staticmethod unwraps to the function on class access; the old
    # staticmethod(staticmethod(f)) double wrap returned the inner object.
    assert inspect.isfunction(Neteffect.block_model)


# --------------------------------------------------------------------------- #
# m7 — dyadic_effect without dyadic_predictors nor shape used to print and
# RETURN the error string, which then propagated as a str into the logit sum.
# --------------------------------------------------------------------------- #
def test_m7_dyadic_effect_without_predictors_or_shape_raises():
    from BayesForge.Network.model_effects import Neteffect

    with pytest.raises(ValueError, match="shape must be defined"):
        Neteffect.dyadic_effect()


# --------------------------------------------------------------------------- #
# m8 — net vs net2 contract drift on 2-D input: net reads 2-D as an edgelist
# (E, K) with one shared scalar, net2 reads it as an (N, N) predictor matrix.
# Aligning would change valid outputs, so the drift is documented instead;
# net2 must also reject a non-triangular shape instead of rounding N.
# --------------------------------------------------------------------------- #
def test_m8_net_dyadic_effect_documents_2d_contract_drift():
    from BayesForge.Network.model_effects import Neteffect
    doc = Neteffect.dyadic_effect.__doc__
    assert "m.net2" in doc and "(N, N)" in doc


def test_m10_dyad_vec_to_mat_still_a_net_method():
    from BayesForge.Network.util import array_manip, dyad_vec_to_mat
    assert array_manip.dyad_vec_to_mat is dyad_vec_to_mat


def test_m8_net_dyadic_effect_still_accepts_3d_edgelist():
    # The SRM path (SRM.import_dyadic_predictors builds (N_dyads, 2, K))
    # must keep working after the 2-D guard.
    from BayesForge.Network.model_effects import Neteffect

    out = Neteffect.dyadic_effect(dyadic_predictors=jnp.ones((6, 2, 1)), sample=True)
    assert out.shape == (6, 2)


def test_m8_net2_dyadic_effect_rejects_non_triangular_shape():
    from BayesForge.Network.model_effects2 import NeteffectMatrix

    with pytest.raises(ValueError, match=r"N\*\(N-1\)/2"):
        NeteffectMatrix.dyadic_effect(shape=5)


def test_m8_net2_dyadic_effect_accepts_triangular_shape():
    from BayesForge.Network.model_effects2 import NeteffectMatrix

    out = NeteffectMatrix.dyadic_effect(shape=6, sample=True)  # 6 = 4*3/2
    assert out.shape == (4, 4)


# --------------------------------------------------------------------------- #
# m9 — nodal_random_effects group codes are 1-based; a 0-based vector used to
# wrap to the last group via JAX gather. Already fixed by commit 456039cf
# (W7 bounds check); kept here as a regression test.
# --------------------------------------------------------------------------- #
def test_m9_nodal_random_effects_rejects_zero_based_group_index():
    from BayesForge.Network.model_effects import Neteffect

    with pytest.raises(ValueError, match="1-based"):
        Neteffect.nodal_random_effects(
            design_random=jnp.eye(2),
            random_group_index=jnp.array([0, 1]),
            num_random=2,
            num_random_groups=2,
        )


# --------------------------------------------------------------------------- #
# m10 — dyad_vec_to_mat was defined twice in util.py (staticmethod + module
# function) and a wrong vector length surfaced as an opaque JAX broadcast
# error. One definition now, with an explicit length check per convention.
# --------------------------------------------------------------------------- #
def test_m10_util_defines_dyad_vec_to_mat_exactly_once():
    import BayesForge.Network.util as util

    tree = ast.parse(_source(util))
    defs = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "dyad_vec_to_mat"]
    assert len(defs) == 1


def test_m10_dyad_vec_to_mat_wrong_length_raises():
    from BayesForge.Network.util import dyad_vec_to_mat

    with pytest.raises(ValueError, match=r"length 5 is not N\*\(N-1\)/2"):
        dyad_vec_to_mat(jnp.zeros(5), symmetric=True)


def test_m10_dyad_vec_to_mat_directed_wrong_length_raises():
    from BayesForge.Network.util import dyad_vec_to_mat

    with pytest.raises(ValueError, match=r"length 5 is not N\*\(N-1\)"):
        dyad_vec_to_mat(jnp.zeros(5))


def test_m10_dyad_vec_to_mat_roundtrips_known_3x3_symmetric():
    from BayesForge.Network.util import dyad_vec_to_mat

    A = jnp.array([[0.0, 1.0, 2.0],
                   [1.0, 0.0, 3.0],
                   [2.0, 3.0, 0.0]])
    vec = A[jnp.triu_indices(3, 1)]  # length 3 = N*(N-1)/2
    assert jnp.allclose(dyad_vec_to_mat(vec, symmetric=True), A)


def test_m10_dyad_vec_to_mat_roundtrips_known_3x3_directed():
    from BayesForge.Network.util import dyad_vec_to_mat

    A = jnp.array([[0.0, 1.0, 2.0],
                   [3.0, 0.0, 4.0],
                   [5.0, 6.0, 0.0]])
    upper = A[jnp.triu_indices(3, 1)]      # i->j: [1, 2, 4]
    lower = A.T[jnp.triu_indices(3, 1)]    # j->i: [3, 5, 6]
    vec = jnp.concatenate([upper, lower])  # length 6 = N*(N-1)
    assert jnp.allclose(dyad_vec_to_mat(vec), A)


# --------------------------------------------------------------------------- #
# m16 — dead state and hygiene: self._cache never read/written.
# Imports and commented-out code are now checked by ruff in CI (F401, F811, ERA001).
# --------------------------------------------------------------------------- #
def test_m16_no_dead_cache_attribute_in_model_effects():
    import BayesForge.Network.model_effects as me

    assert "_cache" not in _source(me)


# --------------------------------------------------------------------------- #
# m18 — the net2 usage docstring masked the diagonal by multiplying the logits
# with (1 - eye): that sets the diagonal to logit 0 -> p = 0.5 (a constant
# -N*log 2 contribution, contradicting a 1 in Y_mat), not an exclusion.
# --------------------------------------------------------------------------- #
def test_m18_net2_usage_example_masks_diagonal_with_where():
    import BayesForge.Network.model_effects2 as me2

    doc = me2.__doc__ or ""
    assert "* (1 - jnp.eye" not in doc
    assert "jnp.where(jnp.eye(N, dtype=bool), -jnp.inf, logits)" in doc


def test_m18_zeroing_logits_is_not_an_exclusion():
    # Numerical reason the old example was wrong: zeroed logits give p = 0.5
    # on the diagonal, -inf masking gives p = 0 (zero log-likelihood term).
    logits = jnp.zeros((3, 3))
    p_mult = jax.nn.sigmoid(logits * (1 - jnp.eye(3))).diagonal()
    assert jnp.allclose(p_mult, 0.5)
    p_where = jax.nn.sigmoid(jnp.where(jnp.eye(3, dtype=bool), -jnp.inf, logits))
    assert jnp.allclose(p_where.diagonal(), 0.0)
