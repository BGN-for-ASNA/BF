"""Regression tests for the 9 minor audit points in BayesForge/Models/PCA.py.

Each test fails on the pre-audit copy and passes after the fix.
Run:
    JAX_PLATFORMS=cpu MPLBACKEND=Agg pytest Test/8.PCA/test_audit_pca_minor.py -q
"""
import inspect

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from BayesForge.Models import PCA as PCA_MODULE
from BayesForge.Models.PCA import pca


# ---------------------------------------------------------------------------
# P1: discarded/returned Warning objects became real ValueErrors
# ---------------------------------------------------------------------------
def test_p1_get_attributes_raises_value_error():
    obj = pca(type="classic")
    X = np.random.default_rng(0).normal(size=(10, 3))
    # Before the fix this returned a Warning instance instead of raising.
    with pytest.raises(ValueError, match="No posterior"):
        obj.get_attributes(X)


def test_p1_individual_correlation_raises_value_error():
    obj = pca(type="classic")
    # Before the fix the Warnings were constructed and discarded, and the
    # call died later with a TypeError on None.
    with pytest.raises(ValueError, match="No X found"):
        obj.individual_correlation()
    X = np.random.default_rng(1).normal(size=(10, 3))
    with pytest.raises(ValueError, match="No Bayesian PCA results"):
        obj.individual_correlation(X=X)


# ---------------------------------------------------------------------------
# P2: epsilon is a scale (standard deviation), not a variance
# ---------------------------------------------------------------------------
def test_p2_epsilon_is_documented_as_scale():
    src = inspect.getsource(PCA_MODULE)
    assert "noise variance" not in src
    assert "standard deviation" in src


# ---------------------------------------------------------------------------
# P3: _scale_sizes helper replaces the 0/0 inline min/max scaling
# ---------------------------------------------------------------------------
def test_p3_scale_sizes():
    # Imported inside the test so the other tests still run on the pre-audit
    # copy (the helper does not exist there).
    from BayesForge.Models.PCA import _scale_sizes

    const = _scale_sizes(jnp.array([3.0, 3.0, 3.0]))
    assert bool(jnp.all(jnp.isfinite(const)))
    assert bool(jnp.all(const == const[0]))
    ramp = _scale_sizes(jnp.array([0.0, 1.0]))
    assert bool(jnp.allclose(ramp, jnp.array([10.0, 110.0])))


# ---------------------------------------------------------------------------
# P4: confidence_ellipse survives a zero-variance group (no NaN radii)
# ---------------------------------------------------------------------------
def test_p4_confidence_ellipse_finite_for_constant_group():
    obj = pca(type="classic")
    fig, ax = plt.subplots()
    try:
        # x has zero variance: the old pearson was 0/0 -> NaN radii.
        patch = obj.confidence_ellipse(np.zeros(5), np.arange(5.0), ax)
        assert np.isfinite(patch.get_width())
        assert np.isfinite(patch.get_height())
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# P5: variable_correlation / individual_correlation need >= 2 components
# ---------------------------------------------------------------------------
def test_p5_two_component_guards_in_source():
    assert "at least 2 components" in inspect.getsource(pca.variable_correlation)
    assert "at least 2 components" in inspect.getsource(pca.individual_correlation)


def test_p5_one_component_raises_value_error():
    obj = pca(type="classic")
    X = np.random.default_rng(3).normal(size=(10, 4))
    results = {
        "components": jnp.ones((4, 1)),
        "explained_variance": jnp.ones((1,)),
        "explained_variance_ratio": jnp.ones((1,)),
    }
    # Before the fix both raised IndexError on ratio[1] / [1] of the ratio.
    with pytest.raises(ValueError, match="at least 2 components"):
        obj.variable_correlation(feature_names=["a", "b", "c", "d"], X=X,
                                 bayesian_results=results)
    with pytest.raises(ValueError, match="at least 2 components"):
        obj.individual_correlation(X=X, bayesian_results=results)


# ---------------------------------------------------------------------------
# P6: components() uses the show_plot pattern, no dead `if ax is None`
# ---------------------------------------------------------------------------
def test_p6_components_uses_show_plot():
    src = inspect.getsource(pca.components)
    assert src.count("if ax is None") == 1
    assert "show_plot" in src


# ---------------------------------------------------------------------------
# P7: __call__ default type matches the constructor default
# ---------------------------------------------------------------------------
def test_p7_call_default_type_matches_constructor():
    assert inspect.signature(pca.__call__).parameters["type"].default == "classic"
    assert inspect.signature(pca.__init__).parameters["type"].default == "classic"


# ---------------------------------------------------------------------------
# P8: unknown type names raise ValueError in __init__ and get_model
# ---------------------------------------------------------------------------
def test_p8_unknown_type_rejected():
    with pytest.raises(ValueError, match="Unknown PCA type"):
        pca(type="nope")
    obj = pca(type="classic")
    # Before the fix get_model("nope") silently returned None.
    with pytest.raises(ValueError, match="Unknown PCA type"):
        obj.get_model("nope")


# ---------------------------------------------------------------------------
# P9: set_deterministic_sign is vectorised and jit-safe
# ---------------------------------------------------------------------------
def _old_sign(components):
    """Pre-audit Python loop, kept as the reference for the new result."""
    components = jnp.array(components)
    for i in range(components.shape[1]):
        max_abs_idx = jnp.argmax(jnp.abs(components[:, i]))
        if components[max_abs_idx, i] < 0:
            components = components.at[:, i].set(components[:, i] * -1)
    return components


def test_p9_deterministic_sign_matches_old_loop_and_jits():
    obj = pca(type="classic")
    M = jnp.array(np.random.default_rng(7).normal(size=(6, 3)))
    expected = _old_sign(M)
    assert bool(jnp.allclose(obj.set_deterministic_sign(M), expected))
    # The old loop branched on a traced value and raised under jit.
    out = jax.jit(obj.set_deterministic_sign)(M)
    assert bool(jnp.allclose(out, expected))
