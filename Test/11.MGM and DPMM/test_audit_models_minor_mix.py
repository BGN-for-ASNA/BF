"""Regression tests for RELEASE_AUDIT.md, Models section 3 minors (GMM, DPMM, BNNC)."""
import ast
import inspect
import types

import numpy as np
import jax.numpy as jnp
import numpyro.handlers
import pytest

import BayesForge.Models.GMM as gmm_mod
import BayesForge.Models.DPMM as dpmm_mod
import BayesForge.Models.BNNC as bnnc_mod
from BayesForge.Models.GMM import gmm, predict_gmm, plot_gmm
from BayesForge.Models.DPMM import dpmm
from BayesForge.Models.BNNC import bnnc


# --------------------------------------------------------------------------
# GMM.py
# --------------------------------------------------------------------------

def test_g1_gmm_validates_data_and_K():
    # 1-D data, K > N and N == 0 must all raise ValueError (not IndexError,
    # not a sklearn error, not a silent pass).
    with numpyro.handlers.seed(rng_seed=0):
        with pytest.raises(ValueError):
            gmm(jnp.zeros(5), 2)
        with pytest.raises(ValueError):
            gmm(jnp.zeros((3, 2)), 5)
        with pytest.raises(ValueError):
            gmm(jnp.zeros((0, 2)), 1)


def test_g2_distance_threshold_exposed():
    assert inspect.signature(predict_gmm).parameters["distance_threshold"].default == 0.5
    assert inspect.signature(plot_gmm).parameters["distance_threshold"].default == 0.5
    # The hard-coded local assignment must be gone (only the keyword default).
    assert "distance_threshold = 0.5" not in inspect.getsource(predict_gmm)


def test_g3_gmm_docstring_label_switching():
    assert "breaks the mixture's label symmetry" not in gmm.__doc__
    assert "label switching" in gmm.__doc__


def test_g4_plot_gmm_requires_2d_data():
    with pytest.raises(ValueError, match="D=2"):
        plot_gmm(np.zeros((5, 3)), None)
    with pytest.raises(ValueError, match="at least 2"):
        plot_gmm(np.zeros((1, 2)), None)


def _import_triples(module):
    """(module, name, asname) triples of every Import/ImportFrom in a module."""
    tree = ast.parse(inspect.getsource(module))
    triples = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                triples.append((node.module, alias.name, alias.asname))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                triples.append((None, alias.name, alias.asname))
    return triples


def test_g5_gmm_no_duplicate_imports():
    triples = _import_triples(gmm_mod)
    assert len(triples) == len(set(triples))


def test_g6_get_cluster_probs_clips_zero_weights():
    # DPMM behaviour: all-zero weights (float32 underflow) must give finite
    # probabilities, not -inf/NaN.
    d = dpmm(None)
    data = jnp.zeros((4, 2))
    w = jnp.zeros(3)
    mu = jnp.zeros((3, 2))
    sigma = jnp.ones((3, 2))
    Lcorr = jnp.broadcast_to(jnp.eye(2), (3, 2, 2))
    p = d.get_cluster_probs(data, w, mu, sigma, Lcorr)
    assert np.all(np.isfinite(np.asarray(p)))
    # GMM's nested get_cluster_probs must use the same clipped log(w).
    assert "jnp.log(w)" not in inspect.getsource(predict_gmm)


# --------------------------------------------------------------------------
# DPMM.py
# --------------------------------------------------------------------------

def test_d1_credible_levels_documented_as_four():
    for mod in (dpmm_mod, gmm_mod):
        src = inspect.getsource(mod)
        assert "ten nested levels" not in src
        assert "5%-95%" not in src


def test_d2_predict_verbose_keyword():
    assert inspect.signature(dpmm.predict).parameters["verbose"].default is False
    assert inspect.signature(bnnc.predict).parameters["verbose"].default is False


def test_d4_inconsistent_draws_rejected():
    # w/mu/sigma carry a chain axis (2 chains x 10 draws); Lcorr has 7 draws:
    # inconsistent posteriors must raise before any plotting.
    posteriors = {
        "w": np.zeros((2, 10, 3)),
        "mu": np.zeros((2, 10, 3, 2)),
        "sigma": np.zeros((2, 10, 3, 2)),
        "Lcorr": np.zeros((7, 3, 2, 2)),
    }
    parent = types.SimpleNamespace(posteriors=posteriors)
    d = dpmm(parent)
    with pytest.raises(ValueError, match="draws"):
        d.proportion_of_data_assigned_to_cluster()


def test_d5_dpmm_plot_requires_2d_data():
    with pytest.raises(ValueError, match="D=2"):
        dpmm(None).plot(np.zeros((5, 3)), None)
    with pytest.raises(ValueError, match="at least 2"):
        dpmm(None).plot(np.zeros((1, 2)), None)


def test_d6_dpmm_no_duplicate_imports():
    triples = _import_triples(dpmm_mod)
    assert len(triples) == len(set(triples))


# --------------------------------------------------------------------------
# BNNC.py
# --------------------------------------------------------------------------

def test_b1_bnnc_model_validation():
    # Validation runs before self.parent is used, so parent=None is fine.
    # K > N is legitimate for BNNC (sparse Dirichlet), so it is NOT tested.
    with numpyro.handlers.seed(rng_seed=0):
        with pytest.raises(ValueError):
            bnnc(None).model(jnp.zeros(5))
        with pytest.raises(ValueError):
            bnnc(None).model(jnp.zeros((0, 2)))


def test_b2_single_log_epsilon():
    assert "1e-8" not in inspect.getsource(bnnc_mod)
    assert bnnc_mod._LOG_EPS == 1e-10


def test_b3_bnnc_plot_requires_2d_data():
    with pytest.raises(ValueError, match="D=2"):
        bnnc(None).plot(np.zeros((5, 3)), None)
    with pytest.raises(ValueError, match="at least 2"):
        bnnc(None).plot(np.zeros((1, 2)), None)
