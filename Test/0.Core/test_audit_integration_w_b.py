"""Regression tests for RELEASE_AUDIT.md, Integration W9-W12."""

import numpy as np
import pytest

from BayesForge import bf
from BayesForge.NBDA.NBDA import NBDA


def _bf():
    """Fresh bf instance: fit() mutates m heavily, tests must not share it."""
    return bf(platform="cpu", print_devices_found=False, rand_seed=7)


# ---------------------------------------------------------------------------
# W9 — pd_to_jax used to jnp.array() every non-numeric default: None raised
# ValueError, str raised TypeError. None/str defaults now stay Python values
# (Data/manip.py, already fixed); this test locks the behaviour in.
# ---------------------------------------------------------------------------
def test_w9_pd_to_jax_keeps_none_and_str_defaults():
    m = _bf()
    rng = np.random.default_rng(0)
    m.data({"x": np.arange(10.0), "data": rng.normal(size=10)})

    def f(x, a=None, method='marginal'):
        pass

    out = m.pd_to_jax(f)
    assert out["a"] is None
    assert out["method"] == "marginal"
    assert "x" in out  # the column itself still converts


# ---------------------------------------------------------------------------
# W10 — m.models exposes the PCA class, the NBDA class and a non-callable
# survival instance; fit() must fail at its entry with a hint naming the
# object to pass, not deep inside pd_to_jax or on model.__name__.
# ---------------------------------------------------------------------------
def test_w10_pca_class_rejected_with_instance_hint():
    m = _bf()
    with pytest.raises(TypeError, match="instance"):
        m.fit(m.models.pca)


def test_w10_nbda_class_rejected_with_nbda_model_hint():
    m = _bf()
    with pytest.raises(TypeError, match=r"nbda\.model"):
        m.fit(m.models.nbda)


def test_w10_survival_instance_rejected():
    m = _bf()
    with pytest.raises(TypeError, match="not a model function"):
        m.fit(m.models.survival)


def test_w10_none_rejected():
    m = _bf()
    with pytest.raises(TypeError, match="model function"):
        m.fit(None)


# ---------------------------------------------------------------------------
# W11 — gmm needs K as a scalar int: a missing K, or a K column arriving as
# (N,) from m.df, must raise ValueError before KMEANS/plate; a Python int
# fits normally.
# ---------------------------------------------------------------------------
def _xy():
    return np.random.default_rng(0).normal(size=(30, 2))


def test_w11_gmm_without_K_raises():
    m = _bf()
    with pytest.raises(ValueError, match=r"\bK\b"):
        m.fit(m.models.gmm, obs={"data": _xy()})


def test_w11_gmm_with_int_K_runs():
    m = _bf()
    m.fit(m.models.gmm, obs={"data": _xy(), "K": 2},
          num_warmup=50, num_samples=50, num_chains=1, progress_bar=False)
    assert m.run_model_name == "gmm"
    assert isinstance(m.data_on_model["K"], int)


# ---------------------------------------------------------------------------
# W12 — the dead self.nbdaModel branch is gone (data still built from m.df
# or obs), and m.fit(nbda.model) must record run_model_name == "nbda"
# instead of the bound method's __name__ "model".
# ---------------------------------------------------------------------------
def _net(n=4):
    rng = np.random.default_rng(0)
    return np.asarray(rng.integers(0, 2, size=(n, n)).astype(float))


def _status(n=4, t=3):
    return np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=float)[:n, :t]


def test_w12_nbda_bound_model_records_nbda_name():
    m = _bf()
    nbda = NBDA(_net(), _status())
    m.fit(nbda.model, num_warmup=50, num_samples=50, num_chains=1,
          progress_bar=False)
    assert m.run_model_name == "nbda"
