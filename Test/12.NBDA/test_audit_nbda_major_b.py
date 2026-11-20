"""Regression tests for RELEASE_AUDIT.md, module NBDA, majors M7-M10 and M13b/c (model structure)."""

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

from BayesForge.NBDA.NBDA import NBDA


def _net(n=4):
    rng = np.random.default_rng(0)
    return np.asarray(rng.integers(0, 2, size=(n, n)).astype(float))


def _status(n=4, t=3):
    return np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=float)[:n, :t]


def _nbda(n=4, t=3):
    return NBDA(_net(n), _status(n, t))


def _trace(nb, CES=False):
    """Trace nb.model the way the critical tests run it (seed + trace handler, no fit)."""
    return numpyro.handlers.trace(numpyro.handlers.seed(nb.model, rng_seed=0)).get_trace(CES=CES)


# ------------------------------- M7: multiprocess status -------------------------------

def test_m7_status_with_two_processes_raises():
    status = np.stack([_status(), _status()], axis=-1)  # (n, t, 2)
    with pytest.raises(NotImplementedError, match="several processes"):
        NBDA(_net(), status)


def test_m7_status_with_one_process_squeezed_and_traces():
    nb = NBDA(_net(), _status()[:, :, None])  # (n, t, 1)
    assert nb.multiprocess is False
    assert nb.N_process == 1
    nb.get_cov()
    tr = _trace(nb)
    # status is 2-D compatible: the observed site is (n, t)
    assert np.asarray(tr["y"]["value"]).shape == (4, 3)


# ------------------------------- M8: multi-network -------------------------------

def test_m8_network_with_two_networks_raises():
    net3 = np.repeat(_net()[:, :, None], 3, axis=2)  # (n, n, t)
    net4 = np.stack([net3, net3], axis=-1)  # (n, n, t, 2)
    with pytest.raises(NotImplementedError, match="several networks"):
        NBDA(net4, _status())


def test_m8_network_with_one_network_builds():
    net4 = np.repeat(_net()[:, :, None, None], 3, axis=2)  # (n, n, t, 1)
    nb = NBDA(net4, _status())
    assert nb.network.shape == (4, 4, 3, 1)
    assert nb.intercept.shape == (4, 4, 3, 1)


# ------------------------------- M9: 3-D network -------------------------------

def test_m9_3d_network_expanded_and_traces():
    net3 = np.repeat(_net()[:, :, None], 3, axis=2)  # (n, n, t)
    nb = NBDA(net3, _status())
    assert nb.network.shape == (4, 4, 3, 1)  # expanded with network[..., None]
    assert nb.intercept.shape == (4, 4, 3, 1)  # 4-D, no concatenate clash
    nb.get_cov()
    tr = _trace(nb)  # model()'s network[:, :, t, 0] must not raise IndexError
    assert np.asarray(tr["y"]["value"]).shape == (4, 3)


# ------------------------------- M10: t = 0 conditions on the initial state -------------------------------

def test_m10_t0_masked_out_of_likelihood():
    nb = _nbda()
    nb.get_cov()
    tr = _trace(nb)
    mask = tr["y"].get("mask")
    if mask is None:  # trace does not expose the mask: check lk directly
        lk = np.asarray(nb.compute_probs(
            D_asocial=jnp.ones((4, 3, 1)),
            asoc=jnp.zeros(1),
            D_social=jnp.zeros((4, 4, 3, 1)),
            soc=jnp.zeros(1),
            network=nb.network,
            status=nb.status,
            N=4,
            T=3,
        ))
        assert np.isnan(lk[:, 0]).all()
    else:
        mask = np.asarray(mask)
        assert not mask[:, 0].any()  # t = 0 is not an event, for every individual
        assert mask[:, 1:].any()  # sanity: later steps still contribute


# ------------------------------- M13b: model() builds the design matrices -------------------------------

def test_m13b_model_calls_get_cov_when_not_called_first():
    nb = _nbda()
    assert nb.T_social is None
    tr = _trace(nb)  # no get_cov() beforehand
    assert nb.T_social is not None
    assert nb.T_asocial is not None
    assert np.asarray(tr["y"]["value"]).shape == (4, 3)


# ------------------------------- M13c: constructor validation -------------------------------

def test_m13c_non_square_network_raises():
    with pytest.raises(ValueError, match="square"):
        NBDA(np.ones((3, 4)), _status())


def test_m13c_network_n_mismatch_raises():
    with pytest.raises(ValueError, match="individuals"):
        NBDA(np.ones((3, 3)), _status(n=4))


def test_m13c_network_t_mismatch_raises():
    net3 = np.repeat(np.ones((4, 4))[:, :, None], 5, axis=2)  # (n, n, 5) vs t = 3
    with pytest.raises(ValueError, match="time steps"):
        NBDA(net3, _status(n=4, t=3))


def test_m13c_status_with_value_2_raises():
    status = _status().copy()
    status[0, 2] = 2.0
    with pytest.raises(ValueError, match="only 0 and 1"):
        NBDA(_net(), status)


def test_m13c_status_with_nan_raises():
    status = _status().copy()
    status[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        NBDA(_net(), status)
