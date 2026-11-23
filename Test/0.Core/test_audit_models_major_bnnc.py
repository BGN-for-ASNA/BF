"""Regression tests for RELEASE_AUDIT.md, module Models, BNNC majors B1-B4.

Pins the fixes:

* B1: ``theta`` (N, K per draw) is no longer a ``deterministic`` site of the
  fit — ``model``/``__call__`` take ``store_theta=False`` by default and
  ``_theta_draws`` recomputes it via ``Predictive``; ``predict`` subsamples
  the draws first, then recomputes theta for the kept draws only;
* B2: ``plot`` uses the same predictive as ``predict``,
  p(k | x) ∝ theta_k(x) N(x | mu_k, sigma_k) with the network gating
  theta(x) = softmax(w2(x) + log pi), not the global pi_k;
* B3: the fixed priors (mu ~ N(0, 5), sigma ~ Exponential(1)) warn on data
  that do not look standardised;
* B4: the grid predictive is computed in float32 chunks of 2000 points
  (checked here only through the plot running end-to-end).

Run:
    pytest Test/0.Core/test_audit_models_major_bnnc.py -q
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from BayesForge import bf
from BayesForge.Models import BNNC


def _tiny_data(seed=0):
    """N=40 standardised 2-D points in 2 blobs."""
    rng = np.random.default_rng(seed)
    X = np.vstack([
        np.array([-2.0, -2.0]) + rng.normal(scale=0.5, size=(20, 2)),
        np.array([2.0, 2.0]) + rng.normal(scale=0.5, size=(20, 2)),
    ])
    return (X - X.mean(0)) / X.std(0)


def _tiny_fit(X, seed=1):
    """Tiny bnnc fit: K=3, D_H1=4, 30+30 draws, 1 chain, cpu."""
    m = bf(platform="cpu", rand_seed=seed, print_devices_found=False)
    m.data_on_model = dict(data=X, K=3, D_H1=4)
    m.fit(m.models.bnnc, num_warmup=30, num_samples=30,
          num_chains=1, progress_bar=False)
    return m


# ---------------------------------------------------------------------------
# B1: theta is not stored, but predict recomputes it and still works
# ---------------------------------------------------------------------------
def test_b1_theta_not_stored_but_predict_works():
    X = _tiny_data()
    m = _tiny_fit(X)

    assert 'theta' not in m.sampler.get_samples()

    theta, mu, sigma, labels = m.models.bnnc.predict(X, m.sampler)
    theta = np.asarray(theta)
    assert theta.shape == (30, 40, 3)
    # theta(x) is a softmax over the K components: rows sum to 1
    assert np.allclose(theta.sum(-1), 1.0, atol=1e-4)


# ---------------------------------------------------------------------------
# B2: plot routes both the grid and the data through _theta_draws
# ---------------------------------------------------------------------------
def test_b2_plot_runs_and_uses_theta(monkeypatch):
    X = _tiny_data(seed=2)
    m = _tiny_fit(X, seed=3)

    calls = []
    orig = BNNC.bnnc._theta_draws

    def counting(self, data, posterior_samples, key=None):
        calls.append(np.asarray(data).shape[0])
        return orig(self, data, posterior_samples, key)

    monkeypatch.setattr(BNNC.bnnc, "_theta_draws", counting)
    m.models.bnnc.plot(X, m.sampler)

    # grid (chunked: >= 1 call) + the data points
    assert len(calls) >= 2
    assert calls[-1] == X.shape[0]  # the last call is the data itself


# ---------------------------------------------------------------------------
# B3: the default priors warn on raw-scale data
# ---------------------------------------------------------------------------
def test_b3_warns_on_raw_scale():
    X = 100.0 * _tiny_data(seed=4)  # std ~ 100 >> 5: not standardised

    m = bf(platform="cpu", rand_seed=5, print_devices_found=False)
    m.data_on_model = dict(data=X, K=3, D_H1=4)
    with pytest.warns(UserWarning, match="standardised"):
        m.fit(m.models.bnnc, num_warmup=30, num_samples=30,
              num_chains=1, progress_bar=False)
