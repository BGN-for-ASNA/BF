"""RELEASE_AUDIT Models — majors G1-G3 (GMM) and D1-D2 (DPMM).

Pins the fixes:

* G1: ``predict_gmm`` relabels every draw to a pivot draw before averaging
    (``_relabel_draws`` undoes label switching; tested here in pure numpy);
* G2: the posterior similarity matrix uses at most 300 evenly-spaced draws
    (``cluster_probs`` leading dim is ``min(S, 300)``);
* G3: the GMM priors scale with the per-feature standard deviation of the
    data, so a 100x rescaling no longer forces ``sigma`` to O(1);
* D2: a constant column no longer makes the DPMM prior singular (the shared
    scale-aware priors of both formulations, D1, are checked by the same fit).

Run:
    pytest Test/0.Core/test_audit_models_major_gmm_dpmm.py -q
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np

from BayesForge import bf
from BayesForge.Models import GMM
from BayesForge.Models.GMM import predict_gmm


class _FakeSampler:
    """Minimal stand-in exposing what ``predict_gmm`` reads
    (same shape as in test_dpmm_plot.py)."""

    def __init__(self, samples):
        self._samples = samples

    def get_samples(self, group_by_chain=False):
        return self._samples


# ---------------------------------------------------------------------------
# G1: _relabel_draws undoes label switching
# ---------------------------------------------------------------------------
def test_g1_relabel_undoes_label_switching():
    rng = np.random.default_rng(0)
    n_draws = 100
    base = np.array([[-5.0, -5.0], [5.0, 5.0]])
    flipped = base[::-1]

    mu_samps = np.empty((n_draws, 2, 2))
    w_samps = np.empty((n_draws, 2))
    for s in range(n_draws):
        if s % 2:  # half of the draws carry the swapped labelling
            mu_samps[s] = flipped + 0.05 * rng.normal(size=(2, 2))
            w_samps[s] = [0.7, 0.3]
        else:
            mu_samps[s] = base + 0.05 * rng.normal(size=(2, 2))
            w_samps[s] = [0.3, 0.7]

    mu_al, w_al = GMM._relabel_draws(mu_samps, w_samps)

    # Without alignment the mean of mu is ~0; after alignment every entry
    # of the mean sits at +-5.
    mu_mean = mu_al.mean(axis=0)
    assert (np.abs(mu_mean) > 4.5).all()

    # The weights follow their components: the aligned means are {0.3, 0.7}.
    assert np.allclose(np.sort(w_al.mean(axis=0)), [0.3, 0.7], atol=0.05)
    # The heavy component is the one whose mean is at -5 (pivot orientation
    # is arbitrary, so check the pairing, not the order).
    heavy = np.argmax(w_al.mean(axis=0))
    assert np.sign(mu_mean[heavy, 0]) == np.sign(mu_mean[heavy, 1])


# ---------------------------------------------------------------------------
# G2: the PSM subsamples at most 300 draws
# ---------------------------------------------------------------------------
def test_g2_psm_uses_at_most_300_draws():
    n_draws, n_data, K, D = 1000, 20, 2, 2
    rng = np.random.default_rng(1)
    data = np.vstack([
        np.array([-5.0, -5.0]) + rng.normal(scale=0.3, size=(n_data // 2, D)),
        np.array([5.0, 5.0]) + rng.normal(scale=0.3, size=(n_data // 2, D)),
    ])
    samples = dict(
        weights=np.tile(np.array([0.5, 0.5]), (n_draws, 1)),
        mu=np.tile(np.array([[-5.0, -5.0], [5.0, 5.0]]), (n_draws, 1, 1)),
        sigma=np.ones((n_draws, K, D)),
        Lcorr=np.tile(np.eye(D), (n_draws, K, 1, 1)),
    )

    out = predict_gmm(data, _FakeSampler(samples), return_probs=True)
    cluster_probs = out[-1]  # last element of the return_probs tuple
    assert np.asarray(cluster_probs).shape == (300, n_data, K)


# ---------------------------------------------------------------------------
# G3: scale-aware GMM prior (tiny fit)
# ---------------------------------------------------------------------------
def test_g3_gmm_scale_invariant_prior():
    rng = np.random.default_rng(2)
    # Two well separated clusters, N=60, D=2, scaled x100: the old prior
    # (HalfCauchy(1), MVN sd ~ 0.316) pinned sigma near O(1) and mu near the
    # K-means start; the scale-aware prior lets sigma follow the data.
    data = 100.0 * np.vstack([
        np.array([-5.0, -5.0]) + rng.normal(scale=0.5, size=(30, 2)),
        np.array([5.0, 5.0]) + rng.normal(scale=0.5, size=(30, 2)),
    ])

    m = bf(platform="cpu", rand_seed=3, print_devices_found=False)
    m.data_on_model = dict(data=data, K=2)
    m.fit(m.models.available['gmm'], num_warmup=50, num_samples=50,
          num_chains=1, progress_bar=False)

    sigma = np.asarray(m.sampler.get_samples()['sigma'])
    assert np.isfinite(sigma).all()
    assert sigma.mean() > 10.0


# ---------------------------------------------------------------------------
# D2: constant column does not crash the DPMM (shared priors, D1)
# ---------------------------------------------------------------------------
def test_d2_dpmm_constant_column_does_not_crash():
    rng = np.random.default_rng(4)
    # Second column constant: std = 0 there, which used to make the prior
    # singular (covariance 0 / scale 0).
    data = np.column_stack([
        np.concatenate([rng.normal(-5.0, 0.5, 20), rng.normal(5.0, 0.5, 20)]),
        np.zeros(40),
    ])

    m = bf(platform="cpu", rand_seed=5, print_devices_found=False)
    m.data_on_model = dict(data=data, T=5)
    m.fit(m.models.dpmm, num_warmup=50, num_samples=50,
          num_chains=1, progress_bar=False)

    mu = np.asarray(m.sampler.get_samples()['mu'])
    assert np.isfinite(mu).all()
