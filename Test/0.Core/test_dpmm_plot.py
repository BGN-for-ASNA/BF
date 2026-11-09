"""DPMM posterior plot: one credible region per consensus cluster.

The previous implementation contoured the posterior-predictive density with
Matplotlib's automatic levels. The lowest level is a tiny *positive* number
that the ``'%.2f'`` label format prints as ``0.00``; combined with the wide,
near-empty components of the DPMM truncation it drew a level set that bridged
unrelated clusters. These tests pin the replacement:

* component covariances and moment matching are exact;
* ``plot`` draws one ellipse per consensus cluster (at each requested
  credible level), with no density contour by default;
* the optional density overlay uses relative levels only.

Run:
    pytest Test/0.Core/test_dpmm_plot.py -q
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
from matplotlib.patches import Ellipse

from BayesForge.Models.DPMM import dpmm


class _FakeSampler:
    """Minimal stand-in exposing what ``dpmm.predict`` reads."""

    def __init__(self, samples):
        self._samples = samples

    def get_samples(self, group_by_chain=False):
        return self._samples


def _synthetic_posterior(n_draws=40, seed=0):
    """3 well-separated blobs + 3 wide, near-empty components."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-5.0, 5.0], [0.0, 0.0], [5.0, -5.0]])
    data = np.vstack([c + 0.5 * rng.normal(size=(40, 2)) for c in centers])

    n_data, dim, T = len(data), 2, 6
    mu = np.zeros((n_draws, T, dim))
    sigma = np.zeros((n_draws, T, dim))
    Lcorr = np.zeros((n_draws, T, dim, dim))
    w = np.zeros((n_draws, T))
    for s in range(n_draws):
        mu[s, :3] = centers + 0.02 * rng.normal(size=(3, dim))
        mu[s, 3:] = rng.uniform(-9.0, 9.0, size=(3, dim))
        sigma[s, :3] = 0.55
        sigma[s, 3:] = 4.0
        Lcorr[s] = np.broadcast_to(np.eye(dim), (T, dim, dim))
        logits = np.array([4.0, 4.0, 4.0, -8.0, -8.0, -8.0])
        w[s] = np.exp(logits) / np.exp(logits).sum()

    return data, dict(w=w, mu=mu, sigma=sigma, Lcorr=Lcorr)


def test_component_covariances_are_diag_sigma_l_lt_diag_sigma():
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    chol = np.linalg.cholesky(corr)
    sigma = np.array([2.0, 3.0])
    Lcorr = chol[None, None, :, :]
    sigma4 = sigma[None, None, :]

    cov = dpmm._component_covariances(sigma4, Lcorr)[0, 0]
    expected = np.diag(sigma) @ corr @ np.diag(sigma)
    assert np.allclose(cov, expected)


def test_cluster_moments_match_analytic_two_component_mixture():
    mu = np.array([[[0.0, 0.0], [10.0, 0.0]]])
    cov = np.broadcast_to(np.eye(2), (1, 2, 2, 2)).copy()
    mass = np.array([[0.75, 0.25]])

    mean, cov_c = dpmm._cluster_moments(mu, cov, mass)

    assert np.allclose(mean, [2.5, 0.0])
    assert np.allclose(cov_c, [[19.75, 0.0], [0.0, 1.0]])


def test_cluster_moments_anchor_uses_within_component_spread_only():
    # Two overlapping components sharing one cluster's mass across draws:
    # the anchored moments must describe the cluster (~component size),
    # not the distance between the components. Without the anchor the
    # between-component term stretches the ellipse across the plot.
    mu = np.array([[[-5.0, 5.0], [-1.0, 3.0]]])
    cov = np.broadcast_to(np.eye(2) * 0.25, (1, 2, 2, 2)).copy()
    mass = np.array([[0.5, 0.5]])
    anchor = np.array([-5.0, 5.0])

    mean, cov_c = dpmm._cluster_moments(mu, cov, mass, anchor=anchor)

    assert np.allclose(mean, anchor)
    assert np.allclose(cov_c, np.eye(2) * 0.25)

    _, cov_free = dpmm._cluster_moments(mu, cov, mass)
    assert cov_free[0, 0] > 4.0  # inflated by the between-component term


def test_cap_covariance_bounds_ellipse_by_point_spread():
    # A posterior covariance far wider than the cluster's points (wide
    # component spanning several consensus clusters) must be capped.
    rng = np.random.default_rng(1)
    pts = np.array([-6.0, 0.0]) + 0.5 * rng.normal(size=(60, 2))
    cov_wide = np.diag([100.0, 1.0])

    cov_c = dpmm._cap_covariance(cov_wide, pts, cap=2.0, floor_std=0.25)

    cap_var = (2.0 ** 2) * np.linalg.eigvalsh(np.cov(pts.T)).max()
    assert np.linalg.eigvalsh(cov_c).max() <= cap_var + 1e-9
    # a covariance already inside the cap passes through unchanged
    cov_tight = np.eye(2) * 0.1
    assert np.allclose(dpmm._cap_covariance(cov_tight, pts, cap=2.0),
                       cov_tight, atol=1e-9)
    # a single point keeps a visible region via the floor
    cov_one = dpmm._cap_covariance(cov_wide, pts[:1], cap=2.0, floor_std=0.5)
    assert np.allclose(cov_one, np.eye(2) * 0.25)


def test_predict_returns_probs_only_when_asked():
    data, samples = _synthetic_posterior()
    m = dpmm(parent=None)

    assert len(m.predict(data, _FakeSampler(samples))) == 5
    out = m.predict(data, _FakeSampler(samples), return_probs=True)
    assert len(out) == 6
    assert np.asarray(out[-1]).shape == (samples["w"].shape[0], len(data), 6)


def test_plot_draws_one_ellipse_per_cluster_and_level():
    data, samples = _synthetic_posterior()
    m = dpmm(parent=None)

    fig, ax = m.plot(data, _FakeSampler(samples), credible=(0.5, 0.9))

    labels = m.predict(data, _FakeSampler(samples))[-1]
    n_clusters = len(np.unique(labels))
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipses) == n_clusters * 2
    # only the scatter collection: no density contour by default
    assert len(ax.collections) == 1

    # each cluster's 90% ellipse is centred on its own point cloud and is
    # strictly wider than its 50% ellipse
    for c in np.unique(labels):
        pts = data[labels == c]
        assert any(np.allclose(np.asarray(p.center), pts.mean(0), atol=0.6)
                   for p in ellipses)

    inner = ellipses[0::2][:n_clusters]
    outer = ellipses[1::2][:n_clusters]
    for wide, narrow in zip(outer, inner):
        assert wide.width > narrow.width


def test_density_overlay_uses_positive_relative_levels():
    data, samples = _synthetic_posterior()
    m = dpmm(parent=None)

    fig, ax = m.plot(data, _FakeSampler(samples), show_density=True)

    # the scatter collection + at least one contour collection
    assert len(ax.collections) > 1
    for coll in ax.collections[1:]:
        levels = np.asarray(coll.levels)
        assert levels.size > 0
        assert np.all(levels > 0.0)  # the old 0.00 level is gone


# ---------------------------------------------------------------------------
# GMM: same credible-ellipse visualization
# ---------------------------------------------------------------------------
from BayesForge.Models.GMM import plot_gmm, predict_gmm


def _synthetic_gmm_posterior(n_draws=40, seed=1):
    """3 well-separated blobs, K=3 components (GMM sample key is 'weights')."""
    rng = np.random.default_rng(seed)
    centers = np.array([[-5.0, 5.0], [0.0, 0.0], [5.0, -5.0]])
    data = np.vstack([c + 0.5 * rng.normal(size=(40, 2)) for c in centers])

    n_data, dim, K = len(data), 2, 3
    mu = np.zeros((n_draws, K, dim))
    sigma = np.zeros((n_draws, K, dim))
    Lcorr = np.zeros((n_draws, K, dim, dim))
    weights = np.zeros((n_draws, K))
    for s in range(n_draws):
        mu[s] = centers + 0.02 * rng.normal(size=(K, dim))
        sigma[s] = 0.55
        Lcorr[s] = np.broadcast_to(np.eye(dim), (K, dim, dim))
        weights[s] = np.array([1 / 3, 1 / 3, 1 / 3])

    return data, dict(weights=weights, mu=mu, sigma=sigma, Lcorr=Lcorr)


def test_gmm_predict_returns_probs_only_when_asked():
    data, samples = _synthetic_gmm_posterior()
    assert len(predict_gmm(data, _FakeSampler(samples))) == 4
    out = predict_gmm(data, _FakeSampler(samples), return_probs=True)
    assert len(out) == 9
    # cluster_probs is the last element: (N_draws, N_data, K)
    assert np.asarray(out[-1]).shape == (samples["weights"].shape[0],
                                         len(data), 3)


def test_gmm_plot_draws_one_ellipse_per_cluster_per_level():
    data, samples = _synthetic_gmm_posterior()
    fig, ax = plot_gmm(data, _FakeSampler(samples), credible=[0.5, 0.9])

    labels = predict_gmm(data, _FakeSampler(samples))[-1]
    n_clusters = len(np.unique(labels))
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipses) == n_clusters * 2
    assert len(ax.collections) == 1  # no density contour by default


def test_gmm_plot_credible_list_controls_ellipse_count():
    # The user-declared list drives the number of nested ellipses per cluster.
    data, samples = _synthetic_gmm_posterior()
    levels = [0.2, 0.4, 0.6, 0.8]
    fig, ax = plot_gmm(data, _FakeSampler(samples), credible=levels)

    labels = predict_gmm(data, _FakeSampler(samples))[-1]
    n_clusters = len(np.unique(labels))
    ellipses = [p for p in ax.patches if isinstance(p, Ellipse)]
    assert len(ellipses) == n_clusters * len(levels)
    # widths strictly increase with the credible level within a cluster
    widths = sorted(e.width for e in ellipses[:len(levels)])
    assert all(b > a for a, b in zip(widths, widths[1:]))
