"""Regression tests for RELEASE_AUDIT.md, module ML, majors M1-M4."""
import numpy as np
import pytest

from BayesForge.ML.KMeans import JAXKMeans, _sq_dist
from BayesForge.ML.ml import ml


def _X():
    rng = np.random.default_rng(0)
    centers = rng.uniform(-10, 10, size=(8, 2))
    return np.concatenate([c + 0.8 * rng.standard_normal((60, 2)) for c in centers]).astype(np.float32)


def _X_sep():
    """Same 8-blob recipe as _X(), but centres redrawn (same rng stream, so
    deterministic) until they are at least 4 apart.

    Deviation note (M1): the _X() draw of test_audit_ml_critical.py contains a
    centre pair only 1.04 apart with std 0.8 — a genuinely multi-basin geometry
    where even sklearn's k-means++ with n_init=10 varies by ~5% across seeds
    (measured with a numpy k-means++: max/min inertia 1.049 over 10 seeds,
    ~22% of n_init=10 runs miss the global basin). The "k-means++ with
    n_init=10 is stable" claim is therefore checked where the optimum is
    unique; _X() still drives the M2-M4 tests.
    """
    rng = np.random.default_rng(0)
    centers = rng.uniform(-10, 10, size=(8, 2))
    for _ in range(1000):
        d = np.sqrt(((centers[:, None, :] - centers[None, :, :]) ** 2).sum(-1))
        if np.min(d + np.eye(8) * 1e9) >= 4.0:
            break
        centers = rng.uniform(-10, 10, size=(8, 2))
    return np.concatenate([c + 0.8 * rng.standard_normal((60, 2)) for c in centers]).astype(np.float32)


# --- M1: k-means++ initialisation + n_init ---

def test_m1_kmeanspp_ninit_stable_across_seeds():
    X = _X_sep()
    inertias = [float(JAXKMeans(X, 8, random_state=rs).inertia_) for rs in range(10)]
    best = min(inertias)
    assert all(i <= 1.01 * best for i in inertias), inertias


def test_m1_kmeanspp_matches_sklearn():
    pytest.importorskip("sklearn")
    from sklearn.cluster import KMeans as SkKMeans

    X = _X_sep()
    bf = float(JAXKMeans(X, 8, random_state=0).inertia_)
    sk = SkKMeans(n_clusters=8, n_init=10, random_state=0).fit(X).inertia_
    assert bf <= 1.01 * float(sk)


def test_m1_init_random_legacy_option():
    X = _X_sep()
    a = JAXKMeans(X, 8, random_state=4, init="random", n_init=1)
    b = JAXKMeans(X, 8, random_state=4, init="random", n_init=1)
    np.testing.assert_array_equal(np.asarray(a.centroids_), np.asarray(b.centroids_))
    assert np.asarray(a.centroids_).shape == (8, 2)


# --- M2: _sq_dist helper replaces the (N, K, D) broadcast ---

def test_m2_sq_dist_matches_broadcast_and_nonnegative():
    import jax.numpy as jnp

    rng = np.random.default_rng(7)
    X = rng.random((40, 3)).astype(np.float32)
    C = rng.random((6, 3)).astype(np.float32)
    # A centroid exactly on a point: true distance 0, cross-term identity
    # could dip below zero without the clip.
    C[0] = X[0]

    ref = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    got = np.asarray(_sq_dist(jnp.asarray(X), jnp.asarray(C)))
    assert got.shape == ref.shape
    np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
    assert np.all(got >= 0.0)


# --- M3: input validation in fit ---

def test_m3_rejects_1d_input():
    X = _X()
    with pytest.raises(ValueError, match=r"1-D data pass X\[:, None\]"):
        JAXKMeans(X[:, 0], n_clusters=3)


def test_m3_rejects_k_zero_and_k_above_n():
    X = _X()
    with pytest.raises(ValueError, match=r"n_clusters=0, n_samples=480"):
        JAXKMeans(X, n_clusters=0)
    with pytest.raises(ValueError, match=r"n_clusters=481, n_samples=480"):
        JAXKMeans(X, n_clusters=X.shape[0] + 1)


def test_m3_rejects_non_finite_X():
    X = _X()
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        JAXKMeans(X_nan, n_clusters=3)
    X_inf = X.copy()
    X_inf[5, 1] = np.inf
    with pytest.raises(ValueError, match="NaN or inf"):
        JAXKMeans(X_inf, n_clusters=3)


# --- M4: fit honours and updates instance state; constructor fit kept ---

def test_m4_fit_override_updates_instance_state():
    # Before the fix, JAXKMeans(X, 4).fit(X, 3) left self.n_clusters == 4
    # while centroids_ had 3 rows (lying attributes).
    X = _X()
    km = JAXKMeans(X, n_clusters=4, random_state=1)
    km.fit(X, n_clusters=3)
    assert km.n_clusters == 3
    assert np.asarray(km.centroids_).shape == (3, 2)
    assert np.asarray(km.results["centroids"]).shape == (3, 2)


def test_m4_constructor_still_fits_for_gmm_path():
    # Backward compatibility (main.py GMM initialisation): constructing the
    # model must leave results["centroids"] ready without an explicit fit().
    X = _X()
    km = JAXKMeans(X, n_clusters=5, random_state=2)
    assert np.asarray(km.results["centroids"]).shape == (5, 2)
    assert np.isfinite(float(km.inertia_))


def test_m4_unfitted_estimator_can_be_constructed():
    # The audit's M4 complaint: "impossible de construire un estimateur non
    # ajusté". X is now optional; fit() then needs data from somewhere.
    km = JAXKMeans(n_clusters=3)
    assert km.centroids_ is None
    assert km.results == {}
    with pytest.raises(ValueError, match="needs data"):
        km.fit()
    km.fit(_X())
    assert np.asarray(km.centroids_).shape == (3, 2)
    # A bare fit() reuses the data the instance now holds.
    km.fit()
    assert np.asarray(km.centroids_).shape == (3, 2)


def test_m4_successive_fits_do_not_reuse_the_same_key():
    # The audit noted self.key was never split: two fits reproduced the exact
    # same initialisation. With init="random", n_init=1, n_iterations=1 the
    # centroids are the means of the clusters formed around the raw random
    # draw, so different inits give different centroids.
    X = _X()
    km = JAXKMeans(X, n_clusters=4, n_iterations=1, random_state=0,
                   init="random", n_init=1)
    first = np.asarray(km.centroids_).copy()
    km.fit(X)
    assert not np.allclose(first, np.asarray(km.centroids_))


def test_m4_handler_results_not_aliased():
    X = _X()
    h = ml()
    h.KMEANS(X, n_clusters=3, random_state=0)
    before = h.results["centroids"]
    h.model.results["centroids"] = None
    assert h.results["centroids"] is before
