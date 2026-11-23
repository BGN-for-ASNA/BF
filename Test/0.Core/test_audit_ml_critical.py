"""Regression tests for RELEASE_AUDIT.md, module 16 (ML), critical C1/C2."""
import numpy as np
import pytest

from BayesForge.ML.KMeans import JAXKMeans
from BayesForge.ML.ml import ml


def _X():
    rng = np.random.default_rng(0)
    centers = rng.uniform(-10, 10, size=(8, 2))
    return np.concatenate([c + 0.8 * rng.standard_normal((60, 2)) for c in centers]).astype(np.float32)


def test_c1_random_state_changes_initialisation():
    X = _X()
    inits = set()
    for rs in range(6):
        km = JAXKMeans(X, n_clusters=8, n_iterations=1, random_state=rs)
        inits.add(tuple(np.round(np.asarray(km.key), 0).ravel().tolist()))
    assert len(inits) == 6


def test_c1_same_random_state_is_reproducible():
    X = _X()
    a = JAXKMeans(X, n_clusters=8, random_state=3)
    b = JAXKMeans(X, n_clusters=8, random_state=3)
    np.testing.assert_array_equal(np.asarray(a.centroids_), np.asarray(b.centroids_))


def test_c2_fluent_fit_without_arguments():
    X = _X()
    h = ml()
    out = h.KMEANS(X, n_clusters=3).fit()
    assert h.results is not None
    assert np.asarray(h.model.centroids_).shape == (3, 2)


def test_c2_kmeans_fit_uses_constructor_settings():
    X = _X()
    km = JAXKMeans(X, n_clusters=4, random_state=1)
    km.fit(X)
    assert np.asarray(km.centroids_).shape == (4, 2)
