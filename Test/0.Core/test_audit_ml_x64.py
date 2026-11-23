"""Regression test: JAXKMeans on float32 data with jax_enable_x64 on.

Found while fixing RELEASE_AUDIT Main M-07: under x64, jax.nn.one_hot
defaulted to float64 and the fori_loop carry changed dtype (TypeError).
"""
import jax
import numpy as np

from BayesForge.ML.KMeans import JAXKMeans


def test_kmeans_float32_data_under_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        rng = np.random.default_rng(0)
        X = np.concatenate([c + 0.5 * rng.standard_normal((40, 2))
                            for c in ([0, 0], [5, 5], [0, 5])]).astype(np.float32)
        km = JAXKMeans(X, n_clusters=3, random_state=0)
        assert np.asarray(km.centroids_).shape == (3, 2)
        assert np.all(np.isfinite(np.asarray(km.centroids_)))
    finally:
        jax.config.update("jax_enable_x64", prev)
