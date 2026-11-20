"""Regression tests for RELEASE_AUDIT.md, module 01 (np_dists.py), critical C1 and C2.

C1: a per-draw ``seed=False`` resolved to the fixed key PRNGKey(0), so every
    such draw returned the same values.
C2: a ``UnifiedDist`` without an attached RNG manager (``rng=None``) drew from
    a literal ``PRNGKey(rand_seed)`` on every call (identical consecutive draws
    for an int seed) or from the clock (non-reproducible), instead of a
    reproducible stream as the ``seed`` docstring promises.
"""
import numpy as np

from BayesForge.Distributions.np_dists import UnifiedDist
from BayesForge.Utils.rng import RNG


def _draw(d, **kw):
    return np.asarray(d.normal(0.0, 1.0, sample=True, shape=(5,), **kw))


def test_c1_seed_false_draws_differ_with_rng_manager():
    d = UnifiedDist(rand_seed=0, rng=RNG(0))
    a, b = _draw(d, seed=False), _draw(d, seed=False)
    assert not np.allclose(a, b)


def test_c1_seed_false_draws_differ_without_rng_manager():
    d = UnifiedDist(rand_seed=0)
    a, b = _draw(d, seed=False), _draw(d, seed=False)
    assert not np.allclose(a, b)


def test_c1_seed_false_different_distributions_not_same_key():
    d = UnifiedDist(rand_seed=0, rng=RNG(0))
    u1 = np.asarray(d.uniform(0.0, 1.0, sample=True, shape=(5,), seed=False))
    u2 = np.asarray(d.uniform(0.0, 1.0, sample=True, shape=(5,), seed=False))
    assert not np.allclose(u1, u2)


def test_c2_int_rand_seed_without_rng_is_a_reproducible_stream():
    d1, d2 = UnifiedDist(rand_seed=123), UnifiedDist(rand_seed=123)
    seq1 = [_draw(d1) for _ in range(3)]
    seq2 = [_draw(d2) for _ in range(3)]
    # reproducible across instances with the same seed
    for a, b in zip(seq1, seq2):
        np.testing.assert_array_equal(a, b)
    # consecutive draws are independent, not the same fixed key
    assert not np.allclose(seq1[0], seq1[1])


def test_c2_rand_seed_false_without_rng_is_reproducible():
    a = _draw(UnifiedDist(rand_seed=False))
    b = _draw(UnifiedDist(rand_seed=False))
    np.testing.assert_array_equal(a, b)


def test_explicit_int_seed_still_honoured_verbatim():
    d = UnifiedDist(rand_seed=0, rng=RNG(0))
    np.testing.assert_array_equal(_draw(d, seed=7), _draw(d, seed=7))
