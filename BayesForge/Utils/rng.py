"""Centralised random-number management for BayesForge.

A single :class:`RNG` instance owns the reproducibility contract for an
``bf`` model. It is built once from the constructor's ``rand_seed`` argument
and is the single source of truth for every random draw in the package.

Two access patterns are exposed, because inference engines and ad-hoc
distribution draws have different needs:

* :meth:`key` -- a *stable* key derived deterministically from the base seed
  and a string label (via ``jax.random.fold_in``). Used by the inference
  engines (``fit``, ``svi``, ``sample``, ``log_prob``) so that repeated calls
  to the same method reproduce, ``fit`` and ``svi`` differ from one another,
  and the result is independent of how many ad-hoc draws happened before it.

* :meth:`stream` -- a *fresh, independent* sub-key on every call, produced by
  splitting a stored key. Used by direct ``m.dist.XXX(sample=True)`` draws so
  that consecutive draws are statistically independent (no correlation between
  e.g. a Normal and an Exponential drawn back-to-back) while the whole
  sequence stays reproducible from a fixed base seed.

Seed semantics (matching the historical ``rand_seed`` contract):

* ``rand_seed=False`` -> base seed ``0`` (reproducible).
* ``rand_seed=<int>`` -> base seed = that integer (reproducible).
* ``rand_seed=True``  -> base seed drawn from entropy (non-reproducible), but
  stored so it can be reported via :meth:`get_seed` and replayed later.
"""

import weakref
import hashlib
import time

import jax


_UINT32_MASK = 0xFFFFFFFF


class RNG:
    # Every live manager, so a read-only model run (m.graph(), m.latex()) can
    # snapshot and restore all streams (RELEASE_AUDIT X15).
    _instances = weakref.WeakSet()

    @classmethod
    def snapshot_streams(cls):
        """Return the stream state of every live RNG manager."""
        return [(r, r._key) for r in list(cls._instances)]

    @staticmethod
    def restore_streams(snapshot):
        """Restore the stream states returned by :meth:`snapshot_streams`."""
        for r, k in snapshot:
            r._key = k

    def __init__(self, rand_seed=False):
        RNG._instances.add(self)
        self.rand_seed = rand_seed
        self.base = self._resolve_base(rand_seed)
        # Live key used by the split-based ``stream`` pattern. Created on the
        # first stream() call: building a PRNGKey here would initialise the JAX
        # backend at import time (module-level UnifiedDist singletons), before
        # setup_device can apply cores/platform.
        self._key = None

        # True once seed_globals() has been called for this instance (the
        # opt-in bf(seed_global_rngs=True) path). set_seed() uses it to keep
        # the numpy/stdlib globals in sync with the JAX stream.
        self._globals_seeded = False

    @staticmethod
    def _resolve_base(rand_seed):
        # ``bool`` must be checked before ``int`` (bool is a subclass of int).
        if isinstance(rand_seed, bool):
            return int(time.time_ns()) & _UINT32_MASK if rand_seed else 0
        if isinstance(rand_seed, int):
            return int(rand_seed) & _UINT32_MASK
        raise ValueError(
            f"Invalid rand_seed type: {type(rand_seed)}. "
            "Expected bool (True=random, False=reproducible) or int."
        )

    @staticmethod
    def _label_hash(label):
        digest = hashlib.sha256(str(label).encode()).hexdigest()
        return int(digest, 16) & _UINT32_MASK

    def key(self, label):
        """Return a stable PRNGKey derived from ``base`` and ``label``."""
        return jax.random.fold_in(
            jax.random.PRNGKey(self.base), self._label_hash(label)
        )

    def stream(self):
        """Return a fresh, independent PRNGKey and advance the stream.

        Reproducibility is order-dependent by design (module docstring): the
        split is mutating, so the sequence of ``sample=True`` draws depends on
        how many draws happened before it — stable for a fixed call order.

        Raises:
            RuntimeError: called while JAX is tracing (e.g. ``sample=True``
                inside a model passed to ``m.fit``/``m.svi``). Storing a traced
                key would leak a tracer and fail later with a cryptic JAX error.

        Note:
            The tracer guard below reads ``jax.core.Tracer``. ``jax.core`` is
            a deprecated alias of the private ``jax._src.core`` module, in the
            process of being moved out of the public surface (RELEASE_AUDIT,
            Utils, stream). Behaviour is intentionally kept, but this line
            must be re-checked on every JAX upgrade: if ``jax.core`` is
            removed, fall back to ``jax._src.core.Tracer``.
        """
        if self._key is None:
            self._key = jax.random.PRNGKey(self.base)
        new_key, sub = jax.random.split(self._key)
        if isinstance(sub, jax.core.Tracer):
            raise RuntimeError(
                "A direct draw (sample=True) was requested while JAX is tracing, "
                "typically inside a model passed to m.fit()/m.svi(). sample=True "
                "draws fixed data for simulation; inside a model declare a latent "
                "parameter instead, e.g. m.dist.normal(0, 1, name='alpha')."
            )
        self._key = new_key
        return sub

    def get_seed(self):
        """Return the resolved integer base seed (useful when ``rand_seed=True``)."""
        return self.base

    def set_seed(self, seed):
        """Reset the manager to a new base seed and rewind the stream.

        Consistency with the constructor path (RELEASE_AUDIT, Utils,
        set_seed): when this RNG was built through ``bf(seed_global_rngs=True)``
        the numpy/stdlib globals were seeded from the base seed, so
        ``set_seed`` re-seeds them with the new base — the JAX and numpy
        halves of the session stay in sync. Without that opt-in the globals
        are never touched (the default ``bf()`` contract).
        """
        self.rand_seed = seed
        self.base = self._resolve_base(seed)
        self._key = None  # rewound: rebuilt from the new base on next stream()
        if self._globals_seeded:
            self.seed_globals()

    def seed_globals(self):
        """Seed the NumPy and stdlib global RNGs so non-JAX paths reproduce too.

        Opt-in: ``bf()`` calls this only when constructed with
        ``seed_global_rngs=True``; by default BF never touches the user's
        global ``numpy.random`` / ``random`` state. Once called, later
        :meth:`set_seed` calls re-apply the new base to the globals (see
        :meth:`set_seed`).
        """
        import random as _pyrand

        try:
            import numpy as _np

            _np.random.seed(self.base)
        except ImportError:
            pass
        _pyrand.seed(self.base)
        self._globals_seeded = True
