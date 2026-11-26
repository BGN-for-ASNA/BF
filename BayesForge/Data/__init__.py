"""Data subpackage of BayesForge: preprocessing and manipulation tools that
prepare pandas tables for JAX-based models.

The module :mod:`BayesForge.Data.manip` defines the ``manip`` class, the base
class of the public ``bf`` model class (``class bf(manip)`` in
``BayesForge/Main/main.py``): loading (``data``), encoding (``OHE``,
``index``), scaling (``scale``, ``z_score``), dtype conversion
(``to_float``, ``to_int``) and conversion to JAX arrays (``pd_to_jax``,
``data_to_model``).

This ``__init__`` is intentionally a docstring only: it
makes ``Data`` a regular package for packaging tools (``find_packages()``)
without importing anything, so importing the subpackage stays side-effect
free and cannot create import cycles with ``BayesForge.Main``.
"""
