"""Distribution backends for BayesForge.

Two interchangeable facades expose a ``UnifiedDist`` class over the same
service arguments (``shape``, ``sample``, ``seed``, ``obs``, ``mask``,
``name``, ``wrap``, ``create_obj``), each on a different library:

- ``BayesForge.Distributions.np_dists``: numpyro backend.
- ``BayesForge.Distributions.tfp_dists``: TensorFlow Probability
  (JAX substrate) backend.

Both modules are imported explicitly by the callers (for example
``from BayesForge.Distributions.tfp_dists import UnifiedDist``); nothing is
re-exported here so that importing the package stays free of heavy
dependencies (jax, numpyro, tensorflow_probability).
"""
