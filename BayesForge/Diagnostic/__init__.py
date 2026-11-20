"""
BayesForge.Diagnostic — posterior diagnostics for BayesForge samplers.

Two public entry points live here; only the first one is supported
(RELEASE_AUDIT bugs 18/21/26):

- :class:`~BayesForge.Diagnostic.Diag2.diagWIP` — the active diagnostic
  object, installed as ``m.diag`` by the model wrapper. Works on the raw
  sampler output (numpyro ``MCMC``, TFP-style ``.posterior`` lists, SVI
  objects, or a plain ``{var: chain-major array}`` dict), and implements
  ESS / R-hat / MCSE / LOO / WAIC and the plot family on top of the pure
  implementations in :mod:`BayesForge.Diagnostic.jax_diagnostics`.

- :class:`~BayesForge.Diagnostic.Diag.diag` — legacy arviz-based wrapper
  kept only so old scripts keep running. New code should not use it.

:mod:`BayesForge.Diagnostic.jax_diagnostics` is the leaf module holding the
shared numerical helpers (``_acf``, ``_ebfmi_per_chain``, ``ess``, ``rhat``,
``summary``, ``loo``, ``filter_posterior_dict``); ``Diag2``, ``patch_diag``
and ``sensitivity`` import from it instead of carrying their own copies.
"""
