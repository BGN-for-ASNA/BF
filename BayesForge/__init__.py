import warnings as _warnings

# JAX compatibility monkey-patch for older NumPyro versions.
# RELEASE_AUDIT X9: when arviz (lazy_loader) is imported first,
# jax.interpreters.pxla sits in sys.modules without being attached to its
# parent package, so the attribute access `jax.interpreters.pxla` raised
# AttributeError and `import BayesForge` crashed. `import ... as` resolves
# the module from sys.modules; re-attach it, and say so if the patch fails.
try:
    import jax.interpreters as _jax_interpreters
    import jax.interpreters.pxla as _pxla
    if not hasattr(_jax_interpreters, "pxla"):
        _jax_interpreters.pxla = _pxla
    if not hasattr(_pxla, "xla_pmap_p"):
        _pxla.xla_pmap_p = getattr(_pxla, "pmap_p", None)
except (ImportError, AttributeError) as _e:
    _warnings.warn(f"BayesForge: JAX pxla compatibility patch not applied ({_e!r}).",
                   RuntimeWarning)

# ArviZ >=1.3 pulls NumPyro through lazy_loader; when arviz is imported before
# numpyro the `numpyro.infer.initialization` submodule ends up in sys.modules
# without being attached to its parent, so arviz_base.io_numpyro.infer_dims
# raises AttributeError. Re-attach it.
try:
    import sys as _sys
    import numpyro.infer as _npi
    import numpyro.infer.initialization  # noqa: F401
    if not hasattr(_npi, "initialization"):
        _npi.initialization = _sys.modules["numpyro.infer.initialization"]
except (ImportError, AttributeError, KeyError) as _e:
    # X9: no longer silent -- without this patch az.from_numpyro fails later.
    _warnings.warn(f"BayesForge: numpyro.infer.initialization patch not applied ({_e!r}).",
                   RuntimeWarning)

from .Main.main import bf, BF
from .SetDevice.set import setup_device
from .Resources.datasets import load
from .Parallel.simulate import grid, run_simulations
import importlib.metadata

BayesForge = BF

try:
    __version__ = importlib.metadata.version("BayesForge")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

# BF_QUIET is set by run_simulations for its workers: with a pool of them the
# banner prints once per process and drowns the actual output.
import os as _os
if _os.environ.get("BF_QUIET", "0") != "1":
    print(f"bf v {__version__} package loaded")
__all__ = ["bf", "BF", "BayesForge", "setup_device", "load",
           "run_simulations", "grid"]
