import inspect

# import jax.random as random
# import jax.numpy as jnp
# from jax import vmap
# from jax import jit
import jax
import jax.numpy as jnp


from numpyro.infer import Predictive
from numpyro import handlers
import numpyro
try:
    import arviz as az
except Exception:
    # matplotlib>=3.11 removed matplotlib.style.core, which arviz_plots imports.
    # Shim it (the paths live on matplotlib.style directly) and retry; arviz is
    # only needed by summary(), so defer to None if it still fails.
    import sys as _sys, types as _t
    import matplotlib.style as _ms
    if not hasattr(_ms, "core"):
        _shim = _t.ModuleType("matplotlib.style.core")
        _shim.USER_LIBRARY_PATHS = getattr(_ms, "USER_LIBRARY_PATHS", [])
        _shim.STYLE_EXTENSION = getattr(_ms, "STYLE_EXTENSION", "mplstyle")
        _shim.reload_library = getattr(_ms, "reload_library", lambda: None)
        _ms.core = _shim
        _sys.modules["matplotlib.style.core"] = _shim
    try:
        import arviz as az
    except Exception:
        az = None

import functools
import cloudpickle
import os
import pickle
import warnings
import numpy as np

from BayesForge.SetDevice.set import setup_device, setup_multi_device


def _bf_version():
    """Installed BayesForge version, stamped on saved objects."""
    import importlib.metadata
    try:
        return importlib.metadata.version("BayesForge")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"

from BayesForge.Data.manip import manip
from BayesForge.Resources.datasets import load

from BayesForge.Utils.Gaussian import gaussian
from BayesForge.Utils.Effects import effects
from BayesForge.Utils.link import link
from BayesForge.Utils.infer_discrete import sample_discrete_posterior

from BayesForge.Diagnostic.Diag2 import diagWIP as diag
#from BayesForge.Diagnostic.Diag import diag as diag

from BayesForge.Network.Net import net, net2
from BayesForge.NBDA.NBDA import NBDA
from BayesForge.Causal.dag import causal, DAG

from BayesForge.Models.models import models
from BayesForge.Models.surv import survival
from BayesForge.Models.GMM import *
from BayesForge.Models.DPMM import *
from BayesForge.ML.ml import ml
from BayesForge.BNN.bnn import bnn


def _as_count(value, name, minimum=0):
    """Return a whole-number float (e.g. 1000.0 passed from R) as an int.

    Rejects non-numeric values (RELEASE_AUDIT M-20: num_chains="4" used to
    pass through and fail deep inside NumPyro) and counts below ``minimum``
    (num_samples=0, thinning=0), with a message naming the argument.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, float)):
        raise ValueError(
            f"{name} must be a whole number >= {minimum}, got {value!r} "
            f"({type(value).__name__}).")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{name} must be a whole number, got {value}")
        value = int(value)
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


class _ExpandObserved(numpyro.primitives.Messenger):
    """Simulate observed sites at the shape of the data they were fitted on.

    During posterior predictive checks or simulations where `obs=None`, sample
    sites that were originally observed are treated as unobserved (`is_observed=False`).
    If likelihood parameters are scalars or broadcasted against the original data
    (e.g., `dist.Poisson(lam, obs=y)` with a scalar `lam`), NumPyro defaults to the
    distribution's intrinsic batch shape, drawing a single value instead of `len(y)`.

    This messenger intercepts sample messages for sites present in `shapes`,
    extracts the target batch shape (accounting for `event_shape`), and expands
    the distribution via `d.expand(...)` so that synthetic draws match the
    dimensions of the fitted observed data.
    """

    def __init__(self, fn, shapes):
        super().__init__(fn)
        self.shapes = shapes

    def process_message(self, msg):
        if (msg["type"] != "sample" or msg["name"] not in self.shapes
                or msg.get("is_observed")):
            return
        d = msg["fn"]
        shape = self.shapes[msg["name"]]
        n_event = len(d.event_shape)
        if len(shape) < n_event:
            # M-39: the negative slice below would silently truncate the
            # batch to shape[:negative] — fewer values than the fitted data,
            # the exact failure this class exists to prevent. Fail loudly.
            raise ValueError(
                f"sample(): observed site '{msg['name']}' was fitted on data "
                f"of shape {shape} (rank {len(shape)}), below the "
                f"distribution's event rank {n_event} "
                f"{tuple(d.event_shape)}: the stored data no longer matches "
                "the model.")
        batch = shape[:len(shape) - n_event] if n_event else shape
        if tuple(d.batch_shape) == tuple(batch):
            return
        try:
            msg["fn"] = d.expand(jnp.broadcast_shapes(tuple(d.batch_shape), tuple(batch)))
        except (ValueError, TypeError, NotImplementedError) as e:
            # M-39: leave the distribution as it is, but say so: the draws
            # keep the distribution's own batch shape and may not match the
            # fitted data.
            warnings.warn(
                f"sample(): could not expand observed site '{msg['name']}' "
                f"from batch shape {tuple(d.batch_shape)} to the fitted "
                f"shape {shape} ({e}); the simulated draws may have a "
                "different shape than the fitted data.",
                UserWarning, stacklevel=2)


class bf(manip):
    def __init__(
        self,
        platform="cpu",
        cores=None,
        gpu_index=None,
        rand_seed=True,
        deallocate=False,
        print_devices_found=True,
        backend="numpyro",
        diag_backend="arviz",
        float_precision=None,
        loss=None,
        optim=None,
        guide=None,
        seed_global_rngs=False,
    ):
        """
        Initialize the BF class with platform, cores, gpu_index, deallocate, print_devices_found, and backend parameters.

        Args:
            platform (str, optional): Platform to use. Defaults to 'cpu'.
            cores (int, optional): Number of cores. Defaults to None.
            gpu_index (int, optional): Index of the GPU to use. Defaults to None.
            float_precision (int or str, optional): JAX floating-point precision — 32 or 64 (also 'float32'/'float64'). Defaults to None: 64 for backend='numpyro', 32 for backend='tfp'. An explicit 64 is honoured on both backends.
            deallocate (bool, optional): Whether to deallocate. Defaults to False.
            print_devices_found (bool, optional): Whether to print devices found. Defaults to True.
            backend (str, optional): Backend to use, 'numpyro' (default) or
                'tfp'. With 'tfp' the model is a generator turned into a
                TFP JointDistributionCoroutine: write every site as
                ``yield m.dist.<dist>(...)``. Sites are matched by position,
                so every observed site (``obs=``) must be yielded after all
                latent parameters; a model may have several observed sites
                (all at the end), each with its own ``name=`` (unnamed sites
                share a default name and TFP rejects the duplicate). A model
                breaking this order raises ValueError at fit().
            diag_backend (str, optional): Engine behind the convergence metrics
                exposed by ``m.summary()`` and ``m.diag.{summary,rhat,ess,mcse,
                loo,WAIC}``. ``'arviz'`` (default) uses ArviZ, the reference
                implementation. ``'jax'`` selects the experimental
                JAX/NumPy diagnostics and emits a warning. Plots, PPC,
                sensitivity and regression overlays are unaffected either way.
                Defaults to 'arviz'.
            shard (bool, optional): Enable EXPERIMENTAL data sharding (split the
                leading axis of data arrays across devices). Requires more than
                one device (cores>1 on CPU or multiple GPUs). Off by default;
                when enabled a warning describes the model classes for which it
                is not yet correct. Equivalent to setting BF_SHARD=1. Defaults to
                False.
            seed_global_rngs (bool): also seed numpy.random and random with the
                BF base seed (off by default: BF uses its own JAX keys and never
                needs the global RNGs).
        """
        manip.__init__(self)
        from BayesForge.Utils.dev_warning import reset_dev_warnings
        reset_dev_warnings()
        # Default precision depends on the backend: TFP samplers run in 32
        # bits unless the user explicitly asks for 64 (RELEASE_AUDIT M-07).
        if float_precision is None:
            float_precision = 32 if backend == "tfp" else 64
        setup_device(
            platform=platform,
            cores=cores,
            gpu_index=gpu_index,
            deallocate=deallocate,
            print_devices_found=print_devices_found,
            float_precision=float_precision,
        )
        # setup_device() above already rejected anything but 32/64 (and their
        # string forms) and configured jax_enable_x64 accordingly. Keep the
        # normalised value so the tfp branch below can honour an explicit
        # float_precision=64 instead of silently forcing x64 off (M-07).
        self._float_precision = (32 if float_precision in (32, "32", "float32")
                                 else 64)

        # Multi-device setup ----------------------------------------------------
        # Works on both CPU (virtual devices via XLA_FLAGS) and GPU (physical).
        # Provides m.n_devices, m.n_gpus, m.shard(), m.replicate().
        import jax
        self._platform = platform
        self.n_gpus    = len(jax.devices('gpu')) if platform == 'gpu' else 0
        all_devices    = jax.devices(platform)
        if cores is not None and int(cores) > 0 and len(all_devices) > int(cores):
            all_devices = all_devices[:int(cores)]
        self.n_devices = len(all_devices)
        self._data_sharding = None
        self._rep_sharding  = None
        self.mesh = None
        self._last_shard_report = None
        if self.n_devices > 1:
            self.mesh, self._data_sharding, self._rep_sharding = \
                setup_multi_device(platform, cores=cores)
        # -----------------------------------------------------------------------

        # Sharding is declared per-fit via fit(shard=True); it is NOT a
        # construction-time concern (the device mesh above is built whenever
        # cores>1 regardless). BF_SHARD=1 only sets the *default* for
        # fit(shard=None). The EXPERIMENTAL-limits warning is printed lazily on
        # the first sharded fit (see _maybe_warn_shard_limits).
        import os as _os
        self._auto_shard = _os.environ.get('BF_SHARD', '0') == '1'
        self._shard_warned = False

        # M-20: R passes whole doubles (rand_seed=1000.0); accept them like
        # the fit counters do (_as_count) instead of raising "Invalid
        # rand_seed type" deep inside RNG, and reject fractional floats here.
        if isinstance(rand_seed, float):
            if not rand_seed.is_integer():
                raise ValueError(
                    f"rand_seed must be a whole number, got {rand_seed}")
            rand_seed = int(rand_seed)
        self.seed = rand_seed
        from BayesForge.Utils.rng import RNG

        self._rng = RNG(rand_seed)
        if seed_global_rngs:
            self._rng.seed_globals()
        self.data_on_model = None
        self._data_from_df = False  # data_on_model was built by pd_to_jax from m.df
        self.priors_name = None
        self.tab_summary = None
        self.posteriors_full = None
        self.posteriors_by_chain_full = None

        self.nbdaModel = False
        self.obs_args = None
        self.model2 = None
        self.trace = None
        self.history = {}
        if backend not in ("numpyro", "tfp"):
            raise ValueError(f"Unknown backend {backend!r}: use 'numpyro' or 'tfp'.")
        self.backend = backend
        # Engine behind the convergence metrics. ArviZ is the default and the
        # reference; "jax" selects the experimental jax_diagnostics path.
        from BayesForge.Diagnostic.patch_diag import _resolve_backend
        self.diag_backend = _resolve_backend(diag_backend)
        self.loss = loss
        self.optim = optim
        self.guide = guide

        self.gaussian = gaussian
        self.survival = survival(self)
        self.effects = effects
        self.link = link

        self.dpmm = dpmm
        self.gmm = gmm
        self.NBDA = NBDA

        self.models = models(self)

        self.model_name = None
        self.run_model_name = None

        self.net = net()
        self.net2 = net2()
        self.ml = ml()
        self.bnn = bnn()
        self.causal = causal()
        self.dag = DAG

        self.load = load()

        if backend == "numpyro":
            from BayesForge.Distributions.np_dists import UnifiedDist as np_dists, DistProxy

            _unified = np_dists(rand_seed=self.seed, rng=self._rng)
            self.dist = DistProxy(_unified)
            # m.bnn layers draw from the same seeded RNG stream as m.dist, so
            # sample=True draws made through BNN layers are reproducible too.
            self.bnn.dist = _unified

            # effects/model_effects/gaussian/DPMM/GMM/surv each hold their own
            # module-level `dist` singleton, instantiated with UnifiedDist's
            # default (rand_seed=True, rng=None). Left alone that: (1) ignores
            # whatever rand_seed was requested here (always time.time_ns() on
            # sample=True), and (2) even once seeded, _draw_key falls back to
            # a literal fixed PRNGKey(seed) reused on every call rather than
            # _rng.stream()'s fresh independent sub-key -- so e.g. lambda_i and
            # delta_ij from separate m.effects.varying_effects(sample=True)
            # calls would be correlated (same key), not independent draws.
            # Share this instance's RNG manager with those singletons so the
            # whole DGP -- not just m.dist -- is both reproducible and
            # properly decorrelated.
            self._bind_library_rng()

        elif backend == "tfp":
            from BayesForge.Distributions.tfp_dists import UnifiedDist as tfp_dists
            from BayesForge.Samplers.mcmc_tfp import mcmc as mcmc_tfp

            self.dist = tfp_dists
            tfp_dists._rng = self._rng
            # RELEASE_AUDIT W13(a): BNN layers emit numpyro sites and are
            # numpyro-only; under tfp only their direct sample=True draws are
            # usable. Make those follow the bf seed instead of the
            # time-based seed of bnn()'s own UnifiedDist.
            self.bnn.dist.seed = self.seed
            self.bnn.dist._rng = self._rng
            self.sampler = mcmc_tfp()
            # M-07: do not silently override float_precision=64 for the whole
            # process. Honour the user's choice; only force x64 off for the
            # usual TFP 32-bit setting.
            if self._float_precision == 64:
                warnings.warn(
                    "backend='tfp' with float_precision=64: TFP samplers run "
                    "in float64 (slower); pass float_precision=32 for the "
                    "usual TFP setting.", UserWarning, stacklevel=2)
            else:
                jax.config.update("jax_enable_x64", False)

    def _resolve_key(self, seed, label):
        """Resolve a JAX PRNGKey for an inference call.

        If ``seed`` is given explicitly it is honoured verbatim; otherwise the
        central RNG manager supplies a stable, label-derived key governed by
        ``rand_seed`` (reproducible for False/int, entropy-based for True).
        Accepts ints (and whole-number floats) and jax PRNGKeys, old-style
        (uint32 pairs) and new-style typed keys; rejects bools and fractional
        floats instead of silently mangling them (RELEASE_AUDIT M-14).
        """
        if seed is None:
            return self._rng.key(label)
        if isinstance(seed, bool):
            raise TypeError(
                f"{label}: seed must be an int or a jax PRNGKey, not bool; "
                "use rand_seed=True on bf() for entropy-based seeds.")
        if hasattr(seed, "shape") and getattr(seed, "shape", ()) != ():
            return seed  # already a PRNGKey (e.g. jax.random.PRNGKey(0))
        try:
            if jax.dtypes.issubdtype(getattr(seed, "dtype", None),
                                     jax.dtypes.prng_key):
                return seed  # new-style typed key (scalar shape)
        except Exception:
            pass  # older jax without jax.dtypes.prng_key: fall through to int
        if isinstance(seed, float) and not float(seed).is_integer():
            raise TypeError(f"{label}: seed must be an integer, got {seed!r}.")
        return jax.random.PRNGKey(int(seed))

    def _check_int_seed(self, seed, label):
        """Validate a seed that must stay an int (e.g. get_samples(seed=)).

        Same bool/float rules as :meth:`_resolve_key`, but returns an int
        instead of a PRNGKey (RELEASE_AUDIT M-14).
        """
        if isinstance(seed, bool):
            raise TypeError(
                f"{label}: seed must be an int, not bool; "
                "use rand_seed=True on bf() for entropy-based seeds.")
        if hasattr(seed, "shape") and getattr(seed, "shape", ()) != ():
            raise TypeError(
                f"{label}: this seed must be an int; a jax PRNGKey cannot be "
                "used here.")
        if isinstance(seed, float) and not float(seed).is_integer():
            raise TypeError(f"{label}: seed must be an integer, got {seed!r}.")
        return int(seed)

    def _check_no_data_clash(self, data, keywords):
        """Reject data entries whose names collide with sampler run() keywords.

        ``**data`` next to an explicit ``kw=...`` of the same name is a raw
        Python "multiple values for keyword argument" TypeError (or, worse, a
        silent rebinding); raise a message naming the offending model
        arguments instead (RELEASE_AUDIT M-13).
        """
        clash = set(data) & set(keywords)
        if clash:
            raise TypeError(
                f"model argument(s) {sorted(clash)} collide with keyword(s) "
                "of the sampler's run(); rename them in the model signature.")

    def _builtin_model(self, model):
        """Identify a BayesForge built-in model by identity: "gmm", "pca", None.

        ``model.__name__`` is user-controlled, so name-based dispatch let a
        plain function named "pca"/"gmm" fall into the built-in branches and
        crash on ``get_model`` (RELEASE_AUDIT M-11). The built-in PCA is
        passed as an *instance* of BayesForge.Models.PCA.pca
        (``m.fit(m.models.pca(type=...))``); the built-in GMM is the ``gmm``
        function of BayesForge.Models.GMM (``m.fit(m.models.gmm, ...)``).
        """
        func = model.func if isinstance(model, functools.partial) else model
        try:
            from BayesForge.Models.PCA import pca as _PCA
        except ImportError:
            _PCA = None
        if _PCA is not None and isinstance(func, _PCA):
            return "pca"
        if (getattr(func, "__name__", None) == "gmm"
                and getattr(func, "__module__", "") == "BayesForge.Models.GMM"):
            return "gmm"
        return None

    def _check_model_object(self, model):
        """Reject registry objects fit() cannot consume, with a usable hint.

        ``m.models`` exposes heterogeneous objects (RELEASE_AUDIT Integration
        W10): ``pca`` and ``nbda`` are classes and ``survival`` is an instance
        without ``__call__``; all of them reached fit() as plausible-looking
        models and crashed deep inside ``pd_to_jax`` or on ``model.__name__``.
        Fail at the entry of fit() instead, naming the object to pass.
        """
        if model is None:
            raise TypeError(
                "fit(model=None): pass a model function, "
                "e.g. m.fit(nbda.model, ...) for NBDA.")
        if inspect.isclass(model):
            try:
                from BayesForge.Models.PCA import pca as _PCA
            except ImportError:
                _PCA = None
            if _PCA is not None and issubclass(model, _PCA):
                raise TypeError(
                    "m.models.pca is a class: pass an instance, e.g. "
                    "m.fit(m.models.pca(type='classic'), ...) "
                    "(type: 'classic', 'ARD', 'robust', 'sparse' or "
                    "'sparse_robust_ard').")
            try:
                from BayesForge.NBDA.NBDA import NBDA as _NBDA
            except ImportError:
                _NBDA = None
            if _NBDA is not None and issubclass(model, _NBDA):
                raise TypeError(
                    "m.models.nbda is the NBDA class: build the NBDA object "
                    "first and pass its model: nbda = m.models.nbda(network, "
                    "status); m.fit(nbda.model).")
            return
        if not callable(model):
            raise TypeError(
                f"{type(model).__name__} object is not a model function; "
                "pass its model method, e.g. m.fit(m.models.survival.model, ...)"
                " for the survival model.")

    def get_seed(self):
        """Return the resolved base seed governing reproducibility.

        Returns:
            int: The resolved base integer seed stored in ``m._rng``.
        """
        return self._rng.get_seed()

    def set_seed(self, seed):
        """Reset the reproducibility base seed for subsequent draws.

        Args:
            seed (int or bool): New seed value (integer for deterministic stream,
                ``True`` for fresh entropy, ``False`` for default seed 0).

        Returns:
            None: Updates ``m`` in-place.
            Updates on m: sets ``m.seed = seed`` and resets the internal random
                number generator ``m._rng`` for all subsequent stochastic calls.
        """
        self._rng.set_seed(seed)
        self.seed = seed
        self._bind_library_rng()

    def _bind_library_rng(self):
        """Bind the library's module-level dist singletons to THIS instance's seed/RNG.

        The built-in model modules (Effects, Gaussian, model_effects,
        model_effects2, DPMM, GMM, surv, PCA) draw through a module-level
        ``dist`` singleton shared by
        every bf instance, so without rebinding the last constructed instance
        would own the RNG for the whole process (RELEASE_AUDIT M-08). Bind
        those singletons to this instance's seed/RNG; called at construction
        and at the start of every inference call (fit, svi, the SBI entry
        points, sample) so the instance that runs owns the RNG.

        For backend="tfp" the same problem lives on the UnifiedDist class:
        _draw_key is a @staticmethod reading the literal ``UnifiedDist._rng``
        (BayesForge/Distributions/tfp_dists.py), so a per-instance subclass
        would be ignored; rebind the class attribute at these same call sites
        instead (RELEASE_AUDIT M-08).
        """
        if self.backend == "tfp":
            from BayesForge.Distributions.tfp_dists import UnifiedDist as tfp_dists
            tfp_dists._rng = self._rng
            return
        if self.backend != "numpyro":
            return
        import BayesForge.Utils.Effects as _effects_mod
        import BayesForge.Utils.Gaussian as _gaussian_mod
        import BayesForge.Network.model_effects as _model_effects_mod
        import BayesForge.Network.model_effects2 as _model_effects2_mod
        import BayesForge.Models.DPMM as _dpmm_mod
        import BayesForge.Models.GMM as _gmm_mod
        import BayesForge.Models.surv as _surv_mod
        import BayesForge.Models.PCA as _pca_mod

        for _mod in (
            _effects_mod,
            _gaussian_mod,
            _model_effects_mod,
            _model_effects2_mod,
            _dpmm_mod,
            _gmm_mod,
            _surv_mod,
            _pca_mod,
        ):
            # RELEASE_AUDIT W2: model_effects2 and PCA were missing from this
            # list. PCA's singleton is named _dist (its module-level ``dist``
            # is still the UnifiedDist class, which must not be rebound).
            for _attr in ("dist", "_dist"):
                _singleton = getattr(_mod, _attr, None)
                if _singleton is not None and not isinstance(_singleton, type):
                    _singleton.seed = self.seed
                    _singleton._rng = self._rng

        # RELEASE_AUDIT MAJ-7: the SBI npe helper holds a module-level bnn()
        # singleton (npe._BNN) with the same shared-RNG problem. Import it
        # lazily so bf keeps working without the SBI dependencies installed.
        try:
            import BayesForge.Samplers.SBI.npe as _npe_mod
        except ImportError:
            _npe_mod = None
        if _npe_mod is not None and getattr(_npe_mod, "_BNN", None) is not None:
            _npe_mod._BNN.dist.seed = self.seed
            _npe_mod._BNN.dist._rng = self._rng

    def latex(self, model=None, data=None):
        """Write the fitted model as LaTeX equations (one line per prior and likelihood).

        .. warning:: Experimental feature (see ``to_latex``).

        The equations are built from the model's Python AST (expanding JAX, link,
        and BF building-block calls into mathematical notation) or from an
        execution trace when Python source is unavailable (e.g. R/Julia models).

        Args:
            model: Optional model function to render. Defaults to ``self.model``.
            data: Optional dict of model inputs (defaults to ``self.data_on_model``
                when rendering ``self.model``).

        Returns:
            str: A LaTeX ``align*`` environment.
        """
        from BayesForge.PostModel.to_latex import to_latex

        if model is None:
            model = self.model
        if data is None and model is getattr(self, "model", None):
            data = getattr(self, "data_on_model", None)
        return to_latex(model, data=data)

    def graph(self, model=None, figsize=(8, 6), title=None, draw=True, data=None):
        """Build (and draw) the directed graph of a probabilistic model.

        .. warning:: Experimental feature (see ``model_to_graph``).

        Combines AST analysis and execution-based JAX tracing to reconstruct
        latent, observed, deterministic and data nodes and the dependencies
        between them.

        Args:
            model: Model function to graph. Defaults to the current model.
            figsize: Figure size passed to the renderer.
            title: Optional plot title.
            draw: If True (default) render with matplotlib; otherwise just
                return the :class:`ModelGraph` object.
            data: Optional dict of model inputs (defaults to ``self.data_on_model``
                when graphing ``self.model``).

        Returns:
            The :class:`ModelGraph` instance (inspect ``.nodes``/``.edges``).
        """
        from BayesForge.PostModel.model_graph import model_to_graph

        if model is None:
            model = self.model
        if data is None and model is getattr(self, "model", None):
            data = getattr(self, "data_on_model", None)
        g = model_to_graph(model, data=data)
        if draw:
            g.draw(figsize=figsize, title=title)
        return g

    @staticmethod
    def _check_data_matches_model(model, data):
        """Raise if data_on_model cannot be the data of ``model`` (stale from a previous fit)."""
        import inspect
        fn = model.func if isinstance(model, functools.partial) else model
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return
        bound = set(getattr(model, "keywords", {}) or {})
        names = set(params)
        extra = sorted(set(data) - names)
        missing = sorted(n for n, p in params.items()
                         if p.default is inspect.Parameter.empty
                         and p.kind not in (inspect.Parameter.VAR_POSITIONAL,)
                         and n not in data and n not in bound)
        if extra or missing:
            raise ValueError(
                f"m.data_on_model does not match {getattr(fn, '__name__', 'the model')}: "
                f"unexpected keys {extra}, missing arguments {missing}. It may be left over "
                "from a previous fit: pass the data with fit(model, obs=...) or reset "
                "m.data_on_model = None to rebuild it from m.df.")

    def fit(
        self,
        model=None,
        obs=None,
        potential_fn=None,
        kinetic_fn=None,
        step_size=1.0,
        inverse_mass_matrix=None,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
        target_accept_prob=0.8,
        trajectory_length=None,
        max_tree_depth=10,
        init_strategy=numpyro.infer.init_to_uniform,
        find_heuristic_step_size=False,
        forward_mode_differentiation=False,
        regularize_mass_matrix=True,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,
        thinning=1,
        postprocess_fn=None,
        chain_method="parallel",
        shard=None,
        progress_bar=True,
        jit_model_args=False,
        seed=None,
        extra_fields=None,
        sampler="nuts",
        gibbs_fn=None,
        gibbs_sites=None,
        num_discrete_updates=None,
        init_params=None,
    ):
        """
        Fit the model using the specified backend and parameters.

        How the model is identified: ``model_name`` is taken from the
        callable's ``__name__`` (a ``functools.partial`` is unwrapped); the
        built-in GMM/PCA are detected by identity, never by name. A
        ``lambda`` therefore gets the name ``"<lambda>"``, which disables
        the built-in special cases and the ``m.plot()`` dispatch (a
        UserWarning is emitted). ``m.models.available`` is a catalogue
        only — fit() does not read it.

        Args:
            model: Model to fit.
            obs: Observed data.
            potential_fn: Potential function.
            kinetic_fn: Kinetic function.
            step_size: Step size for the sampler.
            inverse_mass_matrix: Inverse mass matrix.
            adapt_step_size: Whether to adapt step size.
            adapt_mass_matrix: Whether to adapt mass matrix.
            dense_mass: Whether to use dense mass.
            target_accept_prob: Target acceptance probability.
            trajectory_length: Length of the trajectory.
            max_tree_depth: Maximum tree depth.
            init_strategy: Initialization strategy.
            find_heuristic_step_size: Whether to find heuristic step size.
            forward_mode_differentiation: Whether to use forward mode differentiation.
            regularize_mass_matrix: Whether to regularize mass matrix.
            num_warmup: Number of warmup samples.
            num_samples: Number of samples.
            num_chains: Number of chains.
            thinning: Thinning factor.
            postprocess_fn: Postprocess function.
            chain_method: Chain method.
            progress_bar: Whether to show progress bar.
            jit_model_args: Whether to JIT model arguments.
            seed: Random seed.
            sampler: MCMC kernel, one of `"nuts"` (default; also for models
                whose discrete latents are exactly enumerated), `"mixed_hmc"`
                (HMC + Gibbs for mixed discrete/continuous models),
                `"discrete_gibbs"` (NUTS + automatic Gibbs updates for the
                discrete latents) or `"hmc_gibbs"` (NUTS + user-supplied
                `gibbs_fn` over `gibbs_sites`).
            gibbs_fn: Gibbs update callable for `sampler="hmc_gibbs"`
                (`rng_key, gibbs_sites, hmc_sites` -> dict of new samples).
            gibbs_sites: List of site names covered by `gibbs_fn`.
            num_discrete_updates: Number of discrete updates per MixedHMC
                step (default: number of discrete latent sites).
            init_params: Optional dict of initial values keyed by site name,
                forwarded to the sampler. Needed in particular by the Gibbs
                samplers, which condition each block's initialisation on the
                other blocks' starting values (e.g. discrete latents that
                must satisfy downstream constraints such as N >= y). Give
                every latent site: discrete ones as values, continuous ones in
                unconstrained space (e.g. log for a positive parameter). With
                ``num_chains > 1`` each value needs a leading chain dimension.

        Returns:
            None: Fits the model and populates attributes on ``m`` in-place.
            Attributes set on ``m``:
                - ``m.model``: fitted model callable; ``m.run_model_name``
                  holds its name.
                - ``m.data_on_model``: model data, taken from ``obs``. When
                  ``obs`` is ``None`` the data of the previous fit is reused
                  (its keys are checked against the new model's signature and
                  a mismatch raises ``ValueError``); it is rebuilt from
                  ``m.df`` via ``m.pd_to_jax`` only when no data is set yet
                  or the stored data was itself built that way (RELEASE_AUDIT
                  M-36).
                - ``m.num_chains``: number of chains used for the fit.
                - ``m.sampler`` and ``m.sampler_name``: fitted MCMC sampler
                  instance and kernel name.
                - ``m.posteriors`` and ``m.posteriors_by_chain``: pooled and
                  per-chain posterior draws.
                - ``m.posteriors_full`` and ``m.posteriors_by_chain_full``:
                  unfiltered copies preserved for ``m.filter_posteriors``.
                - ``m.extra_fields`` and ``m.extra_fields_by_chain``: HMC
                  trajectory diagnostics (``num_steps``, ``energy``,
                  ``diverging``, ``accept_prob``, ``adapt_state.step_size``).
                - ``m.diag``: initialized diagnostics object bound to ``m``.
                - ``m.history``: fit snapshot dictionary.
                - ``m._last_shard_report``: sharding report, only when
                  ``shard=True``.
        """
        self._check_model_object(model)
        self._bind_library_rng()
        from BayesForge.Samplers.mcmc_numpyro import SAMPLERS

        if sampler not in SAMPLERS:
            raise ValueError(
                f"Unknown sampler {sampler!r}. Choose from {SAMPLERS}."
            )
        if sampler == "hmc_gibbs" and (gibbs_fn is None or gibbs_sites is None):
            raise ValueError(
                'sampler="hmc_gibbs" requires both `gibbs_fn` and '
                "`gibbs_sites` (or use sampler='discrete_gibbs' for "
                "automatic discrete Gibbs updates)."
            )
        if sampler != "nuts" and self.backend != "numpyro":
            raise ValueError(
                f"sampler={sampler!r} needs the numpyro backend; the "
                f"{self.backend!r} backend only runs its own NUTS sampler."
            )
        # Counts arrive as floats from R (where 1000 is a double): accept
        # whole numbers, reject anything else with a clear message.
        num_warmup, num_samples, num_chains, thinning, max_tree_depth = (
            _as_count(v, n, lo) for v, n, lo in (
                (num_warmup, "num_warmup", 0), (num_samples, "num_samples", 1),
                (num_chains, "num_chains", 1), (thinning, "thinning", 1),
                (max_tree_depth, "max_tree_depth", 1)))
        self.num_chains = num_chains

        # Sharding is declared here, per fit. fit(shard=None) (the default)
        # falls back to the BF_SHARD env default; fit(shard=True/False) is
        # explicit. There is no construction-time shard flag.
        if shard is None:
            shard = self._auto_shard

        if shard:
            if self._data_sharding is None:
                import warnings
                warnings.warn(
                    "shard=True has no effect: no multi-device mesh available "
                    f"(n_devices={self.n_devices}). "
                    "Pass cores>1 when creating bf() to enable sharding.",
                    stacklevel=2,
                )
                shard = False
            else:
                self._maybe_warn_shard_limits()   # print limits once, lazily
                chain_method = "vectorized"

        obs_given = obs is not None
        if obs is None:
            obs = self.data_on_model
        else:
            # Own copy: the pca/gmm branches below add or transpose entries,
            # which must not leak into the caller's dict.
            self.data_on_model = dict(obs)
            self._data_from_df = False

        # model is None is rejected by _check_model_object at fit() entry
        # (RELEASE_AUDIT W10d).
        self.model = model
        # Display/save name only; callables without __name__ (R/Julia
        # wrappers, arbitrary callables) must not raise AttributeError
        # (RELEASE_AUDIT M-11).
        self.model_name = getattr(
            model.func if isinstance(model, functools.partial) else model,
            "__name__", type(model).__name__)
        if self.model_name == "<lambda>":
            # W16: a lambda has no usable name, so the built-in detection
            # and the m.plot() dispatch below silently do nothing; say so.
            # Local import: fit() binds warnings locally further down, which
            # makes the module-level name unreadable here (UnboundLocalError).
            import warnings
            warnings.warn(
                "fit: the model is a lambda; built-in model detection and "
                "m.plot() dispatch use the function name -- define a named "
                "function instead.", UserWarning, stacklevel=2)
        # ``nbda.model`` is a bound method whose __name__ is "model":
        # run_model_name, the m.plot() dispatch and the save() name were
        # meaningless (RELEASE_AUDIT W12). Detect the NBDA owner by identity.
        try:
            from BayesForge.NBDA.NBDA import NBDA as _NBDA
        except ImportError:
            _NBDA = None
        if _NBDA is not None and isinstance(
                getattr(model, "__self__", None), _NBDA):
            self.model_name = "nbda"
        # Built-in dispatch goes on identity, never on the (user-controlled)
        # name: a user function named "pca"/"gmm" must not hit the built-in
        # branches below (RELEASE_AUDIT M-11).
        builtin = self._builtin_model(model)

        # The self.nbdaModel flag was never set to True on bf (NBDA.py sets
        # it on the NBDA instance, never read here): the branch was dead
        # (RELEASE_AUDIT W12). Its body now runs unconditionally.
        if self.data_on_model is None or (not obs_given and self._data_from_df):
            # (re)build from m.df for THIS model: data built for a
            # previous model must not be reused for another signature.
            if self.backend == "numpyro":
                self.data_on_model = self.pd_to_jax(self.model)
            else:
                self.data_on_model = self.pd_to_jax(self.model, bit="32")
            self._data_from_df = True
        elif not obs_given and builtin not in ("gmm", "pca"):
            self._check_data_matches_model(self.model, self.data_on_model)

        if builtin == "gmm":
            # K is a scalar hyper-parameter, not a data column (RELEASE_AUDIT
            # W11): a pd_to_jax column arrives as (N,) and breaks
            # KMEANS(n_clusters=...) and plate("components", K). Accept only a
            # Python int or a 0-d integer array, normalised to int.
            K = self.data_on_model.get("K")
            if isinstance(K, bool):
                K = None
            elif isinstance(K, (int, np.integer)):
                K = int(K)
            elif (getattr(K, "ndim", None) == 0
                    and np.issubdtype(K.dtype, np.integer)):
                K = int(K)
            else:
                K = None
            if K is None:
                raise ValueError(
                    "gmm needs the number of clusters K as an int: "
                    "m.fit(m.models.gmm, obs={'data': X, 'K': 3})")
            self.data_on_model["K"] = K
            if "initial_means" not in self.data_on_model:
                self.ml.KMEANS(self.data_on_model["data"], n_clusters=K)
                self.data_on_model["initial_means"] = self.ml.results["centroids"]

        if builtin == "pca":
            self.model = model.get_model(model.type)
            tmp = self.data_on_model.keys()

            if "X" not in tmp:
                # M-23: `return print(...)` left the caller with a silent
                # None and a half-mutated object; fail loudly instead.
                raise ValueError(
                    "pca needs the data matrix X: "
                    "m.fit(m.models.pca, obs={'X': X})")
            # X is (num_data_points, data_dim) end to end (RELEASE_AUDIT P1):
            # never transpose the user's array in place, or a second fit on the
            # same dict would transpose it twice.
            if "data_dim" not in tmp:
                self.data_on_model["data_dim"] = self.data_on_model["X"].shape[1]
            if "latent_dim" not in tmp:
                # M-42: defaulting the latent dimension to the data dimension
                # makes the probabilistic PCA degenerate (the model is as
                # rich as the data). Keep the default for reproducibility of
                # existing fits, but say so and ask for an explicit value.
                import warnings  # fit() imports it locally elsewhere
                warnings.warn(
                    "pca: latent_dim not given, defaulting to data_dim = "
                    f"{self.data_on_model['X'].shape[1]}; a latent dimension "
                    "equal to the data dimension is a degenerate model (no "
                    "reduction). Pass an explicit latent_dim < data_dim in "
                    "obs={'X': X, 'latent_dim': k}.",
                    UserWarning, stacklevel=2)
                self.data_on_model["latent_dim"] = self.data_on_model["X"].shape[1]
            if "num_data_points" not in tmp:
                self.data_on_model["num_data_points"] = self.data_on_model["X"].shape[0]
            # Init pca class with data
            self.models.pca = self.models.pca(
                X=self.data_on_model["X"],
                latent_dim=self.data_on_model["latent_dim"],
                type=model.type,
            )

        if self.backend == "numpyro":
            from BayesForge.Utils.dev_warning import reset_dev_warnings
            reset_dev_warnings()
            from BayesForge.Samplers.mcmc_numpyro import mcmc_numpyro

            self.sampler = mcmc_numpyro(
                model=self.model,
                potential_fn=potential_fn,
                kinetic_fn=kinetic_fn,
                step_size=step_size,
                inverse_mass_matrix=inverse_mass_matrix,
                adapt_step_size=adapt_step_size,
                adapt_mass_matrix=adapt_mass_matrix,
                dense_mass=dense_mass,
                target_accept_prob=target_accept_prob,
                trajectory_length=trajectory_length,
                max_tree_depth=max_tree_depth,
                init_strategy=init_strategy,
                find_heuristic_step_size=find_heuristic_step_size,
                forward_mode_differentiation=forward_mode_differentiation,
                regularize_mass_matrix=regularize_mass_matrix,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                thinning=thinning,
                postprocess_fn=postprocess_fn,
                chain_method=chain_method,
                progress_bar=progress_bar,
                jit_model_args=jit_model_args,
                sampler=sampler,
                gibbs_fn=gibbs_fn,
                gibbs_sites=gibbs_sites,
                num_discrete_updates=num_discrete_updates,
            )
            self.sampler_name = sampler

            # Improvement 2 — gate data sharding on the explicit *shard* intent
            # only, NEVER on chain_method. Choosing chain_method='vectorized' for
            # its own reasons must not silently trigger data parallelism.
            # Improvement 6 — do not mutate self.data_on_model: keep the canonical
            # (host) copy intact so subsequent fits stay independent. The sharded
            # arrays live only in the local `run_data` passed to this run.
            if shard and self._data_sharding is not None:
                run_data = self._auto_shard_data(self.data_on_model)
                # Safeguards (static HLO + runtime value) run ONCE here, before
                # the MCMC loop — they add a one-time setup cost, no per-step cost.
                # A ✗ runtime verdict (sharding changes the answer) falls back to
                # replicated; ⚠ verdicts (collectives → deadlock/cost) only warn.
                if self._shard_check(self.model, run_data, self.data_on_model):
                    pass  # static-only warnings; keep sharding
                else:
                    run_data = self.data_on_model  # correctness fallback
                    # M-25: the fit runs REPLICATED — rewrite the report
                    # written by _auto_shard_data so shard_report() does not
                    # keep announcing these arrays as sharded.
                    rep = getattr(self, "_last_shard_report", None)
                    if rep and rep.get("sharded"):
                        self._last_shard_report = {
                            **rep,
                            "sharded": [],
                            "replicated_shard_check": list(rep["sharded"]),
                        }
            else:
                run_data = self.data_on_model
            # These were already being requested from NUTS but never read back,
            # so leapfrog counts and energy were collected and thrown away.
            # step_size is added so the adapted value can be reported too.
            # Gibbs-composite kernels (mixed_hmc / discrete_gibbs / hmc_gibbs)
            # carry a different state tree, so these NUTS/HMC-state fields are
            # only requested for the plain NUTS sampler.
            if extra_fields is None:
                if sampler == "nuts":
                    extra_fields = ('num_steps', 'energy', 'diverging',
                                    'accept_prob', 'adapt_state.step_size')
                else:
                    extra_fields = ()
            if isinstance(extra_fields, str):
                # M-19: tuple("diverging") splits a bare string into single
                # characters; NumPyro then lists one-letter fields.
                raise ValueError(
                    f"extra_fields must be a list or tuple of field names, "
                    f"got the string {extra_fields!r} — a bare string is "
                    f"split into single characters. Use ({extra_fields!r},) "
                    f"for a single field.")
            self._extra_fields_requested = tuple(extra_fields)
            self._max_tree_depth = max_tree_depth
            # M-13: a model argument named like an explicit run() keyword
            # would raise a raw "multiple values for keyword argument".
            self._check_no_data_clash(run_data, ("extra_fields", "init_params"))
            try:
                self.sampler.run(
                    self._resolve_key(seed, "fit"),
                    extra_fields=self._extra_fields_requested,
                    init_params=init_params,
                    **run_data,
                )
            except AssertionError as e:
                if "all sites must have unique names" not in str(e):
                    raise
                raise ValueError(
                    f"{e}. Two model parameters have the same name. Give each "
                    "distribution a unique name= (e.g. a = m.dist.normal(0, 1, "
                    "name='a')). Names are only inferred automatically from the "
                    "assigned variable in Python models and Julia @BF models, "
                    "not in models written in R."
                ) from e
            self.extra_fields = self.sampler.get_extra_fields()
            self.extra_fields_by_chain = self.sampler.get_extra_fields(
                group_by_chain=True)
            # V5: get_samples() returns the sampler's cached dict; copy it so
            # mutating m.posteriors never mutates the sampler state.
            self.posteriors = dict(self.sampler.get_samples())
            self.posteriors_by_chain = dict(
                self.sampler.get_samples(group_by_chain=True))
            self.posteriors_full = dict(self.posteriors)
            self.posteriors_by_chain_full = dict(self.posteriors_by_chain)
            self.diag = diag(sampler=self.sampler)
            from BayesForge.Diagnostic.patch_diag import bind_diag_to_model
            bind_diag_to_model(self.diag, self, backend=self.diag_backend)
            self.get_history()

        elif self.backend == "tfp":
            print(
                "[WARNING] This function is still in development. Use it with caution. [WARNING]"
            )
            from BayesForge.Distributions.tfp_dists import UnifiedDist as tfp_dists
            from BayesForge.Samplers.mcmc_tfp import mcmc as mcmc_tfp
            from BayesForge.Samplers.Model_handler import model_handler

            # self.mcmc= mcmc_tfp()
            self.sampler.data_on_model = self.data_on_model
            self.sampler.model = self.model

            ignored = [n for n, v, d in (("dense_mass", dense_mass, False),
                                         ("max_tree_depth", max_tree_depth, 10),
                                         ("extra_fields", extra_fields, None),
                                         ("init_params", init_params, None),
                                         ("sampler", sampler, "nuts"),
                                         ("chain_method", chain_method, "parallel"))
                       if v != d]
            if ignored:
                import warnings  # fit() imports it locally elsewhere
                warnings.warn(f"backend='tfp' ignores {ignored}.",
                              UserWarning, stacklevel=2)

            # Pass the fit parameters to the TFP sampler. Thinning matches the
            # numpyro backend: num_samples draws are run, num_samples // thinning
            # are kept (TFP keeps one draw every num_steps_between_results + 1 steps).
            # M-13: guard the **data splat against the explicit keywords below.
            self._check_no_data_clash(self.data_on_model, (
                "seed", "model", "n_chains", "num_results", "num_burnin_steps",
                "num_steps_between_results", "target_accept_prob", "step_size"))
            self.sampler.run(
                seed=self._resolve_key(seed, "fit"),
                model=self.model,
                n_chains=num_chains,
                num_results=max(num_samples // thinning, 1),
                num_burnin_steps=num_warmup,
                num_steps_between_results=thinning - 1,
                target_accept_prob=target_accept_prob,
                step_size=step_size if step_size != 1.0 else 1e-3,
                **self.data_on_model,
            )
            self.diag = diag(sampler=self.sampler)
            # V5: get_samples() returns the sampler's cached dict; copy it so
            # mutating m.posteriors never mutates the sampler state.
            self.posteriors = dict(self.sampler.get_samples())
            self.posteriors_by_chain = dict(
                self.sampler.get_samples(group_by_chain=True))
            self.posteriors_full = dict(self.posteriors)
            self.posteriors_by_chain_full = dict(self.posteriors_by_chain)
            from BayesForge.Diagnostic.patch_diag import bind_diag_to_model
            bind_diag_to_model(self.diag, self, backend=self.diag_backend)
            self.get_history()

        if builtin == "pca":
            self.models.pca.posterior = self.posteriors
            self.models.pca.get_attributes(self.models.pca.X)

        # Record which model was just run so post-fit helpers (m.plot, m.history)
        # can dispatch on it. svi() does the same; fit() previously did not,
        # which left m.plot() a silent no-op after a normal m.fit(...).
        self.run_model_name = self.model_name
        self.model_name = None

    def svi(
        self,
        model=None,
        guide=None,
        optim=None,
        loss=None,
        num_steps=1000,
        num_samples=1000,
        seed=None,
        **kwargs,
    ):
        """
        Fit the model using SVI (Stochastic Variational Inference).

        Args:
            model: Model callable to fit (defaults to ``m.model``).
            guide: Variational guide callable or name (``"diagonal"``,
                ``"multivariate"``, ``"laplace"``, ``"delta"``, ``"normal"``,
                ``"low_rank"``). Defaults to ``AutoDiagonalNormal``.
            optim: NumPyro optimizer (defaults to ``Adam(1e-3)``).
            loss: ELBO loss instance (defaults to ``Trace_ELBO(num_particles=10)``).
            num_steps (int): Number of SVI optimization steps. Defaults to 1000.
            num_samples (int): Number of posterior draws sampled from the fitted
                guide. Defaults to 1000.
            seed (int, optional): Random seed.
            **kwargs: Additional keyword arguments forwarded to ``svi_numpyro``.

        Returns:
            None: Fits the variational approximation and populates ``m`` in-place.
            Updates on m: sets ``m.model``, ``m.run_model_name``,
                ``m.data_on_model`` (populated via ``m.pd_to_jax`` if ``None``),
                ``m.sampler`` (``svi_numpyro`` instance), ``m.posteriors``
                (draws from the fitted variational guide), ``m.diag``
                (diagnostics instance), and ``m.history``.
        """
        self._bind_library_rng()
        if model is None:
            model = self.model
        else:
            self.model = model
            # M-11: no AttributeError on callables without __name__.
            self.model_name = getattr(
                model.func if isinstance(model, functools.partial) else model,
                "__name__", type(model).__name__)

        if self.data_on_model is None:
            self.data_on_model = self.pd_to_jax(self.model)

        if optim is None:
            optim = self.optim
            if optim is None:
                import numpyro

                optim = numpyro.optim.Adam(step_size=1e-3)

        if loss is None:
            loss = self.loss
            if loss is None:
                from numpyro.infer import Trace_ELBO

                loss = Trace_ELBO(num_particles=10)

        if guide is None:
            guide = self.guide
            if guide is None:
                from numpyro.infer.autoguide import AutoDiagonalNormal

                guide = AutoDiagonalNormal(self.model)
        elif isinstance(guide, str):
            from numpyro.infer import autoguide

            guides = {
                "diagonal": autoguide.AutoDiagonalNormal,
                "multivariate": autoguide.AutoMultivariateNormal,
                "laplace": autoguide.AutoLaplaceApproximation,
                "delta": autoguide.AutoDelta,
                "normal": autoguide.AutoNormal,
                "low_rank": autoguide.AutoLowRankMultivariateNormal,
            }
            if guide.lower() in guides:
                guide = guides[guide.lower()](self.model)
            else:
                raise ValueError(
                    f"Unknown autoguide: {guide}. Available options: {list(guides.keys())}"
                )

        if self.backend != "numpyro":
            raise NotImplementedError(f"svi() is only available with backend='numpyro' (got {self.backend!r}).")
        if self.backend == "numpyro":
            from BayesForge.Samplers.SVI.svi_numpyro import svi_numpyro

            self.sampler = svi_numpyro(
                model=self.model,
                guide=guide,
                optim=optim,
                loss=loss,
                num_steps=num_steps,
                num_samples=num_samples,
                **kwargs,
            )

            # M-13: svi_numpyro.run(rng_key, num_steps=None, *args, **kwargs)
            # — a data entry with either name would shadow the positional key
            # or silently override num_steps.
            self._check_no_data_clash(self.data_on_model,
                                      ("rng_key", "num_steps"))
            self.sampler.run(
                self._resolve_key(seed, "svi"),
                **self.data_on_model,
            )
            self.posteriors = self.sampler.get_samples(
                seed=(self.get_seed() if seed is None
                      else self._check_int_seed(seed, "svi")),
            )
            # SVI has no chains or HMC statistics: drop the previous fit's
            # caches so filter_posteriors()/summary(include=)/sampler_stats()
            # cannot resurrect the MCMC draws of an earlier fit.
            self.posteriors_full = dict(self.posteriors)
            self.posteriors_by_chain = None
            self.posteriors_by_chain_full = None
            self.extra_fields = None
            self.extra_fields_by_chain = None
            self.diag = diag(sampler=self.sampler)
            # X4: same bound diag surface as fit() (ppc_*, summary, density…).
            from BayesForge.Diagnostic.patch_diag import bind_diag_to_model
            bind_diag_to_model(self.diag, self, backend=self.diag_backend)
            self.get_history()

        self.run_model_name = self.model_name
        self.model_name = None

    # Simulation-based inference (ABC, NPE) ----------------------------------------------------
    def _sbi_setup(self, model, data, params, observed, summary_stats, batch_size):
        """Shared set-up for abc()/npe(): resolve model + data, build the simulator."""
        self._bind_library_rng()
        from BayesForge.Samplers.SBI.simulate import Simulator

        if model is None:
            model = self.model
        else:
            self.model = model
            # M-11: no AttributeError on callables without __name__.
            self.model_name = getattr(
                model.func if isinstance(model, functools.partial) else model,
                "__name__", type(model).__name__)

        if data is not None:
            self.data_on_model = dict(data)
        elif self.data_on_model is None:
            self.data_on_model = self.pd_to_jax(self.model)

        return Simulator(
            self.model,
            data=self.data_on_model,
            params=params,
            observed=observed,
            summary_stats=summary_stats,
            batch_size=batch_size,
        )

    def _sbi_finish(self):
        # V5: copy the sampler's cached dicts (see fit()).
        self.posteriors = dict(self.sampler.get_samples())
        self.posteriors_by_chain = dict(
            self.sampler.get_samples(group_by_chain=True))
        self.posteriors_full = dict(self.posteriors)
        self.posteriors_by_chain_full = dict(self.posteriors_by_chain)
        self.diag = diag(sampler=self.sampler)
        # X4: same bound diag surface as fit() (ppc_*, summary, density…).
        from BayesForge.Diagnostic.patch_diag import bind_diag_to_model
        bind_diag_to_model(self.diag, self, backend=self.diag_backend)
        self.get_history()
        self.run_model_name = self.model_name
        self.model_name = None

    def abc(
        self,
        model=None,
        data=None,
        method="rejection",
        params=None,
        observed=None,
        summary_stats="identity",
        distance="euclidean",
        num_simulations=100_000,
        quantile=0.01,
        epsilon=None,
        regression_adjust=False,
        num_samples=1000,
        num_particles=None,
        num_rounds=10,
        alpha=0.5,
        min_acceptance=0.01,
        batch_size=10_000,
        progress_bar=True,
        seed=None,
    ):
        """Approximate Bayesian Computation: likelihood-free inference by simulation.

        The model is used only as a simulator (prior -> data): its likelihood is
        never evaluated, so it works for models whose likelihood is intractable.

        Args:
            model: BF model. Observed sample sites (``obs=``) are simulated.
            data: dict of model arguments. Defaults to the data set with
                ``m.data`` / ``m.df`` (as in ``fit``).
            method: "rejection" (Pritchard et al. 1999) or "smc" (ABC population
                Monte Carlo, Beaumont et al. 2009).
            params: names of the parameters to infer (default: all latent sites;
                remaining latents are nuisance and integrated out by simulation).
            observed: dict ``{site: value}``; use when the simulator output is a
                ``m.dist.track`` site rather than an ``obs=`` likelihood.
            summary_stats: callable applied to one simulated dataset, or one of
                "identity", "moments", "quantiles", "robust".
            distance: "euclidean" (on MAD-scaled stats), "mahalanobis" or a
                callable ``d(s_sim, s_obs)``.
            num_simulations: rejection: number of simulations; smc: total budget.
            quantile: rejection: fraction of simulations accepted; smc: fraction
                kept in round 0.
            epsilon: tolerance. rejection: accept d <= epsilon (overrides quantile);
                smc: stop once reached.
            regression_adjust: rejection only, local-linear adjustment
                (Beaumont et al. 2002).
            num_samples: posterior draws returned.
            num_particles: smc population size (default ``num_samples``).
            num_rounds, alpha, min_acceptance: smc schedule (tolerance of round t
                is the alpha-quantile of round t-1 distances).
            batch_size: simulations per vectorised batch.
        """
        from BayesForge.Samplers.SBI.abc import ABC_BF

        sim = self._sbi_setup(model, data, params, observed, summary_stats, batch_size)
        self.sampler = ABC_BF(
            sim,
            method=method,
            num_simulations=num_simulations,
            quantile=quantile,
            epsilon=epsilon,
            distance=distance,
            regression_adjust=regression_adjust,
            num_samples=num_samples,
            num_particles=num_particles,
            num_rounds=num_rounds,
            alpha=alpha,
            min_acceptance=min_acceptance,
            progress_bar=progress_bar,
        ).run(self._resolve_key(seed, "abc"))
        self._sbi_finish()

    def abc_rf(
        self,
        model=None,
        data=None,
        params=None,
        observed=None,
        summary_stats="identity",
        stat_names=None,
        num_simulations=10_000,
        n_estimators=500,
        min_samples_leaf=5,
        max_features=1 / 3,
        joint=False,
        num_samples=1000,
        n_jobs=-1,
        batch_size=10_000,
        progress_bar=True,
        seed=None,
    ):
        """ABC random forests (Raynal et al. 2019): likelihood-free inference
        without tolerance or distance.

        A reference table of (theta, s(x)) is simulated from the prior
        predictive and a regression random forest is trained to predict each
        parameter from the summary statistics. The forest's leaves define
        weights on the reference table (quantile regression forest), which give
        the posterior at the observed statistics. Requires scikit-learn.

        Args:
            model, data, params, observed, summary_stats: as in ``abc``. Many
                (even weakly informative) statistics can be supplied: the forest
                selects the useful ones.
            stat_names: names of the summary statistics, used in
                ``m.sampler.importance`` and ``m.sampler.plot_importance()``.
            num_simulations: size of the reference table.
            n_estimators, min_samples_leaf, max_features: forest settings
                (max_features=1/3 of the statistics, as in Raynal et al.).
            joint: False (default) trains one forest per scalar parameter
                (accurate marginals, independent draws across parameters);
                True trains one multi-output forest and keeps posterior
                correlations.
            num_samples: posterior draws returned.
            n_jobs: parallel jobs for scikit-learn (-1 = all cores).
        """
        from BayesForge.Samplers.SBI.abc_rf import ABCRF_BF

        sim = self._sbi_setup(model, data, params, observed, summary_stats, batch_size)
        key = self._resolve_key(seed, "abc_rf")
        self.sampler = ABCRF_BF(
            sim,
            num_simulations=num_simulations,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            joint=joint,
            stat_names=stat_names,
            num_samples=num_samples,
            n_jobs=n_jobs,
            random_state=int(jax.random.randint(key, (), 0, 2**31 - 1)),
            progress_bar=progress_bar,
        ).run(key)
        self._sbi_finish()

    def npe(
        self,
        model=None,
        data=None,
        estimator="mdn",
        bayesian=False,
        params=None,
        observed=None,
        summary_stats="identity",
        num_simulations=10_000,
        hidden=(64, 64),
        n_components=5,
        n_layers=6,
        activation="gelu",
        lr=1e-3,
        batch_size=256,
        max_steps=20_000,
        patience=20,
        prior_scale=1.0,
        svi_steps=5_000,
        num_weight_samples=100,
        num_samples=1000,
        sim_batch_size=10_000,
        progress_bar=True,
        seed=None,
    ):
        """Neural Posterior Estimation: learn q(theta | x) from simulations.

        Trains a neural conditional density estimator on (theta, x) pairs drawn
        from the model's prior predictive. The result is *amortised*: use
        ``m.npe_posterior(new_data)`` to get the posterior of other datasets
        without new simulations or retraining.

        Args:
            model, data, params, observed, summary_stats: as in ``abc``.
            estimator: "mdn" (mixture density network) or "flow" (conditional
                affine-coupling normalizing flow).
            bayesian: if True, Bayesian NPE (Delaunoy et al. 2024): a prior is put
                on the network weights and their posterior approximated with SVI;
                the posterior estimate averages over network draws.
            num_simulations: simulation budget for training.
            hidden, n_components, n_layers, activation: network architecture.
            lr, batch_size, max_steps, patience: Adam training with early
                stopping on a 10% validation split.
            prior_scale: Bayesian NPE weight prior scale (N(0, prior_scale/sqrt(fan_in))).
            svi_steps, num_weight_samples: Bayesian NPE SVI steps and number of
                network draws.
            num_samples: posterior draws returned for the observed data.
        """
        from BayesForge.Samplers.SBI.npe import NPE_BF

        sim = self._sbi_setup(model, data, params, observed, summary_stats, sim_batch_size)
        self.sampler = NPE_BF(
            sim,
            estimator=estimator,
            bayesian=bayesian,
            num_simulations=num_simulations,
            hidden=hidden,
            n_components=n_components,
            n_layers=n_layers,
            activation=activation,
            lr=lr,
            batch_size=batch_size,
            max_steps=max_steps,
            patience=patience,
            prior_scale=prior_scale,
            svi_steps=svi_steps,
            num_weight_samples=num_weight_samples,
            num_samples=num_samples,
            progress_bar=progress_bar,
        ).run(self._resolve_key(seed, "npe"))
        self._sbi_finish()

    def npe_posterior(self, x_obs, num_samples=1000, seed=None):
        """Amortised posterior of a new dataset from the trained NPE (no retraining).

        Args:
            x_obs: new observed data: an array (single observed site) or a dict
                ``{site: value}``.
        Returns:
            dict of posterior draws.
        """
        # M-21: self.sampler does not exist at all before the first
        # npe/fit with the numpyro backend — evaluating it inside hasattr
        # raised AttributeError instead of the promised RuntimeError.
        sampler = getattr(self, "sampler", None)
        if not hasattr(sampler, "sample_posterior"):
            raise RuntimeError("Run m.npe(...) first.")
        return sampler.sample_posterior(
            self._resolve_key(seed, "npe_posterior"), x_obs, num_samples
        )

    def sbc_coverage(self, n_test=200, n_post=500, levels=None, plot=True, ax=None,
                     label=None, seed=None):
        """Expected coverage of the NPE posterior (simulation-based calibration).

        Draws n_test (theta*, x*) pairs from the prior predictive and checks how
        often theta* falls in the posterior highest-density region of each
        nominal level. Calibrated: on the diagonal; below: over-confident.
        """
        sampler = getattr(self, "sampler", None)  # M-21, as in npe_posterior
        if not hasattr(sampler, "coverage"):
            raise RuntimeError("Run m.npe(...) first.")
        tab = sampler.coverage(
            self._resolve_key(seed, "coverage"), n_test=n_test, n_post=n_post, levels=levels
        )
        if plot:
            import matplotlib.pyplot as plt

            if ax is None:
                _, ax = plt.subplots(figsize=(4.5, 4.5))
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.plot(tab.nominal, tab.empirical, lw=2, label=label)
            ax.set_xlabel("Nominal credibility level")
            ax.set_ylabel("Empirical coverage")
            if label is not None:
                ax.legend()
        return tab

    # Random number generator ----------------------------------------------------------------

    def randint(self, low, high, shape):
        """
        Generate random integers in the given range.

        Args:
            low: Lowest possible value.
            high: Highest possible value.
            shape: Shape of the output array.

        Returns:
            Array of random integers.
        """
        return jax.random.randint(self._rng.stream(), shape, low, high)

    # Sampler diagnostics -----------------------------------------------------------------------
    def sampler_stats(self, per_chain=False):
        """HMC sampler diagnostics for the last fit: work done and how well it went.

        These describe the *sampler*, not the posterior. `summary()` reports
        ESS and R-hat; this reports what the sampler had to do to get them,
        which is what explains a slow or inefficient run.

        Parameters
        ----------
        per_chain : bool
            Return one row per chain instead of pooled totals.

        Returns
        -------
        dict (or list of dict when per_chain=True) with

          leapfrog_total          summed gradient evaluations over sampling
          leapfrog_per_iter       mean leapfrog steps per iteration
          pct_at_max_treedepth    % of iterations that saturated max_tree_depth.
                                  High values mean the trajectory was truncated
                                  before it turned, so the cost per effective
                                  sample is being capped, not chosen.
          divergences             count of divergent transitions
          ebfmi                   E-BFMI, same definition Stan reports. Below
                                  0.3 indicates poor energy exploration, the
                                  usual signature of a funnel.
          accept_prob             mean Metropolis acceptance probability
          step_size               adapted step size at the end of warmup

        Requires a numpyro-backend fit; `extra_fields` must have included
        'num_steps' and 'energy' (both are on by default).
        """
        import numpy as _np

        ex = getattr(self, "extra_fields_by_chain", None)
        if not ex:
            sampler_name = getattr(self, "sampler_name", None)
            if sampler_name not in (None, "nuts"):
                # Composite kernels (mixed_hmc / discrete_gibbs / hmc_gibbs)
                # carry a different state tree, so fit() never requests the
                # NUTS/HMC extra_fields for them: an empty recording here is
                # the expected outcome of the sampler= choice, not an
                # extra_fields override. Say which sampler was used.
                raise RuntimeError(
                    "sampler_stats() reports NUTS statistics (num_steps, "
                    "energy, ...); the fit used sampler="
                    f"{sampler_name!r}, which does not record them. Refit "
                    "with sampler='nuts' to get them.")
            raise RuntimeError(
                "No sampler diagnostics recorded. sampler_stats() needs a "
                "numpyro-backend fit; if extra_fields was overridden, include "
                "'num_steps' and 'energy'."
            )

        def _get(name):
            v = ex.get(name)
            return None if v is None else _np.asarray(v)

        steps = _get("num_steps")
        energy = _get("energy")
        div = _get("diverging")
        acc = _get("accept_prob")
        ss = _get("adapt_state.step_size")
        cap = 2 ** getattr(self, "_max_tree_depth", 10) - 1

        def _ebfmi(e):
            # mean successive squared energy difference over marginal variance
            var = _np.var(e)
            return float("nan") if var == 0 else float(
                _np.sum(_np.diff(e) ** 2) / (len(e) * var))

        def _row(sl):
            r = {}
            if steps is not None:
                s = steps[sl]
                r["leapfrog_total"] = int(s.sum())
                r["leapfrog_per_iter"] = float(s.mean())
                r["pct_at_max_treedepth"] = float(100.0 * (s >= cap).mean())
            if div is not None:
                r["divergences"] = int(_np.asarray(div[sl]).sum())
            if energy is not None:
                if isinstance(sl, slice):
                    # M-45: pooled row — np.diff must not cross chain
                    # boundaries, so E-BFMI is computed per chain and the
                    # worst chain wins (as Stan warns on). Computing it on
                    # the concatenated chains first would give a different,
                    # wrong value.
                    r["ebfmi"] = round(min(_ebfmi(_np.asarray(c).ravel())
                                           for c in _np.asarray(energy)), 4)
                else:
                    e = _np.asarray(energy[sl]).ravel()
                    r["ebfmi"] = round(_ebfmi(e), 4)
            if acc is not None:
                r["accept_prob"] = round(float(_np.asarray(acc[sl]).mean()), 4)
            if ss is not None:
                r["step_size"] = float(_np.asarray(ss[sl]).ravel()[-1])
            return r

        if per_chain:
            # V6: steps is None when extra_fields lacked 'num_steps'; take the
            # chain count from the first per-chain array that was recorded.
            n_chains = next((len(arr) for arr in (steps, div, energy, acc, ss)
                             if arr is not None), None)
            if n_chains is None:
                warnings.warn(
                    "sampler_stats(per_chain=True): no per-chain sampler "
                    "statistics were recorded (extra_fields)",
                    UserWarning, stacklevel=2)
                return []
            return [dict(chain=i, **_row(i)) for i in range(n_chains)]

        # M-45: _row(slice(None)) already computes E-BFMI per chain (worst
        # chain wins); no post-hoc correction needed here.
        return _row(slice(None))

    # Get posteriors ----------------------------------------------------------------------------
    def summary(
        self,
        round_to=2,
        hdi_prob=0.89,
        include=None,
        exclude=None,
        var_names=None,
        exclude_vars=None,
        filter_regex=None,
        group_by_chain=True,
        backend=None,
        kind="all",
    ):
        """Posterior summary table.

        Args:
            round_to: Decimal places to round.
            hdi_prob: HDI probability mass.
            include: str or list — keep only these parameters.
            exclude: str or list — remove these parameters.
            var_names: Legacy alias for include.
            exclude_vars: Legacy alias for exclude.
            filter_regex: Regex applied to the summary table. On the arviz
                backend it is passed to az.summary as var_names=[filter_regex]
                with filter_vars="regex" (if include is also given, include
                wins and the regex is applied to the resulting table); on the
                jax backend it filters base parameter names.
            group_by_chain: Use chain-structured posteriors (default True).
                Only honoured by the jax backend; the arviz backend always
                uses the chain structure and warns when this is False.
            backend: "arviz" (default) or "jax". Overrides the model's
                ``diag_backend`` for this call; "jax" warns that the
                JAX/NumPy diagnostics are experimental. The two backends
                read different sources: "arviz" summarises the sampler
                trace (``m.diag.trace``), while "jax" summarises
                ``m.posteriors`` / ``m.posteriors_by_chain`` — so after
                ``m.filter_posteriors()``, or with deterministic sites in
                the trace, the two tables can differ. Both use the same
                row labels ("mu[0]", "mu[1]", ...).
            kind: Passed to az.summary ("all", "stats", "diagnostics") when the
                arviz backend is in use.

        Returns:
            pandas.DataFrame: Posterior summary table (``mean``, ``sd``, HDI bounds,
                ``mcse_mean``, ``mcse_sd``, ``ess_bulk``, ``ess_tail``, ``r_hat``).
            Updates on m: stores the returned table in ``m.tab_summary``;
                does not call ``m.filter_posteriors()`` and never modifies
                ``m.posteriors`` / ``m.posteriors_by_chain`` (RELEASE_AUDIT
                M-16) — call ``m.filter_posteriors(include=..., exclude=...)``
                explicitly to shrink them in-place; when using the ``"arviz"``
                backend, initializes ``m.diag.trace`` and ``m.diag.priors_name``
                if not yet built.
        """
        from BayesForge.Diagnostic.patch_diag import (
            ARVIZ, _resolve_backend, _warn_jax_experimental,
        )

        # M-37: before any fit, self.diag / self.posteriors do not exist;
        # raise the same clear error as filter_posteriors() instead of a
        # raw AttributeError.
        if getattr(self, "posteriors_full", None) is None:
            raise RuntimeError("No posteriors found. Run fit() first.")

        _inc = include or var_names
        _exc = exclude or exclude_vars
        # M-16: summary() is a read method — include/exclude only scope the
        # table; they no longer call filter_posteriors(), which would
        # persistently shrink m.posteriors / m.posteriors_by_chain.

        _backend = _resolve_backend(
            backend if backend is not None else getattr(self, "diag_backend", ARVIZ))
        if backend is not None and _backend != ARVIZ:
            _warn_jax_experimental()

        if _backend == ARVIZ:
            if not group_by_chain:
                warnings.warn(
                    "summary(backend='arviz') always uses the chain structure "
                    "(R-hat needs chains); group_by_chain=False only applies "
                    "to backend='jax'.", UserWarning, stacklevel=2)
            if filter_regex is not None and _inc is None:
                self.tab_summary = self.summary_old(
                    round_to=round_to, kind=kind, hdi_prob=hdi_prob,
                    var_names=[filter_regex], filter_vars="regex",
                    exclude_vars=_exc,
                )
            else:
                self.tab_summary = self.summary_old(
                    round_to=round_to, kind=kind, hdi_prob=hdi_prob,
                    var_names=_inc, exclude_vars=_exc,
                )
                if filter_regex is not None:
                    # include wins; the regex is applied to the table after.
                    import re
                    _pat = re.compile(filter_regex)
                    self.tab_summary = self.tab_summary[
                        self.tab_summary.index.map(
                            lambda _s: _pat.search(str(_s)) is not None)]
            return self.tab_summary

        from BayesForge.Diagnostic.jax_diagnostics import summary as _jax_summary
        posteriors = (self.posteriors_by_chain if group_by_chain
                      and getattr(self, 'posteriors_by_chain', None) is not None
                      else self.posteriors)
        if _inc is not None or _exc is not None:
            # Filter a local copy (same base-name semantics as
            # filter_posteriors) instead of mutating m.posteriors.
            def _name_set(arg):
                names = [arg] if isinstance(arg, str) else list(arg)
                return {n.split("[")[0] for n in names}
            _keep = _name_set(_inc) if _inc is not None else None
            _drop = ({n.lstrip("~") for n in _name_set(_exc)}
                     if _exc is not None else set())
            posteriors = {
                k: v for k, v in posteriors.items()
                if (_keep is None or k.split("[")[0] in _keep)
                and k.split("[")[0] not in _drop}
        self.tab_summary = _jax_summary(
            posteriors,
            round_to=round_to,
            hdi_prob=hdi_prob,
            filter_regex=filter_regex,
            group_by_chain=group_by_chain,
        )
        return self.tab_summary

    def summary_jax(self, **kwargs):
        """Posterior summary via the experimental JAX/NumPy diagnostics.

        Args:
            **kwargs: Forwarded to ``m.summary(..., backend="jax")``.

        Returns:
            pandas.DataFrame: Posterior summary table.
            Updates on m: stores the returned table in ``m.tab_summary``;
                like ``m.summary()``, does not modify ``m.posteriors`` /
                ``m.posteriors_by_chain``.
        """
        kwargs.setdefault("backend", "jax")
        return self.summary(**kwargs)

    def summary_old(
        self,
        round_to=2,
        kind="all",
        hdi_prob=0.89,
        var_names=None,
        exclude_vars=None,
        filter_vars=None,
        *args,
        **kwargs,
    ):
        """ArviZ-based posterior summary (the default backend for summary()).

        ``filter_vars="regex"`` makes ``var_names`` a regex applied by
        az.summary itself (RELEASE_AUDIT M-16: summary(filter_regex=...)).
        """
        if az is None:
            raise ImportError(
                "arviz is unavailable, so the arviz summary backend cannot run. "
                "Use m.summary(backend='jax') or fix the arviz import.")
        # M-37: same clear error as summary() when called directly pre-fit.
        if getattr(self, "posteriors_full", None) is None:
            raise RuntimeError("No posteriors found. Run fit() first.")
        if getattr(self.diag, "trace", None) is None:
            self.diag.to_az(backend=self.backend)
        # arviz renamed this argument: 0.x takes hdi_prob=, 1.x takes ci_prob=.
        # Hard-coding ci_prob raised TypeError on every arviz 0.x install.
        import inspect as _inspect
        _prob_kw = ("ci_prob" if "ci_prob" in _inspect.signature(az.summary).parameters
                    else "hdi_prob")
        # Factor sites (m.dist.unit(log_factor=...)) are sampled as empty arrays, which
        # az.summary cannot reduce: leave them out.
        _post = self.diag.trace.posterior
        _empty = [v for v in _post.data_vars if _post[v].size == 0]
        # M-16: summary(filter_regex=...) arrives as var_names=[pattern] with
        # filter_vars="regex"; az.summary applies the regex itself, so the
        # base-name post-filter below must not compare names to the pattern.
        _regex_include = filter_vars == "regex" and var_names is not None
        if filter_vars is not None:
            kwargs["filter_vars"] = filter_vars
        if _regex_include:
            kwargs["var_names"] = var_names
        elif _empty and "var_names" not in kwargs:
            kwargs["var_names"] = [v for v in _post.data_vars if v not in _empty]
        with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in scalar divide",
                category=RuntimeWarning,
            )
            self.tab_summary = az.summary(
                self.diag.trace,
                *args,
                round_to=round_to,
                kind=kind,
                **{_prob_kw: hdi_prob},
                **kwargs,
            )
        if (var_names is not None and not _regex_include) or exclude_vars is not None:
            base_names = self.tab_summary.index.str.split("[").str[0]
            vars_to_keep = base_names.unique().tolist()
            if var_names is not None:
                if isinstance(var_names, str):
                    var_names = [var_names]
                vars_to_keep = [v for v in vars_to_keep if v in var_names]
            if exclude_vars is not None:
                if isinstance(exclude_vars, str):
                    exclude_vars = [exclude_vars]
                vars_to_keep = [v for v in vars_to_keep if v not in exclude_vars]
            self.tab_summary = self.tab_summary.loc[base_names.isin(vars_to_keep)]
        return self.tab_summary

    def filter_posteriors(self, include=None, exclude=None):
        """Filter m.posteriors in-place by parameter base name.

        Always filters from the original full posteriors (m.posteriors_full),
        so repeated calls reset rather than stack.

        Matching is exact on base name: exclude='var' removes 'var' (all its
        elements) but never 'var_1'. Index suffixes are stripped from filter
        names, so 'var[0,0]' and 'var' are equivalent.

        Args:
            include: str or list — keep only these parameters.
            exclude: str or list — remove these parameters.

        Returns:
            None: Modifies ``m`` in-place.
            Updates on m: updates ``m.posteriors`` (pooled across chains) and
                ``m.posteriors_by_chain`` (grouped by chain) in-place with the
                filtered parameter subset, always starting from the unfiltered
                ``m.posteriors_full`` and ``m.posteriors_by_chain_full`` (which
                remain untouched).
        """
        if self.posteriors_full is None:
            raise RuntimeError("No posteriors found. Run fit() first.")

        def _base(name):
            return name.split('[')[0]

        def _as_set(arg):
            if arg is None:
                return None
            if isinstance(arg, str):
                arg = [arg]
            return {_base(n) for n in arg}

        inc = _as_set(include)
        exc = _as_set(exclude)

        def _filter(d):
            return {k: v for k, v in d.items()
                    if (inc is None or k in inc) and (exc is None or k not in exc)}

        self.posteriors = _filter(self.posteriors_full)
        if self.posteriors_by_chain_full is not None:
            self.posteriors_by_chain = _filter(self.posteriors_by_chain_full)

    def get_posterior_means(self):
        """
        Get the posterior means of the variables.

        Returns:
            dict: Dictionary mapping each variable name (with index suffix for
                array parameters) to its posterior mean.
            Updates on m: calls ``m.summary()`` internally, which updates
                ``m.tab_summary`` (and initializes ``m.diag.trace`` when using
                the ``"arviz"`` backend).
        """
        d = self.summary()
        posterior_means = d["mean"].values
        posterior_names = d.index.tolist()
        return {var: mean for var, mean in zip(posterior_names, posterior_means)}

    # Sample model ----------------------------------------------------------------------------
    def find_obs_in_model(self, model_func=None):
        """
        Data arguments that feed an observed site (``obs=...``) of the model.

        Found by tracing the model on its data (:meth:`_observed_sites`), not
        by parsing its source: this also works for models whose source is not
        available (defined in R or Julia, with ``exec``, ``functools.partial``,
        a lambda...). Before any data is attached, returns the last known
        list (empty at first).

        Args:
            model_func: Unused; kept for backward compatibility.

        Returns:
            list[str]: Names of the data entries used as ``obs``.
        """
        if getattr(self, "data_on_model", None):
            self._observed_sites(self.data_on_model)   # sets self.obs_args
        if self.obs_args is None:
            self.obs_args = []
        return list(self.obs_args)

    # Create a new model function with the modified signature
    def build_model_with_Y_None(self, model):
        """
        Modify the original model function to make the observed arguments optional.

        Args:
            model: Model function to modify.

        Returns:
            Modified model function with optional observed arguments.
        """
        # Extract `obs` argument names
        obs = self.find_obs_in_model(model)
        # Modify the function's signature to make the observed argument optional
        sig = inspect.signature(model)
        parameters = []
        seen_default = False
        kw_only = False
        for name, param in sig.parameters.items():
            if name in obs:
                # Make observed variables optional (default=None).
                # If preceding parameters were already forced to KEYWORD_ONLY,
                # this parameter must also be KEYWORD_ONLY to respect Python signature rules
                # (positional parameters cannot follow keyword-only parameters).
                kind = (inspect.Parameter.KEYWORD_ONLY if kw_only
                        else inspect.Parameter.POSITIONAL_OR_KEYWORD)
                param = inspect.Parameter(name, kind, default=None)
            elif (param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                  and (kw_only or (seen_default
                                   and param.default is inspect.Parameter.empty))):
                # A required argument cannot follow an optional one (e.g., model(y, x) where y is observed).
                # Furthermore, once any parameter becomes keyword-only, all subsequent positional
                # parameters must also become keyword-only to preserve a valid inspect.Signature.
                # Since BayesForge always passes model data by keyword, this conversion is transparent.
                param = param.replace(kind=inspect.Parameter.KEYWORD_ONLY)

            # Update flags for subsequent parameter iterations
            kw_only = kw_only or param.kind is inspect.Parameter.KEYWORD_ONLY
            seen_default = seen_default or param.default is not inspect.Parameter.empty
            parameters.append(param)

        def model_with_None(*args, **kwargs):
            # Default values for obs arguments if not passed
            for obs_name in obs:
                if obs_name not in kwargs:
                    kwargs[obs_name] = None
            # Call the original model function with the modified arguments
            return model(*args, **kwargs)

        # Update the signature of the new model
        model_with_None.__signature__ = sig.replace(parameters=parameters)
        self.model2 = model_with_None
        return model_with_None

    def sample(self, data=None, remove_obs=True, posterior=True, samples=None, seed=None, infer_discrete=False,
               posterior_draws=None):
        """
        Sample from the model: posterior predictive, prior predictive, or
        predictive draws at fixed parameter values.

        Args:
            data: Data to use for sampling. Defaults to None (the data the
                model was fitted on). Pass new inputs (e.g. a new design
                matrix) to get predictions for unseen points.
            remove_obs: Whether to remove observed data. Defaults to True,
                which un-conditions the observed sites so that simulated
                values are returned. A site whose obs is a computed value
                rather than a data entry (e.g. obs=y - 1) cannot be
                un-conditioned: it stays conditioned on the data, its draws
                are the conditioning values, and a warning names it. With
                remove_obs=False the observed sites are not returned: the
                result contains the latent (unobserved) sample sites and the
                deterministic sites.
                Forced to False when infer_discrete=True, since exact
                posterior sampling of discrete sites requires conditioning
                on the observed data.
            posterior: Source of the parameter values. Defaults to True.

                - True: use the fitted posterior samples
                  (``self.sampler.get_samples()``). Raises ``RuntimeError``
                  if the model was not fitted.
                - False: draw from the prior (prior predictive).
                - dict: fixed parameter values; each value is treated as a
                  single draw (a leading batch axis of size 1 is added).
            samples: Number of draws. Defaults to None: with posterior=True
                all posterior draws are used, with posterior=False 1000 prior
                draws are made, and with a posterior dict a single draw is
                returned. An integer n with posterior=True takes a random
                subsample of n posterior draws (without replacement); with a
                posterior dict it repeats the fixed draw n times; with
                posterior=False it makes n prior draws. Ignored when
                infer_discrete=True (sample count follows the posterior).
            seed: Random seed. Defaults to None: each call draws a fresh key
                from the model's random stream (reproducible as a sequence when
                the model was built with ``rand_seed=False`` or an integer).
                An explicit seed gives identical draws on every call.
            infer_discrete: If True, draw exact posterior samples of
                discrete latent sites (e.g. auto-enumerated Bernoulli /
                Categorical sites) conditioned on the observed data,
                instead of forward-sampling them from their prior. Use
                this to recover discrete latents (e.g. mixture indicators)
                rather than the plain prior-predictive draw.
                Requires posterior=True (a fitted model).
            posterior_draws: dict of parameter draws to use instead of the
                fitted posterior, with the draws on axis 0 of every value
                (e.g. a stored ``m.posteriors``, or the weight draws of a
                Bayesian NPE). Each draw is used as is, unlike a ``posterior``
                dict, which is one fixed draw. ``samples`` subsamples them as
                with posterior=True. Cannot be combined with a ``posterior``
                other than True.

        Returns:
            dict: Dictionary mapping site names to JAX arrays of simulated draws.
            Updates on m: sets ``m.model2`` (modified model callable with
                optional observed arguments defaulting to ``None``), updates
                ``m.obs_args`` (list of observed data entry names), and advances
                ``m._rng`` when ``seed=None``.
        """
        self._bind_library_rng()
        import jax.numpy as jnp
        # M-22: clear errors instead of AttributeError on a fresh instance.
        if getattr(self, "model", None) is None:
            raise RuntimeError(
                "sample() needs a model: call m.fit(...) first, or set "
                "m.model= for prior-predictive draws.")
        if data is not None and not isinstance(data, dict):
            raise TypeError(
                f"sample(): data must be a dict of model arguments, got "
                f"{type(data).__name__}.")
        if samples is not None:
            # M-32: samples=0 reached Predictive(num_samples=0) or
            # jax.random.choice(..., (0,)), and a negative value crashed in
            # deep indexing; validate like the fit() counters do.
            samples = _as_count(samples, "samples", minimum=1)
        rng_key = (self._rng.stream() if seed is None
                   else self._resolve_key(seed, "sample"))
        self.build_model_with_Y_None(self.model)

        if posterior_draws is not None and posterior is not True:
            raise ValueError("Pass either posterior= or posterior_draws=, not both.")
        if infer_discrete and posterior is not True:
            raise ValueError("infer_discrete=True requires posterior=True (a fitted model).")
        if isinstance(posterior, dict) and not posterior:
            # M-38: an empty posterior dict used to reach next(iter(...)) or
            # Predictive with no sites and die with StopIteration or an
            # obscure NumPyro error.
            raise ValueError(
                "sample(): posterior={} is empty: pass at least one "
                "parameter draw, posterior=False for the prior, or True "
                "for the fitted posterior.")

        own_data = data is None
        if data is None:
            if self.data_on_model is None:
                raise RuntimeError(
                    "sample(): no data on the model — pass data={...} or "
                    "call m.fit(...) first.")
            data = self.data_on_model.copy()

        # Observed sites are found by tracing the model (this also works
        # for models defined in R or Julia, whose source cannot be parsed);
        # their data entries are set to None so that they are simulated
        # rather than conditioned on.
        observed_sites = self._observed_sites(data)
        if "x" in observed_sites and "x" not in data:
            warnings.warn(
                "Observed site has the default name 'x'; pass name='y' "
                "(or similar) to the likelihood to get a meaningful key.",
                UserWarning,
            )

        if infer_discrete:
            remove_obs = False

        if remove_obs:
            # A site conditioned on a computed value (obs=y - 1) has no data
            # entry to set to None: it stays conditioned and Predictive
            # returns the conditioning value, which is not a replicate.
            computed = [name for name, key in observed_sites.items()
                        if key is None]
            if computed:
                warnings.warn(
                    f"Observed site(s) {computed} condition on a computed "
                    "value, not a data entry (e.g. obs=y - 1), so they "
                    "cannot be simulated: remove_obs=True leaves them "
                    "conditioned on the data and their draws are the "
                    "conditioning values. Pass the observed value itself as "
                    "a data entry (obs=y, name='y') to get replicated draws.",
                    UserWarning,
                )

        # Shapes of the fitted data for the observed sites, remembered before
        # their entries are set to None: with obs=None a likelihood whose
        # parameters are scalars would simulate a single value per draw.
        obs_shapes = {}
        if remove_obs and own_data:
            obs_shapes = {site: tuple(np.shape(self.data_on_model[key]))
                          for site, key in observed_sites.items() if key is not None}

        if remove_obs:
            observed = [k for k in observed_sites.values() if k is not None]
            data = {k: (None if k in observed else v) for k, v in data.items()}

        if posterior_draws is not None:
            posterior_samples = {k: jnp.asarray(v) for k, v in posterior_draws.items()}
            sizes = {int(v.shape[0]) if v.ndim else None for v in posterior_samples.values()}
            if len(sizes) != 1 or None in sizes:
                raise ValueError(
                    "posterior_draws: every value needs the same number of draws on "
                    f"axis 0 (got {sorted(sizes, key=str)})."
                )
        elif posterior is False:
            posterior_samples = None
        elif isinstance(posterior, dict):
            posterior_samples = {k: jnp.asarray(v)[None, ...] for k, v in posterior.items()}
        else:
            # M-29: only a missing sampler (or a sampler never run) means
            # "not fitted"; a real error from get_samples() (corrupted
            # state, lost device) must surface with its own message.
            _no_fit = RuntimeError(
                "No posterior samples: call m.fit() first, or use "
                "posterior=False (prior) or a dict of parameter values."
            )
            get_samples = getattr(getattr(self, "sampler", None),
                                  "get_samples", None)
            posterior_samples = None
            if get_samples is not None:
                try:
                    posterior_samples = get_samples()
                except AttributeError as e:
                    raise _no_fit from e  # sampler present but never run
            if not posterior_samples:
                raise _no_fit

        if infer_discrete:
            # numpyro.infer.Predictive(..., infer_discrete=True) unconditionally
            # masks every sample site's log_prob before enumerating (see
            # numpyro/infer/util.py:_predictive), which silently drops the
            # observed-data likelihood and collapses discrete sites to a
            # uniform draw. sample_discrete_posterior conditions correctly.
            return sample_discrete_posterior(
                self.model, posterior_samples, rng_key, model_kwargs=data
            )

        if posterior is False:
            num_samples = 1000 if samples is None else int(samples)
        elif isinstance(posterior, dict):
            if samples is not None:
                n = int(samples)
                posterior_samples = {k: jnp.repeat(v, n, axis=0)
                                     for k, v in posterior_samples.items()}
            num_samples = None
        elif samples is not None:
            # random subsample of the posterior draws, without replacement
            n = int(samples)
            N = int(jnp.asarray(next(iter(posterior_samples.values()))).shape[0])
            if n > N:
                raise ValueError(
                    f"Cannot subsample {n} posterior draws: the posterior "
                    f"has only N = {N} draws."
                )
            key, rng_key = jax.random.split(rng_key)
            idx = jax.random.choice(key, N, (n,), replace=False)
            posterior_samples = {k: v[idx] for k, v in posterior_samples.items()}
            num_samples = None
        else:
            num_samples = None

        return_sites = None
        if not remove_obs:
            # Predictive returns every traced site by default, including the
            # conditioned observed ones; list the sites to keep explicitly:
            # latent sample sites (not observed, not fixed by posterior) plus
            # all deterministic sites.
            observed_names = set(observed_sites)
            fixed_names = set(posterior_samples or {})
            trace_exc = None
            try:
                with handlers.seed(rng_seed=0):
                    trace = handlers.trace(self.model).get_trace(**data)
            except Exception as e:
                # e.g. new data without the outcome: model2 defaults it to None
                trace_exc = e
                try:
                    with handlers.seed(rng_seed=0):
                        trace = handlers.trace(self.model2).get_trace(**data)
                    trace_exc = None
                except Exception as e2:
                    trace_exc = e2
                    trace = {}
            return_sites = [
                name for name, site in trace.items()
                if (site["type"] == "sample" and not site.get("is_observed")
                    and name not in observed_names and name not in fixed_names)
                or site["type"] == "deterministic"
            ]
            if not return_sites:
                if trace_exc is not None:
                    # M-31: an empty result here meant the model could not
                    # be traced at all — name the cause instead of
                    # returning {} as if the model had nothing to return.
                    raise RuntimeError(
                        "sample(remove_obs=False): the model could not be "
                        "traced with the given data to list the sites to "
                        "return; check that data= provides every argument "
                        "the model needs."
                    ) from trace_exc
                return {}

        model_fn = _ExpandObserved(self.model2, obs_shapes) if obs_shapes else self.model2
        predictive = Predictive(
            model_fn,
            posterior_samples=posterior_samples,
            num_samples=num_samples,
            return_sites=return_sites,
        )
        return predictive(rng_key, **data)
    def _observed_sites(self, data=None):
        """Observed sample sites of the model and the data entry each one uses.

        Args:
            data (dict, optional): Model data. Defaults to ``m.data_on_model``.

        Returns:
            dict: ``{site name: data key}``; the key is ``None`` when the
            observed value is not one of the data entries (e.g. ``obs=y - 1``).
        """
        from numpyro import handlers
        if data is None:
            data = self.data_on_model
        try:
            with handlers.seed(rng_seed=0):
                trace = handlers.trace(self.model).get_trace(**data)
        except Exception:
            # e.g. new data without the outcome: fall back to the source scan
            return {k: k for k in (self.obs_args or []) if k in data}

        def same(a, b):
            if a is b:
                return True
            try:
                a, b = np.asarray(a), np.asarray(b)
            except Exception:
                return False
            return a.shape == b.shape and a.dtype.kind in "biuf" and np.array_equal(a, b)

        sites = {}
        for name, site in trace.items():
            if site["type"] == "sample" and site.get("is_observed") and site.get("value") is not None:
                # M-30: identify the data entry by identity first — the trace
                # holds the exact object passed as obs. Value equality alone
                # let the first of two equal entries win, so
                # sample(remove_obs=True) could un-condition the wrong
                # argument and keep conditioning the PPC on the data.
                value = site["value"]
                key = next((k for k, v in data.items() if v is value), None)
                if key is None:
                    matches = [k for k, v in data.items() if same(v, value)]
                    if len(matches) > 1:
                        warnings.warn(
                            f"Observed site '{name}' matches several equal "
                            f"data entries {matches} and none by identity: "
                            "it cannot be identified and stays conditioned "
                            "on the data.",
                            UserWarning, stacklevel=2)
                    elif matches:
                        key = matches[0]
                sites[name] = key
        self.obs_args = [k for k in sites.values() if k is not None]
        return sites

    # Log probability ----------------------------------------------------------------------------
    def log_prob(self, model, seed=None, **kwargs):
        """
        Compute the log probability of a model.

        Args:
            model: Model to compute log probability for.
            seed: Random seed. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple containing init_params, potential_fn, constrain_fn, and model_trace.
        """
        # getting log porbability
        rng_key = self._resolve_key(seed, "log_prob")
        init_params, potential_fn, constrain_fn, model_trace = (
            numpyro.infer.util.initialize_model(rng_key, model, model_kwargs=kwargs)
        )
        print("init_params:  ", init_params)
        print("constrain_fn: ", constrain_fn(init_params.z))
        print("potential_fn: ", -potential_fn(init_params.z))  # log prob
        print("grad:         ", jax.grad(potential_fn)(init_params.z))
        return init_params, potential_fn, constrain_fn, model_trace

    def get_history(self):
        """
        Build and store a snapshot of the current fitted state on ``m.history``.

        Returns:
            None: Populates ``m.history`` in-place.
            Updates on m: sets ``m.history`` to a dictionary with keys
                ``"model"``, ``"model_name"``, ``"data"`` (``m.data_on_model``),
                ``"sampler"``, and ``"posteriors"``.
        """
        # M-24: snapshot copies of the dicts — a later filter_posteriors or
        # refit mutates m.posteriors / m.data_on_model in place and must not
        # retro-edit the history of the previous fit.
        self.history = {
            "model": self.model,
            "model_name": self.run_model_name,
            "data": dict(self.data_on_model) if self.data_on_model is not None else None,
            "sampler": self.sampler,
            "posteriors": dict(self.posteriors) if self.posteriors is not None else None,
        }

    def plot(self, X, y=None, figsize=(10, 6), **kwargs):
        """
        Plot the model results.

        Args:
            X: Data to plot.
            y: Target variable. Defaults to None.
            figsize: Figure size. Defaults to (10, 6).
            **kwargs: Additional keyword arguments. A ``sampler`` keyword
                overrides the fitted sampler held by the model.
        """
        # M-27: self.sampler does not exist before the first fit with the
        # numpyro backend — do not evaluate it when absent.
        sampler = kwargs.pop("sampler", getattr(self, "sampler", None))

        if self.run_model_name == "gmm":
            return plot_gmm(X, sampler=sampler, figsize=figsize,
                            **kwargs)

        elif self.run_model_name == "dpmm":
            return self.models.dpmm.plot(X, sampler=sampler,
                                         figsize=figsize, **kwargs)

        elif self.run_model_name == "bnnc":
            return self.models.bnnc.plot(X, sampler=sampler, figsize=figsize)

        elif self.run_model_name == "pca":
            return self.models.pca.plot()
        else:
            # M-27: the silent no-op for every other model was
            # indistinguishable from success; name it instead. The return
            # value stays None (no valid output changes).
            warnings.warn(
                f"plot() is only defined for the built-in models gmm, dpmm, "
                f"bnnc and pca; nothing was plotted for "
                f"{self.run_model_name!r}.",
                UserWarning, stacklevel=2)

    # Device-bound attributes (JAX Mesh / NamedSharding hold jaxlib Device
    # handles, which cannot be pickled). They are dropped on save and rebuilt
    # from the devices available when the object is loaded.
    _DEVICE_STATE = ("mesh", "_data_sharding", "_rep_sharding")

    def __getstate__(self):
        state = self.__dict__.copy()
        for key in self._DEVICE_STATE:
            state[key] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        platform = state.get("_platform")
        if platform is not None and state.get("n_devices", 1) > 1:
            from BayesForge.SetDevice.set import setup_multi_device
            try:
                self.mesh, self._data_sharding, self._rep_sharding = \
                    setup_multi_device(platform)
            except Exception as e:
                # Saved on a machine with devices this one lacks (e.g. GPU),
                # or the platform no longer resolves: the object stays usable
                # for analysis, without sharding. M-33: drop the stale device
                # state entirely so _auto_shard_data never uses it.
                self.mesh = self._data_sharding = self._rep_sharding = None
                self.n_devices = 1
                warnings.warn(
                    f"Reloaded bf object: could not rebuild the "
                    f"{platform!r} multi-device setup ({type(e).__name__}: "
                    f"{e}); sharding is disabled on this machine.",
                    RuntimeWarning, stacklevel=2)
            else:
                # M-33: this machine may expose a different number of devices
                # than the one that saved the object (setup_multi_device is
                # called without the original cores cap); n_devices must
                # follow the rebuilt mesh, or _auto_shard_data computes
                # divisibility with the wrong divisor.
                self.n_devices = (len(self.mesh.devices)
                                  if self.mesh is not None else 1)
                if self.n_devices != state.get("n_devices", 1):
                    warnings.warn(
                        f"Reloaded bf object: {state.get('n_devices')} "
                        "devices when saved, "
                        f"{self.n_devices} here; sharding follows this "
                        "machine (m.shard() may be unavailable).",
                        RuntimeWarning, stacklevel=2)

    def save(self, path=None):
        """
        Save the BF object state to a file using cloudpickle.
        Includes data_on_model, model, posteriors, and other attributes.

        Args:
            path (str, optional): File path to save the object. Defaults to '{run_model_name}_bf.pkl' or 'bf_model.pkl' in CWD.

        Returns:
            str: The file path the serialized ``bf`` object was written to.
            Updates on m: sets ``m._saved_with_version`` to the current
                BayesForge version string before serializing.
        """
        if path is None:
            # M-17: fit/svi/sbi reset model_name to None at the end, so the
            # old default was dead code — every save wrote bf_model.pkl and
            # two successive fits overwrote each other. run_model_name is
            # the name that survives the fit.
            name = self.run_model_name or self.model_name
            path = f"{name}_bf.pkl" if name else "bf_model.pkl"

        self._saved_with_version = _bf_version()
        # M-28: dump to a temp file first — open(path, "wb") truncated a
        # valid previous save before the pickle could fail on it.
        tmp = f"{path}.tmp"
        try:
            with open(tmp, "wb") as f:
                cloudpickle.dump(self, f)
            os.replace(tmp, path)
        except Exception:
            # Never leave a truncated temp file behind that load() would
            # choke on; the previous save at `path` is untouched.
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        print(f"BF object successfully saved to {path}")
        return path

    @staticmethod
    def load(path=None):
        """
        Load a BF object state from a file using cloudpickle.

        Security: cloudpickle deserialization executes arbitrary code
        embedded in the file. Only load ``.pkl`` files you created yourself
        or fully trust — never a file received from an untrusted source.

        Note (RELEASE_AUDIT M-40): on an instance, ``m.load`` is the BF
        dataset loader (``BayesForge.Resources.datasets.load``), a different
        object that shadows this method; it delegates a path argument back
        here, so ``m.load("x.pkl")`` and ``bf.load("x.pkl")`` both load a
        saved session, while ``m.load.iris()``-style calls load datasets.
        ``bf.load`` on the class is the canonical session entry point.

        Args:
            path (str, optional): File path to load the object from. Defaults to 'bf_model.pkl'.

        Returns:
            BF: The loaded BF instance.

        Raises:
            FileNotFoundError: the file does not exist.
            ValueError: the file is empty, truncated or not a pickle.
            TypeError: the file holds something other than a bf object.

        A file saved with another BayesForge version loads with a warning:
        check the results, or refit with the current version.
        """
        if path is None:
            path = "bf_model.pkl"

        with open(path, "rb") as f:
            try:
                obj = cloudpickle.load(f)
            except (EOFError, pickle.UnpicklingError) as e:
                raise ValueError(f"{path} is not a saved bf object (empty, truncated "
                                 f"or not a pickle file): {e}") from e
        if not isinstance(obj, bf):
            raise TypeError(f"{path} holds a {type(obj).__name__}, not a bf object.")
        saved = getattr(obj, "_saved_with_version", None)
        if saved != _bf_version():
            warnings.warn(f"{path} was saved with BayesForge {saved or '< 0.0.61'} and is "
                          f"loaded with {_bf_version()}; check the results or refit.",
                          stacklevel=2)
        print(f"BF object successfully loaded from {path}")
        return obj

    # ------------------------------------------------------------------
    # Multi-GPU helpers
    # ------------------------------------------------------------------

    def _maybe_warn_shard_limits(self):
        """Print the EXPERIMENTAL data-sharding limits warning once per instance.

        Called lazily from ``fit`` on the first sharded fit (not at
        construction). Sharding splits the leading (axis-0) dimension of each
        eligible data array across devices. This is only numerically correct
        when the model is a composition of *map* (per-row) and *reduce*
        (sum/mean/logsumexp over axis 0) operations. Operations that couple rows
        across the device boundary are NOT yet handled and can crash, deadlock
        (under vectorized chains), or — worst — bias the posterior silently.
        """
        if self._shard_warned:
            return
        self._shard_warned = True
        import warnings
        # M-34: the old "only 1 device" branch was dead — fit() already falls
        # back to shard=False with its own warning when no mesh exists, so
        # this method is only ever reached with a real multi-device mesh.
        warnings.warn(
            "EXPERIMENTAL data sharding is ENABLED (axis-0 split across "
            f"{self.n_devices} devices). It is correct ONLY for models built "
            "from per-row (map) and axis-0 reductions (sum/mean/logsumexp). It "
            "can crash, deadlock, or SILENTLY bias the posterior for models "
            "that couple rows across the shard boundary, including:\n"
            "  1. cross-position ops: diff/lag/cumsum/scan, AR/Kalman/HMM/RNN/"
            "ODE, convolution, sort, FFT;\n"
            "  2. all-to-all/pairwise: outer products on axis 0, Gram/distance/"
            "kernel matrices, attention, reciprocity (Y + Y.T);\n"
            "  3. contraction over the sharded axis (dot products on the wrong "
            "side);\n"
            "  4. whole-array linear algebra: cholesky/inv/solve/qr/svd/eig/det;"
            "\n  5. index/segment ops crossing partitions: scatter, "
            "segment_sum, bincount, permutation gather;\n"
            "  6. a leading axis that is NOT the observation axis (feature- or "
            "time-major arrays).\n"
            "Control placement explicitly with m.shard(arr)/m.replicate(arr), "
            "and inspect m.shard_report() after fitting. These cases are being "
            "addressed incrementally.",
            stacklevel=3,
        )

    def _auto_shard_data(self, data):
        """Shard every eligible array in *data* across available devices.

        Called automatically by :meth:`fit` before passing ``data_on_model``
        to the sampler.  An array is sharded when:

        * ``_data_sharding`` is set (i.e. ``n_devices > 1``), **and**
        * the array has at least one dimension, **and**
        * its leading dimension is divisible by ``n_devices``
          (so the split is exact — no padding needed).

        Scalars, 0-D arrays, strings, and arrays whose leading dimension is
        not divisible by ``n_devices`` are passed through unchanged; XLA
        replicates them automatically on all devices.

        Behaviour notes
        ---------------
        * An array already placed by the caller with :meth:`shard` or
          :meth:`replicate` is left untouched — an explicit placement is never
          overridden (Improvement 4).
        * Arrays whose leading dimension is not divisible by ``n_devices`` are
          left for XLA to replicate, but the decision is reported via a warning
          rather than being silent (Improvement 3 / 5).
        * The per-array decisions are recorded in ``self._last_shard_report``
          (see :meth:`shard_report`).

        .. warning::
           This routine shards purely on leading-axis divisibility; it has no
           knowledge of how the model *consumes* each array. An array whose
           axis 0 is a "needs-all-N" axis (a broadcast partner in an outer
           product, a gather-index source, or a normalisation axis) will be
           sharded if its shape happens to divide, which can crash or, worse,
           silently bias the posterior. Keep such arrays replicated explicitly
           with :meth:`replicate`. Note also that for the matrix-form SRM the
           dyadic random-effect state (``dr_raw`` (2, N_dyads), ``dr_rf_mat``
           (N, N)) is sampled globally and stays replicated on every device, so
           sharding the data does not reduce that O(N^2) parameter memory
           (Improvement 7).

        Parameters
        ----------
        data : dict
            The ``data_on_model`` dictionary.

        Returns
        -------
        dict
            A new dictionary with eligible arrays replaced by sharded copies.
            The input dictionary is not mutated.
        """
        if self._data_sharding is None or data is None:
            return data
        import jax
        import jax.numpy as jnp
        import warnings

        out = {}
        sharded = []            # device-sharded along axis 0
        replicated_user = []    # left as-is because the caller already placed them
        not_divisible = []      # (name, leading_dim) left replicated by XLA
        passthrough = []        # scalars / static args / non-arrays

        for k, v in data.items():
            # Non-array values — python scalars used as static model args (N, K,
            # latent_dim), strings, None — are left untouched. device_put would
            # turn a static arg into a traced array and XLA replicates scalars
            # automatically anyway.
            if not hasattr(v, "shape") and not isinstance(v, (list, tuple)):
                out[k] = v
                passthrough.append(k)
                continue
            try:
                arr = jnp.asarray(v)
            except (TypeError, ValueError):
                out[k] = v
                passthrough.append(k)
                continue

            if arr.ndim == 0:                       # 0-D array → replicated scalar
                out[k] = v
                passthrough.append(k)
                continue

            # Improvement 4 — respect a sharding the caller already chose via
            # m.shard()/m.replicate(). Never override an explicit placement.
            existing = getattr(v, "sharding", None)
            if existing is not None and (
                existing == self._data_sharding or existing == self._rep_sharding
            ):
                out[k] = v
                (sharded if existing == self._data_sharding
                 else replicated_user).append(k)
                continue

            if arr.shape[0] % self.n_devices == 0:
                # device_put errors (e.g. OOM) are intentionally NOT swallowed —
                # they propagate so failures are visible (Improvement 5).
                out[k] = jax.device_put(arr, self._data_sharding)
                sharded.append(k)
            else:
                # Improvement 3 — never a silent no-op. The array is left for
                # XLA to replicate (memory behaviour unchanged) but the decision
                # is recorded and surfaced in the warning below.
                out[k] = v
                not_divisible.append((k, int(arr.shape[0])))

        # Improvement 5 — auditable report instead of silent guesswork.
        if not_divisible:
            details = ", ".join(f"{k}(leading={n})" for k, n in not_divisible)
            warnings.warn(
                f"Auto-shard: {len(not_divisible)} array(s) have a leading "
                f"dimension not divisible by n_devices={self.n_devices} and were "
                f"REPLICATED instead of sharded: {details}. Each device then "
                "computes these in full (no parallel speed-up). Choose an "
                "n_devices that divides the leading dimension to shard them.",
                stacklevel=2,
            )
        self._last_shard_report = {
            "n_devices": self.n_devices,
            "sharded": sharded,
            "replicated_not_divisible": [k for k, _ in not_divisible],
            "replicated_by_caller": replicated_user,
            "passthrough": passthrough,
        }
        return out

    def _shard_check(self, model, sharded_data, replicated_data):
        """Run the static + runtime shard safeguards (fail-closed).

        Returns True if it is safe to proceed with sharding (no ✗ correctness
        verdict), False if the sharded log-density differs from the replicated
        reference (caller should fall back to replicated). Never raises; on any
        internal failure it warns and returns False, so the fit runs REPLICATED
        and stays correct (RELEASE_AUDIT bug 13: it used to return True and
        shard on an unchecked model). BF_SHARD_CHECK=0 skips the check and
        returns True.
        """
        import os as _os
        if _os.environ.get("BF_SHARD_CHECK", "1") == "0":
            return True   # safeguards explicitly disabled (e.g. for benchmarking)
        try:
            from BayesForge.Diagnostic.sharding_safeguards import run_shard_check
            key = self._resolve_key(None, "shard_check")
            report = run_shard_check(model, sharded_data, replicated_data, key,
                                     name=str(self.model_name or ""))
        except Exception as _e:
            import warnings
            warnings.warn(
                f"shard-check could not run ({_e}); running REPLICATED to "
                "stay correct. Set BF_SHARD_CHECK=0 to skip the check and "
                "shard anyway.", stacklevel=2)
            return False
        if report is None:
            import warnings
            warnings.warn(
                "shard-check could not run (the check returned no report); "
                "running REPLICATED to stay correct. Set BF_SHARD_CHECK=0 to "
                "skip the check and shard anyway.", stacklevel=2)
            return False
        if report.get("runtime_unsafe"):
            import warnings
            warnings.warn(
                "shard-check: sharded log-density differs from the replicated "
                "reference — running REPLICATED instead to preserve correctness.",
                stacklevel=2,
            )
            return False
        if report.get("perf_unsafe") and _os.environ.get("BF_SHARD_PERF", "1") != "0":
            import warnings
            back = report.get("backward", {}) or {}
            warnings.warn(
                "shard-check: backward-pass communication is O(N²)-dominant "
                f"(ratio {back.get('ratio', float('nan')):.2f} ≥ "
                f"{back.get('ratio_threshold', 0.25):.2f}) — the gradient all-reduce "
                "of a replicated/coupled parameter will not shrink with device count, "
                "so sharding cannot pay off. Running REPLICATED instead. Set "
                "BF_SHARD_PERF=0 to override.",
                stacklevel=2,
            )
            return False
        return True

    def shard_report(self):
        """Return the per-array sharding decisions from the most recent fit.

        After a ``fit(..., shard=True)`` on a multi-device setup, this returns a
        dict with the keys ``sharded``, ``replicated_not_divisible``,
        ``replicated_by_caller`` and ``passthrough`` listing the array names in
        each category, plus ``n_devices``. When the shard-check falls back to a
        replicated fit, ``sharded`` is empty and the arrays that would have
        been sharded are listed under ``replicated_shard_check``. Returns
        ``None`` if no sharded fit has run yet. Use it to audit what the
        automatic sharder actually did.
        """
        return getattr(self, "_last_shard_report", None)

    def shard(self, array):
        """Shard *array* along its leading axis across all available GPUs.

        Wraps ``jax.device_put(array, data_sharding)`` using the mesh
        created at initialisation (only available when ``platform='gpu'``
        and more than one GPU is present).  Pass the returned sharded array
        as an observation or covariate to the model; XLA will automatically
        compute each GPU's slice of the log-likelihood in parallel.

        On a single-GPU or CPU setup the array is returned unchanged.

        Parameters
        ----------
        array : array-like
            Any array convertible by JAX (numpy, jax, list …).

        Returns
        -------
        jax.Array
            The input array, sharded across GPUs when available.

        Example
        -------
        >>> m = bf(platform='gpu')
        >>> Y_sharded = m.shard(jnp.array(Y))
        >>> m.fit(model, obs=dict(Y=Y_sharded), num_chains=m.n_gpus)
        """
        import jax
        import jax.numpy as jnp
        arr = jnp.asarray(array)
        if self._data_sharding is None:
            return arr
        if arr.ndim == 0 or arr.shape[0] % self.n_devices != 0:
            warnings.warn(
                f"m.shard: leading dimension ({arr.shape[0] if arr.ndim else 0}) is not "
                f"divisible by n_devices={self.n_devices}; replicating array instead.",
                stacklevel=2,
            )
            return self.replicate(arr)
        return jax.device_put(arr, self._data_sharding)

    def replicate(self, array):
        """Place an identical copy of *array* on every GPU.

        Useful for small parameter arrays or constants that every GPU needs
        in full rather than a sharded slice.  On a single-GPU or CPU setup
        the array is returned unchanged.

        Parameters
        ----------
        array : array-like
            Any array convertible by JAX.

        Returns
        -------
        jax.Array
            The input array, replicated on all GPUs when available.
        """
        import jax
        import jax.numpy as jnp
        arr = jnp.asarray(array)
        if self._rep_sharding is None:
            return arr
        return jax.device_put(arr, self._rep_sharding)

    # ------------------------------------------------------------------
    # Parallel simulation scenarios
    # ------------------------------------------------------------------
    # Exposed as staticmethods (like `load` above) because each simulation
    # builds its *own* bf instance in its own process -- there is no `self` to
    # speak of. Imported lazily to keep BayesForge.Parallel free of a circular
    # import back into this module.

    @staticmethod
    def run_simulations(*args, **kwargs):
        """Run independent simulations in parallel. See
        :func:`BayesForge.Parallel.simulate.run_simulations`."""
        from BayesForge.Parallel.simulate import run_simulations as _run
        return _run(*args, **kwargs)

    @staticmethod
    def grid(*args, **kwargs):
        """Build a scenario grid. See
        :func:`BayesForge.Parallel.simulate.grid`."""
        from BayesForge.Parallel.simulate import grid as _grid
        return _grid(*args, **kwargs)


# Alias for backward compatibility
BF = bf
