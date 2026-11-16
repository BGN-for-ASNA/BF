import arviz as az
import inspect

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from numpyro.infer import log_likelihood as _numpyro_log_likelihood
import numpy as np
import pandas as pd
import itertools
import scipy.stats as stats
import re
from BayesForge.Utils.ImportManager import LazyImporter
import plotly.colors as pcolors
from plotly.subplots import make_subplots

importer = LazyImporter()
importer.schedule_import("plotly.graph_objects", "go")
importer.schedule_import("plotly.express", "px")
importer.schedule_import("plotly.figure_factory", "ff")
importer.schedule_import("plotly.colors", "n_colors")
importer.schedule_import("seaborn", "sns")
importer.schedule_import("matplotlib.pyplot", "plt")


def _acf(x, max_lag=40):
    """Autocorrelation function of a 1D chain, lags 0..max_lag-1.

    Centres on the GLOBAL mean and divides by the global variance. Computing
    ``np.corrcoef(x[:-t], x[t:])`` instead re-centres and re-scales each window
    separately, which is not the ACF: it diverges at large lag and the
    resulting sequence is not positive-definite.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    max_lag = int(min(max_lag, n))
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom <= 0:
        return [1.0] + [0.0] * (max_lag - 1)
    return [float(np.dot(xc[: n - t], xc[t:]) / denom) for t in range(max_lag)]


def _az_hdi(data, prob):
    """Call ``az.hdi`` across ArviZ versions.

    ArviZ >= 1.x takes ``prob=``; ArviZ 0.x took ``hdi_prob=``.

    The parameter name is resolved from the SIGNATURE, not by try/except:
    ``az.hdi`` in 0.x accepts ``**kwargs``, so ``prob=0.89`` was swallowed
    silently instead of raising TypeError, the fallback never ran, and every
    HDI came back at ArviZ's DEFAULT probability rather than the requested one
    (e.g. a 94% interval where 89% was asked for).

    Data is coerced to a NumPy array: ArviZ 1.x misreads a raw JAX array (from
    ``get_samples``) as ``(chain, draw)`` and raises a dims error.
    """
    import inspect
    data = np.asarray(data)
    params = inspect.signature(az.hdi).parameters
    name = "prob" if "prob" in params else "hdi_prob"
    return az.hdi(data, **{name: prob})


def _az_from_dict(groups):
    """Build an InferenceData from a ``{group: {var: array}}`` mapping.

    ``az.from_dict`` changed signature between ArviZ majors: 0.x takes one
    keyword per group (``from_dict(posterior=..., sample_stats=...)``) while
    1.x takes the whole mapping as its first argument (``from_dict(data)``)
    and rejects the group keywords with
    "from_dict() got an unexpected keyword argument 'posterior'".
    """
    if "posterior" in inspect.signature(az.from_dict).parameters:
        return az.from_dict(**groups)
    return az.from_dict(groups)


def _numpy_backed_idata(idata, groups=("posterior", "log_likelihood", "sample_stats")):
    """Rebuild an InferenceData with NumPy-backed arrays.

    ArviZ 1.x conversions of a NumPyro sampler keep the arrays as JAX arrays;
    downstream stats (e.g. ``az.loo``'s PSIS smoothing) do in-place assignment,
    which fails on immutable JAX arrays. Re-wrapping every group as NumPy makes
    those routines work.
    """
    # InferenceData.groups is a METHOD in arviz 0.x and an attribute in 1.x;
    # set(<bound method>) raised TypeError, so this helper -- and therefore
    # loo() -- never worked on arviz 0.x at all.
    _groups = getattr(idata, "groups", [])
    if callable(_groups):
        _groups = _groups()
    available = set(_groups or [])
    data = {}
    for group in groups:
        # DataTree groups are exposed as "/posterior" etc.; accept either form.
        if group in available or f"/{group}" in available:
            ds = getattr(idata, group)
            data[group] = {k: np.asarray(v.values) for k, v in ds.data_vars.items()}
    return _az_from_dict(data)


def _numpyro_idata_with_loglik(sampler):
    """NumPy-backed InferenceData (posterior + log_likelihood) from a sampler."""
    idata = az.from_numpyro(sampler, log_likelihood=True)
    return _numpy_backed_idata(idata)


def _pointwise_loglik_from_sampler(sampler, var_name=None):
    """Pointwise log-likelihood ``(S, N)`` computed directly via NumPyro.

    Re-evaluates the model's observed sites over the flattened posterior draws
    with :func:`numpyro.infer.log_likelihood` — **no ArviZ round-trip**. This is
    exactly the quantity ``az.from_numpyro(..., log_likelihood=True)`` builds
    internally, so it yields identical values.

    ``S`` = total draws (chains*draws), ``N`` = number of observations.
    """
    model = sampler.sampler.model
    args = getattr(sampler, "_args", ()) or ()
    kwargs = getattr(sampler, "_kwargs", {}) or {}
    samples = sampler.get_samples()  # flattened across chains: (S, ...) per site
    ll_dict = _numpyro_log_likelihood(model, samples, *args, **kwargs)
    if not ll_dict:
        raise ValueError("Model has no observed sites; cannot compute log-likelihood.")
    if var_name is not None:
        site = var_name
    elif len(ll_dict) == 1:
        site = next(iter(ll_dict))
    else:
        raise ValueError(
            f"Multiple observed sites {list(ll_dict)}; pass var_name to select one."
        )
    ll = np.asarray(ll_dict[site])
    return ll.reshape(ll.shape[0], -1)  # (S, N)


def _pointwise_loglik_from_idata(idata, var_name=None):
    """Pointwise log-likelihood ``(S, N)`` from an InferenceData's group."""
    ll_ds = idata.log_likelihood
    if var_name is not None:
        site = var_name
    elif len(ll_ds.data_vars) == 1:
        site = list(ll_ds.data_vars)[0]
    else:
        raise ValueError(
            f"Multiple log_likelihood vars {list(ll_ds.data_vars)}; pass var_name."
        )
    arr = np.asarray(ll_ds[site].values)      # (chain, draw, *obs)
    return arr.reshape(arr.shape[0] * arr.shape[1], -1)  # (S, N)


def _is_tfp_sampler(sampler):
    """True for the BF TFP MCMC sampler (a JointDistributionCoroutine wrapper)."""
    return (
        hasattr(sampler, "tensor")
        and hasattr(sampler, "model_info")
        and hasattr(sampler, "obs_names")
    )


def _pointwise_loglik_from_tfp(sampler, var_name=None):
    """Pointwise log-likelihood ``(S, N)`` for the TFP backend — **no ArviZ**.

    The TFP sampler stores the model as a ``JointDistributionCoroutine``
    (``sampler.tensor``). We replay it at every posterior draw with
    :meth:`sample_distributions` to recover the observed site's *conditional*
    distribution, unwrap any ``Independent`` wrapping so the leading (datapoint)
    axis survives, and take its per-observation ``log_prob``. This reproduces
    the same pointwise quantity ``numpyro.infer.log_likelihood`` yields for the
    equivalent NumPyro model (verified to floating-point precision).

    ``S`` = total draws (chains*draws), ``N`` = number of observations.
    """
    import jax

    jd = sampler.tensor
    param_names = list(sampler.model_info.keys())
    obs_names = list(sampler.obs_names)
    if var_name is not None:
        obs_names = [var_name]
    obs_tuple = tuple(jnp.asarray(sampler.data_on_model[n]) for n in obs_names)
    n_obs = len(obs_tuple)

    # Flatten chains: {name: (C, S, ...)} -> (C*S, ...), in model_info order.
    draws = sampler.get_samples(group_by_chain=True)
    flat = []
    S = None
    for p in param_names:
        v = jnp.asarray(draws[p])
        v = v.reshape((-1,) + v.shape[2:])
        flat.append(v)
        S = v.shape[0]

    key = jax.random.PRNGKey(0)

    def _pointwise(dist, ob):
        # Unwrap Independent layers so per-datapoint contributions are preserved.
        d = dist
        while (
            getattr(d, "reinterpreted_batch_ndims", None) is not None
            and hasattr(d, "distribution")
        ):
            d = d.distribution
        # Match the observation's dtype to the distribution's (TFP samples in
        # float32; obs coming through data_to_model may be float64/int and would
        # otherwise trip TFP's dtype checks, e.g. binomial lbeta).
        ob_cast = jnp.asarray(ob)
        d_dtype = getattr(d, "dtype", None)
        if d_dtype is not None:
            ob_cast = ob_cast.astype(d_dtype)
        lp = d.log_prob(ob_cast)
        if lp.ndim == 0:
            return lp[None]
        if lp.ndim > 1:  # keep the leading obs axis, sum any event dims
            lp = lp.reshape(lp.shape[0], -1).sum(-1)
        return lp

    def one_draw(params):
        value = tuple(params) + obs_tuple
        dists, _ = jd.sample_distributions(seed=key, value=value)
        obs_dists = dists[-n_obs:]
        outs = [_pointwise(od, ob) for od, ob in zip(obs_dists, obs_tuple)]
        return outs[0] if n_obs == 1 else jnp.concatenate(outs, axis=-1)

    ll = jax.vmap(one_draw)(tuple(flat))
    ll = np.asarray(ll)
    return ll.reshape(S, -1)  # (S, N)


def _pointwise_loglik(sampler, var_name=None):
    """Pointwise log-likelihood ``(S, N)``, ArviZ-free when possible.

    1. TFP backend: replay the ``JointDistributionCoroutine`` natively.
    2. Direct NumPyro path (``sampler.sampler.model`` + stored run args) — no
       ArviZ; the normal MCMC case.
    3. Fallback to ArviZ's ``from_numpyro`` log_likelihood group for samplers
       that don't expose those attributes but that ArviZ can still convert.
    4. If neither works, raise a clear error rather than a cryptic
       ``AttributeError`` from deep inside the extraction.
    """
    if _is_tfp_sampler(sampler):
        return _pointwise_loglik_from_tfp(sampler, var_name=var_name)
    try:
        return _pointwise_loglik_from_sampler(sampler, var_name=var_name)
    except (AttributeError, TypeError):
        pass
    try:
        idata = _numpyro_idata_with_loglik(sampler)
        return _pointwise_loglik_from_idata(idata, var_name=var_name)
    except Exception as exc:
        raise NotImplementedError(
            "Could not compute the pointwise log-likelihood needed for WAIC from "
            "this sampler. WAIC expects a NumPyro MCMC sampler (or an object "
            "ArviZ can convert with a log_likelihood group). "
            f"Underlying error: {exc!r}"
        ) from exc


def _waic(log_likelihood, pointwise=False, scale="log"):
    """Widely Applicable Information Criterion from a pointwise log-likelihood.

    ArviZ >= 1.x removed ``az.waic`` (LOO is preferred), so compute it directly:
    ``elpd_waic = sum(lppd - p_waic)`` with ``lppd_i = logsumexp_s(ll) - log(S)``
    and ``p_waic_i = var_s(ll)`` (Vehtari, Gelman & Gabry 2017).

    Parameters
    ----------
    log_likelihood : array-like, shape (S, N)
        Pointwise log-likelihood: S posterior draws x N observations.

    Returns a ``pandas.Series`` (elpd_waic, se, p_waic, n_samples, n_data_points,
    scale [, waic_i]).
    """
    ll = np.asarray(log_likelihood)
    if ll.ndim != 2:
        ll = ll.reshape(ll.shape[0], -1)
    n_samples = ll.shape[0]
    lppd_i = logsumexp(ll, axis=0) - np.log(n_samples)
    # ddof=0 for both p_waic and the SE, matching ArviZ (xarray .var() default).
    p_waic_i = np.var(ll, axis=0)
    waic_i = lppd_i - p_waic_i                          # pointwise elpd
    elpd_waic = float(np.sum(waic_i))
    p_waic = float(np.sum(p_waic_i))
    n_data = waic_i.size
    se = float(np.sqrt(n_data * np.var(waic_i)))
    sign = {"log": 1.0, "negative_log": -1.0, "deviance": -2.0}.get(scale, 1.0)
    result = pd.Series(
        {
            "elpd_waic": sign * elpd_waic,
            # The pointwise elpds scale by `sign`, so their SE scales by
            # |sign| -- not sqrt(|sign|), which under-reported deviance SE by 2x.
            "se": abs(sign) * se,
            "p_waic": p_waic,
            "n_samples": n_samples,
            "n_data_points": n_data,
            "scale": scale,
        }
    )
    if pointwise:
        result["waic_i"] = sign * waic_i
    return result


class diagWIP:
    """
    The diag class serves as a comprehensive toolkit for diagnosing and visualizing the results of Bayesian models,
    particularly those fitted using MCMC samplers like NumPyro. It is built to provide interactive plotting
    functionalities using Plotly and operates directly on a dictionary of posterior samples. The
    hand-rolled convergence measures used by ``diagnose`` (rank-normalized split R-hat, bulk/tail
    ESS, E-BFMI) and the ``summary`` statistics are computed with NumPy/SciPy (not JAX); the public
    ``rhat``/``ess`` methods delegate to ArviZ.
    """

    def __init__(self, sampler):
        """
        Initialize the diagnostic class.

        Args:
            sampler: A fitted NumPyro MCMC sampler object.
        """
        self.sampler = sampler
        # Get samples with chain information preserved
        self.posterior_samples = sampler.get_samples(group_by_chain=True)
        self.priors_name = list(self.posterior_samples.keys())
        # Determine the number of chains from the shape of the first parameter.
        # Cross-check against the sampler: if group_by_chain was not honoured,
        # shape[0] is n_samples, and every per-chain plot loop below would run
        # thousands of iterations instead of a handful.
        if self.priors_name:
            self.num_chains = int(
                self.posterior_samples[self.priors_name[0]].shape[0])
            declared = getattr(sampler, "num_chains", None)
            if declared is not None and int(declared) != self.num_chains:
                raise ValueError(
                    f"sampler reports num_chains={declared} but samples have "
                    f"leading dim {self.num_chains}; get_samples(group_by_chain=True) "
                    "did not return chain-major draws."
                )
            self.colors = pcolors.qualitative.Plotly
        else:
            self.num_chains = 0
            self.colors = []

    #
    #  Diagnostic with ARVIZ ----------------------------------------------------------------------------
    def to_az(
        self,
        backend="numpyro",
        sample_stats_name=[
            "target_log_prob",
            "log_accept_ratio",
            "has_divergence",
            "energy",
        ],
    ):
        """Convert the sampler output to an arviz trace object.

        This method prepares the trace for use with arviz diagnostic tools.

        Returns:
            self.trace: The arviz trace object containing the diagnostic data
        """
        if backend == "numpyro":
            if hasattr(self.sampler, "svi") or getattr(self.sampler, "az_from_dict", False):
                # Handle SVI / simulation-based (ABC, NPE) wrappers
                posterior_samples = self.sampler.get_samples(group_by_chain=True)
                self.trace = _az_from_dict({"posterior": posterior_samples})
            else:
                self.trace = az.from_numpyro(self.sampler)
            self.priors_name = list(self.trace["posterior"].data_vars.keys())
            return self.trace

        elif backend == "tfp":
            var_names = list(self.sampler.model_info.keys())
            sample_stats = {
                k: jnp.transpose(v)
                for k, v in zip(sample_stats_name, self.sampler.sample_stats)
            }
            trace = {}
            # First dim is the number of chains
            # Second dim is the number of sampling
            # The rest is the shape of the object
            for name, samp in zip(var_names, self.sampler.posterior):
                trace[name] = samp

            self.trace = _az_from_dict({"posterior": trace, "sample_stats": sample_stats})
            self.priors_name = var_names
            return self.trace

        raise ValueError(
            f"unknown backend {backend!r}; expected 'numpyro' or 'tfp'")

    def _ensure_trace(self):
        """Build self.trace on first use rather than requiring an explicit to_az()."""
        if getattr(self, "trace", None) is None:
            self.to_az()
        return self.trace

    # --- Statistical Diagnostics ---

    def loo(self, pointwise=None, var_name=None, reff=None, scale=None):
        """Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).

        Estimates the expected log pointwise predictive density (elpd) using Pareto-smoothed
        importance sampling leave-one-out cross-validation (PSIS-LOO-CV). Also calculates LOO's
        standard error and the effective number of parameters. Read more theory here
        https://arxiv.org/abs/1507.04544 and here https://arxiv.org/abs/1507.02646

        Parameters
        ----------
        pointwise: bool, optional
            If True the pointwise predictive accuracy will be returned. Defaults to
            ``stats.ic_pointwise`` rcParam.
        var_name : str, optional
            The name of the variable in log_likelihood groups storing the pointwise log
            likelihood data to use for loo computation.
        reff: float, optional
            Relative MCMC efficiency, ``ess / n`` i.e. number of effective samples divided by the number
            of actual samples. Computed from trace by default.
        scale: str
            Output scale for loo. Available options are:

            - ``log`` : (default) log-score
            - ``negative_log`` : -1 * log-score
            - ``deviance`` : -2 * log-score

            A higher log-score (or a lower deviance or negative log_score) indicates a model with
            better predictive accuracy.

        Returns
        -------
        ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
        elpd_loo: approximated expected log pointwise predictive density (elpd)
        se: standard error of the elpd
        p_loo: effective number of parameters
        n_samples: number of samples
        n_data_points: number of data points
        warning: bool
            True if the estimated shape parameter of Pareto distribution is greater than
            ``good_k``.
        loo_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
                only if pointwise=True
        pareto_k: array of Pareto shape values, only if pointwise True
        scale: scale of the elpd
        good_k: For a sample size S, the thresold is compute as min(1 - 1/log10(S), 0.7)

            The returned object has a custom print method that overrides pd.Series method.
        """
        # ArviZ 1.x: az.loo needs a NumPy-backed InferenceData with a
        # log_likelihood group (a raw sampler / JAX arrays fail), and no longer
        # takes `scale`. Build one from the sampler and drop unsupported kwargs.
        idata = _numpyro_idata_with_loglik(self.sampler)
        kwargs = {}
        if pointwise is not None:
            kwargs["pointwise"] = pointwise
        if var_name is not None:
            kwargs["var_name"] = var_name
        if reff is not None:
            kwargs["reff"] = reff
        if scale is not None:
            # Forward `scale` when this ArviZ still accepts it (0.x) rather
            # than silently ignoring a documented argument.
            import inspect as _inspect
            if "scale" in _inspect.signature(az.loo).parameters:
                kwargs["scale"] = scale
            elif scale != "log":
                raise NotImplementedError(
                    f"az.loo in arviz {az.__version__} no longer accepts "
                    f"scale={scale!r}; rescale the returned elpd yourself "
                    "(negative_log = -elpd, deviance = -2*elpd)."
                )
        return az.loo(idata, **kwargs)

    def WAIC(self, pointwise=None, var_name=None, scale=None, dask_kwargs=None):
        """
        Compute the widely applicable information criterion.

        Estimates the expected log pointwise predictive density (elpd) using WAIC. Also calculates the
        WAIC's standard error and the effective number of parameters.
        Read more theory here https://arxiv.org/abs/1507.04544 and here https://arxiv.org/abs/1004.2316

        Parameters
        ----------
        pointwise: bool
            If True the pointwise predictive accuracy will be returned. Defaults to
            ``stats.ic_pointwise`` rcParam.
        var_name : str, optional
            The name of the variable in log_likelihood groups storing the pointwise log
            likelihood data to use for waic computation.
        scale: str
            Output scale for WAIC. Available options are:

            - `log` : (default) log-score
            - `negative_log` : -1 * log-score
            - `deviance` : -2 * log-score

            A higher log-score (or a lower deviance or negative log_score) indicates a model with
            better predictive accuracy.
        dask_kwargs : dict, optional
            Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

        Returns
        -------
        ELPDData object (inherits from :class:`pandas.Series`) with the following row/attributes:
        elpd_waic: approximated expected log pointwise predictive density (elpd)
        se: standard error of the elpd
        p_waic: effective number parameters
        n_samples: number of samples
        n_data_points: number of data points
        warning: bool
            True if posterior variance of the log predictive densities exceeds 0.4
        waic_i: :class:`~xarray.DataArray` with the pointwise predictive accuracy,
                only if pointwise=True
        scale: scale of the elpd

            The returned object has a custom print method that overrides pd.Series method.
        """
        # ArviZ 1.x removed az.waic (LOO is preferred); compute it natively from
        # the pointwise log-likelihood, pulled straight from NumPyro (no ArviZ;
        # falls back to the arviz log_likelihood group for non-MCMC samplers).
        ll = _pointwise_loglik(self.sampler, var_name=var_name)
        return _waic(ll, pointwise=bool(pointwise), scale=scale or "log")

    @staticmethod
    def compare(
        compare_dict,
        ic=None,
        method="stacking",
        b_samples=1000,
        alpha=1,
        seed=None,
        scale=None,
        var_name=None,
    ):
        r"""Compare models based on  their expected log pointwise predictive density (ELPD).

        The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
        cross-validation (LOO) or using the widely applicable information criterion (WAIC).
        We recommend loo. Read more theory here - in a paper by some of the
        leading authorities on model comparison dx.doi.org/10.1111/1467-9868.00353

        Parameters
        ----------
        compare_dict: dict of {str: InferenceData or ELPDData}
            A dictionary of model names and :class:`arviz.InferenceData` or ``ELPDData``.
        ic: str, optional
            Method to estimate the ELPD, available options are "loo" or "waic". Defaults to
            ``rcParams["stats.information_criterion"]``.
        method: str, optional
            Method used to estimate the weights for each model. Available options are:

            - 'stacking' : stacking of predictive distributions.
            - 'BB-pseudo-BMA' : pseudo-Bayesian Model averaging using Akaike-type
              weighting. The weights are stabilized using the Bayesian bootstrap.
            - 'pseudo-BMA': pseudo-Bayesian Model averaging using Akaike-type
              weighting, without Bootstrap stabilization (not recommended).

            For more information read https://arxiv.org/abs/1704.02030
        b_samples: int, optional default = 1000
            Number of samples taken by the Bayesian bootstrap estimation.
            Only useful when method = 'BB-pseudo-BMA'.
            Defaults to ``rcParams["stats.ic_compare_method"]``.
        alpha: float, optional
            The shape parameter in the Dirichlet distribution used for the Bayesian bootstrap. Only
            useful when method = 'BB-pseudo-BMA'. When alpha=1 (default), the distribution is uniform
            on the simplex. A smaller alpha will keeps the final weights more away from 0 and 1.
        seed: int or np.random.RandomState instance, optional
            If int or RandomState, use it for seeding Bayesian bootstrap. Only
            useful when method = 'BB-pseudo-BMA'. Default None the global
            :mod:`numpy.random` state is used.
        scale: str, optional
            Output scale for IC. Available options are:

            - `log` : (default) log-score (after Vehtari et al. (2017))
            - `negative_log` : -1 * (log-score)
            - `deviance` : -2 * (log-score)

            A higher log-score (or a lower deviance) indicates a model with better predictive
            accuracy.
        var_name: str, optional
            If there is more than a single observed variable in the ``InferenceData``, which
            should be used as the basis for comparison.

        Returns
        -------
        A DataFrame, ordered from best to worst model (measured by the ELPD).
        The index reflects the key with which the models are passed to this function. The columns are:
        rank: The rank-order of the models. 0 is the best.
        elpd: ELPD estimated either using (PSIS-LOO-CV `elpd_loo` or WAIC `elpd_waic`).
            Higher ELPD indicates higher out-of-sample predictive fit ("better" model).
            If `scale` is `deviance` or `negative_log` smaller values indicates
            higher out-of-sample predictive fit ("better" model).
        pIC: Estimated effective number of parameters.
        elpd_diff: The difference in ELPD between two models.
            If more than two models are compared, the difference is computed relative to the
            top-ranked model, that always has a elpd_diff of 0.
        weight: Relative weight for each model.
            This can be loosely interpreted as the probability of each model (among the compared model)
            given the data. By default the uncertainty in the weights estimation is considered using
            Bayesian bootstrap.
        SE: Standard error of the ELPD estimate.
            If method = BB-pseudo-BMA these values are estimated using Bayesian bootstrap.
        dSE: Standard error of the difference in ELPD between each model and the top-ranked model.
            It's always 0 for the top-ranked model.
        warning: A value of 1 indicates that the computation of the ELPD may not be reliable.
            This could be indication of WAIC/LOO starting to fail see
            http://arxiv.org/abs/1507.04544 for details.
        scale: Scale used for the ELPD.

        References
        ----------
        .. [1] Vehtari, A., Gelman, A. & Gabry, J. Practical Bayesian model evaluation using
            leave-one-out cross-validation and WAIC. Stat Comput 27, 1413–1432 (2017)
            see https://doi.org/10.1007/s11222-016-9696-4
        """
        # Forward every argument this ArviZ still supports rather than dropping
        # documented ones silently (az.compare lost ic/b_samples/alpha/seed/
        # scale in 1.x but keeps them in 0.x).
        import inspect as _inspect
        supported = _inspect.signature(az.compare).parameters
        requested = {"ic": ic, "b_samples": b_samples, "alpha": alpha,
                     "seed": seed, "scale": scale, "var_name": var_name}
        kwargs = {"method": method}
        dropped = []
        for key, val in requested.items():
            if val is None:
                continue
            if key in supported:
                kwargs[key] = val
            elif not (key == "alpha" and val == 1) and not (key == "b_samples" and val == 1000):
                dropped.append(key)
        if dropped:
            raise NotImplementedError(
                f"az.compare in arviz {az.__version__} does not accept "
                f"{sorted(dropped)}; drop them or pin an arviz that supports them."
            )
        # Accept fitted BF objects ({"name": m}) as well as InferenceData/ELPDData:
        # convert them the same way loo()/waic() do (posterior + log_likelihood).
        compare_dict = {
            name: _numpyro_idata_with_loglik(val.sampler) if hasattr(val, "sampler") else val
            for name, val in compare_dict.items()
        }
        return az.compare(compare_dict, **kwargs)

    @staticmethod
    def plot_compare(
        comp_df,
        insample_dev=False,
        plot_standard_error=True,
        plot_ic_diff=False,
        order_by_rank=True,
        legend=False,
        title=True,
        figsize=None,
        textsize=None,
        labeller=None,
        plot_kwargs=None,
        ax=None,
        backend=None,
        backend_kwargs=None,
        show=None,
    ):
        r"""Summary plot for model comparison.

        Models are compared based on their expected log pointwise predictive density (ELPD).
        This plot is in the style of the one used in [2]_. Chapter 6 in the first edition
        or 7 in the second.

        Notes
        -----
        The ELPD is estimated either by Pareto smoothed importance sampling leave-one-out
        cross-validation (LOO) or using the widely applicable information criterion (WAIC).
        We recommend LOO in line with the work presented by [1]_.

        Parameters
        ----------
        comp_df : pandas.DataFrame
            Result of the :func:`arviz.compare` method.
        insample_dev : bool, default False
            Plot in-sample ELPD, that is the value of the information criteria without the
            penalization given by the effective number of parameters (p_loo or p_waic).
        plot_standard_error : bool, default True
            Plot the standard error of the ELPD.
        plot_ic_diff : bool, default False
            Plot standard error of the difference in ELPD between each model
            and the top-ranked model.
        order_by_rank : bool, default True
            If True ensure the best model is used as reference.
        legend : bool, default False
            Add legend to figure.
        figsize : (float, float), optional
            If `None`, size is (6, num of models) inches.
        title : bool, default True
            Show a tittle with a description of how to interpret the plot.
        textsize : float, optional
            Text size scaling factor for labels, titles and lines. If `None` it will be autoscaled based
            on `figsize`.
        labeller : Labeller, optional
            Class providing the method ``make_label_vert`` to generate the labels in the plot titles.
            Read the :ref:`label_guide` for more details and usage examples.
        plot_kwargs : dict, optional
            Optional arguments for plot elements. Currently accepts 'color_ic',
            'marker_ic', 'color_insample_dev', 'marker_insample_dev', 'color_dse',
            'marker_dse', 'ls_min_ic' 'color_ls_min_ic',  'fontsize'
        ax : matplotlib_axes or bokeh_figure, optional
            Matplotlib axes or bokeh figure.
        backend : {"matplotlib", "bokeh"}, default "matplotlib"
            Select plotting backend.
        backend_kwargs : bool, optional
            These are kwargs specific to the backend being used, passed to
            :func:`matplotlib.pyplot.subplots` or :class:`bokeh.plotting.figure`.
            For additional documentation check the plotting method of the backend.
        show : bool, optional
            Call backend show function.

        Returns
        -------
        axes : matplotlib_axes or bokeh_figure

        See Also
        --------
        plot_elpd : Plot pointwise elpd differences between two or more models.
        compare : Compare models based on PSIS-LOO loo or WAIC waic cross-validation.
        loo : Compute Pareto-smoothed importance sampling leave-one-out cross-validation (PSIS-LOO-CV).
        waic : Compute the widely applicable information criterion.

        References
        ----------
        .. [1] Vehtari et al. (2016). Practical Bayesian model evaluation using leave-one-out
           cross-validation and WAIC https://arxiv.org/abs/1507.04544

        .. [2] McElreath R. (2022). Statistical Rethinking A Bayesian Course with Examples in
           R and Stan, Second edition, CRC Press.



        """
        return az.plot_compare(
            comp_df,
            insample_dev=insample_dev,
            plot_standard_error=plot_standard_error,
            plot_ic_diff=plot_ic_diff,
            order_by_rank=order_by_rank,
            legend=legend,
            title=title,
            figsize=figsize,
            textsize=textsize,
            labeller=labeller,
            plot_kwargs=plot_kwargs,
            ax=ax,
            backend=backend,
            backend_kwargs=backend_kwargs,
            show=show,
        )

    def rhat(self, *args, **kwargs):
        """Calculate R-hat statistics for convergence.

        Args:
            *args, **kwargs: Additional arguments for arviz.rhat

        Returns:
            rhat: R-hat values
        """
        # NOTE: results are stored under a private name. Assigning to
        # ``self.rhat`` replaced this bound method with its own return value,
        # so a second call raised "'Dataset' object is not callable".
        self._rhat_result = az.rhat(self._ensure_trace(), *args, **kwargs)
        return self._rhat_result

    def ess(self, *args, **kwargs):
        """Calculate effective sample size (ESS).

        Args:
            *args, **kwargs: Additional arguments for arviz.ess

        Returns:
            ess: Effective sample sizes
        """
        self._ess_result = az.ess(self._ensure_trace(), *args, **kwargs)
        return self._ess_result

    # --- Plotting Functions arviz dependent---
    def plot_ess(self, kind="local", **kwargs):
        """Plot effective sample size across the posterior quantiles.

        ArviZ 1.x ``plot_ess`` supports ``kind`` in {"local", "quantile"} (the
        old "evolution" kind was removed) and returns a PlotCollection.

        Returns:
            fig: ESS plot
        """
        self.ess_plot = az.plot_ess(
            self._ensure_trace(), var_names=self.priors_name, kind=kind, **kwargs
        )
        return self.ess_plot

    def rank(self, **kwargs):
        """Create rank plots for MCMC chains.

        ArviZ 1.x ``plot_rank`` manages its own figure (no ``ax=``) and returns
        a PlotCollection.

        Returns:
            fig: Rank plots
        """
        self._rank_plot = az.plot_rank(
            self._ensure_trace(), var_names=self.priors_name, **kwargs)
        return self._rank_plot

    # --- Plotting Functions  plotly dependent---

    def _summary_row(self, chains: np.ndarray, hdi_prob: float, kind: str) -> dict:
        """Statistics for one scalar parameter. chains: (C, S), chain-major.

        Kept as one function so the scalar and the multi-dimensional branches of
        ``summary`` cannot drift apart.
        """
        chains = np.asarray(chains, dtype=float)
        flat = chains.flatten()
        hdi = _az_hdi(flat, hdi_prob)
        row = {
            "mean": np.mean(flat),
            "median": np.median(flat),
            "sd": np.std(flat, ddof=1),
            f"hdi_{hdi_prob*100}%_lower": hdi[0],
            f"hdi_{hdi_prob*100}%_upper": hdi[1],
        }
        if kind == "stats":
            return row
        # The convergence block needs the chain structure, so it takes `chains`
        # rather than `flat`. Same estimators `diagnose` uses.
        row["mcse_mean"] = self._mcse_mean_manual(chains)
        row["mcse_sd"] = self._mcse_sd_manual(chains)
        row["ess_bulk"] = self._ess_bulk_manual(chains)
        row["ess_tail"] = self._ess_tail_manual(chains)
        row["r_hat"] = self._rhat_manual(chains)
        return row

    def summary(self, round_to=2, hdi_prob=0.89, var_names=None,
                exclude_vars=None, kind="all"):
        """Generate a summary table of posterior statistics.

        Args:
            round_to: Number of decimal places to round the statistics to
            hdi_prob: Credible interval probability (e.g., 0.89 for 89% HDI)
            var_names: List of variable names to include in the summary
            exclude_vars: List of variable names to exclude from the summary
            kind: "all" (default) adds the convergence block -- mcse_mean,
                mcse_sd, ess_bulk, ess_tail, r_hat -- to the point estimates.
                "stats" returns the point estimates only.

        Returns:
            summary_stats: Dictionary containing summary statistics
        """
        import numpy as np
        import arviz as az
        import pandas as pd

        if kind not in ("all", "stats"):
            raise ValueError(f"unknown kind {kind!r}; expected 'all' or 'stats'")

        summary_stats = {}
        vars_to_process = list(self.posterior_samples.keys())

        # --- FIX 2: Add variable filtering ---
        if var_names is not None:
            # Ensure it's a list (handles single strings passed from R)
            if isinstance(var_names, str):
                var_names = [var_names]
            vars_to_process = [v for v in vars_to_process if v in var_names]

        if exclude_vars is not None:
            if isinstance(exclude_vars, str):
                exclude_vars = [exclude_vars]
            vars_to_process = [v for v in vars_to_process if v not in exclude_vars]

        for var_name in vars_to_process:
            samples = self.posterior_samples[var_name]
            param_shape = samples.shape[2:]

            if not param_shape:
                summary_stats[var_name] = self._summary_row(
                    samples, hdi_prob, kind)
            else:
                for idx in np.ndindex(param_shape):
                    idx_str = "[" + ", ".join(map(str, idx)) + "]"
                    full_name = f"{var_name}{idx_str}"
                    element_samples = samples[(slice(None), slice(None)) + idx]
                    summary_stats[full_name] = self._summary_row(
                        element_samples, hdi_prob, kind)

        self.tab_summary = pd.DataFrame(summary_stats).T.round(round_to)
        return self.tab_summary

    def pair(
        self,
        var_names=None,
        colorscale="Viridis",
        max_points=1000,
        point_color="rgba(40, 150, 200, 0.4)",
    ):
        go = importer.get_module("go")

        if var_names is None:
            var_names = self.priors_name

        # Expand multi-dimensional variables
        expanded_vars = []
        expanded_samples = {}
        for var in var_names:
            samples = self.posterior_samples[var]
            param_shape = samples.shape[2:]
            if not param_shape:
                expanded_vars.append(var)
                expanded_samples[var] = samples.flatten()
            else:
                for idx in np.ndindex(param_shape):
                    idx_str = "[" + ", ".join(map(str, idx)) + "]"
                    full_name = f"{var}{idx_str}"
                    expanded_vars.append(full_name)
                    expanded_samples[full_name] = samples[
                        (slice(None), slice(None)) + idx
                    ].flatten()

        n_vars = len(expanded_vars)
        df = pd.DataFrame(expanded_samples)
        plot_df = (
            df.sample(n=max_points, random_state=42) if len(df) > max_points else df
        )
        fig = make_subplots(
            rows=n_vars, cols=n_vars, horizontal_spacing=0.03, vertical_spacing=0.03
        )

        for i in range(n_vars):
            for j in range(n_vars):
                var1, var2 = expanded_vars[i], expanded_vars[j]
                if i == j:
                    fig.add_trace(
                        go.Histogram(
                            x=df[var1], name=f"Hist {var1}", marker_color="#440154"
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
                elif i > j:
                    fig.add_trace(
                        go.Histogram2dContour(
                            x=df[var2],
                            y=df[var1],
                            colorscale=colorscale,
                            showscale=False,
                            name="Density",
                            contours=dict(coloring="lines"),
                            line=dict(width=1),
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df[var2],
                            y=plot_df[var1],
                            mode="markers",
                            name="Samples",
                            marker=dict(size=3, color=point_color),
                        ),
                        row=i + 1,
                        col=j + 1,
                    )
                    median_x, median_y = df[var2].median(), df[var1].median()
                    fig.add_trace(
                        go.Scatter(
                            x=[median_x],
                            y=[median_y],
                            mode="markers",
                            name="Median",
                            marker=dict(symbol="square", color="black", size=8),
                        ),
                        row=i + 1,
                        col=j + 1,
                    )

        fig.update_layout(
            title_text="Pair Plot: Histograms, Density, and Samples",
            height=250 * n_vars,
            width=250 * n_vars,
            showlegend=False,
            plot_bgcolor="white",
        )

        for i in range(n_vars):
            fig.update_yaxes(
                title_text=expanded_vars[i],
                row=i + 1,
                col=1,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            )
        for j in range(n_vars):
            fig.update_xaxes(
                title_text=expanded_vars[j],
                row=n_vars,
                col=j + 1,
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            )

        return fig

    def plot_trace(self, var_names=None):
        go = importer.get_module("go")
        if var_names is None:
            var_names = self.priors_name

        # Expand multi-dimensional variables
        flattened_vars = []
        for var in var_names:
            samples = self.posterior_samples[var]
            param_shape = samples.shape[2:]
            if not param_shape:
                flattened_vars.append((var, var))
            else:
                for idx in np.ndindex(param_shape):
                    idx_str = "[" + ", ".join(map(str, idx)) + "]"
                    full_name = f"{var}{idx_str}"
                    flattened_vars.append((full_name, var, idx))

        subplot_titles = [
            f"{var_label} {suffix}"
            for (var_label, _, *_) in flattened_vars
            for suffix in ["Trace", "Posterior"]
        ]

        fig = make_subplots(
            rows=len(flattened_vars), cols=2, subplot_titles=subplot_titles
        )

        for i, (var_label, orig_var, *idx_info) in enumerate(flattened_vars):
            if not idx_info:
                samples_per_chain = self.posterior_samples[orig_var]
            else:
                idx = idx_info[0]
                samples_per_chain = self.posterior_samples[orig_var][
                    (slice(None), slice(None)) + idx
                ]

            # Trace plot (column 1)
            for chain_idx in range(self.num_chains):
                color = self.colors[chain_idx % len(self.colors)]
                fig.add_trace(
                    go.Scatter(
                        y=samples_per_chain[chain_idx],
                        mode="lines",
                        name=f"Chain {chain_idx}",
                        legendgroup=f"chain{chain_idx}",
                        line=dict(color=color),
                        showlegend=(i == 0),
                    ),
                    row=i + 1,
                    col=1,
                )

            # Histogram (column 2)
            for chain_idx in range(self.num_chains):
                color = self.colors[chain_idx % len(self.colors)]
                fig.add_trace(
                    go.Histogram(
                        x=samples_per_chain[chain_idx],
                        name=f"Chain {chain_idx}",
                        legendgroup=f"chain{chain_idx}",
                        marker_color=color,
                        showlegend=False,
                        opacity=0.6,
                        nbinsx=50,
                    ),
                    row=i + 1,
                    col=2,
                )

        fig.update_layout(
            height=300 * len(flattened_vars),
            title_text="Trace and Posterior Plots",
            barmode="overlay",
        )

        return fig

    def posterior(self, var_names=None, figsize=(800, 400), hdi_prob=0.94):
        go = importer.get_module("go")
        import numpy as np

        if var_names is None:
            var_names = self.priors_name

        # Expand multi-dimensional variables
        flattened_vars = []
        for var in var_names:
            samples = self.posterior_samples[var]
            param_shape = samples.shape[2:]
            if not param_shape:
                flattened_vars.append((var, var))
            else:
                for idx in np.ndindex(param_shape):
                    idx_str = "[" + ", ".join(map(str, idx)) + "]"
                    full_name = f"{var}{idx_str}"
                    flattened_vars.append((full_name, var, idx))

        fig = make_subplots(
            rows=1,
            cols=len(flattened_vars),
            subplot_titles=[v[0] for v in flattened_vars],
        )

        for i, (var_label, orig_var, *idx_info) in enumerate(flattened_vars):
            if not idx_info:
                samples_per_chain = self.posterior_samples[orig_var]
            else:
                idx = idx_info[0]
                samples_per_chain = self.posterior_samples[orig_var][
                    (slice(None), slice(None)) + idx
                ]

            # Plot the histograms for each chain first
            for chain_idx in range(self.num_chains):
                color = self.colors[chain_idx % len(self.colors)]
                fig.add_trace(
                    go.Histogram(
                        x=samples_per_chain[chain_idx],
                        name=f"Chain {chain_idx}",
                        legendgroup=f"chain{chain_idx}",
                        marker_color=color,
                        showlegend=(i == 0),
                        opacity=0.6,
                        nbinsx=50,
                    ),
                    row=1,
                    col=i + 1,
                )

            # Combine all chains to get overall posterior summary statistics
            all_samples = samples_per_chain.flatten()

            # Calculate mean
            mean_val = np.mean(all_samples)

            # Actual HDI (narrowest interval), not an equal-tailed percentile
            # interval -- the two differ for skewed posteriors, and the lines
            # are labelled HDI.
            hdi_lower, hdi_upper = _az_hdi(np.asarray(all_samples), hdi_prob)

            # Add vertical line for the mean
            fig.add_vline(
                x=mean_val,
                line_dash="dash",
                line_color="black",
                annotation_text="",
                annotation_position="top right",
                row=1,
                col=i + 1,
            )

            # Add vertical lines for the HDI
            fig.add_vline(
                x=hdi_lower,
                line_dash="dot",
                line_color="firebrick",
                annotation_text=f"",
                annotation_position="top left",
                row=1,
                col=i + 1,
            )
            fig.add_vline(
                x=hdi_upper,
                line_dash="dot",
                line_color="firebrick",
                annotation_text=f"",
                annotation_position="top right",
                row=1,
                col=i + 1,
            )

        fig.update_layout(
            title_text="Posterior Distributions (Overlaid Chains)",
            width=(
                figsize[0]
                if len(flattened_vars) < 4
                else figsize[0] * len(flattened_vars) // 3
            ),
            height=figsize[1],
            barmode="overlay",
        )

        return fig

    def _rank_normalize(self, chains: np.ndarray) -> np.ndarray:
        """Rank-normalize samples across all chains. chains: (C, S)."""
        flat = chains.flatten()
        ranks = stats.rankdata(flat)
        z = stats.norm.ppf((ranks - 0.375) / (flat.size + 0.25))
        return z.reshape(chains.shape)

    def _split_chains(self, chains: np.ndarray) -> np.ndarray:
        """Split each chain in half -> (2*C, S//2).

        Takes the LAST half, matching ArviZ; for odd S the middle draw is
        dropped rather than reused.
        """
        C, S = chains.shape
        half = S // 2
        return np.concatenate([chains[:, :half], chains[:, -half:]], axis=0)

    def _rhat_core(self, split: np.ndarray) -> float:
        """Gelman-Rubin on already-split, already-transformed chains (m, n)."""
        m, n = split.shape
        chain_means = split.mean(axis=1)
        W = np.mean(np.var(split, axis=1, ddof=1))
        if not np.isfinite(W) or W == 0:
            return np.nan
        B = n * np.var(chain_means, ddof=1)
        var_hat = (n - 1) / n * W + B / n
        return float(np.sqrt(var_hat / W))

    def _rhat_manual(self, chains: np.ndarray) -> float:
        """Rank-normalized *folded* split R-hat (Vehtari et al. 2021). chains: (C, S).

        max of bulk R-hat (rank-normalized draws) and tail R-hat
        (rank-normalized |x - median(x)|). The folded half is what detects a
        chain whose VARIANCE has not converged; without it this returned ~1.00
        for chains ArviZ scores at 1.12.
        """
        split = self._split_chains(np.asarray(chains, dtype=float))
        rhat_bulk = self._rhat_core(self._rank_normalize(split))
        folded = np.abs(split - np.median(split))
        rhat_tail = self._rhat_core(self._rank_normalize(folded))
        if np.isnan(rhat_bulk) and np.isnan(rhat_tail):
            return np.nan
        return float(np.nanmax([rhat_bulk, rhat_tail]))

    def _ess_bulk_manual(self, chains: np.ndarray) -> float:
        """Rank-normalized split bulk ESS. chains: (C, S).

        Splits BEFORE rank-normalizing, as ArviZ does (matters for odd S).
        """
        split = self._split_chains(np.asarray(chains, dtype=float))
        return self._ess_raw_manual(self._rank_normalize(split))

    def _ess_tail_manual(self, chains: np.ndarray, prob: float = 0.05) -> float:
        """Tail ESS at prob and 1-prob quantiles (min of both).

        Splits before quantiling and uses ``<=`` (not ``<``), matching ArviZ's
        _ess_quantile; the strict comparison differed whenever a quantile landed
        exactly on a sample, as it does for discrete outcomes.
        """
        split = self._split_chains(np.asarray(chains, dtype=float))
        q_lo = (split <= np.quantile(split, prob)).astype(float)
        q_hi = (split <= np.quantile(split, 1 - prob)).astype(float)
        return min(self._ess_raw_manual(q_lo), self._ess_raw_manual(q_hi))

    def _ess_mean_manual(self, chains: np.ndarray) -> float:
        """Split ESS of the raw draws, used by the MCSEs. chains: (C, S).

        This is ArviZ's ``_ess_mean``: split the chains, then run the plain
        ESS on them. No rank normalization -- that is bulk ESS, and using it
        here would report the wrong Monte Carlo error.
        """
        split = self._split_chains(np.asarray(chains, dtype=float))
        return self._ess_raw_manual(split)

    def _mcse_mean_manual(self, chains: np.ndarray) -> float:
        """Monte Carlo standard error of the posterior mean. chains: (C, S).

        ``sd / sqrt(ess_mean)`` with ``ddof=1``, matching ArviZ's ``_mcse_mean``.
        """
        chains = np.asarray(chains, dtype=float)
        ess = self._ess_mean_manual(chains)
        if not np.isfinite(ess) or ess <= 0:
            return np.nan
        return float(np.std(chains, ddof=1) / np.sqrt(ess))

    def _mcse_sd_manual(self, chains: np.ndarray) -> float:
        """Monte Carlo standard error of the posterior sd. chains: (C, S).

        Port of ArviZ's ``_mcse_sd``: the delta-method error on the variance of
        the centred squares, halved onto the sd scale (the ``/ 4`` below is the
        square of the ``1 / (2 * sd)`` derivative).
        """
        chains = np.asarray(chains, dtype=float)
        sims_c2 = (chains - chains.mean()) ** 2
        ess = self._ess_mean_manual(sims_c2)
        if not np.isfinite(ess) or ess <= 0:
            return np.nan
        evar = sims_c2.mean()
        if not np.isfinite(evar) or evar <= 0:
            return np.nan
        varvar = ((sims_c2**2).mean() - evar**2) / ess
        return float(np.sqrt(varvar / evar / 4))

    def _ess_raw_manual(self, split: np.ndarray) -> float:
        """ESS from split chains. split: (M, N).

        Faithful port of ArviZ's ``_ess`` (Vehtari et al. 2021). The
        autocorrelation is normalized by the *combined* variance ``var_plus``
        (within-chain W plus between-chain B) — not W alone — and the truncation
        uses Geyer's initial-positive *and* initial-monotone sequences. This is
        what makes bulk and (especially) tail ESS agree with ``az.ess``; the
        earlier W-only / initial-positive-only version over-estimated tail ESS.
        """
        split = np.asarray(split, dtype=float)
        if split.ndim == 1:
            split = np.atleast_2d(split)
        n_chain, n_draw = split.shape
        if n_draw < 4:
            return np.nan
        # A constant array has ess == its size (matches ArviZ).
        if (np.max(split) - np.min(split)) < np.finfo(float).resolution:
            return float(split.size)

        # Per-chain autocovariance via FFT (linear, biased by 1/n).
        acov = []
        for chain in split:
            x = chain - chain.mean()
            f = np.fft.rfft(x, n=2 * n_draw)
            a = np.fft.irfft(f * np.conj(f), n=2 * n_draw)[:n_draw] / n_draw
            acov.append(a)
        acov = np.asarray(acov)  # (M, N)

        chain_mean = split.mean(axis=1)
        mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)          # W
        var_plus = mean_var * (n_draw - 1.0) / n_draw
        if n_chain > 1:
            var_plus += np.var(chain_mean, ddof=1)                       # + between-chain B

        rho_hat_t = np.zeros(n_draw)
        rho_hat_even = 1.0
        rho_hat_t[0] = rho_hat_even
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
        rho_hat_t[1] = rho_hat_odd

        # Geyer's initial positive sequence.
        t = 1
        while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
            rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
            rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
            if (rho_hat_even + rho_hat_odd) >= 0:
                rho_hat_t[t + 1] = rho_hat_even
                rho_hat_t[t + 2] = rho_hat_odd
            t += 2
        max_t = t - 2
        if rho_hat_even > 0:
            rho_hat_t[max_t + 1] = rho_hat_even

        # Geyer's initial monotone sequence.
        t = 1
        while t <= max_t - 2:
            if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
                rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
                rho_hat_t[t + 2] = rho_hat_t[t + 1]
            t += 2

        total = n_chain * n_draw
        tau_hat = (
            -1.0
            + 2.0 * np.sum(rho_hat_t[: max_t + 1])
            + np.sum(rho_hat_t[max_t + 1 : max_t + 2])
        )
        tau_hat = max(tau_hat, 1.0 / np.log10(total))
        if np.isnan(rho_hat_t).any():
            return np.nan
        return float(total / tau_hat)

    def _ebfmi_manual(self, energy: np.ndarray) -> list:
        """E-BFMI per chain. energy: (C, S). Returns list of floats.

        Numerator is the MEAN SQUARED energy difference, matching Stan and
        az.bfmi. Using var(diff, ddof=1) instead subtracts mean(diff)**2 and
        rescales by (N-1)/(N-2), which biased every reported value.
        """
        result = []
        for chain in np.atleast_2d(np.asarray(energy, dtype=float)):
            denom = np.var(chain, ddof=1)
            num = np.mean(np.diff(chain) ** 2)
            result.append(float(num / denom) if denom > 0 else np.nan)
        return result

    def diagnose(
        self,
        max_treedepth: int = 10,
        ebfmi_threshold: float = 0.3,
        rhat_threshold: float = 1.01,
        ess_threshold: float = 400.0,
    ) -> str:
        """
        Run MCMC convergence diagnostics on a fitted BF model.

        Parameters
        ----------
        max_treedepth : int
            Maximum tree depth used during sampling (default 10, matching BF default).
        ebfmi_threshold : float
            E-BFMI values below this trigger a warning (default 0.3).
        rhat_threshold : float
            R-hat values above this trigger a warning (default 1.01).
        ess_threshold : float
            ESS values below this trigger a warning (default 400).

        Returns
        -------
        str
            Human-readable diagnostic report.
        """
        lines = []
        problems = []
        # Not every backend exposes extra_fields (TFP / SVI). Each section below
        # already degrades gracefully on a missing key, so fall back to an empty
        # mapping rather than raising AttributeError here.
        try:
            extra = self.sampler.get_extra_fields(group_by_chain=True) or {}
        except (AttributeError, TypeError):
            extra = {}
        num_chains = self.num_chains
        posteriors_by_chain = self.posterior_samples

        # ------------------------------------------------------------------
        # 1. Treedepth
        # ------------------------------------------------------------------
        lines.append("Checking sampler transitions treedepth.")
        if "num_steps" in extra and extra["num_steps"] is not None:
            num_steps = np.array(extra["num_steps"])  # (C, S)
            if num_steps.ndim == 1:
                num_steps = num_steps.reshape(1, -1)
            tree_depth = np.ceil(np.log2(num_steps + 1)).astype(int)
            per_chain_max_treedepth = np.sum(tree_depth >= max_treedepth, axis=1).tolist()
            saturated = sum(per_chain_max_treedepth)
            chain_vals = " ".join(f"{v}" for v in per_chain_max_treedepth)
            lines.append(f"$num_max_treedepth\n[1] {chain_vals}")
            if saturated > 0:
                msg = (
                    f"{saturated} of {tree_depth.size} transitions hit the "
                    f"maximum treedepth limit of {max_treedepth}. "
                    "Increase max_tree_depth to improve sampling efficiency."
                )
                lines.append(msg)
                problems.append(msg)
            else:
                lines.append("Treedepth satisfactory for all transitions.")
        else:
            lines.append("Treedepth: extra_fields not collected (unavailable).")
        lines.append("")

        # ------------------------------------------------------------------
        # 2. Divergences
        # ------------------------------------------------------------------
        lines.append("Checking sampler transitions for divergences.")
        if "diverging" in extra and extra["diverging"] is not None:
            diverging = np.array(extra["diverging"])
            if diverging.ndim == 1:
                diverging = diverging.reshape(1, -1)
            per_chain_div = np.sum(diverging, axis=1).tolist()
            n_div = int(sum(per_chain_div))
            chain_vals = " ".join(f"{int(v)}" for v in per_chain_div)
            lines.append(f"$num_divergent\n[1] {chain_vals}")
            if n_div > 0:
                msg = (
                    f"{n_div} divergent transition(s) found after warmup. "
                    "Try increasing target_accept_prob or reparameterizing the model."
                )
                lines.append(msg)
                problems.append(msg)
            else:
                lines.append("No divergent transitions found.")
        else:
            lines.append("Divergences: extra_fields not collected (unavailable).")
        lines.append("")

        # ------------------------------------------------------------------
        # 3. E-BFMI
        # ------------------------------------------------------------------
        lines.append("Checking E-BFMI - sampler transitions HMC potential energy.")
        if "energy" in extra and extra["energy"] is not None:
            energy = np.array(extra["energy"])
            if energy.ndim == 1:
                energy = energy.reshape(1, -1)
            ebfmi_vals = self._ebfmi_manual(energy)
            chain_vals = " ".join(f"{v:.6f}" for v in ebfmi_vals)
            lines.append(f"$ebfmi\n[1] {chain_vals}")
            low_chains = [
                i
                for i, v in enumerate(ebfmi_vals)
                if not np.isnan(v) and v < ebfmi_threshold
            ]
            if low_chains:
                detail = ", ".join(
                    f"chain {i}: E-BFMI={ebfmi_vals[i]:.3f}" for i in low_chains
                )
                msg = f"E-BFMI below threshold ({ebfmi_threshold}): {detail}."
                lines.append(msg)
                problems.append(msg)
            else:
                lines.append("E-BFMI satisfactory.")
        else:
            lines.append("E-BFMI: extra_fields not collected (unavailable).")
        lines.append("")

        # ------------------------------------------------------------------
        # 4. ESS
        # ------------------------------------------------------------------
        lines.append("Checking rank-normalized split effective sample size.")
        ess_problems = []
        for param, samples in posteriors_by_chain.items():
            arr = np.array(samples)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            # flatten any extra dims into param indices
            C, S = arr.shape[0], arr.shape[1]
            extra_dims = arr.shape[2:]
            flat = arr.reshape(C, S, -1) if extra_dims else arr.reshape(C, S, 1)
            for idx in range(flat.shape[2]):
                chains = flat[:, :, idx]
                ess = self._ess_bulk_manual(chains)
                label = f"{param}[{idx}]" if extra_dims else param
                if not np.isnan(ess) and ess < ess_threshold:
                    ess_problems.append(f"{label}: ESS={ess:.1f}")
        if ess_problems:
            msg = "Low ESS for: " + "; ".join(ess_problems) + "."
            lines.append(msg)
            problems.append(msg)
        else:
            lines.append(
                "Rank-normalized split effective sample size satisfactory for all parameters."
            )
        lines.append("")

        # ------------------------------------------------------------------
        # 5. R-hat
        # ------------------------------------------------------------------
        lines.append("Checking rank-normalized split R-hat values.")
        rhat_problems = []
        for param, samples in posteriors_by_chain.items():
            arr = np.array(samples)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            C, S = arr.shape[0], arr.shape[1]
            extra_dims = arr.shape[2:]
            flat = arr.reshape(C, S, -1) if extra_dims else arr.reshape(C, S, 1)
            for idx in range(flat.shape[2]):
                chains = flat[:, :, idx]
                rh = self._rhat_manual(chains)
                label = f"{param}[{idx}]" if extra_dims else param
                if not np.isnan(rh) and rh > rhat_threshold:
                    rhat_problems.append(f"{label}: R-hat={rh:.4f}")
        if rhat_problems:
            msg = "High R-hat for: " + "; ".join(rhat_problems) + "."
            lines.append(msg)
            problems.append(msg)
        else:
            lines.append(
                "Rank-normalized split R-hat values satisfactory for all parameters."
            )
        lines.append("")

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        if problems:
            lines.append(f"Processing complete, {len(problems)} problem(s) detected.")
        else:
            lines.append("Processing complete, no problems detected.")

        report = "\n".join(lines)
        # print(report) # Removed print to return report string only, or keep if user wants it printed
        return report

    def autocor(self, var_names=None):
        go = importer.get_module("go")
        if var_names is None:
            var_names = self.priors_name

        # Expand multi-dimensional variables
        flattened_vars = []
        for var in var_names:
            samples = self.posterior_samples[var]
            param_shape = samples.shape[2:]
            if not param_shape:
                flattened_vars.append((var, var))
            else:
                for idx in np.ndindex(param_shape):
                    idx_str = "[" + ", ".join(map(str, idx)) + "]"
                    full_name = f"{var}{idx_str}"
                    flattened_vars.append((full_name, var, idx))

        fig = make_subplots(
            rows=len(flattened_vars),
            cols=1,
            subplot_titles=[f"Autocorrelation of {v[0]}" for v in flattened_vars],
        )
        for i, (var_label, orig_var, *idx_info) in enumerate(flattened_vars):
            if not idx_info:
                samples_per_chain = self.posterior_samples[orig_var]
            else:
                idx = idx_info[0]
                samples_per_chain = self.posterior_samples[orig_var][
                    (slice(None), slice(None)) + idx
                ]

            for chain_idx in range(self.num_chains):
                samples = np.asarray(samples_per_chain[chain_idx], dtype=float)
                autocorr = _acf(samples, max_lag=40)
                color = self.colors[chain_idx % len(self.colors)]
                fig.add_trace(
                    go.Bar(
                        y=autocorr,
                        name=f"Chain {chain_idx}",
                        legendgroup=f"chain{chain_idx}",
                        marker_color=color,
                        showlegend=(i == 0),
                    ),
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(
            height=250 * len(flattened_vars),
            title_text="Autocorrelation Plots by Chain",
            barmode="group",
        )

        return fig

    def forest(self, var_names=None, hdi_prob=0.95):
        go = importer.get_module("go")
        import numpy as np
        import arviz as az
        import plotly.express as px

        if var_names is None:
            var_names = self.priors_name

        # Expand multi-dimensional variables
        flattened_vars = []
        for var in var_names:
            samples = self.posterior_samples[var]
            param_shape = samples.shape[2:]
            if not param_shape:
                flattened_vars.append((var, var))
            else:
                for idx in np.ndindex(param_shape):
                    idx_str = "[" + ", ".join(map(str, idx)) + "]"
                    full_name = f"{var}{idx_str}"
                    flattened_vars.append((full_name, var, idx))

        fig = go.Figure()
        colors = px.colors.qualitative.Plotly

        for i, (var_label, orig_var, *idx_info) in enumerate(flattened_vars):
            color = colors[i % len(colors)]
            if not idx_info:
                samples_per_chain = self.posterior_samples[orig_var]
            else:
                idx = idx_info[0]
                samples_per_chain = self.posterior_samples[orig_var][
                    (slice(None), slice(None)) + idx
                ]

            all_samples = samples_per_chain.flatten()

            fig.add_trace(
                go.Violin(
                    x=all_samples,
                    y=[
                        f" {var_label} "
                    ],  # Extra spaces to avoid naming conflicts with violin group
                    name=var_label,
                    legendgroup=var_label,
                    orientation="h",
                    side="both",
                    points=False,
                    fillcolor=color,
                    opacity=0.4,
                    line_width=0,
                    spanmode="hard",
                )
            )

            mean_val = np.mean(all_samples)
            hdi = _az_hdi(np.array(all_samples), hdi_prob)
            hdi_lower, hdi_upper = hdi[0], hdi[1]

            error_upper = hdi_upper - mean_val
            error_lower = mean_val - hdi_lower

            fig.add_trace(
                go.Scatter(
                    x=[mean_val],
                    y=[f" {var_label} "],
                    mode="markers",
                    legendgroup=var_label,
                    name=var_label,
                    marker=dict(color=color, size=8),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=[error_upper],
                        arrayminus=[error_lower],
                        width=4,
                        color=color,
                    ),
                    showlegend=False,
                )
            )

        fig.add_vline(
            x=0,
            line_dash="dash",
            line_color="black",
            annotation_text="",
            annotation_position="top right",
        )
        fig.update_layout(
            title_text=f"Forest Plot (Posterior Distributions and {hdi_prob*100:.1f}% HDI)",
            xaxis_title="Parameter Value",
            yaxis_title="Parameter",
            violingap=0.1,
            plot_bgcolor="white",
        )
        fig.update_yaxes(autorange="reversed")

        return fig

    def density(self, var_names=None, shade=0.4):
        sns = importer.get_module("sns")
        plt = importer.get_module("plt")
        go = importer.get_module("go")
        if var_names is None:
            var_names = self.priors_name

        # Expand multi-dimensional variables
        flattened_vars = []
        for var in var_names:
            samples = self.posterior_samples[var]
            param_shape = samples.shape[2:]
            if not param_shape:
                flattened_vars.append((var, var))
            else:
                for idx in np.ndindex(param_shape):
                    idx_str = "[" + ", ".join(map(str, idx)) + "]"
                    full_name = f"{var}{idx_str}"
                    flattened_vars.append((full_name, var, idx))

        fig = make_subplots(
            rows=len(flattened_vars),
            cols=1,
            subplot_titles=[f"Density of {v[0]}" for v in flattened_vars],
        )
        for i, (var_label, orig_var, *idx_info) in enumerate(flattened_vars):
            if not idx_info:
                samples_per_chain = self.posterior_samples[orig_var]
            else:
                idx = idx_info[0]
                samples_per_chain = self.posterior_samples[orig_var][
                    (slice(None), slice(None)) + idx
                ]

            for chain_idx in range(self.num_chains):
                color = self.colors[chain_idx % len(self.colors)]
                rgb_color = pcolors.hex_to_rgb(color)
                fill_color = (
                    f"rgba({rgb_color[0]},{rgb_color[1]},{rgb_color[2]},{shade})"
                )
                chain_samples = samples_per_chain[chain_idx]
                with sns.plotting_context(rc={"figure.figsize": (1, 1)}):
                    kde_plot = sns.kdeplot(chain_samples)
                    kde = kde_plot.get_lines()[0].get_data()
                    plt.close()
                fig.add_trace(
                    go.Scatter(
                        x=kde[0],
                        y=kde[1],
                        fill="tozeroy",
                        mode="lines",
                        name=f"Chain {chain_idx}",
                        legendgroup=f"chain{chain_idx}",
                        showlegend=(i == 0),
                        fillcolor=fill_color,
                        line_color=color,
                    ),
                    row=i + 1,
                    col=1,
                )
        fig.update_layout(
            height=300 * len(flattened_vars),
            title_text="Density Plots (Overlaid Chains)",
        )

        return fig

    def model_checks(self):
        """Perform comprehensive model diagnostics with interactive plots."""
        print("Displaying Posterior Plots (Overlaid Chains):")
        self.posterior().show()
        print("\nDisplaying Autocorrelation Plots (by Chain):")
        self.autocor().show()
        print("\nDisplaying Trace Plots (by Chain):")
        self.plot_trace().show()
        print("\nDisplaying Forest Plot (All Chains Combined):")
        self.forest().show()
        print("\nDisplaying Pair Plot (All Chains Combined):")
        self.pair().show()
