  
import arviz as az
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt
import contextlib
import inspect
import warnings
import numpy as np

@contextlib.contextmanager
def _quiet_float_summary():
    """Silence the float-division artifacts az.summary/rhat/ess raise when a
    chain is degenerate (all-equal draws give ESS = 0 and r_hat = NaN). The
    information stays visible as NaNs in the result; only the noise is gone."""
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in (scalar )?divide",
            category=RuntimeWarning,
        )
        yield

# Shared ArviZ-1.x-compatible helpers (NumPy-backed idata for LOO + native,
# ArviZ-free WAIC computed from the NumPyro log-likelihood).
from BayesForge.Diagnostic.Diag2 import (
    _az_from_dict,
    _numpyro_idata_with_loglik,
    _pointwise_loglik,
    _waic,
)

class diag():
    """
    The diag class serves as a comprehensive toolkit for diagnosing and visualizing the results of Bayesian models, particularly those fitted using MCMC samplers like NumPyro or TensorFlow Probability. It is built as a high-level wrapper around the arviz library, simplifying common diagnostic tasks into convenient methods. The class takes a sampler object upon initialization and provides a suite of functions for in-depth model checking, comparison, and visualization.
    """

    def __init__(self, sampler):
        """Initialize the diagnostic class."""
        self.sampler = sampler
        # Must exist so the `if self.trace is None` guards below can run; they
        # previously raised AttributeError before they could ever see None.
        self.trace = None
        self.priors_name = []

    def _ensure_trace(self):
        """Build self.trace on first use rather than requiring an explicit to_az()."""
        if getattr(self, "trace", None) is None:
            self.to_az()
        return self.trace

    # Diagnostic with ARVIZ ----------------------------------------------------------------------------
    def to_az(self, backend="numpyro", sample_stats_name=['target_log_prob','log_accept_ratio','has_divergence','energy']):
        """Convert the sampler output to an arviz trace object.
        
        This method prepares the trace for use with arviz diagnostic tools.
        
        Returns:
            self.trace: The arviz trace object containing the diagnostic data
        """
        if backend == "numpyro":
            self.trace = az.from_numpyro(self.sampler)
            self.priors_name = list(self.trace['posterior'].data_vars.keys())
            return self.trace
        
        elif backend == "tfp":
            var_names= list(self.sampler.model_info.keys())
            sample_stats = {k:jnp.transpose(v) for k, v in zip(sample_stats_name, self.sampler.sample_stats)}
            trace = {}
            #First dim is the number of chains
            #Second dim is the number of sampling
            #The rest is the shape of the object
            for name, samp in zip(var_names, self.sampler.posterior):
                trace[name] = samp
    
            self.trace = _az_from_dict({"posterior": trace, "sample_stats": sample_stats})
            self.priors_name = var_names
            return self.trace

        raise ValueError(f"unknown backend {backend!r}; expected 'numpyro' or 'tfp'")

    def summary(self, *args, round_to=2, kind="stats", hdi_prob=0.89, **kwargs):
        """Calculate summary statistics for the posterior distribution.

        Args:
            round_to (int): Number of decimal places to round results
            kind (str): Type of statistics to compute (default: "stats")
            hdi_prob (float): Probability for highest posterior density interval
            *args, **kwargs: Additional arguments for arviz.summary

        Returns:
            pd.DataFrame: Summary statistics of the posterior distribution
        """
        trace = self._ensure_trace()
        # arviz renamed this argument: 0.x takes hdi_prob=, 1.x takes ci_prob=.
        # Hard-coding ci_prob raised TypeError on every arviz 0.x install.
        params = inspect.signature(az.summary).parameters
        prob_kw = "ci_prob" if "ci_prob" in params else "hdi_prob"
        with _quiet_float_summary():
            self.tab_summary = az.summary(
                trace, *args, round_to=round_to, kind=kind,
                **{prob_kw: hdi_prob}, **kwargs)
        return self.tab_summary

    def plot_trace(self, var_names= None, kind="rank_bars", *args, **kwargs):
        """Create a trace plot for visualizing MCMC diagnostics.

        Args:
            var_names (list): List of variable names to include
            kind (str): Type of plot (default: "rank_bars")
            *args, **kwargs: Additional arguments for arviz.plot_trace

        Returns:
            plot: The trace plot object
        """
        trace = self._ensure_trace()
        self._trace_plot = az.plot_trace(
            trace, *args, var_names=var_names or self.priors_name,
            kind=kind, **kwargs)
        return self._trace_plot

    def posterior(self, figsize=(8, 4)):
        """Create posterior distribution plots.
        
        Args:
            figsize (tuple): Size of the figure (width, height)
            
        Returns:
            fig: Matplotlib figure containing posterior plots
        """        
        fig, axes = plt.subplots(1, len(self.priors_name), figsize=figsize)
        az.plot_posterior(self._ensure_trace(), var_names=self.priors_name, ax=axes)
        self._posterior_plot = fig
        return fig

    def autocor(self, *args, **kwargs):
        """Plot autocorrelation of the MCMC chains.

        Args:
            *args, **kwargs: Additional arguments for arviz.plot_autocorr

        Returns:
            fig: Autocorrelation plot
        """
        self._autocor_plot = az.plot_autocorr(
            self._ensure_trace(), *args, var_names=self.priors_name, **kwargs)
        return self._autocor_plot

    def rank(self, *args, **kwargs):
        """Create rank plots for MCMC chains.

        Args:
            *args, **kwargs: Additional arguments for arviz.plot_rank

        Returns:
            fig: Rank plots
        """
        fig, axes = plt.subplots(1, len(self.priors_name))
        az.plot_rank(self._ensure_trace(), *args,
                     var_names=self.priors_name, ax=axes, **kwargs)
        self._rank_plot = fig
        return fig

    def forest(self, data=None, kind="ridgeplot", ess=True, var_names=None,
               *args, **kwargs):
        """Create a forest plot of estimated values.

        Args:
            data: Data to plot (default: self.trace)
            kind (str): Type of plot (default: "ridgeplot")
            ess (bool): Include effective sample size
            var_names (list): Variables to include
            *args, **kwargs: Additional arguments for arviz.plot_forest

        Returns:
            fig: Forest plot
        """
        if var_names is None:
            var_names = self.priors_name
        if data is None:
            data = self._ensure_trace()
        self._forest_plot = az.plot_forest(
            data, *args, var_names=var_names, kind=kind, ess=ess, **kwargs)
        return self._forest_plot

    def rhat(self, *args, **kwargs):
        """Calculate R-hat statistics for convergence.
        
        Args:
            *args, **kwargs: Additional arguments for arviz.rhat
            
        Returns:
            rhat: R-hat values
        """        
        # Stored under a private name: assigning to self.rhat replaced this
        # bound method with its own result, breaking every later call.
        with _quiet_float_summary():
            self._rhat_result = az.rhat(self._ensure_trace(), *args, **kwargs)
        return self._rhat_result

    def ess(self, *args, **kwargs):
        """Calculate effective sample size (ESS).
        
        Args:
            *args, **kwargs: Additional arguments for arviz.ess
            
        Returns:
            ess: Effective sample sizes
        """        
        with _quiet_float_summary():
            self._ess_result = az.ess(self._ensure_trace(), *args, **kwargs)
        return self._ess_result

    def pair(self, var_names = None,
                  kind=["scatter", "kde"],
                  kde_kwargs={"fill_last": False},
                  marginals=True,
                  point_estimate="median",
                  figsize=(11.5, 5),
                  *args, **kwargs):
        """Create pairwise plots of the posterior distribution.
        
        Args:
            var_names (list): Variables to include
            kind (list): Type of plots ("scatter" and/or "kde")
            kde_kwargs (dict): Additional arguments for KDE plots
            marginals (bool): Include marginal distributions
            point_estimate (str): Point estimate to plot
            figsize (tuple): Size of the figure
            *args, **kwargs: Additional arguments for arviz.plot_pair
            
        Returns:
            fig: Pair plot
        """                  
        if var_names is None:
            var_names = self.priors_name
        self.pair_plot = az.plot_pair(self._ensure_trace(), var_names = var_names,
                                      kind=kind,
                                      kde_kwargs=kde_kwargs,
                                      marginals=marginals,
                                      point_estimate=point_estimate,
                                      figsize=figsize,
                                      *args, **kwargs)
        return self.pair_plot   
    
    def density(self, var_names=None, shade=0.2, *args, **kwargs):
        """Plot density plots of the posterior distribution.
        
        Args:
            var_names (list): Variables to include
            shade (float): Transparency of the filled area
            *args, **kwargs: Additional arguments for arviz.plot_density
            
        Returns:
            fig: Density plots
        """        
        if var_names is None:
            var_names = self.priors_name

        self._density_plot = az.plot_density(
                            self._ensure_trace(), *args,
                            var_names=var_names,
                            shade=shade,
                            **kwargs
                        )
        return self._density_plot

    def plot_ess(self, kind="evolution", **kwargs):
        """Plot effective sample size.

        Args:
            kind (str): "evolution" (arviz 0.x), "local" or "quantile".

        Returns:
            fig: ESS plot
        """
        self.ess_plot = az.plot_ess(
            self._ensure_trace(), var_names=self.priors_name, kind=kind, **kwargs)
        return self.ess_plot

    def model_checks(self):
        """Perform comprehensive model diagnostics.
        
        Creates various diagnostic plots including:
        - Posterior distributions
        - Autocorrelation plots
        - Trace plots
        - Rank plots
        - Forest plots
        
        Stores plots under self._posterior_plot, self._autocor_plot,
        self._traces_plot, self._rank_plot, self._forest_plot.
        """
        params = self.priors_name
        trace = self._ensure_trace()

        fig_post, axes = plt.subplots(1, len(params), figsize=(8, 4))
        az.plot_posterior(trace, var_names=params, ax=axes)

        autocor = az.plot_autocorr(trace, var_names=params)
        traces = az.plot_trace(trace, compact=False)

        fig_rank, axes = plt.subplots(1, len(params))
        az.plot_rank(trace, var_names=params, ax=axes)

        forest = az.plot_forest(trace, var_names=params)

        # NOTE: these must not be named after the methods above -- assigning to
        # self.autocor / self.rank / self.forest replaced those bound methods.
        self._posterior_plot = fig_post
        self._autocor_plot = autocor
        self._traces_plot = traces
        self._rank_plot = fig_rank
        self._forest_plot = forest

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
        # ArviZ 1.x: az.loo needs a NumPy-backed idata with log_likelihood and
        # no longer accepts `scale`.
        idata = _numpyro_idata_with_loglik(self.sampler)
        kwargs = {}
        if pointwise is not None:
            kwargs["pointwise"] = pointwise
        if var_name is not None:
            kwargs["var_name"] = var_name
        if reff is not None:
            kwargs["reff"] = reff
        if scale is not None:
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
    
    def WAIC(self,  pointwise=None, var_name=None, scale=None, dask_kwargs=None):
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
        # ArviZ 1.x removed az.waic; compute it natively from the NumPyro
        # pointwise log-likelihood (no ArviZ round-trip; falls back to the arviz
        # log_likelihood group for non-MCMC samplers).
        ll = _pointwise_loglik(self.sampler, var_name=var_name)
        return _waic(ll, pointwise=bool(pointwise), scale=scale or "log")

    @staticmethod
    def compare(compare_dict, ic=None, method='stacking', b_samples=1000, alpha=1, seed=None, scale=None, var_name=None):
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
        # Forward everything this ArviZ still supports instead of dropping
        # documented arguments silently. Shared with Diag2.compare.
        from BayesForge.Diagnostic.Diag2 import diagWIP as _diagWIP
        return _diagWIP.compare(
            compare_dict, ic=ic, method=method, b_samples=b_samples,
            alpha=alpha, seed=seed, scale=scale, var_name=var_name)

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
        return az.plot_compare(comp_df, insample_dev=insample_dev, plot_standard_error=plot_standard_error, plot_ic_diff=plot_ic_diff, order_by_rank=order_by_rank, legend=legend, title=title, figsize=figsize, textsize=textsize, labeller=labeller, plot_kwargs=plot_kwargs, ax=ax, backend=backend, backend_kwargs=backend_kwargs, show=show)
