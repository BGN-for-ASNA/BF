from BayesForge.Utils.ImportManager import LazyImporter
import numpy as np
import jax.numpy as jnp
from jax import tree_util
import itertools
import scipy.stats as stats

# Create a global instance of the importer
importer = LazyImporter()

# Register the heavy plotting libraries for lazy import on first use (no threads).
importer.schedule_import("plotly.graph_objects", "go")
importer.schedule_import("plotly.express", "px")
importer.schedule_import("plotly.figure_factory", "ff")
importer.schedule_import("plotly.colors", "n_colors")
importer.schedule_import("seaborn", "sns")
importer.schedule_import("matplotlib.pyplot", "plt")


def _autocorr_1d(series, lags):
    """Autocorrelation of a 1D numpy array for lags 0..lags-1, NaN-free.

    RELEASE_AUDIT (Utils, autocorr): ``np.corrcoef(series[:-k], series[k:])``
    returned NaN with a RuntimeWarning for ``k >= len(series)`` (empty slices)
    and for constant series (zero variance), and the interactive 1D path used
    ``np.corrcoef`` while its 2D non-interactive twin used ``jnp.corrcoef``
    (two different NaN semantics). Every path now goes through this single
    numpy helper: lag 0 is 1.0 and any lag whose overlap has fewer than two
    distinct points is 0.0 (no correlation information), never NaN.
    """
    lags = max(int(lags), 0)
    acf = [1.0] if lags > 0 else []
    for k in range(1, lags):
        a, b = series[:-k], series[k:]
        if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
            acf.append(0.0)
        else:
            acf.append(float(np.corrcoef(a, b)[0, 1]))
    return acf


class SampledData:
    """
    A wrapper for a JAX numpy array that adds interactive plotting methods using Plotly
    while preserving all other JAX array functionalities through attribute delegation.

    Public contract (RELEASE_AUDIT, Utils, "Contrat public"): the plotting
    methods above the delegation layer are the real added value; the
    statistical/export surface is deliberately thin and documented here
    rather than expanded:

    * First-class statistics are only :meth:`hdi` and :meth:`to_jax`.
      ``sd.mean()``, ``sd.std()``, ``sd.quantile()`` etc. are delegated to
      the wrapped JAX array, so they use ``ddof=0`` (population, not sample,
      statistics — biased on few draws), have no ``nanmean``-style variants,
      and never treat axis 0 as a chain axis: there is no ``(chains, draws)``
      semantics anywhere in this class.
    * There is no CSV/ArviZ export; convert with :meth:`to_jax` or
      ``np.asarray`` and export yourself.
    * Plotting methods accept data containing NaN/inf without warning
      (``np.histogram``/KDE simply ignore or fail on them); :meth:`density`
      is the only method that filters non-finite values.
    * ``__eq__`` is element-wise (it returns an array), so — like numpy
      arrays — SampledData is **unhashable**: ``dict[sd] = ...`` and ``set``
      membership raise ``TypeError``.
    * The wrapped JAX array is immutable: ``sd[idx] = v`` raises ``TypeError``
      (use ``sd.at[idx].set(v)``), which also neutralises the views-vs-copies
      risk on the source posterior.
    """

    def __init__(self, data):
        """
        Initializes the SampledData object.

        Args:
            data (jnp.ndarray): The sampled JAX array.
        """
        # Ensure data is a JAX array for consistency
        self._data = jnp.asarray(data)

    def _wrap_result(self, result):
        """Wraps the result in a SampledData object if it's a JAX array with dimension > 0."""
        if isinstance(result, jnp.ndarray) and result.ndim > 0:
            return SampledData(result)
        return result

    def _extract_data(self, other):
        """Extracts the JAX array if the other object is a SampledData instance."""
        if isinstance(other, SampledData):
            return other._data
        return other

    # NOTE: _tree_flatten / _tree_unflatten are defined once, near the pytree
    # registration at the bottom of the class. An earlier duplicate pair (whose
    # unflatten routed through __init__/jnp.asarray and broke vmap/scan) used
    # to sit here; it was shadowed by the second pair and has been removed.

    def __repr__(self):
        # Mirror jax's own repr so the dtype (int/float + precision 8/16/32/64)
        # is visible, just re-labelled with the wrapper name. jax formats arrays
        # as "Array(<values>, dtype=...)"; swap the leading class name for ours.
        inner = repr(self._data)
        if inner.startswith("Array("):
            return "SampledData(" + inner[len("Array("):]
        # Fallback for any array type whose repr doesn't start with "Array(":
        # append the dtype explicitly so type info is never lost.
        return f"SampledData({self._data}, dtype={self._data.dtype})"

    # --- Rich display (Jupyter / IPython) -----------------------------------
    # type() is honest (SampledData), so IPython resolves these on the real
    # type. _register_ipython_pprinter still pins the text/plain path to repr()
    # for belt-and-braces (older IPython dispatch quirks).
    def _repr_pretty_(self, p, cycle):
        p.text(repr(self))

    def _repr_html_(self):
        return f"<pre>{repr(self)}</pre>"

    def hist(
        self, 
        title="Histogram of Sampled Data", 
        nbinsx=30, xaxis_title="Value", yaxis_title="Frequency", 
        template="plotly_white", 
        interactive = True,
        figsize=(6, 4),**kwargs
    ):
        """Histogram of the draws.

        For 3D data the array is read as ``(n_samples, n_groups, n_times)``:
        axis 1 indexes groups (one colour / z-offset each), axis 2 indexes
        times (one line each, plotted on y). The shared bin range covers the
        data — it used to be frozen at [-4, 4], which silently dropped every
        value outside it (``np.histogram`` ignores out-of-bin values) and
        produced wrong densities for non-standardised draws.
        """
        if interactive:
            """Interactive histogram visualization."""
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")

            fig = go.Figure()

            if self._data.ndim == 1:
                 fig.add_trace(go.Histogram(x=np.array(self._data), nbinsx=nbinsx, **kwargs))

            elif self._data.ndim == 2:
                for i in range(self._data.shape[1]):
                    fig.add_trace(go.Histogram(x=np.array(self._data[:, i]), nbinsx=nbinsx, name=f"Var {i}  ", **kwargs))

            elif self._data.ndim == 3:
                # (n_samples, n_groups, n_times): groups live on axis 1, times
                # on axis 2. The old code sliced [:, t, g] while iterating t
                # over range(n_times) — IndexError when n_times > n_groups,
                # silently wrong slices otherwise.
                n_samples, n_groups, n_times = self._data.shape
                arr = np.asarray(self._data)
                bins = nbinsx
                # Data-driven bin range that keeps the historical [-4, 4]
                # whenever the data fits inside it (identical output for
                # standardised draws) but grows to cover outliers instead of
                # dropping them.
                lo = min(-4.0, float(arr.min()))
                hi = max(4.0, float(arr.max()))
                bin_edges = np.linspace(lo, hi, bins + 1)
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

                colorscale = px.colors.qualitative.Plotly
                colors = [colorscale[g % len(colorscale)] for g in range(n_groups)]

                for g in range(n_groups):
                    for t in range(n_times):
                        values = arr[:, g, t]
                        counts, _ = np.histogram(values, bins=bin_edges, density=True)

                        x = bin_centers
                        y = np.full_like(bin_centers, t)
                        z = counts + g

                        fig.add_trace(go.Scatter3d(
                            x=x, y=y, z=z,
                            mode="lines",
                            line=dict(width=5, color=colors[g]),
                            name=f"Group {g}, Time {t}"
                        ))
            fig.update_layout(title=title, template=template,
                             xaxis_title=xaxis_title, yaxis_title=yaxis_title)
            fig.show()
        else:
            if self._data.ndim > 2:
                raise ValueError(f"Density plot is only supported for 1D or 2D data. Your data has {self._data.ndim} dimensions.")
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            
            plt.figure(figsize=figsize)
            if self._data.ndim == 1:
               sns.histplot(self._data, bins=nbinsx, kde=False)
            elif self._data.ndim == 2:
               for i in range(self._data.shape[1]):
                   sns.histplot(self._data[:, i], bins=nbinsx, kde=False, label=f"Var {i}", alpha=0.5)
               plt.legend()
            plt.title(title)
            plt.show()


    def corr_heatmap(self, title="Correlation Matrix", template="plotly_white", digits=5, interactive = True, figsize=(6, 5), **kwargs):
        """
        Visualizes the Pearson correlation matrix of the sampled data as a heatmap
        with annotations. The input data are draws with shape
        (n_samples, n_variables); the plotted matrix is their Pearson correlation,
        with shape (n_variables, n_variables).
        Values are explicitly formatted as strings to ensure consistent rounding in the plot.
        The y-axis is inverted to match standard matrix representation.
        """
     
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if self._data.ndim != 2:
                raise ValueError(f"Correlation heatmap is only supported for 2D data. Your data has   {self.  _data.ndim} dimensions.")
            
            # The data are draws [n_samples, n_variables]; plot their Pearson
            # correlation [n_variables, n_variables], never the raw draws.
            if self._data.shape[0] < 2:
                raise ValueError("Cannot calculate correlation with fewer than 2 samples.")
            corr_matrix = np.corrcoef(np.asarray(self._data), rowvar=False)
        
            # Format annotations into strings for consistent display.
            formatted_text = np.vectorize(lambda v: f"{v:.{digits}f}")(corr_matrix)
        
            x = [f'Var {i}' for i in range(corr_matrix.shape[1])]
            y = [f'Var {i}' for i in range(corr_matrix.shape[0])]
            
            # go.Heatmap + texttemplate replaces ff.create_annotated_heatmap,
            # removed in plotly >= 6.
            fig = go.Figure(data=go.Heatmap(
                z=np.asarray(corr_matrix),
                x=x,
                y=y,
                text=formatted_text,
                texttemplate="%{text}",
                colorscale='Viridis',
                **kwargs,
            ))

            # Invert the y-axis to have Var 0 at the bottom
            fig.update_yaxes(autorange='reversed')
            
            fig.update_layout(title_text=title, template=template)
            fig.show()

        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Heatmap requires 2D data.")
            if self._data.shape[0] < 2:
                raise ValueError("Cannot calculate correlation with fewer than 2 samples.")
            corr_matrix = np.corrcoef(np.asarray(self._data), rowvar=False)
            plt.figure(figsize=figsize)
            sns.heatmap(corr_matrix, cmap="viridis")
            plt.title(title)
            plt.show()


    def boxplot(self, title="Boxplot of Matrix Samples", template="plotly_white", interactive = True,figsize=(6, 4),  **kwargs):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim != 2:
                raise ValueError("Boxplot requires 2D data.")
            fig = go.Figure()
            for i in range( self._data.shape[1]):
                fig.add_trace(go.Box(y= self._data[:, i], name=f"Var {i}", **kwargs))
            fig.update_layout(title=title, template=template)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if  self._data.ndim != 2:
                raise ValueError("Boxplot requires 2D data.")
            plt.figure(figsize=figsize)
            sns.boxplot(data= self._data)
            plt.title(title)
            plt.show()

    def violinplot(self, title="Violin Plot of Matrix Samples", template="plotly_white", interactive = True, figsize=(6, 4), **kwargs):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim != 2:
                raise ValueError("Violin plot requires 2D data.")
            fig = go.Figure()
            for i in range( self._data.shape[1]):
                fig.add_trace(go.Violin(y= self._data[:, i], name=f"Var {i}", box_visible=True,     meanline_visible=True, **kwargs))
            fig.update_layout(title=title, template=template)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Violin plot requires 2D data.")
            plt.figure(figsize=figsize)
            sns.violinplot(data=self._data, inner="box")
            plt.title(title)
            plt.show()

    def pairplot(self, max_vars=5, title="Pairwise Scatter Plots", interactive = True):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim != 2:
                raise ValueError("Pairplot requires 2D data.")
            n_vars = min( self._data.shape[1], max_vars)
            fig = go.Figure()
            for i, j in itertools.combinations(range(n_vars), 2):
                fig.add_trace(go.Scatter(
                    x= self._data[:, i],
                    y= self._data[:, j],
                    mode="markers",
                    name=f"Var {i} vs Var {j}",
                    opacity=0.5
                ))
            fig.update_layout(title=title)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Pairplot requires 2D data.")
            n_vars = min(self._data.shape[1], max_vars)
            import pandas as pd
            df = pd.DataFrame(self._data[:, :n_vars], columns=[f"Var {i}" for i in range(n_vars)])
            sns.pairplot(df, diag_kind="kde")
            plt.suptitle(title, y=1.02)
            plt.show()

    def timeseries(self, credible_interval=0.9, title="Sampled Time Series", interactive = True,figsize=(8, 4)):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")

            if  self._data.ndim != 2:
                raise ValueError("Timeseries requires shape [n_samples, n_time].")
            mean =  self._data.mean(axis=0)
            lower = np.percentile( self._data, (1 - credible_interval) / 2 * 100, axis=0)
            upper = np.percentile( self._data, (1 + credible_interval) / 2 * 100, axis=0)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=mean, mode="lines", name="Mean"))
            fig.add_trace(go.Scatter(y=upper, mode="lines", name="Upper", line=dict(dash="dash")))
            fig.add_trace(go.Scatter(y=lower, mode="lines", name="Lower", line=dict(dash="dash"),
                                     fill="tonexty", fillcolor="rgba(0,100,200,0.2)"))
            fig.update_layout(title=title)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Timeseries requires shape [n_samples, n_time].")
            mean = self._data.mean(axis=0)
            lower = np.percentile(self._data, (1 - credible_interval) / 2 * 100, axis=0)
            upper = np.percentile(self._data, (1 + credible_interval) / 2 * 100, axis=0)

            plt.figure(figsize=figsize)
            plt.plot(mean, label="Mean")
            plt.fill_between(np.arange(len(mean)), lower, upper, alpha=0.3, label=f"{int    (credible_interval*100)}% CI")
            plt.title(title)
            plt.legend()
            plt.show()

    def scatter3d(self, title="3D Scatter of Samples", interactive = True,figsize=(6, 5)):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if self._data.ndim != 2 or self._data.shape[1] < 3:
                raise ValueError("Need shape [n_samples, >=3] for 3D scatter.")
            fig = go.Figure(data=[go.Scatter3d(
            x=self._data[:, 0], y=self._data[:, 1], z=self._data[:, 2],
            mode="markers",
            marker=dict(size=3, opacity=0.5)
            )])
            fig.update_layout(title=title)
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2 or self._data.shape[1] < 3:
                raise ValueError("Need shape [n_samples, >=3] for 3D scatter.")
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(self._data[:, 0], self._data[:, 1], self._data[:, 2], s=5, alpha=0.5)
            ax.set_title(title)
            plt.show()

    def traceplot(self, title="Trace Plot of Samples", interactive = True,figsize=(10, 6)):
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim != 2:
                raise ValueError("Trace plot requires [n_samples, n_chains/variables].")
            fig = go.Figure()
            for i in range( self._data.shape[1]):
                fig.add_trace(go.Scatter(y= self._data[:, i], mode="lines", name=f"Var {i}", opacity=0.7))
            fig.update_layout(title=title, xaxis_title="Iteration", yaxis_title="Value")
            fig.show()
        else:
            sns=importer.get_module("sns")
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Trace plot requires [n_samples, n_chains/variables].")
            plt.figure(figsize=figsize)
            for i in range(self._data.shape[1]):
                plt.plot(self._data[:, i], label=f"Var {i}", alpha=0.7)
            plt.title(title)
            plt.xlabel("Iteration")
            plt.ylabel("Value")
            plt.legend()
            plt.show()
    
    def autocorr(self, lags=50, title="Autocorrelation Plot", interactive = True,figsize=(8, 4)):
        """Bar plot of the autocorrelation of each column (or of a 1D series).

        All four branches (interactive/non-interactive x 1D/2D) share the
        single numpy helper :func:`_autocorr_1d`: the data is converted to
        numpy once (no per-lag device sync) and short or constant series
        yield 0.0 instead of NaN + RuntimeWarning (see the helper's
        docstring).
        """
        if interactive:
            go=importer.get_module("go")
            px=importer.get_module("px")
            ff=importer.get_module("ff")
            n_colors=importer.get_module("n_colors")
            if  self._data.ndim == 1:
                series = np.asarray(self._data)
                acf = _autocorr_1d(series, lags)
                fig = go.Figure([go.Bar(x=list(range(lags)), y=acf)])
                fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="Autocorrelation")
                fig.show()
            elif  self._data.ndim == 2:
                fig = go.Figure()
                arr = np.asarray(self._data)
                for i in range( self._data.shape[1]):
                    series = arr[:, i]
                    acf = _autocorr_1d(series, lags)
                    fig.add_trace(go.Bar(x=list(range(lags)), y=acf, name=f"Var {i}", opacity=0.5))
                fig.update_layout(title=title, xaxis_title="Lag", yaxis_title="Autocorrelation")
                fig.show()
            else:
                raise ValueError("Autocorrelation requires 1D or 2D data.")
        else:
            plt=importer.get_module("plt")
            plt.figure(figsize=figsize)
            if self._data.ndim == 1:
                series = np.asarray(self._data)
                acf = _autocorr_1d(series, lags)
                plt.bar(range(lags), acf)
            elif self._data.ndim == 2:
                arr = np.asarray(self._data)
                for i in range( self._data.shape[1]):
                    series = arr[:, i]
                    acf = _autocorr_1d(series, lags)
                    plt.bar(range(lags), acf, alpha=0.5, label=f"Var {i}")
                plt.legend()
            else:
                raise ValueError("Autocorrelation requires 1D or 2D data.")
            plt.title(title)
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.show()
    # -----------------
    # Delegation
    
    def density(self, title="Density Plot", template="plotly_white", nbinsx=40, **kwargs):
        """
        Visualizes the distribution of the data as a histogram with a
        Gaussian-KDE overlay.

        Replaces ``plotly.figure_factory.create_distplot``, removed in
        plotly >= 6.
        """
        import numpy as np
        import plotly.graph_objects as go
        try:
            from scipy.stats import gaussian_kde
        except Exception:
            gaussian_kde = None

        if self._data.ndim > 2:
            raise ValueError(f"Density plot is only supported for 1D or 2D data. Your data has {self._data.ndim} dimensions.")

        if self._data.ndim == 1:
            series = [np.asarray(self._data)]
            labels = ['Sample']
        else:  # 2D
            series = [np.asarray(self._data[:, i]) for i in range(self._data.shape[1])]
            labels = [f'Var {i}' for i in range(self._data.shape[1])]

        fig = go.Figure()
        for s, lab in zip(series, labels):
            s = s[np.isfinite(s)]
            fig.add_trace(go.Histogram(x=s, histnorm='probability density',
                                       name=lab, opacity=0.55, nbinsx=nbinsx))
            if gaussian_kde is not None and s.size > 1 and np.ptp(s) > 0:
                xs = np.linspace(s.min(), s.max(), 200)
                fig.add_trace(go.Scatter(x=xs, y=gaussian_kde(s)(xs),
                                         mode='lines', name=f'{lab} KDE'))
        fig.update_layout(title_text=title, template=template, barmode='overlay')
        fig.show()
        
    def ridgeline(self, title="Ridgeline Plot", template="plotly_white",interactive = True,category_labels=None, offset=2):
        """Ridgeline (joy) plot of the columns of a (n_samples, n_categories) array.

        Each *column* is one category (one violin / one density ridge); rows
        are draws. The interactive branch used to ``zip(self._data, colors)``,
        which iterates the rows and therefore plotted the first
        ``n_categories`` draws as if they were the variables, and it accepted
        3D input that ``go.Violin`` cannot render. Both fixed: 2D input is
        required and the columns are plotted.
        """
        if interactive:
            go=importer.get_module("go")
            # importer alias "n_colors" resolves to the plotly.colors MODULE, not
            # the function; import the callable directly.
            from plotly.colors import n_colors
            if self._data.ndim != 2:
                raise ValueError(f"Ridgeline plot requires 2D [samples, categories] data. Your data has {self._data.ndim}       dimensions.")

            n_samples, n_categories = self._data.shape
            colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', n_categories, colortype='rgb')
            arr = np.asarray(self._data)

            fig = go.Figure()
            for i, color in enumerate(colors):
                # One violin per category (column), not per draw (row).
                fig.add_trace(
                    go.Violin(x=arr[:, i], line_color='black', name=i, fillcolor=color)
                    )

            # use negative ... cuz I'm gonna flip things later
            fig = fig.update_traces(orientation='h', side='negative', width=3, points=False, opacity=1)
            # reverse the (z)-order of the traces

            # flip the y axis (negative violin is now positive and traces on the top are now on the bottom)
            fig.update_layout(legend_traceorder='reversed', yaxis_autorange='reversed').show()

            fig.update_layout(
                title=title,
                template=template,
                showlegend=False,
                yaxis=dict(showticklabels=False, title="Categories"),
                xaxis_title="Value"
            )
            
        else:
            plt=importer.get_module("plt")
            if self._data.ndim != 2:
                raise ValueError("Ridgeline requires 2D [samples, categories].")
            n_samples, n_categories = self._data.shape
            if category_labels is None:
                category_labels = [f"Category {i}" for i in range(n_categories)]

            plt.figure(figsize=(8, n_categories * 0.7))
            for i in range(n_categories):
                values = self._data[:, i]
                kde = stats.gaussian_kde(values)
                x_range = np.linspace(values.min() - 1, values.max() + 1, 300)
                y_vals = kde(x_range)
                plt.fill_between(x_range, y_vals + i * offset, i * offset, alpha=0.6)
                plt.plot(x_range, y_vals + i * offset, lw=1)
            plt.yticks([i * offset for i in range(n_categories)], category_labels)
            plt.title(title)
            plt.show()


    def surface_3d(self, title="3D Surface Plot", template="plotly_white", **kwargs):
        """
        Visualizes 3D data as a surface plot.
        """
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.figure_factory as ff
        from plotly.colors import n_colors
        if self._data.ndim != 2:
            raise ValueError(f"3D Surface plot is only supported for 2D data. Your data has {self._data.ndim} dimensions.")

        fig = go.Figure(data=[go.Surface(z=self._data, colorscale='Viridis', **kwargs)])
        fig.update_layout(
            title=title, template=template,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Value'
            )
        )
        fig.show()

    def ppc_plot(self, observed_data, n_samples_to_plot=50, title="Posterior Predictive Check", template="plotly_white", seed=0):
        """
        Creates a posterior predictive check plot.

        The subset of draws to plot is chosen with a local
        ``np.random.default_rng(seed)`` (never the global NumPy RNG).

        KDE guards (RELEASE_AUDIT, Utils, ppc_plot): ``stats.gaussian_kde``
        raises ``LinAlgError`` (singular covariance) on constant data with no
        useful message. Predictive draws that are constant (or have fewer
        than two points) carry no density information and are skipped;
        constant observed data raises a clear ``ValueError`` instead of the
        bare ``LinAlgError``.
        """
        from BayesForge.Utils.dev_warning import warn_dev
        warn_dev("ppc")
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.figure_factory as ff
        from plotly.colors import n_colors
        if self._data.ndim != 2:
            raise ValueError(f"PPC plot expects 2D sampled data [n_draws, n_observations]. Your data has {self._data.ndim} dimensions.")

        observed_data = np.array(observed_data)
        sampled_data = np.array(self._data)

        if observed_data.size < 2 or np.ptp(observed_data) == 0:
            raise ValueError(
                "ppc_plot needs observed data with at least two distinct "
                f"values; got {observed_data.size} value(s) with zero spread. "
                "A Gaussian KDE is undefined for constant data (singular "
                "covariance matrix)."
            )

        fig = go.Figure()

        # Plot densities of posterior predictive samples (thin, semi-transparent lines)
        rng = np.random.default_rng(seed)
        subset_indices = rng.choice(sampled_data.shape[0], size=min(n_samples_to_plot, sampled_data.shape[0]), replace=False)
        for i in subset_indices:
            row = sampled_data[i, :]
            if row.size < 2 or np.ptp(row) == 0:
                # Constant predictive draw: gaussian_kde would raise
                # LinAlgError; the draw carries no density information.
                continue
            density = stats.gaussian_kde(row)
            x_vals = np.linspace(min(observed_data.min(), sampled_data.min()), max(observed_data.max(), sampled_data.max()), 200)
            fig.add_trace(go.Scatter(
                x=x_vals, y=density(x_vals),
                mode='lines',
                line=dict(width=1, color='rgba(70, 130, 180, 0.5)'), # SteelBlue with alpha
                showlegend=False
            ))

        # Plot density of observed data (thick, solid line)
        density_observed = stats.gaussian_kde(observed_data)
        x_vals = np.linspace(min(observed_data.min(), sampled_data.min()), max(observed_data.max(), sampled_data.max()), 200)
        fig.add_trace(go.Scatter(
            x=x_vals, y=density_observed(x_vals),
            mode='lines',
            line=dict(width=3, color='black'),
            name='Observed Data Density'
        ))

        # Update layout for a clean, modern look
        fig.update_layout(
            title=title,
            xaxis_title="Value",
            yaxis_title="Density",
            template=template,
            legend=dict(x=0.01, y=0.98) # Position legend inside the plot
        )
        fig.show()

# ================== Array basics ==================
    def hdi(self, cred_mass=0.95):
        """
        Compute highest density interval (HDI) from samples.

        Axis 0 holds the draws. A 1-D array gives one interval; an array of
        shape (draws, ...) gives one interval per column, computed
        independently along axis 0 (e.g. shape (100, 4): 4 intervals).

        Args:
            cred_mass: float, credible mass (default 0.95)

        Returns:
            (hdi_min, hdi_max): scalars for 1-D draws, otherwise arrays of
            shape ``data.shape[1:]``.
        """
        data = jnp.asarray(self._data)
        if data.ndim == 0:
            raise ValueError("hdi() needs an array of draws (axis 0), got a scalar.")
        if data.shape[0] == 0:
            raise ValueError("hdi() got no draws.")
        if bool(jnp.any(jnp.isnan(data))):
            raise ValueError("hdi() got NaN draws; drop them (e.g. divergent draws) first.")
        samples = jnp.sort(data, axis=0)
        n = samples.shape[0]
        interval_idx_inc = int(jnp.floor(cred_mass * n))
        if interval_idx_inc < 1:
            raise ValueError(f"cred_mass * n_draws = {cred_mass * n:.3g} < 1: too few draws for this cred_mass.")

        lows  = samples[: n - interval_idx_inc]
        highs = samples[interval_idx_inc:]
        widths = highs - lows

        # Narrowest window per column (a scalar index for 1-D draws).
        min_idx = jnp.argmin(widths, axis=0)[None]
        lo = jnp.take_along_axis(lows, min_idx, axis=0)[0]
        hi = jnp.take_along_axis(highs, min_idx, axis=0)[0]
        return lo, hi

    def to_jax(self):
        return jnp.array(self._data)
# ================== ARITHMETIC OPERATORS ==================

    def __add__(self, other):
        return self._wrap_result(self._data + self._extract_data(other))
    def __radd__(self, other):
        return self._wrap_result(self._extract_data(other) + self._data)
    def __sub__(self, other):
        return self._wrap_result(self._data - self._extract_data(other))
    def __rsub__(self, other):
        return self._wrap_result(self._extract_data(other) - self._data)
    def __mul__(self, other):
        return self._wrap_result(self._data * self._extract_data(other))
    def __rmul__(self, other):
        return self._wrap_result(self._extract_data(other) * self._data)
    def __truediv__(self, other):
        return self._wrap_result(self._data / self._extract_data(other))
    def __rtruediv__(self, other):
        return self._wrap_result(self._extract_data(other) / self._data)
    def __floordiv__(self, other):
        return self._wrap_result(self._data // self._extract_data(other))
    def __rfloordiv__(self, other):
        return self._wrap_result(self._extract_data(other) // self._data)
    def __mod__(self, other):
        return self._wrap_result(self._data % self._extract_data(other))
    def __rmod__(self, other):
        return self._wrap_result(self._extract_data(other) % self._data)
    def __pow__(self, other):
        return self._wrap_result(self._data ** self._extract_data(other))
    def __rpow__(self, other):
        return self._wrap_result(self._extract_data(other) ** self._data)
    def __matmul__(self, other):
        return self._wrap_result(self._data @ self._extract_data(other))
    def __rmatmul__(self, other):
        return self._wrap_result(self._extract_data(other) @ self._data)
    def __divmod__(self, other):
        d = self._extract_data(other)
        return self._wrap_result(self._data // d), self._wrap_result(self._data % d)
    def __rdivmod__(self, other):
        d = self._extract_data(other)
        return self._wrap_result(d // self._data), self._wrap_result(d % self._data)

    # Unary operators
    def __neg__(self):
        return self._wrap_result(-self._data)
    def __pos__(self):
        return self._wrap_result(+self._data)
    def __abs__(self):
        return self._wrap_result(jnp.abs(self._data))

    # Bitwise operators
    def __and__(self, other):
        return self._wrap_result(self._data & self._extract_data(other))
    def __rand__(self, other):
        return self._wrap_result(self._extract_data(other) & self._data)
    def __or__(self, other):
        return self._wrap_result(self._data | self._extract_data(other))
    def __ror__(self, other):
        return self._wrap_result(self._extract_data(other) | self._data)
    def __xor__(self, other):
        return self._wrap_result(self._data ^ self._extract_data(other))
    def __rxor__(self, other):
        return self._wrap_result(self._extract_data(other) ^ self._data)
    def __invert__(self):
        return self._wrap_result(~self._data)
    def __lshift__(self, other):
        return self._wrap_result(self._data << self._extract_data(other))
    def __rlshift__(self, other):
        return self._wrap_result(self._extract_data(other) << self._data)
    def __rshift__(self, other):
        return self._wrap_result(self._data >> self._extract_data(other))
    def __rrshift__(self, other):
        return self._wrap_result(self._extract_data(other) >> self._data)

    # Comparison operators
    # __eq__ is element-wise and returns an array, so Python sets __hash__
    # to None: SampledData is unhashable (like a numpy array) — see the
    # class docstring.
    def __eq__(self, other):
        return self._data == self._extract_data(other)
    def __ne__(self, other):
        return self._data != self._extract_data(other)
    def __lt__(self, other):
        return self._data < self._extract_data(other)
    def __le__(self, other):
        return self._data <= self._extract_data(other)
    def __gt__(self, other):
        return self._data > self._extract_data(other)
    def __ge__(self, other):
        return self._data >= self._extract_data(other)

    # Built-in functions
    def __len__(self):
        return len(self._data)
    def __iter__(self):
        for item in self._data:
            yield self._wrap_result(item) if isinstance(item, jnp.ndarray) else item
    def __contains__(self, item):
        """Membership test with numpy semantics (RELEASE_AUDIT, Utils, contracts).

        ``x in sd`` asks whether the value ``x`` occurs in the wrapped array.
        Like ``x in np_array``, testing a *multi-element* array raises
        ``ValueError`` (the elementwise result has no truth value — see
        ``__bool__``); use ``np.isin`` for that.
        """
        return self._extract_data(item) in self._data
    def __bool__(self):
        """Truth value of the wrapped array (numpy semantics).

        Only valid for 0- or 1-element data; a multi-element array raises
        ``ValueError`` instead of an ambiguous True/False. This is what makes
        ``multi_element_array in sd`` raise (see ``__contains__``).
        """
        return bool(self._data)
    def __int__(self):
        return int(self._data)
    def __float__(self):
        return float(self._data)
    def __complex__(self):
        return complex(self._data)
    def __format__(self, format_spec):
        return format(self._data, format_spec)

    # Array protocol methods
    def __array__(self, dtype=None, copy=None):
        # numpy 2 passes `copy`; accept it (conversion from a jax array always
        # materialises a fresh numpy array anyway) so np.asarray(sd, copy=...)
        # does not raise a DeprecationWarning.
        return np.array(self._data, dtype=dtype) if dtype else np.array(self._data)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # out=/where= cannot work on the immutable wrapped JAX array (jax's
        # ufunc paths reject them); computing them in numpy instead would
        # silently write into a detached copy. Fail loudly (RELEASE_AUDIT,
        # Utils, type contracts).
        if "out" in kwargs or "where" in kwargs:
            raise NotImplementedError(
                f"SampledData does not support out=/where= for numpy ufuncs "
                f"(np.{ufunc.__name__}(...)): the wrapped JAX array is "
                "immutable. Unwrap with sd.to_jax() (or np.asarray(sd)) and "
                "run the ufunc on the plain array."
            )
        converted_inputs = [self._extract_data(inp) for inp in inputs]
        result = getattr(ufunc, method)(*converted_inputs, **kwargs)
        if isinstance(result, np.ndarray):
            # Non-__call__ methods (reduce/accumulate/outer/at) fall through
            # to numpy because jax's Array returns NotImplemented for them;
            # wrap the bare ndarray so np.add.reduce(sd) matches the
            # SampledData that sd + x would return (same values, jnp.asarray
            # is a dtype-preserving host copy).
            result = jnp.asarray(result)
        return self._wrap_result(result)

    # Indexing and slicing
    def __getitem__(self, idx):
        result = self._data[idx]
        return self._wrap_result(result) if isinstance(result, jnp.ndarray) and result.ndim > 0 else result

    def __setitem__(self, idx, value):
        # Match jax: arrays are immutable. In-place item assignment silently
        # broke value semantics (alias writes, mutation seen by a cached jit,
        # float->int truncation). Use `sd = sd.at[idx].set(value)` instead.
        raise TypeError(
            "SampledData wraps an immutable JAX array and does not support "
            "in-place item assignment. Instead of x[idx] = y, use "
            "x = x.at[idx].set(y) or another .at[] method."
        )

    @property
    def at(self):
        # Route .at[...] .set/.add/... back through the wrapper so the result
        # stays a SampledData rather than decaying to a bare jax array.
        return _SampledDataAtHelper(self)

    def __getattr__(self, name):
        """
        Dynamically delegates attribute access to the underlying JAX array.
        This is the most direct and robust way to mimic the array's behavior.

        Guards (RELEASE_AUDIT, Utils, __getattr__): reading ``self._data``
        unguarded here recursed forever on a partially constructed object
        (an exception in ``__init__``, or ``object.__new__`` outside the
        pytree path) — every attribute lookup, ``_data`` included, re-entered
        ``__getattr__`` and a RecursionError masked the original error.
        ``__getattr__`` only runs after normal lookup fails, so probing
        ``self.__dict__`` is the recursion-free way to find ``_data``.
        Typos (``sd.meann()``) now raise an AttributeError naming
        SampledData instead of the wrapped jax type.
        """
        data = self.__dict__.get("_data")
        if data is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute {name!r} "
                "and is not initialised: '_data' was never assigned (the "
                "constructor raised before completing)."
            )
        try:
            attr = getattr(data, name)
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object (wrapping "
                f"{type(data).__name__}) has no attribute {name!r}"
            ) from None

        if callable(attr):
            # If the attribute is a method (like .sum(), .reshape()), return
            # a new function that calls the original method and wraps the result.
            def wrapper(*args, **kwargs):
                unwrapped_args = [self._extract_data(arg) for arg in args]
                unwrapped_kwargs = {k: self._extract_data(v) for k, v in kwargs.items()}

                result = attr(*unwrapped_args, **unwrapped_kwargs)
                return self._wrap_result(result)
            return wrapper
        else:
            # If the attribute is a property (like .T, .shape), just return its value,
            # wrapping it if it's an array.
            return self._wrap_result(attr)

    def __dir__(self):
            # Start with the attributes of this class
            own_attrs = set(super().__dir__())
            # Add the attributes from the wrapped data object (guarded: an
            # uninitialised instance must stay introspectable, see
            # __getattr__ above).
            data = self.__dict__.get("_data")
            if data is None:
                return sorted(own_attrs)
            data_attrs = set(dir(data))
            return sorted(list(own_attrs | data_attrs))

    # --- jax interop without a __class__ lie ---
    # This class does NOT spoof __class__, so `type(sd) is SampledData` and
    # `isinstance(sd, jnp.ndarray)` is False. jax compatibility comes from
    # __jax_array__ (coercion hook every jnp.*/lax.* path honours), the pytree
    # registration below (jit/vmap/grad/scan), and __array__ (numpy). Code that
    # must branch on "is this an array" should test
    # `isinstance(x, (jnp.ndarray, SampledData))` or unwrap via x.to_jax() first.
    __array_priority__ = 100  # win binops against raw numpy arrays on the left

    # Reconstruct via the real class for pickle/cloudpickle (BayesForge model
    # save/load); the default would also work now, this keeps it explicit.
    def __reduce__(self):
        return (SampledData, (self._data,))

    # --- ADD THIS METHOD ---
    def __jax_array__(self):
        """
        Allows the class to be treated as a JAX array directly.
        This is the key to robust compatibility with transformations like vmap.
        """
        return self._data
        
    # --- KEEP YOUR PYTREE METHODS ---
    def _tree_flatten(self):
        children = (self._data,)
        aux_data = None
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        # Do NOT route through __init__: it calls jnp.asarray(data), and JAX's
        # pytree contract requires unflatten to accept ARBITRARY leaf objects.
        # During vmap / lax.scan (and transpose) JAX rebuilds a treedef with
        # placeholder leaves -- tracers, None, or a bare object() sentinel --
        # and jnp.asarray on those raises
        #   TypeError: Value '<object object ...>' with dtype object is not a
        #   valid JAX array type
        # which broke vmap over SampledData and SampledData as a scan carry.
        # Construct without validation and let real coercion stay in __init__.
        obj = object.__new__(cls)
        obj._data = children[0]
        return obj


class _SampledDataAtIndex:
    """Result of `sd.at[idx]`. Delegates .set/.add/.mul/... to the underlying
    jax `.at[idx]` ref and re-wraps the (new) array as a SampledData."""

    def __init__(self, parent, idx):
        self._parent = parent
        self._ref = parent._data.at[idx]

    def __getattr__(self, name):
        method = getattr(self._ref, name)

        def wrapper(*args, **kwargs):
            args = [self._parent._extract_data(a) for a in args]
            kwargs = {k: self._parent._extract_data(v) for k, v in kwargs.items()}
            return self._parent._wrap_result(method(*args, **kwargs))

        return wrapper


class _SampledDataAtHelper:
    """The `sd.at` object; `sd.at[idx]` -> _SampledDataAtIndex."""

    def __init__(self, parent):
        self._parent = parent

    def __getitem__(self, idx):
        return _SampledDataAtIndex(self._parent, idx)


tree_util.register_pytree_node(
    SampledData,
    lambda x: x._tree_flatten(),    # The flatten function is the instance method
    SampledData._tree_unflatten     # The unflatten function is the class method
)


def _register_ipython_pprinter():
    """Pin IPython/Jupyter text/plain display for SampledData to repr().

    Historically SampledData spoofed __class__ -> jnp.ndarray, which made
    IPython's text/plain pretty-printer dispatch on the spoofed class and print
    a bare id ("<jax.Array at 0x...>"). The spoof is gone (type() is now honest),
    so IPython already resolves _repr_pretty_ on the real type; this registration
    just belt-and-braces it by delegating the SampledData text/plain path to
    repr(). Genuine jax arrays are left untouched. No-op outside a running
    IPython.
    """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            return
        text_formatter = ip.display_formatter.formatters.get("text/plain")
        if text_formatter is not None:
            text_formatter.for_type(SampledData, lambda obj, p, cycle: p.text(repr(obj)))
    except Exception:
        # Display is a convenience; never let it break importing BayesForge.
        pass


_register_ipython_pprinter()