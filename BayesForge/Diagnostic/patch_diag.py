import inspect
"""
JAX+Plotly diagnostic methods for BF models.

Two entry points:
  patch_diag_class(cls)       — patches a tool class; methods take (self, m, ...)
  bind_diag_to_model(diag, m) — patches a diag *instance*; methods take (include=, exclude=, ...)
                                 and close over m, so m.diag.density() respects m.posteriors
"""
import BayesForge.Diagnostic.jax_diagnostics as jd
from BayesForge.Diagnostic.jax_diagnostics import iter_expanded, filter_posterior_dict
import BayesForge.Diagnostic.ppc as _ppc
import BayesForge.Diagnostic.sensitivity as _sens
from BayesForge.Diagnostic.regression_plot import plot_regression as _plot_regression
import plotly.colors as pcolors

_COLORS = pcolors.qualitative.Plotly


def _acf(x, max_lag=40):
    """Autocorrelation of a single chain, lags 0..max_lag-1, on the global mean."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    n = x.size
    max_lag = int(min(max_lag, n))
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return [1.0] + [0.0] * (max_lag - 1)
    return [float(np.dot(xc[: n - t], xc[t:]) / denom) for t in range(max_lag)]


# =============================================================================
# Shared helpers
# =============================================================================

def _has_chains(m) -> bool:
    """True when m carries chain-major posteriors.

    Truthiness, not hasattr: the attribute can exist and be None/{}, in which
    case passing group_by_chain=True fed flat draws to chain-major code.
    """
    return bool(getattr(m, 'posteriors_by_chain', None))


def _get_posteriors(m, by_chain=False):
    if by_chain and _has_chains(m):
        return m.posteriors_by_chain
    if getattr(m, 'posteriors', None):
        return m.posteriors
    if getattr(m, 'posterior_samples', None):
        return m.posterior_samples
    if isinstance(m, dict):
        return m
    raise AttributeError(
        "Cannot find posteriors. Expected m.posteriors, m.posterior_samples, or a dict.")


def _source(m, filtered=True, by_chain=False):
    """Select posteriors dict: filtered=False uses posteriors_full (ignores active filter)."""
    if not filtered:
        if by_chain:
            full = getattr(m, 'posteriors_by_chain_full', None)
        else:
            full = getattr(m, 'posteriors_full', None)
        if full is not None:
            return full
    return _get_posteriors(m, by_chain=by_chain)


def _expand(m, include=None, exclude=None, filter_regex=None, filtered=True):
    """Return list of (label, 1D_samples).

    filtered=True  — uses m.posteriors (active filter) plus per-call include/exclude.
    filtered=False — uses m.posteriors_full (all parameters) plus per-call include/exclude.
    filter_regex applied on expanded labels.

    NOTE: chains are concatenated here. Use :func:`_expand_by_chain` for any
    plot whose meaning depends on chain identity (trace, autocorrelation).
    """
    posteriors = filter_posterior_dict(_source(m, filtered=filtered),
                                       include=include, exclude=exclude)
    return list(iter_expanded(posteriors, filter_regex=filter_regex))


def _expand_by_chain(m, include=None, exclude=None, filter_regex=None,
                     filtered=True):
    """Return list of (label, chains_2d) keeping the chain axis.

    A trace plot exists to show whether chains MIX, and an ACF is only defined
    within a chain -- both are meaningless on draws spliced end to end, which is
    what _expand produces.
    """
    import re
    import numpy as np

    posteriors = _source(m, filtered=filtered, by_chain=True)
    if not posteriors:
        posteriors = _source(m, filtered=filtered, by_chain=False)
    posteriors = filter_posterior_dict(posteriors, include=include, exclude=exclude)

    expanded = []
    for key, arr in posteriors.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]          # flat draws -> a single chain
        if arr.ndim == 2:
            expanded.append((key, arr))
        else:
            for idx in np.ndindex(arr.shape[2:]):
                label = f"{key}[{', '.join(map(str, idx))}]"
                expanded.append((label, arr[(slice(None), slice(None)) + idx]))

    if filter_regex:
        expanded = [(l, a) for l, a in expanded if re.search(filter_regex, l)]
    return expanded


# =============================================================================
# Module-level plot functions  (take m directly)
# =============================================================================

def _plot_trace(m, include=None, exclude=None, filter_regex=None, filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    expanded = _expand_by_chain(m, include, exclude, filter_regex, filtered)
    if not expanded:
        return go.Figure()
    titles = [f'{name} {s}' for (name, _) in expanded for s in ['Trace', 'Posterior']]
    fig = make_subplots(rows=len(expanded), cols=2, subplot_titles=titles)
    for i, (name, chains) in enumerate(expanded):
        # One line per chain, so mixing is visible. Concatenating the chains
        # into a single series (the old behaviour) hid exactly the failure a
        # trace plot is drawn to reveal.
        for c in range(chains.shape[0]):
            color = _COLORS[c % len(_COLORS)]
            fig.add_trace(go.Scatter(y=chains[c], mode='lines', name=f'Chain {c}',
                                     legendgroup=f'chain{c}', line=dict(color=color),
                                     showlegend=(i == 0)), row=i+1, col=1)
            fig.add_trace(go.Histogram(x=chains[c], name=f'Chain {c}',
                                       legendgroup=f'chain{c}', marker_color=color,
                                       showlegend=False, opacity=0.6, nbinsx=50),
                          row=i+1, col=2)
    fig.update_layout(height=300*len(expanded), title_text="Trace and Posterior Plots",
                      barmode='overlay')
    return fig


def _plot_posterior(m, include=None, exclude=None, filter_regex=None,
                    figsize=(800, 400), hdi_prob=0.94, filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    fig = make_subplots(rows=1, cols=len(expanded), subplot_titles=[e[0] for e in expanded])
    for i, (name, samples) in enumerate(expanded):
        color = _COLORS[i % len(_COLORS)]
        fig.add_trace(go.Histogram(x=samples, name=name, marker_color=color,
                                   showlegend=False, opacity=0.7, nbinsx=50), row=1, col=i+1)
        mean_val = float(np.mean(samples))
        hdi_vals = jd.hdi(samples, hdi_prob=hdi_prob)
        fig.add_vline(x=mean_val, line_dash="dash", line_color="black", row=1, col=i+1)
        fig.add_vline(x=float(hdi_vals[0]), line_dash="dot", line_color="firebrick", row=1, col=i+1)
        fig.add_vline(x=float(hdi_vals[1]), line_dash="dot", line_color="firebrick", row=1, col=i+1)
    n = len(expanded)
    fig.update_layout(title_text=f"Posterior Distributions ({hdi_prob*100:.0f}% HDI)",
                      width=figsize[0] if n < 4 else figsize[0]*n//3,
                      height=figsize[1], barmode='overlay')
    return fig


def _plot_autocor(m, include=None, exclude=None, filter_regex=None, max_lag=40, filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    expanded = _expand_by_chain(m, include, exclude, filter_regex, filtered)
    if not expanded:
        return go.Figure()
    fig = make_subplots(rows=len(expanded), cols=1,
                        subplot_titles=[f"Autocorrelation of {e[0]}" for e in expanded])
    for i, (name, chains) in enumerate(expanded):
        # ACF per chain, centred on that chain's mean. Computed across
        # concatenated chains it correlated the tail of one chain with the head
        # of the next; np.corrcoef(x[:-t], x[t:]) also re-centres each window,
        # which is not the ACF.
        for c in range(chains.shape[0]):
            color = _COLORS[c % len(_COLORS)]
            fig.add_trace(go.Bar(y=_acf(np.asarray(chains[c], dtype=float), max_lag),
                                 name=f'Chain {c}', legendgroup=f'chain{c}',
                                 marker_color=color, showlegend=(i == 0)),
                          row=i+1, col=1)
    fig.update_layout(height=250*len(expanded), title_text="Autocorrelation Plots",
                      barmode='group')
    return fig


def _plot_forest(m, include=None, exclude=None, filter_regex=None, hdi_prob=0.95, filtered=True):
    import plotly.graph_objects as go
    import jax.numpy as jnp

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    fig = go.Figure()
    for i, (name, samples) in enumerate(expanded):
        color = _COLORS[i % len(_COLORS)]
        fig.add_trace(go.Violin(x=samples, y=[f" {name} "], name=name, legendgroup=name,
                                orientation='h', side='both', points=False, fillcolor=color,
                                opacity=0.4, line_width=0, spanmode='hard'))
        mean_val = float(jnp.mean(samples))
        hdi_vals = jd.hdi(samples, hdi_prob=hdi_prob)
        lo, hi = float(hdi_vals[0]), float(hdi_vals[1])
        fig.add_trace(go.Scatter(x=[mean_val], y=[f" {name} "], mode='markers',
                                 legendgroup=name, name=name,
                                 marker=dict(color=color, size=8),
                                 error_x=dict(type='data', symmetric=False,
                                              array=[hi-mean_val], arrayminus=[mean_val-lo],
                                              width=4, color=color),
                                 showlegend=False))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(title_text=f'Forest Plot ({hdi_prob*100:.1f}% HDI)',
                      xaxis_title="Parameter Value", yaxis_title="Parameter",
                      violingap=0.1, plot_bgcolor='white')
    fig.update_yaxes(autorange="reversed")
    return fig


def _plot_density(m, include=None, exclude=None, filter_regex=None, shade=0.4, filtered=True):
    import plotly.graph_objects as go
    import plotly.colors as pcolors_
    from plotly.subplots import make_subplots
    import seaborn as sns
    import matplotlib.pyplot as plt

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    fig = make_subplots(rows=len(expanded), cols=1,
                        subplot_titles=[f"Density of {e[0]}" for e in expanded])
    for i, (name, samples) in enumerate(expanded):
        color = _COLORS[i % len(_COLORS)]
        rgb = pcolors_.hex_to_rgb(color)
        fill = f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{shade})'
        with sns.plotting_context(rc={"figure.figsize": (1, 1)}):
            kde_plot = sns.kdeplot(samples)
            kde = kde_plot.get_lines()[0].get_data()
            plt.close()
        fig.add_trace(go.Scatter(x=kde[0], y=kde[1], fill='tozeroy', mode='lines',
                                 name=name, showlegend=False, fillcolor=fill, line_color=color),
                      row=i+1, col=1)
    fig.update_layout(height=300*len(expanded), title_text="Density Plots")
    return fig


def _plot_rank(m, include=None, exclude=None, filter_regex=None, bins=20, filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import scipy.stats as stats

    posteriors_by_chain = _source(m, filtered=filtered, by_chain=True)
    if not posteriors_by_chain:
        posteriors_by_chain = _source(m, filtered=filtered, by_chain=False)

    from BayesForge.Diagnostic.jax_diagnostics import filter_posterior_dict
    posteriors_by_chain = filter_posterior_dict(posteriors_by_chain, include=include, exclude=exclude)

    # expand to (label, chains_2d) pairs
    expanded = []
    for key, arr in posteriors_by_chain.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]   # (1, S)
        if arr.ndim == 2:
            expanded.append((key, arr))
        else:
            C, S = arr.shape[0], arr.shape[1]
            for idx in np.ndindex(arr.shape[2:]):
                label = key + "[" + ",".join(map(str, idx)) + "]"
                expanded.append((label, arr[(slice(None), slice(None)) + idx]))

    if filter_regex:
        import re
        expanded = [(l, a) for l, a in expanded if re.search(filter_regex, l)]

    if not expanded:
        return go.Figure()

    colors = _COLORS

    fig = make_subplots(rows=len(expanded), cols=1,
                        subplot_titles=[f"Rank plot: {l}" for l, _ in expanded])

    for i, (label, chains) in enumerate(expanded):
        flat = chains.flatten()
        ranks = stats.rankdata(flat).reshape(chains.shape)
        # Per-parameter chain count; a global one taken from expanded[0]
        # IndexErrors on any parameter with fewer chains.
        for c in range(chains.shape[0]):
            color = colors[c % len(colors)]
            fig.add_trace(go.Histogram(
                x=ranks[c], name=f"Chain {c}",
                legendgroup=f"chain{c}", showlegend=(i == 0),
                marker_color=color, opacity=0.6, nbinsx=bins,
            ), row=i+1, col=1)

    fig.update_layout(height=300*len(expanded), title_text="Rank Plots", barmode="overlay")
    return fig


def _plot_ess_evolution(m, include=None, exclude=None, filter_regex=None,
                        steps=10, filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

    posteriors_by_chain = _source(m, filtered=filtered, by_chain=True)
    if not posteriors_by_chain:
        posteriors_by_chain = _source(m, filtered=filtered, by_chain=False)

    from BayesForge.Diagnostic.jax_diagnostics import filter_posterior_dict, _ess_1d
    posteriors_by_chain = filter_posterior_dict(posteriors_by_chain, include=include, exclude=exclude)

    expanded = []
    for key, arr in posteriors_by_chain.items():
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 2:
            expanded.append((key, arr))
        else:
            C, S = arr.shape[0], arr.shape[1]
            for idx in np.ndindex(arr.shape[2:]):
                label = key + "[" + ",".join(map(str, idx)) + "]"
                expanded.append((label, arr[(slice(None), slice(None)) + idx]))

    if filter_regex:
        import re
        expanded = [(l, a) for l, a in expanded if re.search(filter_regex, l)]

    if not expanded:
        return go.Figure()

    S_total = expanded[0][1].shape[1]
    fracs = np.linspace(1/steps, 1.0, steps)
    ns = [max(4, int(f * S_total)) for f in fracs]

    fig = make_subplots(rows=len(expanded), cols=1,
                        subplot_titles=[f"ESS evolution: {l}" for l, _ in expanded])

    for i, (label, chains) in enumerate(expanded):
        color = _COLORS[i % len(_COLORS)]
        ess_vals = []
        for n in ns:
            ess_vals.append(float(_ess_1d(chains[:, :n])))
        fig.add_trace(go.Scatter(x=ns, y=ess_vals, mode="lines+markers",
                                 name=label, line=dict(color=color), showlegend=False),
                      row=i+1, col=1)
        fig.update_xaxes(title_text="Samples", row=i+1, col=1)
        fig.update_yaxes(title_text="ESS bulk", row=i+1, col=1)

    fig.update_layout(height=300*len(expanded), title_text="ESS Evolution Plots")
    return fig


def _plot_pair(m, include=None, exclude=None, filter_regex=None,
               colorscale="Viridis", max_points=1000,
               point_color='rgba(40, 150, 200, 0.4)', filtered=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    import numpy as np

    expanded = _expand(m, include, exclude, filter_regex, filtered)
    labels = [e[0] for e in expanded]
    samples_dict = {name: np.asarray(s) for (name, s) in expanded}
    n_vars = len(labels)
    df = pd.DataFrame(samples_dict)
    plot_df = df.sample(n=max_points, random_state=42) if len(df) > max_points else df

    fig = make_subplots(rows=n_vars, cols=n_vars, horizontal_spacing=0.03, vertical_spacing=0.03)
    for i in range(n_vars):
        for j in range(n_vars):
            v1, v2 = labels[i], labels[j]
            if i == j:
                fig.add_trace(go.Histogram(x=df[v1], marker_color='#440154'), row=i+1, col=j+1)
            elif i > j:
                fig.add_trace(go.Histogram2dContour(
                    x=df[v2], y=df[v1], colorscale=colorscale, showscale=False,
                    contours=dict(coloring='lines'), line=dict(width=1)), row=i+1, col=j+1)
                fig.add_trace(go.Scatter(x=plot_df[v2], y=plot_df[v1], mode='markers',
                                         marker=dict(size=3, color=point_color)), row=i+1, col=j+1)
                fig.add_trace(go.Scatter(x=[df[v2].median()], y=[df[v1].median()], mode='markers',
                                         marker=dict(symbol='square', color='black', size=8)),
                              row=i+1, col=j+1)
    fig.update_layout(title_text="Pair Plot: Histograms, Density, and Samples",
                      height=250*n_vars, width=250*n_vars, showlegend=False, plot_bgcolor='white')
    for i in range(n_vars):
        fig.update_yaxes(title_text=labels[i], row=i+1, col=1,
                         showline=True, linewidth=1, linecolor='black', mirror=True)
    for j in range(n_vars):
        fig.update_xaxes(title_text=labels[j], row=n_vars, col=j+1,
                         showline=True, linewidth=1, linecolor='black', mirror=True)
    return fig


# =============================================================================
# patch_diag_class  — patches a tool class (methods take self + m)
# =============================================================================

def patch_diag_class(cls):
    """Patch cls so all diagnostic/plot methods take (self, m, include=, exclude=, ...)."""

    def para_names(self, m):
        return list(_get_posteriors(m).keys())

    def summary_jax(self, m, round_to=2, hdi_prob=0.89,
                    include=None, exclude=None, filter_regex=None,
                    var_names=None, exclude_vars=None):
        has_chains = _has_chains(m)
        self.tab_summary = jd.summary(
            _get_posteriors(m, by_chain=has_chains),
            round_to=round_to, hdi_prob=hdi_prob,
            include=include or var_names, exclude=exclude or exclude_vars,
            filter_regex=filter_regex, group_by_chain=has_chains)
        return self.tab_summary

    def rhat_jax(self, m, include=None, exclude=None,
                 var_names=None, exclude_vars=None):
        return jd.rhat(_get_posteriors(m, by_chain=True),
                       include=include or var_names, exclude=exclude or exclude_vars)

    def ess_jax(self, m, include=None, exclude=None, kind="bulk",
                var_names=None, exclude_vars=None):
        has_chains = _has_chains(m)
        return jd.ess(_get_posteriors(m, by_chain=has_chains),
                      include=include or var_names, exclude=exclude or exclude_vars, kind=kind)

    def mcse_jax(self, m, include=None, exclude=None,
                 var_names=None, exclude_vars=None):
        has_chains = _has_chains(m)
        return jd.mcse(_get_posteriors(m, by_chain=has_chains),
                       include=include or var_names, exclude=exclude or exclude_vars)

    def plot_trace_jax(self, m, include=None, exclude=None, filter_regex=None):
        return _plot_trace(m, include, exclude, filter_regex)

    def posterior_jax(self, m, include=None, exclude=None, filter_regex=None,
                      figsize=(800, 400), hdi_prob=0.94):
        return _plot_posterior(m, include, exclude, filter_regex, figsize, hdi_prob)

    def autocor_jax(self, m, include=None, exclude=None, filter_regex=None, max_lag=40):
        return _plot_autocor(m, include, exclude, filter_regex, max_lag)

    def forest_jax(self, m, include=None, exclude=None, filter_regex=None, hdi_prob=0.95):
        return _plot_forest(m, include, exclude, filter_regex, hdi_prob)

    def density_jax(self, m, include=None, exclude=None, filter_regex=None, shade=0.4):
        return _plot_density(m, include, exclude, filter_regex, shade)

    def pair_jax(self, m, include=None, exclude=None, filter_regex=None,
                 colorscale="Viridis", max_points=1000,
                 point_color='rgba(40, 150, 200, 0.4)'):
        return _plot_pair(m, include, exclude, filter_regex, colorscale, max_points, point_color)

    def model_checks_jax(self, m, include=None, exclude=None, filter_regex=None):
        kw = dict(include=include, exclude=exclude, filter_regex=filter_regex)
        print("Posterior Plots:");      self.posterior(m, **kw).show()
        print("\nAutocorrelation:");    self.autocor(m, **kw).show()
        print("\nTrace Plots:");        self.plot_trace(m, **kw).show()
        print("\nForest Plot:");        self.forest(m, **kw).show()
        print("\nPair Plot:");          self.pair(m, **kw).show()

    def compute_ll(self, m, model, *args, obs_name=None, **kwargs):
        return jd.compute_log_likelihood(model, _get_posteriors(m), *args,
                                          obs_name=obs_name, **kwargs)

    def loo_jax(self, m=None, model=None, *args, log_likelihood=None,
                obs_name=None, pointwise=False, scale="log", reff=None, **kwargs):
        if log_likelihood is None:
            if m is None or model is None:
                raise ValueError("Pass m+model+data args or log_likelihood=.")
            log_likelihood = jd.compute_log_likelihood(
                model, _get_posteriors(m), *args, obs_name=obs_name, **kwargs)
        if reff is None and m is not None and _has_chains(m):
            reff = jd.relative_eff(_get_posteriors(m, by_chain=True))
        return jd.loo(log_likelihood, pointwise=pointwise, scale=scale, reff=reff)

    def waic_jax(self, m=None, model=None, *args, log_likelihood=None,
                 obs_name=None, pointwise=False, scale="log", **kwargs):
        if log_likelihood is None:
            if m is None or model is None:
                raise ValueError("Pass m+model+data args or log_likelihood=.")
            log_likelihood = jd.compute_log_likelihood(
                model, _get_posteriors(m), *args, obs_name=obs_name, **kwargs)
        return jd.waic(log_likelihood, pointwise=pointwise, scale=scale)

    cls.para_names           = para_names
    cls.summary              = summary_jax
    cls.rhat                 = rhat_jax
    cls.ess                  = ess_jax
    cls.mcse                 = mcse_jax
    cls.plot_trace           = plot_trace_jax
    cls.posterior            = posterior_jax
    cls.autocor              = autocor_jax
    cls.forest               = forest_jax
    cls.density              = density_jax
    cls.pair                 = pair_jax
    cls.model_checks         = model_checks_jax
    cls.loo                  = loo_jax
    cls.WAIC                 = waic_jax
    cls.compute_log_likelihood = compute_ll
    # NOTE: named compare_ll, not compare. jd.compare takes
    # {name: log_likelihood_array} whereas Diag/Diag2.compare take
    # {name: InferenceData}; overwriting `compare` silently changed the
    # argument type of a public method that kept its name.
    cls.compare_ll           = staticmethod(jd.compare)
    return cls


# =============================================================================
# bind_diag_to_model — patches a diag *instance* to use m.posteriors directly
# =============================================================================

ARVIZ = "arviz"
JAX = "jax"
_VALID_BACKENDS = (ARVIZ, JAX)

_JAX_EXPERIMENTAL_MSG = (
    "[WARNING] backend='jax' selects the experimental JAX/NumPy diagnostics "
    "(BayesForge.Diagnostic.jax_diagnostics) instead of ArviZ. They are "
    "verified equal to ArviZ to 1e-6 on R-hat, ESS (bulk/tail/mean/sd), "
    "MCSE (mean/sd), HDI, WAIC and LOO, but ArviZ remains the reference. "
    "Use it with caution. [WARNING]"
)


def _warn_jax_experimental():
    import warnings
    warnings.warn(_JAX_EXPERIMENTAL_MSG, UserWarning, stacklevel=3)


def _resolve_backend(backend):
    if backend is None:
        return ARVIZ
    backend = str(backend).lower()
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"backend must be one of {_VALID_BACKENDS}; got {backend!r}")
    return backend


def _az_diag_for(m):
    """The existing ArviZ-backed diag class, bound to m's sampler.

    Reuses BayesForge.Diagnostic.Diag.diag rather than reimplementing the
    ArviZ calls: that class already wraps az.summary / az.rhat / az.ess /
    az.loo and handles the 0.x-vs-1.x kwarg differences.
    """
    from BayesForge.Diagnostic.Diag import diag as _ArvizDiag
    cached = getattr(m, "_az_diag", None)
    if cached is None or cached.sampler is not m.sampler:
        cached = _ArvizDiag(sampler=m.sampler)
        try:
            cached.to_az(backend=getattr(m, "backend", "numpyro"))
        except Exception:
            pass                      # _ensure_trace() will retry on first use
        m._az_diag = cached
    return cached


def _as_var_names(include):
    if include is None:
        return None
    return [include] if isinstance(include, str) else list(include)


class _SiteFigures(dict):
    """dict mapping observed site -> figure; Jupyter shows the figures, not the repr."""

    def _ipython_display_(self):
        from IPython.display import display
        for fig in self.values():
            display(fig)


def _resolve_obs_y(m, site=None):
    """Observed outcome array for the PPC plots.

    ``m.obs_args`` is None for some models, and the old
    ``if y is None and m.obs_args:`` guard then left y as None -- which
    produced a silently EMPTY figure (np.asarray(None) -> nan -> every KDE
    skipped) rather than an error. Fall back to the lone non-parameter entry
    in data_on_model, and raise a clear message when it is genuinely ambiguous.
    """
    data = getattr(m, "data_on_model", None) or {}

    if site is not None:
        if hasattr(m, "_observed_sites"):
            sites = m._observed_sites(data)
            if site in sites and sites[site] is not None and sites[site] in data:
                return data[sites[site]]
        if site in data:
            return data[site]
        raise ValueError(
            f"Observed site {site!r} not found in model data. Available data keys: {sorted(data)}"
        )

    # 1. Declared explicitly.
    obs_args = getattr(m, "obs_args", None) or []
    if len(obs_args) == 1 and obs_args[0] in data:
        return data[obs_args[0]]

    # 2. Ask the MODEL. Observed sample sites carry their observed value, so
    #    this works even when the site was never given a name= and obs_args is
    #    empty -- which is the common case for `m.dist.normal(..., obs=y)`.
    model = getattr(m, "model", None)
    if model is not None and data:
        try:
            from numpyro import handlers
            with handlers.seed(rng_seed=0):
                trace = handlers.trace(model).get_trace(**data)
            observed = [s for s in trace.values()
                        if s.get("type") == "sample"
                        and s.get("is_observed", False)
                        and s.get("value") is not None]
            if len(observed) == 1:
                return observed[0]["value"]
            if len(observed) > 1:
                names = [s['name'] for s in observed]
                raise ValueError(
                    f"This model has several observed sites ({names}); specify "
                    f"which one to plot via site='{names[0]}' or pass y= directly."
                )
        except ValueError:
            raise
        except Exception:
            pass                      # tracing failed; fall through to (3)

    # 3. A single non-parameter entry in data_on_model.
    params = set((getattr(m, "posteriors", None) or {}).keys())
    candidates = [k for k in data if k not in params]
    if len(candidates) == 1:
        return data[candidates[0]]

    raise ValueError(
        "Cannot identify the observed outcome for this PPC: m.obs_args="
        f"{obs_args!r} and data_on_model has {sorted(data)}. "
        "Pass y= explicitly."
    )


_PPC_NOTE = ("``y`` and ``yrep`` default to the model's observed outcome and its "
             "posterior predictive draws (``m.diag.get_yrep()``); ``name`` "
             "defaults to the observed site name. For ``ppc_loo_pit``, "
             "``log_likelihood`` is computed from the model when ``yrep`` is "
             "not given.")
_M_NOTE = "The fitted model ``m`` is supplied automatically."


def _documented(method, target, note, defaults_none=(), drop=()):
    """Give a bound diag method the docstring and signature of ``target``.

    The signature shown to users drops arguments the binding supplies
    (``drop``) and marks arguments it fills in by default as optional
    (``defaults_none``).
    """
    sig = inspect.signature(target)
    params, seen_default = [], False
    for p in sig.parameters.values():
        if p.name in drop:
            continue
        if p.default is p.empty and (p.name in defaults_none or seen_default):
            # a required arg after one the binding fills in (e.g. status_y
            # after y/yrep) must still show a default to keep the signature
            # valid; the binding raises the function's own error if missing
            p = p.replace(default=None)
        seen_default = seen_default or p.default is not p.empty
        params.append(p)
    method.__signature__ = sig.replace(parameters=params)
    method.__name__ = target.__name__
    doc = inspect.getdoc(target) or ""
    summary, _, rest = doc.partition("\n")
    method.__doc__ = f"{summary}\n\n{note}\n{rest}".rstrip() + "\n"
    return method


def _bind_plot(m, plot_fn, summary):
    """m.diag version of a posterior plot taking (m, include, exclude, filtered)."""
    def method(include=None, exclude=None, filtered=True, x_label=None,
               y_label=None, **kw):
        fig = plot_fn(m, include, exclude, filtered=filtered, **kw)
        return _ppc._axis_labels(fig, x_label, y_label)
    method.__name__ = plot_fn.__name__.lstrip("_")
    method.__doc__ = f"""{summary}

    Args:
        include (str or list of str, optional): Parameters to plot (default: all).
        exclude (str or list of str, optional): Parameters to leave out.
        filtered (bool): Use ``m.posteriors`` (True, default) or every sampled
            site, ``m.posteriors_full`` (False).
        x_label (str or list of str, optional): X-axis title; a list gives one
            per subplot, in order. Defaults to the plot's own labels.
        y_label (str or list of str, optional): Y-axis title, as x_label.
        **kw: Passed to the plotting function (e.g. ``figsize``).

    Returns:
        plotly.graph_objects.Figure
    """
    return method


def bind_diag_to_model(diag_obj, m, backend=ARVIZ):
    """Bind plotting and diagnostic methods onto a diag instance.

    Args:
        diag_obj: the diag instance to patch (methods are set on the instance).
        m: the fitted BF model the methods close over.
        backend: which engine backs the *convergence metrics*
            (summary / rhat / ess / mcse / loo / WAIC):

            "arviz" (default) — delegate to BayesForge.Diagnostic.Diag.diag,
                i.e. az.summary / az.rhat / az.ess / az.mcse / az.loo.
            "jax"             — use BayesForge.Diagnostic.jax_diagnostics.
                Emits an experimental-feature warning.

    Plotting (trace, posterior, forest, pair, rank, density, autocorrelation),
    PPC, sensitivity and regression overlays are backend-independent and are
    bound identically either way.

    All methods accept:
      filtered=True  (default) — use m.posteriors (respects active filter)
      filtered=False           — use m.posteriors_full (all parameters)
    Per-call include/exclude further restrict on top of whichever source is chosen.
    """
    backend = _resolve_backend(backend)
    if backend == JAX:
        _warn_jax_experimental()
    diag_obj.backend = backend

    # ---- convergence metrics: ArviZ (default) or JAX --------------------
    def _summary(include=None, exclude=None, round_to=2, hdi_prob=0.89,
                 filter_regex=None, filtered=True, kind="all"):
        """Posterior summary table (mean, sd, credible interval, ESS, R-hat, MCSE).

        Args:
            include (str or list of str, optional): Parameters to summarise (default: all).
            exclude (str or list of str, optional): Parameters to leave out.
            round_to (int): Number of decimals. Defaults to 2.
            hdi_prob (float): Probability mass of the credible interval. Defaults to 0.89.
            filter_regex (str, optional): Regular expression on parameter names (JAX backend).
            filtered (bool): Use ``m.posteriors`` (True) or every sampled site,
                ``m.posteriors_full`` (False). Defaults to True.
            kind (str): ``"all"``, ``"stats"`` or ``"diagnostics"``. Defaults to ``"all"``.

        Returns:
            pandas.DataFrame: One row per parameter element.
        """
        if backend == ARVIZ:
            var_names = _as_var_names(include)
            kw = {}
            if var_names is not None:
                kw["var_names"] = var_names
            if _as_var_names(exclude) is not None:
                kw["filter_vars"] = None
            return _az_diag_for(m).summary(
                round_to=round_to, kind=kind, hdi_prob=hdi_prob, **kw)
        src = _source(m, filtered=filtered, by_chain=True)
        return jd.summary(src, include=include, exclude=exclude,
                          filter_regex=filter_regex, round_to=round_to,
                          hdi_prob=hdi_prob, group_by_chain=True)

    def _rhat(include=None, exclude=None, filtered=True, method="rank"):
        """Potential scale reduction factor (R-hat) of each parameter.

        Values close to 1 (below 1.01) indicate that the chains have mixed.
        Requires more than one chain.

        Args:
            include (str or list of str, optional): Parameters to check (default: all).
            exclude (str or list of str, optional): Parameters to leave out.
            filtered (bool): Use ``m.posteriors`` (True) or ``m.posteriors_full`` (False).
            method (str): R-hat variant, ``"rank"`` (rank-normalised split R-hat,
                default) or ``"split"``.

        Returns:
            R-hat per parameter.
        """
        if backend == ARVIZ:
            var_names = _as_var_names(include)
            kw = {"method": method}
            if var_names is not None:
                kw["var_names"] = var_names
            return _az_diag_for(m).rhat(**kw)
        return jd.rhat(_source(m, filtered=filtered, by_chain=True),
                       include=include, exclude=exclude)

    def _ess(include=None, exclude=None, kind="bulk", filtered=True):
        """Effective sample size of each parameter.

        Args:
            include (str or list of str, optional): Parameters to check (default: all).
            exclude (str or list of str, optional): Parameters to leave out.
            kind (str): ``"bulk"`` (default) or ``"tail"`` ESS.
            filtered (bool): Use ``m.posteriors`` (True) or ``m.posteriors_full`` (False).

        Returns:
            ESS per parameter.
        """
        if backend == ARVIZ:
            var_names = _as_var_names(include)
            kw = {"method": kind}
            if var_names is not None:
                kw["var_names"] = var_names
            return _az_diag_for(m).ess(**kw)
        return jd.ess(_source(m, filtered=filtered, by_chain=True),
                      include=include, exclude=exclude, kind=kind)

    def _mcse(include=None, exclude=None, kind="mean", filtered=True):
        """Monte Carlo standard error of each parameter's posterior summary.

        Args:
            include (str or list of str, optional): Parameters to check (default: all).
            exclude (str or list of str, optional): Parameters to leave out.
            kind (str): Summary whose error is estimated: ``"mean"`` (default),
                ``"sd"`` or a quantile.
            filtered (bool): Use ``m.posteriors`` (True) or ``m.posteriors_full`` (False).

        Returns:
            MCSE per parameter.
        """
        if backend == ARVIZ:
            import arviz as az
            var_names = _as_var_names(include)
            kw = {"method": kind}
            if var_names is not None:
                kw["var_names"] = var_names
            return az.mcse(_az_diag_for(m)._ensure_trace(), **kw)
        return jd.mcse(_source(m, filtered=filtered, by_chain=True),
                       include=include, exclude=exclude, kind=kind)

    def _model_checks(include=None, exclude=None, filter_regex=None, filtered=True):
        """Show the standard convergence plots in one call.

        Displays posterior densities, autocorrelation, trace, forest and pair plots.

        Args:
            include (str or list of str, optional): Parameters to plot (default: all).
            exclude (str or list of str, optional): Parameters to leave out.
            filter_regex (str, optional): Regular expression on parameter names.
            filtered (bool): Use ``m.posteriors`` (True) or ``m.posteriors_full`` (False).
        """
        kw = dict(include=include, exclude=exclude, filter_regex=filter_regex, filtered=filtered)
        print("Posterior Plots:");   diag_obj.posterior(**kw).show()
        print("\nAutocorrelation:"); diag_obj.autocor(**kw).show()
        print("\nTrace Plots:");     diag_obj.plot_trace(**kw).show()
        print("\nForest Plot:");     diag_obj.forest(**kw).show()
        print("\nPair Plot:");       diag_obj.pair(**kw).show()

    def _loo(log_likelihood=None, pointwise=False, scale="log", reff=None,
             var_name=None):
        """Pareto-smoothed importance-sampling leave-one-out cross-validation (PSIS-LOO).

        Estimates the expected log pointwise predictive density (ELPD) of the
        fitted model. Use ``m.diag.compare({"m1": m1, "m2": m2})`` to rank models.

        Args:
            log_likelihood (array, optional): Pointwise log-likelihood
                (chains, draws, observations). Computed from the model by default.
            pointwise (bool): Also return per-observation values and Pareto k. Defaults to False.
            scale (str): ``"log"`` (default), ``"negative_log"`` or ``"deviance"``.
            reff (float, optional): Relative MCMC efficiency; estimated from the chains by default.
            var_name (str, optional): Observed variable to use when the model has several.

        Returns:
            ELPD estimate, its standard error and the effective number of parameters (p_loo).
        """
        if backend == ARVIZ and log_likelihood is None:
            return _az_diag_for(m).loo(pointwise=pointwise, var_name=var_name,
                                       reff=reff, scale=scale)
        if log_likelihood is None:
            if not hasattr(m, 'model') or m.model is None:
                raise ValueError("No model stored on m. Pass log_likelihood= directly.")
            log_likelihood = jd.compute_log_likelihood(
                m.model, _source(m, filtered=False, by_chain=True), **m.data_on_model)
        if reff is None and _has_chains(m):
            # az.loo derives reff from the POSTERIOR; without it the PSIS tail
            # is sized as if the draws were independent, biasing pareto_k.
            reff = jd.relative_eff(_source(m, filtered=False, by_chain=True))
        return jd.loo(log_likelihood, pointwise=pointwise, scale=scale, reff=reff)

    def _waic(log_likelihood=None, pointwise=False, scale="log", var_name=None):
        """Widely applicable information criterion (WAIC).

        Args:
            log_likelihood (array, optional): Pointwise log-likelihood
                (chains, draws, observations). Computed from the model by default.
            pointwise (bool): Also return per-observation values. Defaults to False.
            scale (str): ``"log"`` (default), ``"negative_log"`` or ``"deviance"``.
            var_name (str, optional): Observed variable to use when the model has several.

        Returns:
            ELPD estimate (``elpd_waic``), its standard error and ``p_waic``.
        """
        if backend == ARVIZ and log_likelihood is None:
            return _az_diag_for(m).WAIC(pointwise=pointwise, var_name=var_name,
                                        scale=scale)
        if log_likelihood is None:
            if not hasattr(m, 'model') or m.model is None:
                raise ValueError("No model stored on m. Pass log_likelihood= directly.")
            log_likelihood = jd.compute_log_likelihood(
                m.model, _source(m, filtered=False, by_chain=True), **m.data_on_model)
        return jd.waic(log_likelihood, pointwise=pointwise, scale=scale)

    # ---- PPC helpers that auto-generate yrep from m ---------------------
    def _get_yrep(site=None, seed=None):
        """Posterior predictive draws of the observed outcome(s).

        Args:
            site (str, optional): Name of the observed site when multiple
                observed sites exist. If None and there are multiple sites,
                a dict mapping site names to their draws is returned.
            seed (int, optional): Seed of the predictive simulation. Defaults
                to None (a fresh draw from the model's random stream).

        Returns:
            Array of shape (draws, observations) or dict[str, Array].
        """
        return _ppc.get_yrep(m, site=site, seed=seed)

    # ---- regression overlay ---------------------------------------------
    def _plot_regression_bound(x_var, y_obs=None, n=20, link_inv=None,
                               x_range=None, n_points=200, seed=42, **kw):
        return _plot_regression(m, x_var, y_obs=y_obs, n=n, link_inv=link_inv,
                                x_range=x_range, n_points=n_points, seed=seed, **kw)
    _documented(_plot_regression_bound, _plot_regression, _M_NOTE, drop=("m",))

    # ---- PPC bindings (y + yrep) ----------------------------------------
    def _single_site():
        """Name of the observed site when the model has exactly one."""
        try:
            sites = m._observed_sites(m.data_on_model)
        except Exception:
            return None
        return next(iter(sites)) if len(sites) == 1 else None

    def _site_log_likelihood(site, shape):
        """Pointwise log-likelihood of the observed site, (draws, n_obs) in
        the draw order of get_yrep, and the relative efficiency for PSIS."""
        import numpy as np
        ll = jd.compute_log_likelihood(m.model, m.posteriors_by_chain,
                                       obs_name=site, **m.data_on_model)
        if isinstance(ll, dict):
            ll = ll[site]
        ll = np.asarray(ll, dtype=np.float64)
        ll = ll.reshape(ll.shape[0] * ll.shape[1], -1) if ll.ndim >= 3 else ll
        if ll.shape != tuple(shape):
            raise ValueError(
                f"log-likelihood of {site!r} has shape {ll.shape} but its "
                f"predictions have {tuple(shape)}; pass log_likelihood= explicitly.")
        return ll, float(np.asarray(jd.relative_eff(m.posteriors_by_chain)))

    def _bind_ppc(plot_fn):
        """m.diag version of a ppc_* plot: y and yrep default to the model's
        observed outcome and its posterior predictive draws. With several
        observed sites and no site=, returns a dict mapping each site to its
        figure."""
        params = inspect.signature(plot_fn).parameters
        takes_name = "name" in params
        # LOO-PIT needs the pointwise log-likelihood of the SAME draws as yrep;
        # it is only computed here when yrep is the model's own draws.
        wants_ll = "log_likelihood" in params

        def call(y, yrep, s, own_yrep, kw):
            kw = dict(kw)
            if takes_name and s is not None:
                kw.setdefault("name", s)
            if wants_ll and own_yrep and kw.get("log_likelihood") is None:
                kw["log_likelihood"], reff = _site_log_likelihood(s, yrep.shape)
                kw.setdefault("reff", reff)
            # x=/group= may be given as the name of a model data entry or a
            # column of m.df; resolve it to the array and use the name as the
            # axis/group label.
            import numpy as np
            for key, label_key in (("x", "x_label"), ("group", "group_name"),
                                   ("status_y", None)):
                value = kw.get(key)
                if isinstance(value, str) and key in params:
                    data = getattr(m, "data_on_model", None) or {}
                    if value in data:
                        arr = np.asarray(data[value])
                    else:
                        df = getattr(m, "df", None)
                        if df is not None and value in getattr(df, "columns", ()):
                            arr = np.asarray(df[value])
                        else:
                            raise ValueError(
                                f"{key}= '{value}': no such entry in "
                                "m.data_on_model or column in m.df")
                    kw[key] = arr
                    if label_key and label_key in params:
                        kw.setdefault(label_key, value)
            return plot_fn(y, yrep, **kw)

        def method(y=None, yrep=None, site=None, **kw):
            own_yrep = yrep is None
            if yrep is None:
                res = _get_yrep(site=site)
                if isinstance(res, dict):
                    if y is not None:
                        raise ValueError(
                            "This model has several observed sites and y= was given, "
                            "so the site is ambiguous; pass site= or yrep= explicitly."
                        )
                    figs = _SiteFigures()
                    for s in sorted(res):
                        fig = call(_resolve_obs_y(m, site=s), res[s], s, True, kw)
                        if not takes_name:
                            title = fig.layout.title.text or ""
                            fig.update_layout(title_text=f"{title} ({s})" if title else s)
                        figs[s] = fig
                    return figs
                yrep = res
            if y is None:
                y = _resolve_obs_y(m, site=site)
            return call(y, yrep, site or _single_site(), own_yrep, kw)
        return _documented(method, plot_fn, _PPC_NOTE, defaults_none=("y", "yrep"))

    # ---- sensitivity bindings -------------------------------------------
    def _influence(log_likelihood=None, x=None, y=None, **kw):
        return _sens.influence_plot(m=m, log_likelihood=log_likelihood,
                                    x=x, y=y, **kw)
    _documented(_influence, _sens.influence_plot, _M_NOTE, drop=("m",))

    def _divergence_energy(**kw):
        return _sens.divergence_energy_plot(m, **kw)
    _documented(_divergence_energy, _sens.divergence_energy_plot, _M_NOTE, drop=("m",))

    def _multimodality(param_names=None, **kw):
        return _sens.multimodality_check(m=m, param_names=param_names, **kw)
    _documented(_multimodality, _sens.multimodality_check, _M_NOTE, drop=("m",))

    # ---- wire everything up --------------------------------------------
    diag_obj.summary        = _summary
    diag_obj.rhat           = _rhat
    diag_obj.ess            = _ess
    diag_obj.mcse           = _mcse
    diag_obj.plot_trace     = _bind_plot(m, _plot_trace, "Trace plot of each chain (mixing and stationarity).")
    diag_obj.posterior      = _bind_plot(m, _plot_posterior, "Posterior density of each parameter with its credible interval.")
    diag_obj.autocor        = _bind_plot(m, _plot_autocor, "Autocorrelation of the draws of each parameter, per chain.")
    diag_obj.forest         = _bind_plot(m, _plot_forest, "Forest plot: posterior mean and credible interval of each parameter.")
    diag_obj.density        = _bind_plot(m, _plot_density, "Posterior density of each parameter, one curve per chain.")
    diag_obj.pair           = _bind_plot(m, _plot_pair, "Pairwise scatter plots of the draws (correlations, funnels, multimodality).")
    diag_obj.rank           = _bind_plot(m, _plot_rank, "Rank plots: histograms of draw ranks per chain (uniform when chains agree).")
    diag_obj.plot_ess       = _bind_plot(m, _plot_ess_evolution, "Evolution of the effective sample size as draws accumulate.")
    diag_obj.loo            = _loo
    diag_obj.waic           = _waic
    diag_obj.WAIC           = _waic  # backward-compatible alias
    diag_obj.model_checks   = _model_checks
    # regression
    diag_obj.plot_regression = _plot_regression_bound
    diag_obj.get_yrep        = _get_yrep
    # PPC — distributions
    diag_obj.ppc_density     = _bind_ppc(_ppc.ppc_density)
    diag_obj.ppc_hist        = _bind_ppc(_ppc.ppc_hist)
    diag_obj.ppc_boxplot     = _bind_ppc(_ppc.ppc_boxplot)
    # PPC — statistics
    diag_obj.ppc_stat        = _bind_ppc(_ppc.ppc_stat)
    diag_obj.ppc_stat_2d     = _bind_ppc(_ppc.ppc_stat_2d)
    # PPC — intervals
    diag_obj.ppc_intervals   = _bind_ppc(_ppc.ppc_intervals)
    diag_obj.ppc_ribbon      = _bind_ppc(_ppc.ppc_ribbon)
    # PPC — errors
    diag_obj.ppc_error_scatter = _bind_ppc(_ppc.ppc_error_scatter)
    diag_obj.ppc_error_hist  = _bind_ppc(_ppc.ppc_error_hist)
    diag_obj.ppc_error_binned = _bind_ppc(_ppc.ppc_error_binned)
    # PPC — survival
    diag_obj.ppc_km_overlay  = _bind_ppc(_ppc.ppc_km_overlay)
    # PPC — scatter
    diag_obj.ppc_scatter     = _bind_ppc(_ppc.ppc_scatter)
    # PPC — discrete
    diag_obj.ppc_rootogram   = _bind_ppc(_ppc.ppc_rootogram)
    diag_obj.ppc_bars        = _bind_ppc(_ppc.ppc_bars)
    # PPC — LOO
    diag_obj.ppc_loo_pit     = _bind_ppc(_ppc.ppc_loo_pit)
    diag_obj.ppc_loo_intervals = _bind_ppc(_ppc.ppc_loo_intervals)
    diag_obj.ppc_intervals_sorted = _bind_ppc(_ppc.ppc_intervals_sorted)
    # sensitivity
    diag_obj.influence       = _influence
    diag_obj.calibration     = _bind_ppc(_sens.calibration_plot)
    diag_obj.divergence_energy = _divergence_energy
    diag_obj.multimodality   = _multimodality
    # A staticmethod object stored on an INSTANCE is not unwrapped by the
    # descriptor protocol; it is only callable at all on Python >= 3.10.
    diag_obj.prior_sensitivity = _sens.prior_sensitivity_plot
