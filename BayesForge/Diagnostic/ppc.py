"""Posterior Predictive Check (PPC) plots for BF models.

All PPC functions take:
    y    : 1D array of observed values, shape (n_obs,)
    yrep : 2D matrix of replicated datasets, shape (n_rep, n_obs)

Generate yrep with:
    pred = m.sample()
    yrep = np.asarray(pred[m.obs_args[0]])   # (S, n_obs)

or use the helper:
    yrep = get_yrep(m)

Categories implemented
-----------------------
Distributions  : ppc_density, ppc_hist, ppc_boxplot
Statistics     : ppc_stat, ppc_stat_2d
Intervals      : ppc_intervals, ppc_ribbon, ppc_intervals_sorted
Errors         : ppc_error_scatter, ppc_error_hist, ppc_error_binned
Survival       : ppc_km_overlay
Scatterplots   : ppc_scatter
Discrete       : ppc_rootogram, ppc_bars
LOO            : ppc_loo_pit (ppc_loo_intervals: deprecated alias of ppc_intervals_sorted)
"""
import re

import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots

_COLORS = pcolors.qualitative.Plotly
_REP_COLOR = "rgba(100,149,237,0.25)"
_REP_COLOR_SOLID = "rgba(100,149,237,0.6)"
_OBS_COLOR = "black"


def _color_to_rgba(color, alpha=0.3):
    """Convert hex `#rrggbb` or `rgb(...)` color string to `rgba(...)` with given alpha."""
    if color.startswith("#"):
        c = color.lstrip("#")
        if len(c) == 3:
            c = "".join([x * 2 for x in c])
        r, g, b = tuple(int(c[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("rgb("):
        return color.replace("rgb(", "rgba(").replace(")", f",{alpha})")
    if color.startswith("rgba("):
        parts = [p.strip() for p in color.strip("rgba()").split(",")]
        return f"rgba({parts[0]},{parts[1]},{parts[2]},{alpha})"
    return color


# =============================================================================
# Utility
# =============================================================================

def get_yrep(m, site=None, seed=None):
    """Return posterior predictive matrix yrep of shape (S, n_obs), or a dict
    of such matrices if the model has multiple observed sites and ``site=None``.

    The observed sites are simulated from every posterior draw (their data
    entries are set to None), so yrep holds genuine replicated data -- never the
    observed values themselves.

    Args:
        m (bf): Fitted BayesForge model.
        site (str, optional): Name of the observed site. If None and the model
            has a single observed site, that site is used. If None and the model
            has multiple observed sites, a dict ``{site_name: yrep}`` is returned.
        seed (int, optional): Seed of the predictive simulation. Defaults to None.

    Returns:
        np.ndarray or dict[str, np.ndarray]: Matrix of shape (S, n_obs) or
        dict mapping each observed site name to its replicated data matrix.
    """
    sites = m._observed_sites(m.data_on_model)
    if not sites:
        raise ValueError("No observed sites found in the model.")

    if site is not None:
        if site not in sites:
            raise ValueError(
                f"Unknown observed site {site!r}. Available sites: {sorted(sites)}"
            )
        target_sites = {site: sites[site]}
    else:
        target_sites = sites

    for s_name, key in target_sites.items():
        if key is None:
            raise ValueError(
                f"The observed site {s_name!r} is not one of the model's data entries "
                "(e.g. obs=y - 1), so it cannot be simulated. Pass yrep= explicitly."
            )

    pred = m.sample(remove_obs=True, posterior=True, seed=seed)

    yreps = {}
    for s_name in target_sites:
        arr = np.asarray(pred[s_name])
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        yreps[s_name] = arr

    if site is not None or len(sites) == 1:
        return next(iter(yreps.values())) if site is None else yreps[site]
    return yreps


def _as_np(*arrays):
    return [np.asarray(a, dtype=float).flatten() for a in arrays]


def _kde(stats, data, bw_adjust=1.0):
    """gaussian_kde with bw_adjust as a MULTIPLIER on Scott's rule.

    scipy's ``bw_method=`` is the absolute bandwidth factor, not an adjustment,
    so passing bw_adjust straight through made the default (1.0) about 3.5x
    wider than Scott at n=500 -- every density overlay came out oversmoothed.
    """
    kde = stats.gaussian_kde(data)
    if bw_adjust is not None and bw_adjust != 1.0:
        kde.set_bandwidth(kde.factor * float(bw_adjust))
    return kde


def _stat_fn(name):
    fns = {
        "mean":   np.mean,
        "median": np.median,
        "sd":     lambda x: float(np.std(x, ddof=1)),
        "var":    lambda x: float(np.var(x, ddof=1)),
        "min":    np.min,
        "max":    np.max,
        "q25":    lambda x: np.percentile(x, 25),
        "q75":    lambda x: np.percentile(x, 75),
        "skew":   lambda x: float(np.mean(((x - np.mean(x)) / (np.std(x) + 1e-30)) ** 3)),
        "n_zero": lambda x: int(np.sum(x == 0)),
        "n_pos":  lambda x: int(np.sum(x > 0)),
    }
    if callable(name):
        return name
    if name not in fns:
        raise ValueError(f"Unknown stat '{name}'. Choose from {list(fns)}")
    return fns[name]


# Every plot labels the outcome by ``name`` -- m.diag passes the observed site
# name (e.g. "height") -- so axes and legends read "Predictions of height"
# rather than the bayesplot jargon "yrep".
def _obs(name):
    return f"{name} (observed)"


def _preds(name):
    return f"Predictions of {name}"


def _axis_labels(fig, x_label=None, y_label=None):
    """Override the axis titles of every plot: a string sets all the x (y)
    axes, a list sets them one per subplot in layout order (None keeps one)."""
    for label, axes in ((x_label, "xaxis"), (y_label, "yaxis")):
        if label is None:
            continue
        names = sorted((k for k in fig.layout.to_plotly_json() if re.fullmatch(axes + r"\d*", k)),
                       key=lambda k: int(k[len(axes):] or 1)) or [axes]
        labels = [label] * len(names) if isinstance(label, str) else list(label)
        for k, text in zip(names, labels):
            if text is not None:
                fig.layout[k].title.text = text
    return fig


def _title(title, kind, name):
    return f"PPC: {kind} — predictions of {name}" if title is None else title


def _interval_bounds(y, yrep, x, prob, prob_outer):
    """Rows sorted by x and the type-7 quantiles bayesplot's
    ``ppc_intervals_data`` draws: (x, y, ll, l, m, h, hh)."""
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    if prob > prob_outer:          # bayesplot swaps them too
        prob, prob_outer = prob_outer, prob
    x = np.arange(len(y)) if x is None else np.asarray(x).flatten()
    order = np.argsort(x, kind="stable")
    x, y, yrep = x[order], y[order], yrep[:, order]

    def q(p):
        return np.percentile(yrep, 100 * p, axis=0)
    return (x, y, q((1 - prob_outer) / 2), q((1 - prob) / 2), q(0.5),
            q((1 + prob) / 2), q((1 + prob_outer) / 2))


def _segments(x, lo, hi):
    """Vertical segments x_i: lo_i -> hi_i as one line trace (None-separated)."""
    n = len(x)
    xs = np.empty(3 * n, dtype=object)
    ys = np.empty(3 * n, dtype=object)
    xs[0::3], xs[1::3], xs[2::3] = x, x, None
    ys[0::3], ys[1::3], ys[2::3] = lo, hi, None
    return list(xs), list(ys)


def _group_levels(group, n):
    """Validate a grouping vector and return ``(levels, masks)``: the sorted
    unique levels and one boolean mask per level, or ``(None, None)`` when
    ``group is None``. Levels may be any comparable hashable values."""
    if group is None:
        return None, None
    g = np.asarray(group, dtype=object).ravel()
    if len(g) != n:
        raise ValueError(
            f"group has length {len(g)}, expected {n} "
            "(one entry per observation).")
    levels = sorted(set(g.tolist()))
    return levels, [g == lv for lv in levels]


# Above this many distinct values, x is treated as continuous and keeps
# plotly's default linear ticks.
_MAX_DISCRETE_X = 12


def _level_labels(levels):
    """Format ``levels`` with the fewest decimals that still tells them apart.

    A scaled predictor reaches the panel titles and the x ticks as a raw
    float ("shade = 1.2018504251546631"); this shortens it to the readable
    form ("-1.2", "0", "1.2"). Non-numeric levels (strings, dates) are
    returned as ``str(level)``.
    """
    levels = list(levels)
    try:
        values = [float(lv) for lv in levels]
    except (TypeError, ValueError):
        return [str(lv) for lv in levels]
    if not all(np.isfinite(values)):
        return [str(lv) for lv in levels]
    whole = all(abs(v - round(v)) < 1e-12 for v in values)
    # Whole levels may be rounded to 0 decimals; anything else keeps at least
    # one, so "1.0" is never printed as "1" next to "1.2".
    for nd in (range(0, 7) if whole else range(1, 7)):
        rounded = [round(v, nd) for v in values]
        if len(set(rounded)) == len(rounded):
            if nd == 0:
                return [f"{r:.0f}" for r in rounded]
            return [f"{r:.{nd}f}".rstrip("0").rstrip(".") for r in rounded]
    return [f"{v:.6g}" for v in values]


def _level_label(level):
    """Single-level form of :func:`_level_labels`."""
    return _level_labels([level])[0]


def _discrete_ticks(values):
    """``(tickvals, ticktext)`` when ``values`` holds a few distinct numbers.

    A discrete predictor (a scaled 1/2/3 design) would otherwise get plotly's
    default linear ticks, which fall between the observations and make
    ``ppc_intervals`` unreadable: the bars sit at -1.2 and 1.2 while the axis
    is labelled -1.0, -0.5, 0.0, 0.5, 1.0. Continuous, non-numeric or empty x
    returns None and keeps the default ticks.
    """
    arr = np.asarray(values).ravel()
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        return None
    v = arr.astype(float)
    if not np.all(np.isfinite(v)):
        return None
    uniq = np.unique(v)
    if not 2 <= len(uniq) <= _MAX_DISCRETE_X:
        return None
    return uniq.tolist(), _level_labels(uniq)


def _apply_discrete_ticks(fig, axes, panel_x):
    """Label each of ``axes`` with one tick per distinct x value in ``panel_x``.

    Axes whose x is continuous are left untouched, so a panel-by-panel x (one
    per subplot) may mix discrete and continuous panels.
    """
    for axis, x in zip(axes, panel_x):
        ticks = _discrete_ticks(x)
        if ticks is not None:
            fig.layout[axis].update(tickmode="array", tickvals=ticks[0],
                                    ticktext=ticks[1])


def _add_trace(fig, trace, row=None, col=None):
    """``fig.add_trace`` that also works on plain (non-subplot) figures.

    ``row``/``col`` are only meaningful on figures from ``make_subplots``;
    passing real ones to a plain ``go.Figure`` raises. The panel helpers
    here are written once and receive ``row=None`` for a single-panel
    figure, ``row``/``col`` for a grid; this keeps the ``if row is None``
    branch out of every call site::

        fig = go.Figure()                            # single-panel figure
        _add_trace(fig, trace)                       # row=None -> add_trace(trace)
        # fig.add_trace(trace, row=1, col=1) would raise here

        fig = make_subplots(rows=1, cols=2)
        _add_trace(fig, trace, 1, 2)                 # -> right-hand panel
    """
    if row is None:
        fig.add_trace(trace)
    else:
        fig.add_trace(trace, row=row, col=col)


# =============================================================================
# Distributions
# =============================================================================

def _ppc_density_panel(fig, row, col, y_g, yrep_g, idx, bw_adjust, name,
                       show_legend=True):
    """Draw one ppc_density panel: the KDE of each sampled yrep row (blue)
    and of y (black), all evaluated on one shared x grid. ``row``/``col`` are
    None for a single-panel figure; only the first panel carries the legend."""
    import scipy.stats as stats
    y_g = np.asarray(y_g, dtype=np.float64).flatten()
    yrep_g = np.asarray(yrep_g, dtype=np.float64)

    p1    = float(np.percentile(y_g, 0.5))
    p99_y = float(np.percentile(y_g, 99.5))
    p1_r  = float(np.percentile(yrep_g[idx], 0.5))
    p99_r = float(np.percentile(yrep_g[idx], 99.5))
    x_min = float(min(p1, p1_r))
    x_max = float(max(p99_y, p99_r))
    if x_min >= x_max:
        x_max = x_min + 1.0
    xs = np.linspace(x_min, x_max, 300, dtype=np.float64)

    for i, s in enumerate(idx):
        row_vals = yrep_g[s]
        row_vals = row_vals[np.isfinite(row_vals)]
        if len(row_vals) < 2 or float(np.std(row_vals)) < 1e-12:
            continue
        try:
            kde = _kde(stats, row_vals, bw_adjust)
            vals = kde(xs)
            if not np.isfinite(vals).all():
                continue
        except Exception:
            continue
        _add_trace(fig, go.Scatter(
            x=xs, y=vals, mode="lines",
            line=dict(color=_REP_COLOR, width=1),
            showlegend=(i == 0 and show_legend), name=_preds(name),
        ), row, col)

    y_clean = y_g[np.isfinite(y_g)]
    if len(y_clean) >= 2 and float(np.std(y_clean)) > 1e-12:
        kde_y = _kde(stats, y_clean, bw_adjust)
        obs = go.Scatter(
            x=xs, y=kde_y(xs), mode="lines",
            line=dict(color=_OBS_COLOR, width=2.5),
            name=_obs(name),
        )
        if not show_legend:
            obs.showlegend = False
        _add_trace(fig, obs, row, col)


def ppc_density(y, yrep, n=50, bw_adjust=1.0, title=None, name="y", x_label=None,
                y_label=None, group=None, group_name="group"):
    """KDE of y (black) vs n random draws from yrep rows (blue).

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        n: Max number of yrep lines to show.
        bw_adjust: Bandwidth multiplier passed to scipy KDE.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels ("Predictions of
            <name>"); m.diag passes the observed site name.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
        group: Optional grouping vector, shape (n_obs,) (any hashable
            values). With a group, one density-overlay panel per level
            (bayesplot's ``ppc_dens_overlay_grouped``); observed and
            predicted curves within a panel share its x range.
        group_name: Name of the grouping variable, used in panel titles
            ("{group_name} = {level}").
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)

    finite_mask = np.isfinite(yrep).all(axis=1)
    yrep_clean = yrep[finite_mask]
    if len(yrep_clean) == 0:
        raise ValueError("yrep contains no finite rows.")
    n_dropped = len(yrep) - len(yrep_clean)
    if n_dropped:
        import warnings
        warnings.warn(
            f"ppc_density dropped {n_dropped} of {len(yrep)} yrep draws "
            "containing non-finite values.", stacklevel=2)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep_clean), size=min(n, len(yrep_clean)), replace=False)

    levels, masks = _group_levels(group, len(y))
    if levels is None:
        fig = go.Figure()
        _ppc_density_panel(fig, None, None, y, yrep_clean, idx, bw_adjust, name)
        fig.update_layout(
            title=_title(title, "Density overlay", name), xaxis_title=name,
            yaxis_title="Density",
            plot_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
    else:
        ncols = min(4, len(levels))
        nrows = int(np.ceil(len(levels) / ncols))
        fig = make_subplots(rows=nrows, cols=ncols,
                            shared_xaxes="all",
                            vertical_spacing=(min(0.12, 0.3 / nrows)
                                              if nrows > 1 else None),
                            subplot_titles=[f"{group_name} = {_level_label(lv)}"
                                            for lv in levels])
        for i, mask in enumerate(masks):
            _ppc_density_panel(fig, i // ncols + 1, i % ncols + 1,
                               y[mask], yrep_clean[:, mask], idx, bw_adjust, name,
                               show_legend=(i == 0))
        # axis titles once: x on the bottom row (empty cells of the last row
        # take the title on the row above), y on the first column only
        n_last = len(levels) - ncols * (nrows - 1)
        for c in range(1, ncols + 1):
            fig.update_xaxes(title_text=name,
                             row=nrows if c <= n_last else max(nrows - 1, 1),
                             col=c)
        fig.update_yaxes(title_text="Density", col=1)
        fig.update_layout(
            title=title or f"PPC: Density overlay of {name} by {group_name}",
            height=max(360, 260 * nrows + 120), plot_bgcolor="white",
            legend=dict(orientation="h", xanchor="center", x=0.5,
                        yanchor="top", y=-0.08),
            margin=dict(b=90),
        )
    return _axis_labels(fig, x_label, y_label)


def ppc_hist(y, yrep, n=8, bins=30, title=None, name="y", x_label=None, y_label=None):
    """Histograms: observed y plus n random yrep draws.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        n: Number of yrep draws to show.
        bins: Number of histogram bins.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep), size=min(n, len(yrep)), replace=False)

    total = 1 + len(idx)
    ncols = min(4, total)
    nrows = int(np.ceil(total / ncols))
    subtitles = [_obs(name)] + [f"Prediction {i + 1}" for i in range(len(idx))]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subtitles)

    # One shared x scale and identical bin edges in every panel: edges are
    # computed once from the observed data and all plotted draws, so the
    # panels are comparable (bayesplot's ppc_hist facets share the axes).
    lo = min(float(np.min(y)), float(np.min(yrep[idx])))
    hi = max(float(np.max(y)), float(np.max(yrep[idx])))
    xbins = dict(start=lo, end=hi, size=(hi - lo) / bins)

    def _add_hist(data, row, col, color):
        fig.add_trace(go.Histogram(
            x=data, xbins=xbins,
            marker_color=color, opacity=0.75, showlegend=False,
        ), row=row, col=col)

    _add_hist(y, 1, 1, _OBS_COLOR)
    for i, s in enumerate(idx):
        r = (i + 1) // ncols + 1
        c = (i + 1) % ncols + 1
        _add_hist(yrep[s], r, c, "cornflowerblue")

    fig.update_xaxes(range=[lo, hi])
    fig.update_layout(title=_title(title, "Histograms", name),
                      height=280 * nrows, showlegend=False, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


def ppc_boxplot(y, yrep, n=20, title=None, name="y", x_label=None, y_label=None):
    """Box-and-whisker: y (black) vs n yrep draws.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        n: Number of yrep draws.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep), size=min(n, len(yrep)), replace=False)

    fig = go.Figure()
    fig.add_trace(go.Box(y=y, marker_color=_OBS_COLOR,
                         name="observed", showlegend=False))
    for i, s in enumerate(idx):
        fig.add_trace(go.Box(y=yrep[s], marker_color=_REP_COLOR_SOLID,
                             showlegend=False, name=f"Prediction {i + 1}"))
    fig.update_layout(title=_title(title, "Boxplots", name), yaxis_title=name,
                      xaxis_title=f"Observed {name} and {_preds(name).lower()}",
                      plot_bgcolor="white", boxmode="overlay")
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Test statistics
# =============================================================================

def _ppc_stat_panel(fig, row, col, t_y, t_rep, stat_name, name,
                    show_legend=True):
    """Draw one ppc_stat panel: KDE of t_rep filled to zero with a vertical
    line at t_y. ``row``/``col`` are None for a single-panel figure; only the
    first panel carries the legend."""
    import scipy.stats as scipy_stats
    xs = np.linspace(t_rep.min(), t_rep.max(), 300)
    kde = _kde(scipy_stats, t_rep)
    trace = go.Scatter(
        x=xs, y=kde(xs), mode="lines", fill="tozeroy",
        fillcolor="rgba(100,149,237,0.3)",
        line=dict(color="cornflowerblue", width=1.5),
        name=f"{stat_name} of each prediction of {name}",
    )
    if not show_legend:
        trace.showlegend = False
    _add_trace(fig, trace, row, col)
    if row is None:
        fig.add_vline(x=t_y, line_color=_OBS_COLOR, line_width=2.5,
                      annotation_text=f"observed {stat_name} = {t_y:.3g}",
                      annotation_position="top right")
    else:
        # grouped panels: no vline annotation (it would overflow into the
        # neighbouring panels); the value stays readable via hover instead
        fig.add_vline(x=t_y, row=row, col=col,
                      line_color=_OBS_COLOR, line_width=2.5)
        _add_trace(fig, go.Scatter(
            x=[t_y], y=[0], mode="markers",
            marker=dict(color=_OBS_COLOR, size=8, symbol="line-ns-open"),
            name=f"observed {stat_name}", showlegend=show_legend,
            hovertemplate=f"observed {stat_name} = {t_y:.3g}<extra></extra>",
        ), row, col)


def ppc_stat(y, yrep, stat="mean", title=None, name="y", x_label=None, y_label=None,
             group=None, group_name="group"):
    """Distribution of test statistic over yrep vs observed value.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        stat: Statistic name or callable. Options: 'mean', 'median', 'sd',
              'var', 'min', 'max', 'q25', 'q75', 'skew', 'n_zero', 'n_pos'.
              A callable takes one 1D array and returns a number, e.g.
              ``lambda v: np.percentile(v, 90)``.
        title: Plot title. Defaults to one naming the statistic, the outcome
            and the Bayesian p-values.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
        group: Optional grouping vector, shape (n_obs,) (any hashable
            values). With a group, one panel per level showing T(y[g]) and
            T(yrep[:, g]) (bayesplot's ``ppc_stat_grouped``); each panel has
            its own x range and its two-sided p-value in the subplot title.
        group_name: Name of the grouping variable, used in the titles
            ("{group_name} = {level}").
    Returns:
        plotly Figure with Bayesian p-value annotation.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    fn = _stat_fn(stat)
    stat_name = stat if isinstance(stat, str) else getattr(stat, "__name__", "T")
    if stat_name == "<lambda>":
        stat_name = "T"

    levels, masks = _group_levels(group, len(y))
    if levels is None:
        t_y = float(fn(y))
        t_rep = np.array([fn(yrep[s]) for s in range(len(yrep))])

        # One-sided P(T(yrep) >= T(y)); the two-sided value is what people usually
        # mean by "the" Bayesian p-value, so report both rather than an
        # unqualified number that reads as 0 or 1 for a good fit.
        pval = float(np.mean(t_rep >= t_y))
        pval_2s = float(2 * min(pval, 1.0 - pval))

        fig = go.Figure()
        _ppc_stat_panel(fig, None, None, t_y, t_rep, stat_name, name)
        fig.update_layout(
            title=title or (f"PPC: {stat_name} of {name}  |  "
                            f"P(predicted ≥ observed) = {pval:.3f}"
                            f"  (two-sided p = {pval_2s:.3f})"),
            xaxis_title=f"{stat_name}({name})",
            yaxis_title="Density",
            plot_bgcolor="white",
        )
        return _axis_labels(fig, x_label, y_label)

    panels = []
    for mask in masks:
        t_y = float(fn(y[mask]))
        t_rep = np.array([fn(yrep[s][mask]) for s in range(len(yrep))])
        pval = float(np.mean(t_rep >= t_y))
        pval_2s = float(2 * min(pval, 1.0 - pval))
        panels.append((t_y, t_rep, pval_2s))

    ncols = min(4, len(levels))
    nrows = int(np.ceil(len(levels) / ncols))
    subtitles = [f"{group_name} = {_level_label(lv)} (p = {p:.2f})"
                 for lv, (_, _, p) in zip(levels, panels)]
    # no shared_xaxes here: the statistic's range differs by group
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subtitles)
    for i, (t_y, t_rep, _) in enumerate(panels):
        _ppc_stat_panel(fig, i // ncols + 1, i % ncols + 1, t_y, t_rep,
                        stat_name, name, show_legend=(i == 0))
    # axis titles once: x on the bottom row (empty cells of the last row take
    # the title on the row above), y on the first column only
    n_last = len(levels) - ncols * (nrows - 1)
    for c in range(1, ncols + 1):
        fig.update_xaxes(title_text=f"{stat_name}({name})",
                         row=nrows if c <= n_last else max(nrows - 1, 1),
                         col=c)
    fig.update_yaxes(title_text="Density", col=1)
    fig.update_layout(
        title=title or f"PPC: {stat_name} of {name} by {group_name}",
        height=max(360, 260 * nrows + 120), plot_bgcolor="white",
        legend=dict(orientation="h", xanchor="center", x=0.5,
                    yanchor="top", y=-0.08),
        margin=dict(b=90),
    )
    return _axis_labels(fig, x_label, y_label)


def ppc_stat_2d(y, yrep, stat1="mean", stat2="sd", title=None, name="y", x_label=None, y_label=None):
    """Scatter of two test statistics over yrep vs observed point.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        stat1, stat2: Statistic names or callables.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    fn1, fn2 = _stat_fn(stat1), _stat_fn(stat2)
    s1_name = stat1 if isinstance(stat1, str) else "T1"
    s2_name = stat2 if isinstance(stat2, str) else "T2"

    t1_rep = np.array([fn1(yrep[s]) for s in range(len(yrep))])
    t2_rep = np.array([fn2(yrep[s]) for s in range(len(yrep))])
    t1_y, t2_y = float(fn1(y)), float(fn2(y))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t1_rep, y=t2_rep, mode="markers",
        marker=dict(color="cornflowerblue", opacity=0.4, size=5),
        name=_preds(name),
    ))
    fig.add_trace(go.Scatter(
        x=[t1_y], y=[t2_y], mode="markers",
        marker=dict(color=_OBS_COLOR, size=12, symbol="x"),
        name=_obs(name),
    ))
    fig.update_layout(
        title=title or f"PPC: {s1_name} vs {s2_name} — predictions of {name}",
        xaxis_title=f"{s1_name}({name})", yaxis_title=f"{s2_name}({name})",
        plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Intervals
# =============================================================================

def _ppc_intervals_panel(fig, row, col, x, y, ll, l, m, h, hh, name,
                         prob, prob_outer, show_legend=True):
    """Draw one ppc_intervals panel: outer/inner interval bars, median and
    observed points. ``row``/``col`` are None for a single-panel figure; only
    the first panel carries the legend."""
    def _add(trace):
        if not show_legend:
            trace.showlegend = False
        _add_trace(fig, trace, row, col)

    xs, ys = _segments(x, ll, hh)
    _add(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color=_REP_COLOR_SOLID, width=1.5),
        name=f"{int(round(prob_outer * 100))}% interval", hoverinfo="skip",
    ))
    xs, ys = _segments(x, l, h)
    _add(go.Scatter(
        x=xs, y=ys, mode="lines", line=dict(color="cornflowerblue", width=5),
        name=f"{int(round(prob * 100))}% interval", hoverinfo="skip",
    ))
    _add(go.Scatter(
        x=x, y=m, mode="markers",
        marker=dict(color="white", size=7,
                    line=dict(color="cornflowerblue", width=1.5)),
        name=f"Median prediction of {name}",
    ))
    _add(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(color=_OBS_COLOR, size=5),
        name=_obs(name),
    ))


def ppc_intervals(y, yrep, x=None, prob=0.5, prob_outer=0.9, title=None,
                  name="y", x_label=None, y_label=None, group=None,
                  group_name="group"):
    """Predictive interval of each observation, with the observed value.

    One vertical bar per observation, as bayesplot's ``ppc_intervals``: a
    thin line for the outer (prob_outer) interval, a thick line for the inner
    (prob) interval, a light point for the median prediction and a dark
    point for the observed value. About prob of the observed points should
    fall on the thick bars and prob_outer on the thin ones.

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        x: Optional x-axis variable, shape (n_obs,) (e.g. a predictor).
            Defaults to the observation index. A discrete x (up to
            ``_MAX_DISCRETE_X`` distinct values) gets one tick per level,
            labelled with the fewest decimals that tells the levels apart.
        prob: Inner interval probability (e.g., 0.5).
        prob_outer: Outer interval probability (e.g., 0.9).
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
        group: Optional grouping vector, shape (n_obs,) (any hashable
            values). With a group, one panel per level holding that group's
            observations (bayesplot's ``ppc_intervals_grouped``); x within a
            panel is x[g] when x is given, else 0..n_g-1, and the y axis is
            shared across panels.
        group_name: Name of the grouping variable, used in panel titles
            ("{group_name} = {level}").
    Returns:
        plotly Figure.
    """
    x_was_none = x is None
    levels, masks = _group_levels(group, len(np.asarray(y).flatten()))
    if levels is None:
        x_, y_, ll, l, m, h, hh = _interval_bounds(y, yrep, x, prob, prob_outer)
        prob, prob_outer = sorted((prob, prob_outer))

        fig = go.Figure()
        _ppc_intervals_panel(fig, None, None, x_, y_, ll, l, m, h, hh,
                             name, prob, prob_outer)
        fig.update_layout(title=_title(title, "Predictive intervals", name),
                          xaxis_title="Observation" if x_was_none else "x",
                          yaxis_title=name, plot_bgcolor="white")
        if not x_was_none:
            _apply_discrete_ticks(fig, ["xaxis"], [x_])
        return _axis_labels(fig, x_label, y_label)

    y_arr, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    x_arr = None if x is None else np.asarray(x).flatten()
    panels = [_interval_bounds(y_arr[mask], yrep[:, mask],
                               None if x_arr is None else x_arr[mask],
                               prob, prob_outer) for mask in masks]
    prob, prob_outer = sorted((prob, prob_outer))

    ncols = min(4, len(levels))
    nrows = int(np.ceil(len(levels) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, shared_yaxes=True,
                        subplot_titles=[f"{group_name} = {_level_label(lv)}"
                                        for lv in levels])
    for i, panel in enumerate(panels):
        _ppc_intervals_panel(fig, i // ncols + 1, i % ncols + 1, *panel,
                             name=name, prob=prob, prob_outer=prob_outer,
                             show_legend=(i == 0))
    fig.update_xaxes(title_text="Observation" if x_was_none else "x")
    fig.update_yaxes(title_text=name, col=1)   # y axis is shared, title once
    if not x_was_none:
        # make_subplots names the axes xaxis, xaxis2, ... in panel order, and
        # panel i holds level i of the group, so panel_x lines up with axes.
        _apply_discrete_ticks(fig, ["xaxis"] + [f"xaxis{k}"
                                                for k in range(2, len(levels) + 1)],
                              [panel[0] for panel in panels])
    fig.update_layout(
        title=title or f"PPC: Predictive intervals of {name} by {group_name}",
        height=280 * nrows + 125, plot_bgcolor="white",
        # one shared legend under the panels: on the right it would eat a
        # third of the width. Low enough to clear the x axis titles.
        legend=dict(orientation="h", xanchor="center", x=0.5,
                    yanchor="top", y=-0.30),
        margin=dict(b=165))
    return _axis_labels(fig, x_label, y_label)


def ppc_ribbon(y, yrep, x=None, prob=0.5, prob_outer=0.9, title=None,
               name="y", x_label=None, y_label=None):
    """Predictive intervals drawn as continuous ribbons along x.

    Same intervals as :func:`ppc_intervals`, as bayesplot's ``ppc_ribbon``:
    a light outer ribbon, a darker inner ribbon, a thin line for the median
    prediction and the observed values as points. Easier to read than the
    bars when x is continuous (time, dose, a predictor).

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        x: Optional x-axis variable, shape (n_obs,). Defaults to the
            observation index. A discrete x gets one tick per level, as in
            :func:`ppc_intervals`.
        prob: Inner interval probability (e.g., 0.5).
        prob_outer: Outer interval probability (e.g., 0.9).
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    x_was_none = x is None
    x, y, ll, l, m, h, hh = _interval_bounds(y, yrep, x, prob, prob_outer)
    prob, prob_outer = sorted((prob, prob_outer))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]), y=np.concatenate([hh, ll[::-1]]),
        fill="toself", fillcolor="rgba(100,149,237,0.2)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name=f"{int(round(prob_outer * 100))}% interval",
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]), y=np.concatenate([h, l[::-1]]),
        fill="toself", fillcolor="rgba(100,149,237,0.45)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        name=f"{int(round(prob * 100))}% interval",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=m, mode="lines", line=dict(color="cornflowerblue", width=1),
        name=f"Median prediction of {name}",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", marker=dict(color=_OBS_COLOR, size=5),
        name=_obs(name),
    ))
    fig.update_layout(title=_title(title, "Predictive ribbon", name),
                      xaxis_title="Observation" if x_was_none else "x",
                      yaxis_title=name, plot_bgcolor="white")
    if not x_was_none:
        _apply_discrete_ticks(fig, ["xaxis"], [x])
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Errors
# =============================================================================

def ppc_error_scatter(y, yrep, title=None, name="y", x_label=None, y_label=None):
    """Scatter of the average predictive error e_i = y_i - mean_s yrep[s, i]
    (x) against the observed y_i (y), like bayesplot's ppc_error_scatter_avg.

    Read it: no pattern is good; a slope means the errors depend on the
    observed value.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    mu_rep = yrep.mean(axis=0)
    err = y - mu_rep

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=err, y=y, mode="markers",
        marker=dict(color="cornflowerblue", size=6, opacity=0.7),
        name="observations",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="black")
    fig.update_layout(title=_title(title, "Average predictive error", name),
                      xaxis_title=f"Average error ({name} − mean prediction)",
                      yaxis_title=_obs(name), plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


def ppc_error_hist(y, yrep, bins=40, title=None, name="y", x_label=None, y_label=None):
    """Distribution of mean residuals r_i = y_i - E[yrep_i].

    Plots the mean residual per observation; bayesplot's ppc_error_hist
    instead shows the per-draw errors y - yrep[s, i] faceted by draw.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        bins: Number of histogram bins.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    residuals = y - yrep.mean(axis=0)

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=residuals, nbinsx=bins,
                               marker_color="cornflowerblue", opacity=0.8,
                               showlegend=False))
    fig.add_vline(x=0, line_color="black", line_dash="dash")
    fig.update_layout(title=_title(title, "Residuals", name),
                      xaxis_title=f"Observed {name} − mean prediction",
                      yaxis_title="Count", plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


def _n_bins(N):
    """bayesplot's default bin count: floor(N/2) for N <= 10, 10 for
    N < 100, floor(sqrt(N)) otherwise."""
    if N <= 10:
        return N // 2
    if N < 100:
        return 10
    return int(np.floor(np.sqrt(N)))


def _bin_errors(ey, r, bins):
    """One bayesplot ``bin_errors`` pass: group observations into ``bins``
    bins by ``ey`` and return the per-bin (ey_bar, err_bar, se2) arrays.

    Mirrors bayesplot 1.16's ``bin_errors`` exactly: breaks at the midpoints
    of the sorted ``ey`` at the 1-indexed positions floor(N*i/bins) (with the
    tie/-Inf adjustments and ``unique``), right-closed ``(a, b]`` intervals,
    sd with n-1, se2 = 2 * sd / sqrt(n_bin), and NaN rows for empty bins
    (R's mean/sd of nothing).
    """
    ey = np.asarray(ey, dtype=float).flatten()
    r = np.asarray(r, dtype=float).flatten()
    N = len(ey)
    break_ids = [int(np.floor(N * i / bins)) for i in range(1, bins)]
    if any(b == 0 for b in break_ids):
        bins = 1
    if bins == 1:
        breaks = np.array([-np.inf, (ey.min() + ey.max()) / 2, np.inf])
    else:
        ey_sort = np.sort(ey)
        breaks = [-np.inf]
        for break_i in break_ids:
            lo, hi = ey_sort[break_i - 1], ey_sort[break_i]   # R is 1-indexed
            if lo == hi:
                if lo == ey.min():
                    lo = -np.inf
                else:
                    lo = ey[ey < lo].max()
            breaks.append((lo + hi) / 2)
        breaks = np.unique(np.append(np.asarray(breaks, dtype=float), np.inf))
    binned = np.digitize(ey, breaks[1:-1], right=True) + 1    # (a, b] like cut
    nb = len(breaks) - 1
    ey_bar = np.full(nb, np.nan)
    err_bar = np.full(nb, np.nan)
    se2 = np.full(nb, np.nan)
    for i in range(1, nb + 1):
        mark = np.flatnonzero(binned == i)
        k = len(mark)
        if k:
            ey_bar[i - 1] = ey[mark].mean()
            err_bar[i - 1] = r[mark].mean()
            s = r[mark].std(ddof=1) if k > 1 else 0.0
            se2[i - 1] = 2 * s / np.sqrt(k)
    return ey_bar, err_bar, se2


def ppc_error_binned(y, yrep, x=None, bins=None, n=3, title=None, name="y",
                     x_label=None, y_label=None):
    """Binned residual plot (bayesplot's ppc_error_binned): one panel per draw.

    Intended for binary (or proportion) outcomes: observations are grouped
    into bins by their expected value and each panel plots the average error
    per bin (points) with a +/-2 SE ribbon around the dashed line at 0. For
    the classic binned-residual plot pass the expected values (posterior
    probabilities, one row per draw) as ``yrep``; the first ``n`` rows are
    used. Read it: points inside the +/-2 SE band and no trend across bins
    is good; points systematically off the band or a slope mean the
    predicted probabilities are mis-calibrated.

    Args:
        y: Observed data (binary 0/1 or proportions).
        yrep: Expected values matrix (posterior probabilities, one row per
            draw); the first ``n`` rows are plotted.
        x: Values to bin on (and plot on x) instead of the rows of yrep.
        bins: Number of bins. Defaults to bayesplot's n_bins(length(y)).
        n: Number of draws (panels).
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)[:n]
    if bins is None:
        bins = _n_bins(len(y))
    n = len(yrep)

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[f"Draw {s + 1}" for s in range(n)])
    for s in range(n):
        ey = yrep[s] if x is None else np.asarray(x, dtype=float).flatten()
        ey_bar, err_bar, se2 = _bin_errors(ey, y - yrep[s], bins)
        r, c = s // ncols + 1, s % ncols + 1
        fig.add_trace(go.Scatter(
            x=ey_bar, y=err_bar + se2, mode="lines",
            line=dict(color="cornflowerblue", width=1), hoverinfo="skip",
            showlegend=(s == 0), name="±2 SE"), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=ey_bar, y=err_bar - se2, mode="lines",
            line=dict(color="cornflowerblue", width=1),
            fill="tonexty", fillcolor="rgba(100,149,237,0.2)",
            showlegend=False), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=ey_bar, y=err_bar, mode="markers",
            marker=dict(color="cornflowerblue", size=6),
            showlegend=(s == 0), name="Average error"), row=r, col=c)
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=r, col=c)
        fig.update_xaxes(title_text="Predicted proportion", row=r, col=c)
        fig.update_yaxes(title_text="Average errors (with 2SE bounds)", row=r, col=c)

    fig.update_layout(title=_title(title, "Binned residuals", name),
                      height=300 * nrows, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Survival
# =============================================================================

def _km(time, event):
    """Kaplan-Meier estimate of (time, event), as survival::survfit.

    Returns the sorted unique event times and S(t) = prod_{t_j <= t}
    (1 - d_j / n_j), where d_j is the number of events at t_j and n_j the
    number at risk at t_j (observations censored at t_j are still at risk
    at t_j). The returned arrays start with (0, 1.0) prepended.
    """
    time = np.asarray(time, dtype=float).flatten()
    event = np.asarray(event, dtype=float).flatten()
    t = np.unique(time[event > 0])
    s = np.array([1.0 - np.sum((time == tj) & (event > 0)) / np.sum(time >= tj)
                  for tj in t])
    return (np.concatenate(([0.0], t)),
            np.concatenate(([1.0], np.cumprod(s))))


def _ppc_km_panel(fig, row, col, y, status_y, yrep, t_cut, name,
                  show_legend=True, obs_color=_OBS_COLOR,
                  rep_color="rgba(100,149,237,0.35)",
                  obs_name=None, pred_name=None, legendgroup=None):
    """One KM-overlay panel: the KM of each yrep row (all events, cut at
    t_cut), then the KM of the observed (y, status_y) drawn last on top."""
    if pred_name is None:
        pred_name = _preds(name)
    if obs_name is None:
        obs_name = _obs(name)
    for s in range(len(yrep)):
        t, surv = _km(yrep[s], np.ones(len(yrep[s])))
        keep = t <= t_cut
        tr = go.Scatter(
            x=t[keep], y=surv[keep], mode="lines", line_shape="hv",
            line=dict(color=rep_color),
            showlegend=(show_legend and s == 0), name=pred_name)
        if legendgroup is not None:
            tr.legendgroup = f"{legendgroup}_pred"
        _add_trace(fig, tr, row, col)
    t, surv = _km(y, status_y)
    obs = go.Scatter(
        x=t, y=surv, mode="lines", line_shape="hv",
        line=dict(color=obs_color, width=2.5), name=obs_name)
    if not show_legend:
        obs.showlegend = False
    if legendgroup is not None:
        obs.legendgroup = f"{legendgroup}_obs"
    _add_trace(fig, obs, row, col)


def ppc_km_overlay(y, yrep, status_y, n=50, extrapolation_factor=1.2,
                   title=None, name="y", x_label=None, y_label=None,
                   group=None, group_name="group", facet=False):
    """Kaplan-Meier overlay (bayesplot's ppc_km_overlay): observed survival
    curve against the curves of n predictive replications.

    The observed curve is the Kaplan-Meier estimate of (y, status_y), where
    status_y marks each observation as an event (1) or censored (0); each
    predicted curve is the Kaplan-Meier estimate of (yrep[s, ], all events),
    i.e. replications carry no censoring. Both are drawn as step functions.
    Read it: the observed curve should lie inside the band of predicted
    curves; a curve systematically above or below the band means the model
    mis-predicts survival times. Predicted curves are cut at
    extrapolation_factor * max(y) (use Inf to keep all of them).

    Args:
        y: Observed times.
        yrep: Posterior predictive matrix of times; the first ``n`` rows
            are plotted.
        status_y: Event indicator (1 = event, 0 = censored), same length as y.
        n: Number of predictive curves.
        extrapolation_factor: Predicted curves are cut at this multiple of
            max(y). Use Inf to draw them in full.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
        group: Optional grouping vector, shape (n_obs,) (any hashable
            values). With a group, each group's observations and predicted
            columns are plotted (each group's predicted curves are cut at
            extrapolation_factor * max(y[group])).
        group_name: Name of the grouping variable, used in labels and panel
            titles ("{group_name} = {level}").
        facet: When False (the default when group is given), overlays the KM
            curves of all groups on a single plot with distinct colors per
            group. When True, separates the groups into subplots (one panel
            per level, matching bayesplot's grouped layout).
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    status_y, = _as_np(status_y)
    if len(status_y) != len(y) or not np.all((status_y == 0) | (status_y == 1)):
        raise ValueError("`status_y` must be a numeric vector of 0s and 1s "
                         "the same length as `y`.")
    yrep = np.asarray(yrep, dtype=float)[:n]

    levels, masks = _group_levels(group, len(y))
    if levels is None:
        fig = go.Figure()
        _ppc_km_panel(fig, None, None, y, status_y, yrep,
                      extrapolation_factor * y.max(), name)
        fig.update_layout(title=_title(title, "Kaplan-Meier overlay", name),
                          xaxis_title="Time", yaxis_title="Survival probability",
                          yaxis=dict(range=[0, 1.02]), plot_bgcolor="white")
    elif not facet:
        fig = go.Figure()
        formatted = _level_labels(levels)
        for i, (mask, lv) in enumerate(zip(masks, levels)):
            col = _COLORS[i % len(_COLORS)]
            r_col = _color_to_rgba(col, 0.35)
            lbl = formatted[i]
            t_cut = extrapolation_factor * y[mask].max()
            _ppc_km_panel(fig, None, None, y[mask], status_y[mask], yrep[:, mask],
                          t_cut, name, show_legend=True,
                          obs_color=col, rep_color=r_col,
                          obs_name=f"{group_name} = {lbl} (observed)",
                          pred_name=f"{group_name} = {lbl} (predicted)",
                          legendgroup=f"{group_name}_{i}")
        fig.update_layout(
            title=title or f"PPC: Kaplan-Meier overlay of {name} by {group_name}",
            xaxis_title="Time", yaxis_title="Survival probability",
            yaxis=dict(range=[0, 1.02]), plot_bgcolor="white",
            legend=dict(orientation="h", xanchor="center", x=0.5,
                        yanchor="top", y=-0.12),
            margin=dict(b=90))
    else:
        ncols = min(4, len(levels))
        nrows = int(np.ceil(len(levels) / ncols))
        fig = make_subplots(rows=nrows, cols=ncols,
                            shared_yaxes="all",
                            vertical_spacing=(min(0.12, 0.3 / nrows)
                                              if nrows > 1 else None),
                            subplot_titles=[f"{group_name} = {_level_label(lv)}"
                                            for lv in levels])
        for i, mask in enumerate(masks):
            _ppc_km_panel(fig, i // ncols + 1, i % ncols + 1,
                          y[mask], status_y[mask], yrep[:, mask],
                          extrapolation_factor * y[mask].max(), name,
                          show_legend=(i == 0))
        # axis titles once: Time on the bottom row (empty cells of the last
        # row take the title on the row above), survival on the first column
        n_last = len(levels) - ncols * (nrows - 1)
        for c in range(1, ncols + 1):
            fig.update_xaxes(title_text="Time",
                             row=nrows if c <= n_last else max(nrows - 1, 1),
                             col=c)
        fig.update_yaxes(title_text="Survival probability", col=1)
        fig.update_yaxes(range=[0, 1.02])
        fig.update_layout(
            title=title or f"PPC: Kaplan-Meier overlay of {name} by {group_name}",
            height=max(360, 260 * nrows + 120), plot_bgcolor="white",
            legend=dict(orientation="h", xanchor="center", x=0.5,
                        yanchor="top", y=-0.08),
            margin=dict(b=90))
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Scatterplots
# =============================================================================

def ppc_scatter(y, yrep, n_reps=9, title=None, name="y", x_label=None, y_label=None):
    """Small-multiple scatterplots of y vs n_reps random rows of yrep.

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        n_reps: Number of replications to show.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(yrep), size=min(n_reps, len(yrep)), replace=False)

    ncols = 3
    nrows = int(np.ceil(len(idx) / ncols))
    subtitles = [f"Prediction {i + 1}" for i in range(len(idx))]
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=subtitles)

    lim = [y.min(), y.max()]
    for i, s in enumerate(idx):
        r = i // ncols + 1
        c = i % ncols + 1
        fig.add_trace(go.Scatter(
            x=yrep[s], y=y, mode="markers",
            marker=dict(color="cornflowerblue", size=5, opacity=0.6),
            showlegend=False,
        ), row=r, col=c)
        fig.add_trace(go.Scatter(
            x=lim, y=lim, mode="lines",
            line=dict(color="black", dash="dash", width=1),
            showlegend=False,
        ), row=r, col=c)
        fig.update_xaxes(title_text=f"predicted {name}", row=r, col=c)
        fig.update_yaxes(title_text=f"observed {name}", row=r, col=c)

    fig.update_layout(title=_title(title, "Observed vs individual predictions", name),
                      height=300 * nrows, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# Discrete
# =============================================================================

def ppc_rootogram(y, yrep, title=None, name="y", x_label=None, y_label=None):
    """Hanging rootogram for count outcomes.

    Compares sqrt of observed counts (bars) to expected sqrt counts (curve).
    Bars hanging below zero indicate over-prediction; above zero: under-prediction.

    Args:
        y: Observed integer counts.
        yrep: Posterior predictive matrix of counts.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y = np.rint(np.asarray(y, dtype=float).flatten()).astype(int)
    yrep = np.rint(np.asarray(yrep, dtype=float)).astype(int)
    if y.min() < 0 or yrep.min() < 0:
        raise ValueError("ppc_rootogram expects non-negative counts.")

    max_val = max(int(y.max()), int(yrep.max()))
    counts_obs = np.bincount(y, minlength=max_val + 1).astype(float)

    counts_rep = np.array([
        np.bincount(yrep[s], minlength=max_val + 1)
        for s in range(len(yrep))
    ], dtype=float)
    expected = counts_rep.mean(axis=0)

    xs = np.arange(len(counts_obs))
    sqrt_obs = np.sqrt(counts_obs)
    sqrt_exp = np.sqrt(expected)

    fig = go.Figure()
    # Hanging rootogram: bars of height sqrt(observed) are suspended FROM the
    # sqrt(expected) curve, so their distance from y=0 is the misfit and the
    # zero line is the reference. Basing them at sqrt_exp instead (as before)
    # drew bars spanning expected->observed, leaving add_hline(y=0) meaningless.
    fig.add_trace(go.Bar(
        x=xs, y=sqrt_obs, base=sqrt_exp - sqrt_obs,
        marker_color="cornflowerblue", opacity=0.7,
        name="√Observed (hanging from √Expected)",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=sqrt_exp, mode="lines+markers",
        line=dict(color="black", width=2),
        name=f"√Expected count ({_preds(name).lower()})",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="grey")
    fig.update_layout(
        title=_title(title, "Rootogram", name), xaxis_title=name,
        yaxis_title="√Frequency",
        plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


def ppc_bars(y, yrep, title=None, name="y", x_label=None, y_label=None):
    """Bar chart comparing observed y to posterior predictive yrep.

    For ordinal, categorical, or discrete outcomes.
    Shows observed counts and, per category, the median predicted count with
    a 90% interval, as in bayesplot's ppc_bars.

    Args:
        y: Observed discrete values.
        yrep: Posterior predictive matrix.
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure.
    """
    y = np.asarray(y).flatten()
    yrep = np.asarray(yrep)
    # bayesplot's ppc_bars puts no cap on the number of categories; it only
    # requires whole-number outcomes (NaN counts as not whole).
    for v in (y, yrep):
        if np.any(v != np.round(v)):
            raise ValueError(
                f"ppc_bars expects discrete outcomes (whole numbers); {name} "
                "has non-integer values. Use ppc_density or ppc_hist for "
                "continuous data."
            )
    cats = np.unique(np.concatenate([y, yrep.flatten()]))

    obs_counts = np.bincount(np.searchsorted(cats, y),
                             minlength=len(cats)).astype(float)
    idx = np.searchsorted(cats, yrep)  # every yrep value is in cats
    rep_counts = np.stack([np.bincount(row, minlength=len(cats))
                           for row in idx]).astype(float)

    rep_med  = np.median(rep_counts, axis=0)
    rep_lo   = np.percentile(rep_counts, 5, axis=0)
    rep_hi   = np.percentile(rep_counts, 95, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cats, y=obs_counts,
        marker_color=_OBS_COLOR, opacity=0.8,
        name=_obs(name),
    ))
    fig.add_trace(go.Scatter(
        x=cats, y=rep_med, mode="markers",
        marker=dict(color="cornflowerblue", size=10, symbol="diamond"),
        error_y=dict(type="data", symmetric=False,
                     array=rep_hi - rep_med,
                     arrayminus=rep_med - rep_lo),
        name=f"{_preds(name)}: median count ± 90% interval",
    ))
    fig.update_layout(title=_title(title, "Bar chart", name), xaxis_title=name,
                      yaxis_title="Count", plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# LOO-based PPC
# =============================================================================

def _pit_density(pit, xs):
    """KDE of values in [0, 1] (Scott bandwidth) with reflection at 0 and 1.

    A plain KDE leaks mass outside [0, 1], so even perfectly uniform PIT
    values would draw a curve that drops to ~0.5 at both ends -- the shape
    of an over-dispersed model. Reflecting the sample (bayesplot's
    boundary correction) removes that artefact.
    """
    pit = np.asarray(pit, dtype=np.float64)
    n = len(pit)
    bw = np.std(pit, ddof=1) * n ** (-0.2)
    pts = np.concatenate([pit, -pit, 2.0 - pit])
    z = (np.asarray(xs, dtype=np.float64)[:, None] - pts[None, :]) / bw
    return np.exp(-0.5 * z * z).sum(axis=1) / (n * bw * np.sqrt(2.0 * np.pi))


def ppc_loo_pit(y, yrep, log_likelihood=None, reff=None, n_unif=100,
                title=None, name="y", x_label=None, y_label=None):
    """LOO probability integral transform check.

    PIT value for each observation: ``F(y_i) = P(yrep[:, i] <= y_i)``.
    Under a well-calibrated model these are Uniform(0, 1).

    When ``log_likelihood`` is given, the draws are reweighted by
    Pareto-smoothed importance sampling so each ``F(y_i)`` is a *leave-one-out*
    predictive CDF — this is what makes the check meaningful. Without it the
    same draws that were fitted to ``y`` are used to score ``y``, which pulls
    the PITs toward 0.5 and makes a miscalibrated model look calibrated; a
    warning is emitted in that case.

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        log_likelihood: Pointwise log-likelihood, shape (C, S, n_obs) or
            (S, n_obs), with S matching yrep's rows. Enables true LOO-PIT.
        reff: Relative MCMC efficiency for PSIS; see
            ``jax_diagnostics.relative_eff``. Defaults to 1.0.
        n_unif: Number of reference curves, each the density of n_obs
            Uniform(0, 1) draws, drawn in grey: how far from flat a curve
            from a perfectly calibrated model can be by chance.
        title: Plot title. Defaults to one naming the outcome and saying
            whether the PIT is leave-one-out.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.
    Returns:
        plotly Figure. Both densities are boundary-corrected (reflected at 0
        and 1), as in bayesplot.
    """
    import warnings
    y, = _as_np(y)
    yrep = np.asarray(yrep, dtype=float)
    n_obs = len(y)

    if log_likelihood is None:
        warnings.warn(
            "ppc_loo_pit called without log_likelihood: computing a plain "
            "posterior-predictive PIT, not LOO-PIT. The same draws are used to "
            "fit and to score y, so the PITs are biased toward 0.5 and a "
            "miscalibrated model can look calibrated. Pass log_likelihood= for "
            "the leave-one-out version.",
            stacklevel=2,
        )
        pit = np.array([np.mean(yrep[:, i] <= y[i]) for i in range(n_obs)],
                       dtype=np.float64)
    else:
        from BayesForge.Diagnostic.jax_diagnostics import _psis_weights
        ll = np.asarray(log_likelihood, dtype=np.float64)
        if ll.ndim == 3:
            ll = ll.reshape(ll.shape[0] * ll.shape[1], ll.shape[2])
        if ll.shape != yrep.shape:
            raise ValueError(
                f"log_likelihood shape {ll.shape} does not match yrep "
                f"{yrep.shape}; both need (n_draws, n_obs) with the same draws."
            )
        pit = np.empty(n_obs, dtype=np.float64)
        for i in range(n_obs):
            log_w, _ = _psis_weights(ll[:, i], reff=1.0 if reff is None else reff)
            w = np.exp(np.asarray(log_w, dtype=np.float64))
            w /= w.sum()
            pit[i] = float(np.sum(w * (yrep[:, i] <= y[i])))

    kind = "LOO-PIT" if log_likelihood is not None else "PIT (not leave-one-out)"
    title = _title(title, kind, name)
    pit = pit[np.isfinite(pit)]
    if len(pit) < 2:
        return go.Figure().update_layout(title=title + " (insufficient data)")

    xs = np.linspace(0, 1, 300)
    fig = go.Figure()
    rng = np.random.default_rng(0)
    for j in range(int(n_unif)):
        fig.add_trace(go.Scatter(
            x=xs, y=_pit_density(rng.uniform(size=len(pit)), xs), mode="lines",
            line=dict(color="rgba(150,150,150,0.25)", width=1),
            hoverinfo="skip", legendgroup="unif", showlegend=(j == 0),
            name=f"Uniform samples of size {len(pit)}",
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[1, 1], mode="lines",
        line=dict(color="black", dash="dash", width=1.5),
        name="Uniform(0,1)",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=_pit_density(pit, xs), mode="lines",
        line=dict(color="cornflowerblue", width=2.5),
        name=f"{kind} of {name}",
    ))
    fig.update_layout(
        title=title, xaxis_title=kind, yaxis_title="Density",
        xaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


def ppc_intervals_sorted(y, yrep, x=None, prob=0.5, prob_outer=0.9,
                         title=None, name="y", x_label=None, y_label=None):
    """Predictive intervals with the observations ordered by observed value.

    This is :func:`ppc_intervals` with observation i placed at the rank of
    y_i, so the observed points rise from left to right and systematic
    over/under-prediction across the range of the outcome is easy to read.
    It performs no leave-one-out computation (``ppc_loo_intervals`` is a
    deprecated alias).

    Args:
        y: Observed data.
        yrep: Posterior predictive matrix.
        x: Optional ordering variable. Defaults to the rank of y.
        prob: Inner interval probability (e.g., 0.5).
        prob_outer: Outer interval probability (e.g., 0.9).
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title. Defaults to the plot's own label.
        y_label: Y-axis title. Defaults to the outcome name.
    Returns:
        plotly Figure.
    """
    sort_by_y = x is None
    if sort_by_y:
        # The rank of each observation; argsort(y) alone is the inverse
        # permutation and left the observations unsorted.
        x = np.argsort(np.argsort(np.asarray(y, dtype=float).flatten(),
                                  kind="stable"), kind="stable")
    fig = ppc_intervals(y, yrep, x=x, prob=prob, prob_outer=prob_outer,
                        title=_title(title, "Predictive intervals sorted by observed value", name),
                        name=name)
    if sort_by_y:
        fig.update_layout(xaxis_title=f"Observations ranked by observed {name}")
    return _axis_labels(fig, x_label, y_label)


def ppc_loo_intervals(y, yrep, x=None, prob=0.9, title=None, name="y",
                      x_label=None, y_label=None):
    """Deprecated alias for :func:`ppc_intervals_sorted` (it is not LOO-based).

    ``prob`` is the outer interval, as before; the inner one is 50%.
    """
    import warnings
    warnings.warn(
        "ppc_loo_intervals performs no leave-one-out computation; it is "
        "ppc_intervals sorted by y. Use ppc_intervals_sorted instead.",
        DeprecationWarning, stacklevel=2,
    )
    return ppc_intervals_sorted(y, yrep, x=x, prob=min(0.5, prob),
                                prob_outer=prob, title=title, name=name,
                                x_label=x_label, y_label=y_label)
