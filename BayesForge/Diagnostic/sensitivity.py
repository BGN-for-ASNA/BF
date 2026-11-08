"""Posterior sensitivity analysis and model checking utilities for BF.

Functions
---------
prior_sensitivity_plot   — overlay posteriors fitted with different priors
prior_sensitivity        — same, or declare prior variations and refit in parallel
influence_plot           — Pareto-k scatter (LOO influential observations)
calibration_plot         — empirical coverage vs nominal level
divergence_energy_plot   — HMC energy / divergence check
multimodality_check      — cluster-based mode detection
"""
import numpy as np
import jax.numpy as jnp
import plotly.graph_objects as go
from BayesForge.Diagnostic.ppc import _axis_labels
import plotly.colors as pcolors
from plotly.subplots import make_subplots
import plotly.express as px

_COLORS = pcolors.qualitative.Plotly

# Hover for a density curve: density at x, plus the posterior summary.
_HOVER = ("density %{y:.3g}<br>mean %{customdata[0]:.3g} "
          "(sd %{customdata[1]:.3g})<br>"
          "HDI [%{customdata[2]:.3g}, %{customdata[3]:.3g}]<extra>%{fullData.name}</extra>")


def _hdi(samples, prob):
    """Highest-density interval (lower, upper) of a 1D sample."""
    from BayesForge.Diagnostic.jax_diagnostics import hdi as _jd_hdi
    bounds = np.asarray(_jd_hdi(np.asarray(samples).flatten(), hdi_prob=prob))
    return float(bounds[0]), float(bounds[1])


# =============================================================================
# 1. Prior sensitivity
# =============================================================================

def prior_sensitivity_plot(
    posteriors_list,
    labels,
    param_names=None,
    hdi_prob=0.89,
    title="Prior Sensitivity Analysis",
    x_label=None,
    y_label=None,
):
    """Overlay posteriors from models fitted with different priors/assumptions.

    Compares posterior distributions across multiple model fits to assess
    how much conclusions change when priors (or likelihood assumptions) vary.

    Args:
        posteriors_list: List of posterior dicts, each {param: array(S,)}.
                         Typically [m1.posteriors, m2.posteriors, ...].
        labels: List of labels for each fit (e.g. ['weakly informative',
                'strong prior', 'alternative likelihood']).
        param_names: Subset of parameters to compare. Defaults to all
                     parameters present in all fits.
        hdi_prob: HDI probability for interval markers.
        title: Plot title.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

    Returns:
        plotly Figure with one subplot per parameter.

    Example:
        # Refit with a tighter prior on b and compare:
        m1.fit(model_weak, ...); m2.fit(model_strong, ...)
        fig = prior_sensitivity_plot(
            [m1.posteriors, m2.posteriors],
            labels=['weak prior', 'strong prior'],
        )
    """
    import scipy.stats as stats

    if param_names is None:
        all_keys = [set(p.keys()) for p in posteriors_list]
        param_names = sorted(set.intersection(*all_keys))

    n_params = len(param_names)
    if n_params == 0:
        raise ValueError("No common parameters found across posteriors.")

    ncols = min(3, n_params)
    nrows = int(np.ceil(n_params / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=param_names)

    for pi, param in enumerate(param_names):
        r = pi // ncols + 1
        c = pi % ncols + 1
        for mi, (posteriors, label) in enumerate(zip(posteriors_list, labels)):
            if param not in posteriors:
                continue
            samples = np.asarray(posteriors[param]).flatten()
            color = _COLORS[mi % len(_COLORS)]

            xs = np.linspace(samples.min(), samples.max(), 300)
            kde = stats.gaussian_kde(samples)
            fig.add_trace(go.Scatter(
                x=xs, y=kde(xs), mode="lines",
                line=dict(color=color, width=2),
                name=label, showlegend=(pi == 0),
                legendgroup=label,
                customdata=np.tile([samples.mean(), samples.std(),
                                    *_hdi(samples, hdi_prob)], (len(xs), 1)),
                hovertemplate=_HOVER,
            ), row=r, col=c)

            # HDI marker -- the actual highest-density interval, not an
            # equal-tailed percentile interval. The two differ for the skewed
            # posteriors this plot exists to compare.
            lo, hi = _hdi(samples, hdi_prob)
            y_max = float(kde(xs).max())
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[y_max * 0.05, y_max * 0.05],
                mode="lines", line=dict(color=color, width=4),
                showlegend=False, legendgroup=label,
                hoverinfo="skip",
            ), row=r, col=c)

    fig.update_layout(title=title, height=300 * nrows, plot_bgcolor="white",
                      hovermode="x unified")
    return _axis_labels(fig, x_label, y_label)


def prior_sensitivity_by_prior(posts, labels, varied, param_names=None,
                               hdi_prob=0.89, title="Prior sensitivity, by prior"):
    """One row per varied prior, one column per parameter.

    Each cell overlays the baseline posterior (grey, dashed) with the
    posteriors obtained when *that* prior alone is varied; the other priors
    stay at baseline. ``labels[0]`` must be the baseline fit.
    """
    import scipy.stats as stats
    base = posts[0]
    rows = []
    for v in varied:
        if v != "baseline" and v not in rows:
            rows.append(v)
    if param_names is None:
        param_names = sorted(set.intersection(*[set(p) for p in posts]))
    nr, nc = len(rows), len(param_names)
    fig = make_subplots(
        rows=nr, cols=nc,
        column_titles=list(param_names),
        row_titles=[f"vary {r}" for r in rows],
        vertical_spacing=0.08 / max(nr, 1) * 3)
    for ri, row in enumerate(rows, start=1):
        idx = [i for i, v in enumerate(varied) if v == row]
        for ci, param in enumerate(param_names, start=1):
            groups = [(0, "baseline", dict(color="grey", dash="dash", width=2))]
            for k, i in enumerate(idx):
                groups.append((i, labels[i],
                               dict(color=_COLORS[k % len(_COLORS)], width=2)))
            for i, lab, line in groups:
                smp = np.asarray(posts[i][param]).flatten()
                xs = np.linspace(smp.min(), smp.max(), 300)
                lo, hi = _hdi(smp, hdi_prob)
                cd = np.tile([smp.mean(), smp.std(), lo, hi], (len(xs), 1))
                fig.add_trace(go.Scatter(
                    x=xs, y=stats.gaussian_kde(smp)(xs), mode="lines",
                    line=line, name=lab, legendgroup=f"{row}|{lab}",
                    showlegend=(ci == 1 and i != 0),
                    customdata=cd, hovertemplate=_HOVER), row=ri, col=ci)
    fig.update_layout(title=title, height=260 * nr + 80, plot_bgcolor="white",
                      hovermode="x unified")
    return fig


def prior_sensitivity_shift(table, title="Shift of the posterior mean vs baseline"):
    """Dot plot of ``shift_sd`` (in baseline posterior sd) per variation.

    One panel per parameter, one line per prior variation. A dot near 0
    means the posterior ignores that prior change; |shift| > ~0.5 sd is a
    sensitive parameter. ``table`` is the DataFrame from
    ``prior_sensitivity(..., return_results=True)``.
    """
    t = table[table["label"] != "baseline"]
    params = list(dict.fromkeys(t["param"]))
    fig = make_subplots(rows=1, cols=len(params), shared_yaxes=True,
                        subplot_titles=params)
    for ci, param in enumerate(params, start=1):
        d = t[t["param"] == param]
        fig.add_trace(go.Scatter(
            x=d["shift_sd"], y=d["label"], mode="markers",
            marker=dict(size=10, color=_COLORS[0]), showlegend=False),
            row=1, col=ci)
        fig.add_vline(x=0, line_dash="dash", line_color="grey", row=1, col=ci)
        fig.update_xaxes(title_text="shift (sd)", row=1, col=ci)
    fig.update_layout(title=title, height=120 + 28 * t["label"].nunique() + 120,
                      plot_bgcolor="white")
    return fig


def _prior_scenarios(priors, mode="one_at_a_time", baseline=None, joint=None):
    """Build the scenario list for :func:`prior_sensitivity`.

    ``priors`` maps a model argument to the list of values to try, e.g.
    ``{"b_sd": [0.1, 0.5, 2.0], "a_sd": [5, 50]}``. Each scenario dict holds
    a ``label``, the ``varied`` argument name and the full ``priors`` dict
    that the model receives.

    mode="one_at_a_time": a baseline fit, then each argument varied alone
        with the others left at ``baseline`` (the model defaults when
        absent). 1 + sum(len(values)) fits. Standard practice: the shift of
        the posterior is attributable to one prior at a time.
    mode="grid": Cartesian product of every list (interactions, but the
        number of fits is the product of the list lengths).
    joint: optional {label: {arg: value, ...}} extra scenarios in which
        several priors move together (e.g. "all tighter").
    """
    import itertools
    baseline = dict(baseline or {})
    out = [dict(label="baseline", varied="baseline", priors=dict(baseline))]
    if mode == "one_at_a_time":
        for arg, values in priors.items():
            for v in values:
                out.append(dict(label=f"{arg}={v}", varied=arg,
                                priors={**baseline, arg: v}))
    elif mode == "grid":
        out = []
        keys = list(priors)
        for combo in itertools.product(*(priors[k] for k in keys)):
            d = dict(zip(keys, combo))
            out.append(dict(label=", ".join(f"{k}={v}" for k, v in d.items()),
                            varied="grid", priors={**baseline, **d}))
    else:
        raise ValueError("mode must be 'one_at_a_time' or 'grid'.")
    for label, d in (joint or {}).items():
        out.append(dict(label=label, varied="joint", priors={**baseline, **d}))
    return out


def _prior_sens_sim(label, varied, priors, seed, model, obs, fit_kwargs, **_):
    """One prior-sensitivity fit (runs inside a run_simulations worker)."""
    import functools
    import numpy as np
    from BayesForge import bf
    m = bf(rand_seed=seed, print_devices_found=False)
    bound = functools.partial(model, m, **priors)
    m.fit(bound, obs=obs, progress_bar=False, **fit_kwargs)
    post = {k: np.asarray(v).reshape(-1) for k, v in m.posteriors.items()}
    return {"posteriors": post}


def prior_sensitivity(
    model_or_posteriors,
    labels=None,
    priors=None,
    obs=None,
    mode="one_at_a_time",
    baseline=None,
    joint=None,
    fit_kwargs=None,
    workers=None,
    seed=0,
    param_names=None,
    hdi_prob=0.89,
    title="Prior Sensitivity Analysis",
    x_label=None,
    y_label=None,
    return_results=False,
    plot="by_prior",
    **run_kwargs,
):
    """Prior sensitivity: from already-fitted posteriors, or by refitting in parallel.

    Two ways to call it:

    1. **Posteriors you already have** (unchanged behaviour)::

           m.diag.prior_sensitivity([post_weak, post_strong],
                                    labels=["weak", "strong"])

    2. **Declare the prior variations and let BF run the fits in parallel**
       (through :func:`BayesForge.run_simulations`)::

           def model(m, weight, height, a_sd=20.0, b_sd=1.0):
               a = m.dist.normal(178, a_sd, name="a")
               b = m.dist.log_normal(0, b_sd, name="b")
               s = m.dist.uniform(0, 50, name="s")
               m.dist.normal(a + b * weight, s, obs=height,
                             shape=(weight.shape[0],))

           m.diag.prior_sensitivity(
               model,
               priors={"b_sd": [0.1, 0.5, 2.0], "a_sd": [5, 50]},
               obs=dict(weight=w, height=h),
           )

       The model takes the bf instance ``m`` first, then the data, then the
       prior hyper-parameters (with their baseline values as defaults). Each
       fit is a separate worker with its own bf instance and seed.

    Args:
        model_or_posteriors: list of posterior dicts (mode 1) or a model
            function ``model(m, <obs names>, <prior args>)`` (mode 2).
        labels: labels for mode 1; ignored in mode 2 (generated).
        priors: {prior arg: [values]}; each value is one variation.
        obs: dict of data passed to ``m.fit(obs=...)``.
        mode: ``'one_at_a_time'`` (default; baseline + each prior varied
            alone, 1 + sum(len) fits) or ``'grid'`` (Cartesian product).
        baseline: {prior arg: value} used when an arg is not varied. Defaults
            to the model's own defaults.
        joint: {label: {arg: value}} extra scenarios where several priors
            move together.
        fit_kwargs: extra ``m.fit`` arguments (num_samples, num_chains...).
        workers, seed, **run_kwargs: forwarded to ``run_simulations``.
        plot: ``'by_prior'`` (default for one_at_a_time: one row per varied
            prior, baseline in grey dashed + that prior's variations),
            ``'overlay'`` (every fit in one panel per parameter) or
            ``'shift'`` (dot plot of the posterior-mean shift vs baseline).
        return_results: also return the summary table (posterior mean/sd/HDI
            per scenario and shift from the baseline in posterior sd).

    Returns:
        plotly Figure (or ``(Figure, DataFrame)`` with ``return_results``).
    """
    # A sequence of posteriors, not a model. Checked first: a Julia vector
    # reaches Python as a wrapper that is both callable and a Sequence.
    from collections.abc import Sequence
    if isinstance(model_or_posteriors, Sequence) or not callable(model_or_posteriors):
        if labels is None:
            labels = [f"fit {i + 1}" for i in range(len(model_or_posteriors))]
        return prior_sensitivity_plot(
            model_or_posteriors, labels, param_names=param_names,
            hdi_prob=hdi_prob, title=title, x_label=x_label, y_label=y_label)

    if not priors and not joint:
        raise ValueError("Pass `priors={arg: [values]}` (or `joint=`).")
    import pandas as pd
    from BayesForge.Parallel.simulate import run_simulations

    scen = _prior_scenarios(priors or {}, mode, baseline, joint)
    res = run_simulations(
        _prior_sens_sim, scen, workers=workers, seed=seed,
        fn_kwargs=dict(model=model_or_posteriors, obs=obs or {},
                       fit_kwargs=dict(fit_kwargs or {})),
        **run_kwargs)
    if "error" in res and res["error"].notna().any():
        bad = res[res["error"].notna()][["label", "error"]]
        raise RuntimeError(f"prior_sensitivity: {len(bad)} fit(s) failed:\n{bad}")

    posts = list(res["posteriors"])
    lbls = list(res["label"])
    rows, base = [], posts[0] if lbls[0] == "baseline" else None
    for lab, varied, post in zip(lbls, res["varied"], posts):
        for k, v in post.items():
            if param_names and k not in param_names:
                continue
            lo, hi = _hdi(v, hdi_prob)
            row = dict(label=lab, varied=varied, param=k, mean=float(np.mean(v)),
                       sd=float(np.std(v)), hdi_low=lo, hdi_high=hi)
            if base is not None and k in base:
                row["shift_sd"] = (row["mean"] - float(np.mean(base[k]))) / (
                    float(np.std(base[k])) or np.nan)
            rows.append(row)
    table = pd.DataFrame(rows)

    if plot == "by_prior" and mode == "one_at_a_time" and base is not None:
        fig = prior_sensitivity_by_prior(posts, lbls, list(res["varied"]),
                                         param_names, hdi_prob)
    elif plot == "shift" and base is not None:
        fig = prior_sensitivity_shift(table)
    else:
        fig = prior_sensitivity_plot(posts, lbls, param_names=param_names,
                                     hdi_prob=hdi_prob, title=title,
                                     x_label=x_label, y_label=y_label)
    if not return_results:
        return fig
    return fig, table


# =============================================================================
# 2. LOO influence
# =============================================================================

def influence_plot(
    m=None,
    log_likelihood=None,
    x=None,
    y=None,
    good_k=None,
    title="LOO Influence: Pareto-k Diagnostics",
    x_label=None,
    y_label=None,
):
    """Scatter of Pareto-k values per observation.

    High k values (k > 0.7) indicate influential observations where the model
    is sensitive — potential outliers or model misspecification.

    Args:
        m: Fitted BF model. Used to auto-compute log_likelihood if not given.
        log_likelihood: Pre-computed log-likelihood array, shape (C, S, N) or
                        (S, N). If None, computed from m.model + m.data_on_model.
        x: Optional 1D array of x-axis values (e.g. predictor). Defaults to 1..N.
        y: Optional 1D array of observed values for hover text.
        good_k: Threshold for flagging bad k (default 0.7).
        title: Plot title.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

    Returns:
        plotly Figure with horizontal reference lines at k=0.5, 0.7.
    """
    from BayesForge.Diagnostic.jax_diagnostics import (
        compute_log_likelihood, _psis_weights, relative_eff
    )

    reff = 1.0
    if log_likelihood is None:
        if m is None:
            raise ValueError("Pass m or log_likelihood=.")
        log_likelihood = compute_log_likelihood(
            m.model, m.posteriors_by_chain, **m.data_on_model
        )
    if m is not None and getattr(m, "posteriors_by_chain", None):
        # PSIS sizes its Pareto tail as 3*sqrt(S/reff); leaving reff at 1 for
        # autocorrelated draws biases exactly the k values this plot thresholds.
        reff = relative_eff(m.posteriors_by_chain)

    ll_arr = np.asarray(log_likelihood)
    if ll_arr.ndim == 3:
        C, S, N = ll_arr.shape
        ll_flat = ll_arr.reshape(C * S, N)
    else:
        ll_flat, N = ll_arr, ll_arr.shape[1]

    k_vals = np.array([_psis_weights(ll_flat[:, i], reff=reff)[1] for i in range(N)])

    if x is None:
        x = np.arange(N)
    x = np.asarray(x).flatten()
    if good_k is None:
        total_S = ll_flat.shape[0]
        good_k = min(1 - 1 / max(float(np.log10(total_S)), 1.01), 0.7)

    colors = np.where(k_vals > good_k, "red", "cornflowerblue")
    hover = [f"obs {i}: k={k_vals[i]:.3f}" +
             (f", y={float(y[i]):.3f}" if y is not None else "")
             for i in range(N)]

    fig = go.Figure()
    for clr, label in [("cornflowerblue", f"k ≤ {good_k:.2f} (good)"),
                        ("red",            f"k > {good_k:.2f} (influential)")]:
        mask = (colors == clr)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=x[mask], y=k_vals[mask], mode="markers",
                marker=dict(color=clr, size=6, opacity=0.8),
                text=[hover[i] for i in np.where(mask)[0]],
                hoverinfo="text", name=label,
            ))

    fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                  annotation_text="k=0.5", annotation_position="right")
    fig.add_hline(y=good_k, line_dash="dash", line_color="red",
                  annotation_text=f"k={good_k:.2f}", annotation_position="right")
    fig.update_layout(
        title=title, xaxis_title="Observation index",
        yaxis_title="Pareto k", plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# 3. Calibration / coverage
# =============================================================================

def calibration_plot(
    y,
    yrep,
    levels=None,
    title=None,
    name="y",
    x_label=None,
    y_label=None,
):
    """Empirical coverage vs nominal credible interval levels.

    For each level α, computes the fraction of observed y values that fall
    inside the α-level predictive interval. A well-calibrated model has
    empirical ≈ nominal.

    Args:
        y: Observed data, shape (n_obs,).
        yrep: Posterior predictive matrix, shape (n_rep, n_obs).
        levels: List of nominal coverage levels. Defaults to
                [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99].
        title: Plot title. Defaults to one naming the outcome.
        name: Name of the outcome used in the labels.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

    Returns:
        plotly Figure.
    """
    if levels is None:
        levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    y, = [np.asarray(y, dtype=float).flatten()]
    yrep = np.asarray(yrep, dtype=float)
    n_obs = len(y)

    empirical = []
    for level in levels:
        tail = (1 - level) / 2
        lo = np.percentile(yrep, 100 * tail, axis=0)
        hi = np.percentile(yrep, 100 * (1 - tail), axis=0)
        coverage = float(np.mean((y >= lo) & (y <= hi)))
        empirical.append(coverage)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="black", dash="dash"),
        name="Perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=levels, y=empirical, mode="lines+markers",
        line=dict(color="cornflowerblue", width=2.5),
        marker=dict(size=8),
        name="Observed coverage",
        text=[f"{l:.0%} → {e:.1%}" for l, e in zip(levels, empirical)],
        hoverinfo="text",
    ))
    fig.update_layout(
        title=title or f"Calibration — predictions of {name}",
        xaxis_title="Nominal level of the predictive interval",
        yaxis_title=f"Share of observed {name} inside it",
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
        plot_bgcolor="white",
    )
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# 4. HMC diagnostics: energy + divergences
# =============================================================================

def divergence_energy_plot(m, title="HMC Diagnostics: Energy & Divergences",
                           x_label=None, y_label=None):
    """Plot the HMC energy diagnostic and the divergent transitions per chain.

    Top row, one panel per chain (as bayesplot's ``mcmc_nuts_energy``): the
    distribution of the energy E around its mean (marginal energy) and of its
    change between successive iterations, Delta E (energy transition). When
    the Delta E histogram is about as wide as the E histogram the sampler can
    move across the whole posterior; a much narrower Delta E histogram means
    it explores the tails badly. The panel title gives E-BFMI =
    sum(Delta E^2) / sum((E - mean E)^2); values below 0.3 are flagged.

    Bottom row: number of divergent transitions per chain (ideal: 0).

    Args:
        m: Fitted BF model with NumPyro MCMC sampler.
        title: Plot title.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

    Returns:
        plotly Figure.
    """
    # TFP / SVI samplers do not expose extra_fields; the panels below already
    # render an "unavailable" annotation, so degrade rather than raise.
    try:
        extra = m.sampler.get_extra_fields(group_by_chain=True) or {}
    except (AttributeError, TypeError):
        extra = {}

    def _per_chain(key):
        v = extra.get(key)
        if v is None:
            return None
        v = np.asarray(v)
        return v[None, :] if v.ndim == 1 else v

    energy, div = _per_chain("energy"), _per_chain("diverging")
    n_chains = next((len(v) for v in (energy, div) if v is not None), 1)

    subtitles = []
    for c in range(n_chains):
        if energy is None:
            subtitles.append(f"Chain {c}")
            continue
        e = energy[c].astype(float)
        bfmi = float(np.sum(np.diff(e) ** 2) / np.sum((e - e.mean()) ** 2))
        subtitles.append(f"Chain {c}: E-BFMI = {bfmi:.2f}" + (" ⚠" if bfmi < 0.3 else ""))
    subtitles.append("Divergent transitions per chain")
    fig = make_subplots(
        rows=2, cols=n_chains, subplot_titles=subtitles,
        specs=[[{}] * n_chains, [{"colspan": n_chains}] + [None] * (n_chains - 1)],
        vertical_spacing=0.18,
    )

    if energy is not None:
        for c in range(n_chains):
            e = energy[c].astype(float)
            for vals, label, color in [(e - e.mean(), "E − mean(E)  (marginal energy)", "cornflowerblue"),
                                       (np.diff(e), "ΔE  (energy transition)", "orange")]:
                fig.add_trace(go.Histogram(
                    x=vals, histnorm="probability density", nbinsx=40,
                    marker_color=color, opacity=0.55, name=label,
                    legendgroup=label, showlegend=(c == 0),
                ), row=1, col=c + 1)
            fig.update_xaxes(title_text="Energy", row=1, col=c + 1)
    else:
        fig.add_annotation(text="Energy data not available", showarrow=False,
                           xref="x domain", yref="y domain", x=0.5, y=0.5)

    if div is not None:
        n_div = div.sum(axis=1).astype(int)
        fig.add_trace(go.Bar(
            x=[f"Chain {c}" for c in range(n_chains)], y=n_div.tolist(),
            marker_color=["firebrick" if d else "cornflowerblue" for d in n_div],
            text=[f"{d} / {div.shape[1]}" for d in n_div], textposition="outside",
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Divergences", rangemode="tozero",
                         range=[0, max(1, int(n_div.max())) * 1.25], row=2, col=1)
        if n_div.sum() == 0:
            fig.add_annotation(text="No divergent transitions", showarrow=False,
                               row=2, col=1, x=0.5, y=0.5,
                               xref="x domain", yref="y domain",
                               font=dict(color="seagreen", size=14))
    else:
        fig.add_annotation(text="Divergence data not available", showarrow=False,
                           row=2, col=1, x=0.5, y=0.5,
                           xref="x domain", yref="y domain")

    fig.update_layout(title=title, barmode="overlay", plot_bgcolor="white",
                      height=700,
                      legend=dict(orientation="h", yanchor="top", y=-0.08))
    return _axis_labels(fig, x_label, y_label)


# =============================================================================
# 5. Multimodality detection
# =============================================================================

def multimodality_check(
    m=None,
    posteriors=None,
    param_names=None,
    n_modes=2,
    title="Multimodality Check",
    x_label=None,
    y_label=None,
):
    """Detect multiple modes in posterior marginals using KDE + local maxima.

    Flags parameters whose KDE has more than one local maximum (potential
    multimodality). Also overlays chain-separated density to show if chains
    explore different regions.

    Args:
        m: Fitted BF model. Uses m.posteriors and m.posteriors_by_chain.
        posteriors: Dict {param: array(S,)} — override m.posteriors.
        param_names: Subset of parameters to check.
        n_modes: Number of modes to highlight.
        title: Plot title.
        x_label: X-axis title (a list gives one per subplot, in order).
            Defaults to the plot's own label.
        y_label: Y-axis title, as x_label.

    Returns:
        plotly Figure, one subplot per parameter.
    """
    import scipy.stats as stats
    from scipy.signal import find_peaks

    if m is not None:
        if posteriors is None:
            posteriors = m.posteriors
        by_chain = getattr(m, 'posteriors_by_chain', None)
    else:
        by_chain = None

    if param_names is None:
        param_names = list(posteriors.keys())

    ncols = min(3, len(param_names))
    nrows = int(np.ceil(len(param_names) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=param_names)

    for pi, param in enumerate(param_names):
        r = pi // ncols + 1
        c = pi % ncols + 1

        samples = np.asarray(posteriors[param]).flatten()
        xs = np.linspace(samples.min(), samples.max(), 500)
        kde = stats.gaussian_kde(samples)
        ys = kde(xs)

        # find modes
        peaks, _ = find_peaks(ys, prominence=ys.max() * 0.1)

        # overall density
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="tozeroy",
            fillcolor="rgba(100,149,237,0.2)",
            line=dict(color="cornflowerblue", width=2),
            name=param, showlegend=False,
        ), row=r, col=c)

        # mark peaks
        for pk in peaks[:n_modes]:
            fig.add_trace(go.Scatter(
                x=[xs[pk]], y=[ys[pk]], mode="markers",
                marker=dict(color="red", size=10, symbol="triangle-up"),
                showlegend=False,
            ), row=r, col=c)

        # chains overlay if available
        if by_chain is not None and param in by_chain:
            chain_samples = np.asarray(by_chain[param])
            for ci in range(chain_samples.shape[0]):
                cs = chain_samples[ci].flatten()
                kde_c = stats.gaussian_kde(cs)
                color = _COLORS[ci % len(_COLORS)].replace("#", "")
                fig.add_trace(go.Scatter(
                    x=xs, y=kde_c(xs), mode="lines",
                    line=dict(color=_COLORS[ci % len(_COLORS)], width=1,
                               dash="dot"),
                    name=f"Chain {ci}", showlegend=(pi == 0),
                    legendgroup=f"chain{ci}",
                ), row=r, col=c)

        n_peaks = len(peaks)
        flag = " ⚠ multimodal" if n_peaks > 1 else " ✓"
        # Index this subplot's own annotation. update_annotations(selector=...)
        # rewrote EVERY annotation whose text matched, including the figure
        # title and any duplicate subplot title.
        if pi < len(fig.layout.annotations):
            fig.layout.annotations[pi].text = param + flag

    fig.update_layout(title=title, height=300 * nrows, plot_bgcolor="white")
    return _axis_labels(fig, x_label, y_label)
