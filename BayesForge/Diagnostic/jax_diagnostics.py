"""
Pure JAX replacements for ArviZ diagnostic functions.

These operate directly on posterior samples shaped as (num_chains, num_samples, ...).
All functions are JIT-compilable for maximum performance.
"""

import warnings

import numpy as np

import jax
import jax.numpy as jnp
from functools import partial


# =============================================================================
# Posterior dict filtering
# =============================================================================

def _base_name(name: str) -> str:
    """Strip index suffix: 'var[0,1]' → 'var'."""
    return name.split('[')[0]


def filter_posterior_dict(posteriors: dict, include=None, exclude=None) -> dict:
    """Filter a posterior samples dict by parameter base name.

    Matching is on base name only: 'var' matches key 'var' (and covers all
    'var[i,j,...]') but never matches 'var_1'. Index suffixes in filter names
    are stripped, so 'var[0,0]' and 'var' are equivalent filters.

    Args:
        posteriors: Dict of {name: array}.
        include: str or list of str — keep only these base names.
        exclude: str or list of str — remove these base names.

    Returns:
        Filtered dict (shallow copy of matched entries).
    """
    def _as_set(arg):
        if arg is None:
            return None
        if isinstance(arg, str):
            arg = [arg]
        return {_base_name(n) for n in arg}

    inc = _as_set(include)
    exc = _as_set(exclude)

    # Match on the base name of the *key* too, so a filter of 'var' selects a
    # dict keyed 'var[0]' / 'var[0,1]' as well as one keyed plain 'var'.
    keys = [k for k in posteriors
            if (inc is None or _base_name(k) in inc)
            and (exc is None or _base_name(k) not in exc)]
    return {k: posteriors[k] for k in keys}


# =============================================================================
# R-hat (Split R-hat, Vehtari et al. 2021)
# =============================================================================

def _rhat_core(split) -> float:
    """Gelman-Rubin R-hat on already-split, already-transformed chains.

    Args:
        split: (m, n) array of m split-chains of length n.

    Returns:
        Scalar R-hat, or nan when the within-chain variance is zero.
    """
    import numpy as _np
    split = _np.asarray(split, dtype=float)
    m, n = split.shape

    chain_means = split.mean(axis=1)
    W = _np.mean(_np.var(split, axis=1, ddof=1))
    if not _np.isfinite(W) or W == 0:
        return float("nan")
    B = n * _np.var(chain_means, ddof=1)
    var_hat = (n - 1) / n * W + B / n
    return float(_np.sqrt(var_hat / W))


def _rhat_1d(chains) -> float:
    """Rank-normalized folded-split R-hat (Vehtari et al. 2021).

    This is the estimator ``az.rhat(method="rank")`` computes, and the one the
    conventional 1.01 threshold is calibrated against. It is the maximum of

      * bulk R-hat  — split R-hat on rank-normalized draws, and
      * tail R-hat  — split R-hat on rank-normalized *folded* draws
                      ``|x - median(x)|``.

    The folded half is what detects a chain whose **variance**, rather than its
    mean, has not converged; plain split R-hat is blind to that failure mode.

    Args:
        chains: Array of shape (num_chains, num_samples).

    Returns:
        Scalar R-hat value, or NaN with fewer than 2 chains (ArviZ behaviour;
        RELEASE_AUDIT bug 3, owner decision 2026-11-13), with a UserWarning
        whose text matches ``Diag2._rhat_manual``.
    """
    import numpy as _np
    chains = _np.asarray(chains, dtype=float)
    if chains.shape[0] < 2:
        # RELEASE_AUDIT bug 3 (owner decision 2026-11-13): a single chain only
        # compares its own two halves; that is not a between-chain convergence
        # check, so return NaN like ArviZ and say why.
        warnings.warn(
            "R-hat with a single chain only compares its two halves "
            "(within-chain stationarity); run >= 2 chains to check "
            "convergence between chains.",
            UserWarning, stacklevel=2)
        return float("nan")
    split = _split_chains_np(chains)

    rhat_bulk = _rhat_core(_z_scale_np(split))
    folded = _np.abs(split - _np.median(split))
    rhat_tail = _rhat_core(_z_scale_np(folded))

    if _np.isnan(rhat_bulk) and _np.isnan(rhat_tail):
        return float("nan")
    return float(_np.nanmax([rhat_bulk, rhat_tail]))


def rhat(posterior_samples: dict, var_names=None,
         include=None, exclude=None) -> dict:
    """Compute R-hat for all parameters.

    Args:
        posterior_samples: Dict of {name: array}, shape (num_chains, num_samples, ...).
        include: str or list — keep only these base names.
        exclude: str or list — remove these base names.
        var_names: legacy alias for include.

    Returns:
        Dict of {name: R-hat value(s)}.
    """
    import numpy as np

    posterior_samples = filter_posterior_dict(
        posterior_samples, include=include or var_names, exclude=exclude)
    var_names = list(posterior_samples.keys())

    results = {}
    for name in var_names:
        samples = _as_chain_major(posterior_samples[name])
        if samples.ndim == 2:
            # Scalar parameter: (chains, samples)
            results[name] = float(_rhat_1d(samples))
        else:
            # Multi-dimensional parameter: (chains, samples, ...)
            param_shape = samples.shape[2:]
            rhat_vals = np.empty(param_shape, dtype=float)
            for idx in np.ndindex(param_shape):
                chain_slice = samples[(slice(None), slice(None)) + idx]  # (C, S)
                rhat_vals[idx] = _rhat_1d(chain_slice)
            results[name] = rhat_vals
    return results


# =============================================================================
# Effective Sample Size (ESS) — bulk and tail
# =============================================================================

def _as_chain_major(samples, group_by_chain: bool = True):
    """Coerce posterior samples to chain-major ``(C, S, ...)``.

    ``m.posteriors`` holds flat ``(N, ...)`` draws while
    ``m.posteriors_by_chain`` holds ``(C, S, ...)``. Every estimator here wants
    the latter.

    A 1-D ``(N,)`` array is unambiguous and is always promoted to ``(1, N)``.
    A 2-D array is genuinely ambiguous — ``(N, K)`` flat draws of a K-vector
    look exactly like ``(C, S)`` draws of a scalar — so the caller must say
    which it is via ``group_by_chain``; guessing from the shape is what used to
    make ``ess``/``mcse`` raise IndexError on flat input.
    """
    import numpy as _np
    arr = _np.asarray(samples)
    if arr.ndim == 1:
        return arr[None, :]
    if not group_by_chain:
        return arr[None, ...]
    return arr


def _autocov_np(ary):
    """Autocovariance for 2D array (n_chain, n_draw) at all lags, matching ArviZ."""
    import numpy as _np
    ary = _np.asarray(ary, dtype=float)
    n = ary.shape[1]
    from scipy.fft import next_fast_len
    m = next_fast_len(2 * n)
    ary = ary - ary.mean(axis=1, keepdims=True)
    fft_x = _np.fft.rfft(ary, n=m, axis=1)
    fft_x *= _np.conj(fft_x)
    acov = _np.fft.irfft(fft_x, n=m, axis=1)[:, :n]
    acov /= n
    return acov


def _split_chains_np(ary):
    """Split each chain in half and stack (ArviZ convention)."""
    import numpy as _np
    ary = _np.asarray(ary)
    half = ary.shape[1] // 2
    return _np.concatenate([ary[:, :half], ary[:, -half:]], axis=0)


def _z_scale_np(ary):
    """Rank-normalize array using Blom (1958) formula, matching ArviZ _z_scale."""
    import numpy as _np
    from scipy import stats as _stats
    ary = _np.asarray(ary, dtype=float)
    shape = ary.shape
    flat = ary.flatten()
    rank = _stats.rankdata(flat, method="average")
    c = 3.0 / 8.0
    n = len(rank)
    rank = (rank - c) / (n - 2.0 * c + 1.0)
    z = _stats.norm.ppf(rank)
    return z.reshape(shape)


def _ess_raw(ary):
    """Core ESS on pre-split, pre-transformed chains.

    Matches ArviZ _ess exactly: cross-chain pooled rho_hat, Geyer's initial
    positive sequence + initial monotone sequence, tau lower bound.

    Args:
        ary: (n_chain, n_draw) numpy array, already split and transformed.

    Returns:
        Scalar ESS float.
    """
    import numpy as _np
    ary = _np.asarray(ary, dtype=float)
    n_chain, n_draw = ary.shape
    # RELEASE_AUDIT bug 5: same guards as Diag2._ess_raw_manual (Diag2.py
    # L1341-1346) -- too few draws to estimate an autocorrelation, and the
    # constant-array special case.
    if n_draw < 4:
        return _np.nan
    # A constant array has ess == its size (matches ArviZ and Diag2).
    if (_np.max(ary) - _np.min(ary)) < _np.finfo(float).resolution:
        return float(ary.size)

    acov = _autocov_np(ary)
    chain_mean = ary.mean(axis=1)
    mean_var = _np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += _np.var(chain_mean, ddof=1)

    rho_hat_t = _np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - _np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence
    t = 1
    while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - _np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - _np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even

    # Geyer's initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    ess_total = float(n_chain * n_draw)
    tau_hat = -1.0 + 2.0 * _np.sum(rho_hat_t[:max_t + 1]) + _np.sum(rho_hat_t[max_t + 1:max_t + 2])
    tau_hat = max(tau_hat, 1.0 / _np.log10(ess_total))
    return ess_total / tau_hat


def _ess_1d(chains) -> float:
    """Bulk ESS: split chains → rank-normalize → _ess_raw (matches ArviZ _ess_bulk)."""
    import numpy as _np
    chains = _np.asarray(chains, dtype=float)
    split = _split_chains_np(chains)
    z = _z_scale_np(split)
    return float(_ess_raw(z))


def _acf(x: np.ndarray, max_lag: int = 40) -> list:
    """Autocorrelation function of a 1D chain, lags 0..max_lag-1.

    Canonical implementation, shared by ``Diag2`` (autocorr plot) and
    ``patch_diag`` (kernel density bandwidth) so the three copies cannot
    drift apart (RELEASE_AUDIT bug 15).

    Centres on the GLOBAL mean and divides by the global variance. Computing
    ``np.corrcoef(x[:-t], x[t:])`` instead re-centres and re-scales each window
    separately, which is not the ACF: it diverges at large lag and the
    resulting sequence is not positive-definite.

    Parameters
    ----------
    x : array-like
        1D chain.
    max_lag : int
        Number of lags to return (lags 0 through max_lag-1).

    Returns
    -------
    list of float
        Autocorrelation values for lags 0 through ``max_lag - 1``.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    max_lag = int(min(max_lag, n))
    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom <= 0:
        return [1.0] + [0.0] * (max_lag - 1)
    return [float(np.dot(xc[: n - t], xc[t:]) / denom) for t in range(max_lag)]


def _ebfmi_per_chain(energy) -> np.ndarray:
    """Energy Bayesian Fraction of Missing Information per chain.

    Computes ``mean(diff(E)^2) / var(E, ddof=1)`` per chain — the quantity
    Stan and ``az.bfmi`` report. Shared by ``Diag2._ebfmi_manual`` and
    ``sensitivity.divergence_energy_plot`` so the two implementations cannot
    drift apart (RELEASE_AUDIT bug 16). ``sum(diff^2)/sum((E-mean)^2)`` is
    algebraically the same ratio; a constant-energy chain yields ``nan``
    (zero variance), not a fabricated value.

    Parameters
    ----------
    energy : array-like
        Energy draws, 1D (single chain) or 2D chain-major ``(C, S)``.

    Returns
    -------
    np.ndarray
        One E-BFMI value per chain (``nan`` where the energy is constant).
    """
    e = np.atleast_2d(np.asarray(energy, dtype=float))
    result = []
    for chain in e:
        denom = np.var(chain, ddof=1)
        num = np.mean(np.diff(chain) ** 2)
        result.append(float(num / denom) if denom > 0 else np.nan)
    return np.asarray(result)


def _ess_tail_1d(chains, prob: tuple[float, float] = (0.05, 0.95)) -> float:
    """Tail ESS: min(ESS(I(x<=q_lo)), ESS(I(x<=q_hi))) (matches ArviZ _ess_tail).

    ArviZ splits the chains *before* taking the quantiles, which matters when
    ``n_draw`` is odd (splitting drops the middle draw).

    Parameters
    ----------
    chains : array-like
        Draws with shape (n_chains, n_draws).
    prob : tuple[float, float]
        Lower and upper tail probabilities, as in ArviZ's ``ess(prob=...)``.
        Default ``(0.05, 0.95)``.
    """
    import numpy as _np
    chains = _np.asarray(chains, dtype=float)
    split = _split_chains_np(chains)
    q_lo, q_hi = _np.percentile(split, [prob[0] * 100, prob[1] * 100])
    ess_low = _ess_raw((split <= q_lo).astype(float))
    ess_high = _ess_raw((split <= q_hi).astype(float))
    return float(min(ess_low, ess_high))


def _ess_mean_1d(chains) -> float:
    """ESS of the mean: split chains, no rank-normalization (ArviZ _ess_mean).

    This — not bulk ESS — is the ESS that ``mcse_mean`` divides by.
    """
    import numpy as _np
    return float(_ess_raw(_split_chains_np(_np.asarray(chains, dtype=float))))


def _ess_sd_1d(chains) -> float:
    """ESS of the sd (ArviZ _ess_sd): ESS of the squared deviations."""
    import numpy as _np
    ary = _np.asarray(chains, dtype=float)
    return float(_ess_raw(_split_chains_np((ary - ary.mean()) ** 2)))


def _mcse_sd_1d(chains) -> float:
    """MCSE of the posterior sd (ArviZ _mcse_sd).

    Delta-method form: the sd is a smooth function of the mean squared
    deviation, so its MC error follows from that mean's MC error. The old
    ``sd / sqrt(2*(ess-1))`` normal approximation was off by 20-90%.
    """
    import numpy as _np
    ary = _np.asarray(chains, dtype=float)
    if ary.size < 4:
        return float('nan')
    sims_c2 = (ary - ary.mean()) ** 2
    e = _ess_mean_1d(sims_c2)
    if not _np.isfinite(e) or e <= 0:
        return float('nan')
    evar = sims_c2.mean()
    if evar <= 0:
        return float('nan')
    varvar = ((sims_c2 ** 2).mean() - evar ** 2) / e
    return float(_np.sqrt(varvar / evar / 4))


def ess(posterior_samples: dict, var_names=None, kind="bulk",
        include=None, exclude=None, group_by_chain: bool = True,
        prob: tuple[float, float] = (0.05, 0.95)) -> dict:
    """Compute effective sample size for all parameters.

    Args:
        posterior_samples: Dict of {name: array}.
            group_by_chain=True (default): shape (chains, samples, ...).
            group_by_chain=False: flat shape (n_samples, ...).
        include: str or list — keep only these base names.
        exclude: str or list — remove these base names.
        var_names: legacy alias for include.
        kind: "bulk", "tail", "mean" or "sd".
        group_by_chain: whether the leading axis is the chain axis.
        prob: Tail probabilities for ``kind="tail"``, as in ArviZ's
            ``ess(prob=...)``. Ignored for other kinds.

    Returns:
        Dict of {name: ESS value(s)}.
    """
    import numpy as np

    posterior_samples = filter_posterior_dict(
        posterior_samples, include=include or var_names, exclude=exclude)

    try:
        ess_fn = {"bulk": _ess_1d, "tail": _ess_tail_1d,
                  "mean": _ess_mean_1d, "sd": _ess_sd_1d}[kind]
    except KeyError:
        raise ValueError(
            f"kind must be one of 'bulk', 'tail', 'mean', 'sd'; got {kind!r}")
    if kind == "tail" and tuple(prob) != (0.05, 0.95):
        ess_fn = partial(ess_fn, prob=tuple(prob))

    results = {}
    for name, samples in posterior_samples.items():
        samples = _as_chain_major(samples, group_by_chain=group_by_chain)
        if samples.ndim == 2:
            results[name] = ess_fn(samples)
        else:
            param_shape = samples.shape[2:]
            ess_vals = np.empty(param_shape, dtype=float)
            for idx in np.ndindex(param_shape):
                chain_slice = samples[(slice(None), slice(None)) + idx]
                ess_vals[idx] = ess_fn(chain_slice)
            results[name] = ess_vals
    return results


# =============================================================================
# HDI (Highest Density Interval)
# =============================================================================

def hdi(samples: jnp.ndarray, hdi_prob: float = 0.94) -> jnp.ndarray:
    """Compute the Highest Density Interval (narrowest credible interval).

    Args:
        samples: 1D array of posterior samples.
        hdi_prob: Probability mass of the HDI (default 0.94).

    Returns:
        Array of [lower, upper] bounds.
    """
    # NumPy, not JAX: JAX defaults to 32-bit, which cost ~3e-08 of accuracy on
    # the returned bounds for no benefit (this is a sort plus an argmin).
    import numpy as _np
    samples = _np.sort(_np.asarray(samples, dtype=_np.float64).ravel())
    n = samples.shape[0]
    # ArviZ uses floor here; ceil produced a systematically wider interval
    # whenever hdi_prob * n was not an integer.
    interval_size = int(_np.floor(hdi_prob * n))

    # Slide a window of size interval_size and find the narrowest
    if interval_size >= n or interval_size < 1:
        return _np.array([samples[0], samples[-1]])

    # Width of every possible interval
    widths = samples[interval_size:] - samples[:n - interval_size]
    best_idx = int(_np.argmin(widths))
    return _np.array([samples[best_idx], samples[best_idx + interval_size]])


# =============================================================================
# Summary table (replaces az-dependent summary)
# =============================================================================

def summary(posterior_samples: dict, round_to: int = 2, hdi_prob: float = 0.89,
            var_names=None, exclude_vars=None, filter_regex=None,
            include=None, exclude=None,
            group_by_chain: bool = False):
    """Compute summary statistics matching az.summary() output format.

    Args:
        posterior_samples: Dict of {name: array}.
            When group_by_chain=False (default): shape (n_samples, ...) — flat.
            When group_by_chain=True: shape (chains, samples, ...).
        round_to: Decimal places to round to.
        hdi_prob: HDI probability mass (default 0.89 matches az.summary default).
        var_names: Optional list of variable names to include.
        exclude_vars: Optional list of variable names to exclude.
        filter_regex: Optional regex string to filter variable names.
        group_by_chain: If True, input is (chains, samples, ...). Default False.

    Returns:
        pandas DataFrame matching az.summary() columns:
        mean, sd, hdi_5.5%, hdi_94.5%, mcse_mean, mcse_sd, ess_bulk, ess_tail, r_hat
    """
    import pandas as pd
    import numpy as np
    import re as _re

    # include/exclude take priority; var_names/exclude_vars are legacy aliases
    _inc = include if include is not None else var_names
    _exc = exclude if exclude is not None else exclude_vars
    posterior_samples = filter_posterior_dict(posterior_samples,
                                              include=_inc, exclude=_exc)

    vars_to_process = list(posterior_samples.keys())

    if filter_regex is not None:
        vars_to_process = [v for v in vars_to_process if _re.search(filter_regex, v)]

    # Column names match ArviZ convention: hdi_{lo:.1f}% / hdi_{hi:.1f}%
    lo_pct = (1 - hdi_prob) / 2 * 100   # 5.5 for hdi_prob=0.89
    hi_pct = 100 - lo_pct                # 94.5
    hdi_lo_col = f'hdi_{lo_pct:.1f}%'
    hdi_hi_col = f'hdi_{hi_pct:.1f}%'

    def _stats_for(chains_2d):
        """All summary columns for one scalar parameter, chains (C, S)."""
        all_samples = np.asarray(chains_2d, dtype=float).flatten()

        mean_val = float(np.mean(all_samples))
        # ArviZ reports the sample sd with ddof=1.
        sd_val = float(np.std(all_samples, ddof=1)) if all_samples.size > 1 else float('nan')
        hdi_vals = hdi(all_samples, hdi_prob=hdi_prob)

        # Split R-hat is well defined for a single chain (that is the point of
        # splitting), so it is reported rather than forced to nan.
        rhat_val = _rhat_1d(chains_2d)
        ess_val = _ess_1d(chains_2d)
        ess_tail_val = _ess_tail_1d(chains_2d)

        # mcse_mean divides by ess_mean (split, NOT rank-normalized), not by
        # ess_bulk; mcse_sd uses ArviZ's exact factor, not the normal approx.
        ess_mean_val = _ess_mean_1d(chains_2d)
        mcse_mean_val = (sd_val / ess_mean_val ** 0.5
                         if ess_mean_val > 0 else float('nan'))
        mcse_sd_val = _mcse_sd_1d(chains_2d)

        return {
            'mean': mean_val, 'sd': sd_val,
            hdi_lo_col: float(hdi_vals[0]),
            hdi_hi_col: float(hdi_vals[1]),
            'mcse_mean': mcse_mean_val,
            'mcse_sd': mcse_sd_val,
            'ess_bulk': ess_val,
            'ess_tail': ess_tail_val,
            'r_hat': rhat_val,
        }

    summary_stats = {}

    for var_name in vars_to_process:
        samples = _as_chain_major(posterior_samples[var_name],
                                  group_by_chain=group_by_chain)

        # samples is now (C, S, ...)
        param_shape = samples.shape[2:]

        if not param_shape:
            summary_stats[var_name] = _stats_for(samples)
        else:
            for idx in np.ndindex(param_shape):
                idx_str = "[" + ", ".join(map(str, idx)) + "]"
                full_name = f"{var_name}{idx_str}"
                element_samples = samples[(slice(None), slice(None)) + idx]
                summary_stats[full_name] = _stats_for(element_samples)

    return pd.DataFrame(summary_stats).T.round(round_to)


# =============================================================================
# MCSE (Monte Carlo Standard Error)
# =============================================================================

def mcse(posterior_samples: dict, var_names=None,
         include=None, exclude=None, kind: str = "mean",
         group_by_chain: bool = True) -> dict:
    """Compute Monte Carlo Standard Error for all parameters.

    Matches ``az.mcse``:

      * ``kind="mean"`` — ``sd / sqrt(ess_mean)``, where ess_mean is the split
        (NOT rank-normalized) ESS. Dividing by ess_bulk, as this used to, gives
        a materially different number for skewed posteriors.
      * ``kind="sd"``  — ``sd * sqrt(e * (1 - 1/ess_sd)**(ess_sd - 1) - 1)``.

    Args:
        posterior_samples: Dict of {name: array}.
            group_by_chain=True (default): shape (chains, samples, ...).
            group_by_chain=False: flat shape (n_samples, ...).
        include: str or list — keep only these base names.
        exclude: str or list — remove these base names.
        var_names: legacy alias for include.
        kind: "mean" (default) or "sd".
        group_by_chain: whether the leading axis is the chain axis.

    Returns:
        Dict of {name: MCSE value(s)}.
    """
    import numpy as np

    if kind not in ("mean", "sd"):
        raise ValueError(f"kind must be 'mean' or 'sd'; got {kind!r}")

    posterior_samples = filter_posterior_dict(
        posterior_samples, include=include or var_names, exclude=exclude)

    def _mcse_1d(chains_2d):
        flat = np.asarray(chains_2d, dtype=float).flatten()
        if flat.size < 2:
            return float('nan')
        sd_val = float(np.std(flat, ddof=1))
        if kind == "mean":
            e = _ess_mean_1d(chains_2d)
            return sd_val / e ** 0.5 if e > 0 else float('nan')
        return _mcse_sd_1d(chains_2d)

    results = {}
    for name, samples in posterior_samples.items():
        samples = _as_chain_major(samples, group_by_chain=group_by_chain)
        if samples.ndim == 2:
            results[name] = _mcse_1d(samples)
        else:
            param_shape = samples.shape[2:]
            vals = np.empty(param_shape, dtype=float)
            for idx in np.ndindex(param_shape):
                vals[idx] = _mcse_1d(samples[(slice(None), slice(None)) + idx])
            results[name] = vals
    return results


# =============================================================================
# PSIS (Pareto Smoothed Importance Sampling)
# Reference: Vehtari, Gelman, Gabry (2017). arxiv.org/abs/1507.02646
# =============================================================================

def _gpdfit(ary):
    """Fit GPD via empirical Bayes (Zhang & Stephens 2009), matching ArviZ exactly.

    ary: sorted 1D positive array (tail exceedances above cutoff).
    Returns (k, sigma).
    """
    import numpy as np
    prior_bs = 3
    prior_k = 10
    n = len(ary)
    m_est = 30 + int(n ** 0.5)

    b_ary = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1, dtype=float) - 0.5))
    b_ary /= prior_bs * ary[int(n / 4 + 0.5) - 1]
    b_ary += 1 / ary[-1]

    k_ary = np.log1p(-b_ary[:, None] * ary).mean(axis=1)
    len_scale = n * (np.log(-(b_ary / k_ary)) - k_ary - 1)
    weights = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)

    real_idxs = weights >= 10 * np.finfo(float).eps
    if not np.all(real_idxs):
        weights = weights[real_idxs]
        b_ary = b_ary[real_idxs]
    weights /= weights.sum()

    b_post = np.sum(b_ary * weights)
    k_post = np.log1p(-b_post * ary).mean()
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return float(k_post), float(sigma)


def _gpinv(probs, k, sigma):
    """Inverse GPD quantile function, matching ArviZ."""
    import numpy as np
    x = np.full_like(probs, np.nan)
    if sigma <= 0:
        return x
    ok = (probs > 0) & (probs < 1)
    if np.all(ok):
        if np.abs(k) < np.finfo(float).eps:
            x = -np.log1p(-probs)
        else:
            x = np.expm1(-k * np.log1p(-probs)) / k
        x *= sigma
    else:
        if np.abs(k) < np.finfo(float).eps:
            x[ok] = -np.log1p(-probs[ok])
        else:
            x[ok] = np.expm1(-k * np.log1p(-probs[ok])) / k
        x[ok] *= sigma
    return x


def _psis_weights(log_likelihood_i: jnp.ndarray, reff: float = 1.0) -> tuple:
    """PSIS-smoothed log weights for a single data point, matching ArviZ's _psislw.

    Args:
        log_likelihood_i: shape (S,) — log p(y_i | theta_s) for all draws.
        reff: relative MCMC efficiency, ESS / S. It sets how many draws enter
            the generalized-Pareto tail fit: ``3 * sqrt(S / reff)``. Leaving it
            at 1.0 for autocorrelated draws fits the tail on too many points
            and biases the returned Pareto k.

    Returns:
        (log_weights_normalized, pareto_k)
    """
    import numpy as np
    from scipy.special import logsumexp as _sp_logsumexp

    x = -np.asarray(log_likelihood_i, dtype=np.float64)  # log importance ratios
    S = len(x)

    max_x = np.max(x)
    x -= max_x

    reff = float(reff) if reff and np.isfinite(reff) and reff > 0 else 1.0
    # Tail length as in arviz-stats >= 1.3 (``_get_ps_tails``, right tail):
    # floor(3 * sqrt(S / reff)) when S * reff > 225, else floor(S / 5).
    # The older ceil(min(S/5, 3 sqrt(S/reff))) rule (ArviZ 0.x) kept one more
    # draw and shifted every Pareto k-hat by up to ~1e-2.
    if S * reff > 225:
        n_tail = int(np.floor(3 * (S / reff) ** 0.5))
    else:
        n_tail = int(np.floor(S / 5.0))
    cutoff_ind = -n_tail - 1
    cutoffmin = np.log(np.finfo(float).tiny)

    x_sort_ind = np.argsort(x)
    xcutoff = max(x[x_sort_ind[cutoff_ind]], cutoffmin)
    expxcutoff = np.exp(xcutoff)

    (tailinds,) = np.where(x > xcutoff)
    x_tail = x[tailinds]
    tail_len = len(x_tail)

    k = np.inf
    if tail_len > 4:
        x_tail_si = np.argsort(x_tail)
        x_tail_exc = np.exp(x_tail) - expxcutoff
        k, sigma = _gpdfit(x_tail_exc[x_tail_si])

        if np.isfinite(k):
            sti = np.arange(0.5, tail_len) / tail_len
            smoothed_tail = _gpinv(sti, k, sigma)
            smoothed_tail = np.log(smoothed_tail + expxcutoff)
            x[tailinds[x_tail_si]] = smoothed_tail
            x[x > 0] = 0

    x -= _sp_logsumexp(x)
    return jnp.array(x), float(k)


# =============================================================================
# LOO-CV (PSIS-LOO)
# =============================================================================

class ELPDData:
    """Container for ELPD results, similar to ArviZ ELPDData."""

    def __init__(self, kind, elpd, se, p, n_samples, n_data_points,
                 pointwise_elpd, pareto_k, scale, warning):
        self.kind = kind
        if kind == "loo":
            self.elpd_loo = elpd
            self.p_loo = p
        else:
            self.elpd_waic = elpd
            self.p_waic = p
        self.elpd = elpd
        self.se = se
        self.p = p
        self.n_samples = n_samples
        self.n_data_points = n_data_points
        self.pointwise_elpd = pointwise_elpd
        self.pareto_k = pareto_k
        self.scale = scale
        self.warning = warning
        import math
        self.good_k = (min(1 - 1 / max(math.log10(n_samples), 1.01), 0.7)
                       if n_samples > 10 else 0.7)

    def __repr__(self):
        kind_upper = self.kind.upper()
        lines = [
            f"Computed from {self.n_samples} posterior samples and "
            f"{self.n_data_points} observations (log scale).",
            "",
            f"         Estimate    SE",
            f"elpd_{self.kind:4s}  {self.elpd:10.2f}  {self.se:.2f}",
            f"p_{self.kind:4s}     {self.p:10.2f}        -",
            "",
        ]
        if self.pareto_k is not None:
            import numpy as _np
            n_bad = int(_np.sum(_np.asarray(self.pareto_k) > self.good_k))
            if n_bad > 0:
                lines.append(
                    f"WARNING: {n_bad} Pareto k values > {self.good_k:.2f}. "
                    f"LOO estimate may be unreliable."
                )
            else:
                lines.append(
                    f"All Pareto k estimates OK (k < {self.good_k:.2f})."
                )
        return "\n".join(lines)


def relative_eff(posterior_samples: dict) -> float:
    """Relative MCMC efficiency ``reff = mean(ess_mean) / n_samples``.

    This is what ``az.loo`` computes to size the PSIS tail, and it is derived
    from the **posterior**, not from the log-likelihood. Pass the result to
    :func:`loo` as ``reff=``; without it ``loo`` assumes 1.0 (the value ArviZ
    uses for a single chain), which for autocorrelated draws fits the
    generalized-Pareto tail on too many points and biases the reported k.

    Args:
        posterior_samples: Dict of {name: array} shaped (chains, samples, ...).

    Returns:
        Scalar reff in (0, 1]; 1.0 when there is no usable chain structure.
    """
    import numpy as np

    effs = []
    n_samples = None
    for arr in posterior_samples.values():
        arr = _as_chain_major(arr)
        if arr.ndim < 2 or arr.shape[0] < 2:
            continue
        C, S = arr.shape[0], arr.shape[1]
        if C * S < 4:
            continue
        n_samples = C * S
        flat = arr.reshape(C, S, -1)
        for i in range(flat.shape[2]):
            effs.append(_ess_mean_1d(flat[:, :, i]))

    if not effs or not n_samples:
        return 1.0
    reff = float(np.nanmean(effs) / n_samples)
    return reff if np.isfinite(reff) and reff > 0 else 1.0


def loo(log_likelihood: jnp.ndarray, pointwise: bool = False,
        scale: str = "log", reff: float = None) -> ELPDData:
    """Compute PSIS-LOO-CV from pointwise log-likelihood values.

    Implements Pareto-smoothed importance sampling leave-one-out
    cross-validation (Vehtari et al., 2017).

    Args:
        log_likelihood: Array of shape (chains, samples, n_data_points)
            or (total_samples, n_data_points) containing pointwise
            log-likelihood values log p(y_i | theta_s).
        pointwise: If True, include pointwise values in the result.
        scale: "log" (default), "negative_log", or "deviance".
        reff: Relative MCMC efficiency (ESS / n_samples) used to size the PSIS
            tail. Defaults to 1.0, matching ArviZ's single-chain case. Compute
            it from the posterior with :func:`relative_eff` and pass it here to
            reproduce ``az.loo`` exactly on autocorrelated draws.

    Returns:
        ELPDData object with elpd_loo, p_loo, SE, etc.
    """
    if reff is None:
        reff = 1.0

    # Flatten chains if needed: (C, S, N) -> (C*S, N)
    if log_likelihood.ndim == 3:
        C, S, N = log_likelihood.shape
        ll_flat = log_likelihood.reshape(C * S, N)
    elif log_likelihood.ndim == 2:
        ll_flat = log_likelihood
    else:
        raise ValueError(
            f"log_likelihood must be 2D or 3D, got shape {log_likelihood.shape}")

    total_S, N = ll_flat.shape

    # float64 throughout; see the note in waic().
    import math
    import numpy as _np
    from scipy.special import logsumexp as _sp_logsumexp
    ll_flat = _np.asarray(ll_flat, dtype=_np.float64)

    # Compute PSIS weights and LOO-elpd for each data point
    loo_lppd_i = _np.zeros(N, dtype=_np.float64)
    pareto_k = _np.zeros(N, dtype=_np.float64)

    for i in range(N):
        log_w, k_i = _psis_weights(ll_flat[:, i], reff=reff)
        # LOO log predictive density for point i:
        # log( sum_s w_s * p(y_i | theta_s) ) where w_s are normalized PSIS weights
        # = logsumexp(log_w + log_lik_i)
        loo_lppd_i[i] = _sp_logsumexp(
            _np.asarray(log_w, dtype=_np.float64) + ll_flat[:, i])
        pareto_k[i] = k_i

    # elpd_loo = sum of pointwise loo_lppd
    elpd_loo = _np.sum(loo_lppd_i)

    # p_loo (effective number of parameters)
    # = lppd - elpd_loo, where lppd = sum_i log(mean_s p(y_i|theta_s))
    lppd_i = _sp_logsumexp(ll_flat, axis=0) - _np.log(total_S)
    lppd = _np.sum(lppd_i)
    p_loo = lppd - elpd_loo

    # Standard error
    se = _np.sqrt(N * _np.var(loo_lppd_i))

    # Warning
    good_k = min(1 - 1 / max(math.log10(total_S), 1.01), 0.7)
    has_warning = bool(_np.any(pareto_k > good_k))

    # Scale
    elpd_out = float(elpd_loo)
    if scale == "negative_log":
        elpd_out = -elpd_out
    elif scale == "deviance":
        elpd_out = -2 * elpd_out

    return ELPDData(
        kind="loo",
        elpd=elpd_out,
        se=float(se),
        p=float(p_loo),
        n_samples=int(total_S),
        n_data_points=int(N),
        pointwise_elpd=loo_lppd_i if pointwise else None,
        pareto_k=pareto_k,
        scale=scale,
        warning=has_warning,
    )


# =============================================================================
# WAIC (Widely Applicable Information Criterion)
# =============================================================================

def waic(log_likelihood: jnp.ndarray, pointwise: bool = False,
         scale: str = "log") -> ELPDData:
    """Compute WAIC from pointwise log-likelihood values.

    The Widely Applicable Information Criterion (Watanabe, 2010)
    estimates the out-of-sample predictive accuracy using the
    computed log pointwise predictive density and a correction
    for the effective number of parameters.

    Args:
        log_likelihood: Array of shape (chains, samples, n_data_points)
            or (total_samples, n_data_points) containing pointwise
            log-likelihood values log p(y_i | theta_s).
        pointwise: If True, include pointwise values in result.
        scale: "log" (default), "negative_log", or "deviance".

    Returns:
        ELPDData object with elpd_waic, p_waic, SE, etc.
    """
    # Flatten chains if needed
    if log_likelihood.ndim == 3:
        C, S, N = log_likelihood.shape
        ll_flat = log_likelihood.reshape(C * S, N)
    elif log_likelihood.ndim == 2:
        ll_flat = log_likelihood
    else:
        raise ValueError(
            f"log_likelihood must be 2D or 3D, got shape {log_likelihood.shape}")

    total_S, N = ll_flat.shape

    # float64 throughout: JAX defaults to 32-bit, which cost ~2e-6 of relative
    # accuracy on the SE compared with az.waic.
    import numpy as _np
    from scipy.special import logsumexp as _sp_logsumexp
    ll_flat = _np.asarray(ll_flat, dtype=_np.float64)

    # lppd = sum_i log( mean_s( p(y_i | theta_s) ) )
    # In log space: logsumexp over samples - log(S)
    lppd_i = _sp_logsumexp(ll_flat, axis=0) - _np.log(total_S)

    # p_waic = sum_i var_s( log p(y_i | theta_s) )
    # Variance of log-lik across posterior samples, for each data point.
    # ddof=0, matching ArviZ (xarray .var() default).
    p_waic_i = _np.var(ll_flat, axis=0)

    # elpd_waic = lppd - p_waic (pointwise)
    elpd_waic_i = lppd_i - p_waic_i
    elpd_waic = _np.sum(elpd_waic_i)
    p_waic = _np.sum(p_waic_i)

    # Standard error
    se = _np.sqrt(N * _np.var(elpd_waic_i))

    # Warning: if any p_waic_i > 0.4
    has_warning = bool(_np.any(p_waic_i > 0.4))

    # Scale
    elpd_out = float(elpd_waic)
    if scale == "negative_log":
        elpd_out = -elpd_out
    elif scale == "deviance":
        elpd_out = -2 * elpd_out

    return ELPDData(
        kind="waic",
        elpd=elpd_out,
        se=float(se),
        p=float(p_waic),
        n_samples=int(total_S),
        n_data_points=int(N),
        pointwise_elpd=elpd_waic_i if pointwise else None,
        pareto_k=None,
        scale=scale,
        warning=has_warning,
    )


# =============================================================================
# Model comparison (compare)
# =============================================================================

def compare(compare_dict: dict, ic: str = "loo", method: str = "stacking",
            scale: str = "log") -> "pd.DataFrame":
    """Compare models based on ELPD (LOO or WAIC).

    Args:
        compare_dict: Dict of {model_name: log_likelihood_array}.
            Each array has shape (chains, samples, n_data) or
            (total_samples, n_data).
        ic: "loo" or "waic".
        method: Weight estimation method — "stacking" (recommended)
            or "pseudo-BMA".
        scale: "log", "negative_log", or "deviance".

    Returns:
        pandas DataFrame ordered from best to worst model with columns:
        rank, elpd, pIC, elpd_diff, weight, SE, dSE, warning.
    """
    import pandas as pd
    import numpy as np

    ic_fn = loo if ic == "loo" else waic

    # Compute IC for all models
    results = {}
    pointwise_elpds = {}
    for name, ll in compare_dict.items():
        res = ic_fn(ll, pointwise=True, scale="log")  # always compute in log first
        results[name] = res
        pointwise_elpds[name] = np.array(res.pointwise_elpd)

    model_names = list(results.keys())
    K = len(model_names)
    N = results[model_names[0]].n_data_points

    sizes = {m: pointwise_elpds[m].shape[0] for m in model_names}
    if len(set(sizes.values())) > 1:
        raise ValueError(
            "All models must be compared on the same observations; got "
            f"differing n_data_points: {sizes}")

    # Stack pointwise elpds: (K, N)
    elpd_matrix = np.stack([pointwise_elpds[m] for m in model_names])

    # Compute weights
    if method == "stacking":
        weights = _stacking_weights(elpd_matrix)
    elif method == "pseudo-BMA":
        weights = _pseudo_bma_weights(elpd_matrix)
    else:
        weights = np.ones(K) / K

    # Build comparison table
    rows = []
    elpd_values = np.array([float(results[m].elpd) for m in model_names])
    ranks = np.argsort(-elpd_values)  # higher elpd = better

    # Reference is the best model
    best_idx = ranks[0]
    best_elpd_pointwise = elpd_matrix[best_idx]

    for idx, name in enumerate(model_names):
        res = results[name]
        # ArviZ convention: elpd_diff = best - model, so it is 0 for the
        # top-ranked model and POSITIVE for worse ones. The reverse sign made
        # this table disagree with az.compare and with az.plot_compare.
        diff_pointwise = best_elpd_pointwise - elpd_matrix[idx]
        elpd_diff = np.sum(diff_pointwise)
        dse = np.sqrt(N * np.var(diff_pointwise)) if idx != best_idx else 0.0

        rows.append({
            'rank': int(np.where(ranks == idx)[0][0]),
            f'elpd_{ic}': float(res.elpd),
            f'p_{ic}': float(res.p),
            'elpd_diff': float(elpd_diff),
            'weight': float(weights[idx]),
            'SE': float(res.se),
            'dSE': float(dse),
            'warning': res.warning,
        })

    df = pd.DataFrame(rows, index=model_names)
    df = df.sort_values('rank')

    # Apply scale
    if scale != "log":
        multiplier = -1.0 if scale == "negative_log" else -2.0
        for col in [f'elpd_{ic}', 'elpd_diff']:
            df[col] = df[col] * multiplier

    return df


def _stacking_weights(elpd_matrix: "np.ndarray") -> "np.ndarray":
    """Compute stacking weights by optimizing the combined LOO predictive
    density (Yao et al., 2018).

    This solves:  max_w  sum_i log( sum_k w_k * exp(elpd_k_i) )
                  s.t.   w >= 0, sum(w) = 1

    Uses scipy's minimize with SLSQP.

    Args:
        elpd_matrix: (K, N) array of pointwise elpds.

    Returns:
        Array of shape (K,) with stacking weights.
    """
    import numpy as np
    from scipy.optimize import minimize

    K, N = elpd_matrix.shape

    if K == 1:
        return np.array([1.0])

    # Normalize per data point for numerical stability
    elpd_max = elpd_matrix.max(axis=0, keepdims=True)
    exp_elpd = np.exp(elpd_matrix - elpd_max)  # (K, N)

    def neg_log_score(w):
        # Weighted combination: for each data point i, compute sum_k w_k * exp_elpd_ki
        combined = w @ exp_elpd  # (N,)
        combined = np.clip(combined, 1e-30, None)
        return -np.sum(np.log(combined))

    def neg_log_score_grad(w):
        combined = w @ exp_elpd  # (N,)
        combined = np.clip(combined, 1e-30, None)
        # grad_k = -sum_i exp_elpd_ki / combined_i
        grad = -np.sum(exp_elpd / combined[np.newaxis, :], axis=1)
        return grad

    # Initial weights: uniform
    w0 = np.ones(K) / K
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * K

    result = minimize(neg_log_score, w0, jac=neg_log_score_grad,
                      method='SLSQP', constraints=constraints, bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-10})

    weights = result.x
    weights = np.clip(weights, 0, None)
    weights = weights / weights.sum()
    return weights


def _pseudo_bma_weights(elpd_matrix: "np.ndarray") -> "np.ndarray":
    """Compute pseudo-BMA weights (Akaike-type weighting).

    w_k = exp(elpd_k) / sum_k exp(elpd_k)
    where elpd_k = sum_i elpd_k_i

    Args:
        elpd_matrix: (K, N) array of pointwise elpds.

    Returns:
        Array of shape (K,) with pseudo-BMA weights.
    """
    import numpy as np

    elpd_totals = elpd_matrix.sum(axis=1)  # (K,)
    elpd_max = elpd_totals.max()
    weights = np.exp(elpd_totals - elpd_max)
    weights = weights / weights.sum()
    return weights


# =============================================================================
# Log-likelihood extraction from any NumPyro model (post-fitting)
# =============================================================================

def compute_log_likelihood(model, posterior_samples, *model_args,
                           obs_name=None, has_chains=None, **model_kwargs):
    """Compute pointwise log-likelihood by replaying a fitted model.

    This works with ANY NumPyro model — no need to modify the model
    definition. It traces the model with each posterior sample and
    extracts the log-probability of the observed site(s).

    Args:
        model: The NumPyro model function (same one passed to MCMC).
        posterior_samples: Dict of {name: array} with shape
            (num_chains, num_samples, ...) or (num_samples, ...).
            Typically from sampler.get_samples(group_by_chain=True).
        *model_args: Positional arguments passed to the model
            (same as during fitting).
        obs_name: Name(s) of the observed sample site(s). If None,
            auto-detects all sites that have obs= set.
            Can be a string or list of strings.
        has_chains: Whether the leading axis of posterior_samples is the
            chain axis (RELEASE_AUDIT bug 6). True = (chains, samples, ...),
            False = flat (samples, ...). If None (default), a heuristic
            guesses and warns when it decides the leading axis is chains.
        **model_kwargs: Keyword arguments passed to the model
            (same as during fitting).

    Returns:
        If obs_name is a single string or auto-detected as one site:
            jnp.ndarray of shape (num_chains, num_samples, n_data_points)
        If multiple observed sites:
            dict of {site_name: jnp.ndarray} each shaped
            (num_chains, num_samples, n_data_points)

    Example:
        # After fitting:
        mcmc.run(rng_key, X, y)
        posterior = mcmc.get_samples(group_by_chain=True)

        # Compute log-likelihood (pass same args as model):
        ll = compute_log_likelihood(my_model, posterior, X, y)

        # Now use it:
        result = loo(ll)
        print(result)
    """
    import numpyro
    from numpyro import handlers
    import numpyro.distributions as dist

    # --- Flatten chains if present ---
    first_key = list(posterior_samples.keys())[0]
    first_val = posterior_samples[first_key]
    num_chains, num_samples = 1, first_val.shape[0]

    if has_chains is None:
        # Heuristic: if first dim is small (1-20) and second dim is large,
        # it's likely (chains, samples, ...). But we also check if all params
        # share the same first two dims.
        has_chains = False
        if first_val.ndim >= 2 and first_val.shape[0] <= 20:
            # Check consistency across all parameters
            shapes_consistent = all(
                v.ndim >= 2 and v.shape[0] == first_val.shape[0]
                for v in posterior_samples.values()
            )
            if shapes_consistent:
                has_chains = True
                # RELEASE_AUDIT bug 6: a flat posterior with <= 20 draws is
                # silently misread as chain-major; make the guess explicit.
                warnings.warn(
                    "compute_log_likelihood: guessed that the leading axis "
                    "of the posterior samples is the chain axis (size %d "
                    "<= 20); pass has_chains=True/False to make it explicit."
                    % first_val.shape[0], UserWarning, stacklevel=2)
    if has_chains:
        num_chains = first_val.shape[0]
        num_samples = first_val.shape[1]

    # Flatten to (total_samples, ...) for tracing
    flat_samples = {}
    if has_chains:
        for k, v in posterior_samples.items():
            flat_samples[k] = v.reshape((-1,) + v.shape[2:])
    else:
        flat_samples = dict(posterior_samples)

    total_samples = flat_samples[first_key].shape[0]

    # --- Discover observed sites if obs_name not given ---
    if obs_name is None:
        # Trace the model once to find observed sites
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
        obs_sites = [
            name for name, site in trace.items()
            if site['type'] == 'sample'
            and site.get('is_observed', False)
        ]
        if not obs_sites:
            raise ValueError(
                "No observed sites found in model trace. "
                "Pass obs_name explicitly."
            )
    elif isinstance(obs_name, str):
        obs_sites = [obs_name]
    else:
        obs_sites = list(obs_name)

    # --- Compute log-likelihood for each posterior sample ---
    # We use numpyro's log_likelihood utility which is vectorized
    from numpyro.infer import log_likelihood as numpyro_log_likelihood

    log_liks = numpyro_log_likelihood(
        model, flat_samples, *model_args, **model_kwargs
    )
    # log_liks is dict of {site_name: (total_samples, n_data_points)}

    # Filter to requested sites
    log_liks = {k: v for k, v in log_liks.items() if k in obs_sites}

    # Reshape back to (chains, samples, n_data_points) if needed
    if has_chains:
        log_liks = {
            k: v.reshape(num_chains, num_samples, *v.shape[1:])
            for k, v in log_liks.items()
        }

    # Return single array if only one observed site
    if len(log_liks) == 1:
        return list(log_liks.values())[0]
    return log_liks


# =============================================================================
# iter_expanded — expand parameters to (label, 1D_samples) pairs
# Used by patch_diag.py for plotting and filtering
# =============================================================================

def iter_expanded(posterior_samples: dict, var_names=None, exclude_vars=None,
                  filter_regex=None):
    """Iterate over parameters, expanding multi-dimensional arrays into scalar entries.

    For a scalar param 'a' with shape (N,), yields ('a', flat_1D_array).
    For a vector param 'mu' with shape (N, K), yields ('mu[0]', ...), ('mu[K-1]', ...).

    Supports flat input (N, ...) from m.posteriors (group_by_chain=False).

    Args:
        posterior_samples: Dict of {name: array} with shape (N, ...) or (C, S, ...).
        var_names: If given, only include variables whose expanded label matches.
        exclude_vars: If given, exclude variables whose expanded label matches.
        filter_regex: If given, include only labels matching this regex.

    Yields:
        (label, 1D_samples) tuples where 1D_samples is a flat numpy array.
    """
    import numpy as np
    import re as _re

    if isinstance(var_names, str):
        var_names = [var_names]
    if isinstance(exclude_vars, str):
        exclude_vars = [exclude_vars]

    for name, samples in posterior_samples.items():
        samples = np.asarray(samples)

        # Flatten chain dim if chain-structured (C, S, ...) → (N, ...)
        if samples.ndim >= 3:
            C, S = samples.shape[0], samples.shape[1]
            samples = samples.reshape(C * S, *samples.shape[2:])

        # Determine parameter shape (dims beyond the sample dim)
        if samples.ndim == 1:
            param_shape = ()
        else:
            param_shape = samples.shape[1:]

        if not param_shape:
            entries = [(name, samples.flatten())]
        else:
            entries = []
            for idx in np.ndindex(param_shape):
                label = f"{name}[{', '.join(map(str, idx))}]"
                element = samples[(slice(None),) + idx].flatten()
                entries.append((label, element))

        for label, flat_samples in entries:
            # Filtering: match against full label or base name
            base = label.split('[')[0]
            if var_names is not None:
                if not any(label == v or base == v or label.startswith(v + '[')
                           for v in var_names):
                    continue
            if exclude_vars is not None:
                if any(label == v or base == v or label.startswith(v + '[')
                       for v in exclude_vars):
                    continue
            if filter_regex is not None and not _re.search(filter_regex, label):
                continue
            yield label, flat_samples


def compute_log_likelihood_manual(model, posterior_samples, *model_args,
                                  obs_name=None, batch_size=100,
                                  has_chains=None, **model_kwargs):
    """Compute pointwise log-likelihood sample by sample.

    Slower fallback for models where numpyro.infer.log_likelihood
    doesn't work (e.g. complex control flow, custom distributions).
    Traces the model once per posterior sample.

    Args:
        model: The NumPyro model function.
        posterior_samples: Dict with shape (chains, samples, ...) or
            (samples, ...).
        *model_args: Args passed to model.
        obs_name: Observed site name(s). Auto-detected if None.
        batch_size: Process this many samples at a time (memory control).
        has_chains: Whether the leading axis of posterior_samples is the
            chain axis (RELEASE_AUDIT bug 6). True = (chains, samples, ...),
            False = flat (samples, ...). If None (default), a heuristic
            guesses and warns when it decides the leading axis is chains.
        **model_kwargs: Kwargs passed to model.

    Returns:
        Same as compute_log_likelihood.
    """
    from numpyro import handlers

    # Flatten chains
    first_key = list(posterior_samples.keys())[0]
    first_val = posterior_samples[first_key]
    num_chains = 1

    if has_chains is None:
        has_chains = False
        if first_val.ndim >= 2 and first_val.shape[0] <= 20:
            shapes_consistent = all(
                v.ndim >= 2 and v.shape[0] == first_val.shape[0]
                for v in posterior_samples.values()
            )
            if shapes_consistent:
                has_chains = True
                # RELEASE_AUDIT bug 6: a flat posterior with <= 20 draws is
                # silently misread as chain-major; make the guess explicit.
                warnings.warn(
                    "compute_log_likelihood_manual: guessed that the leading "
                    "axis of the posterior samples is the chain axis (size "
                    "%d <= 20); pass has_chains=True/False to make it "
                    "explicit." % first_val.shape[0],
                    UserWarning, stacklevel=2)
    if has_chains:
        num_chains = first_val.shape[0]

    flat_samples = {}
    if has_chains:
        for k, v in posterior_samples.items():
            flat_samples[k] = v.reshape((-1,) + v.shape[2:])
    else:
        flat_samples = dict(posterior_samples)

    total_samples = flat_samples[first_key].shape[0]

    # Discover obs sites
    if obs_name is None:
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(model).get_trace(*model_args, **model_kwargs)
        obs_sites = [
            name for name, site in trace.items()
            if site['type'] == 'sample' and site.get('is_observed', False)
        ]
        if not obs_sites:
            raise ValueError("No observed sites found. Pass obs_name explicitly.")
    elif isinstance(obs_name, str):
        obs_sites = [obs_name]
    else:
        obs_sites = list(obs_name)

    # Trace sample by sample
    log_liks = {site: [] for site in obs_sites}

    for s in range(total_samples):
        # Get single sample
        single_sample = {k: v[s] for k, v in flat_samples.items()}

        # Condition model on this posterior sample and trace
        conditioned = handlers.condition(model, data=single_sample)
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(conditioned).get_trace(
                *model_args, **model_kwargs
            )

        for site in obs_sites:
            site_data = trace[site]
            fn = site_data['fn']
            value = site_data['value']
            ll_i = fn.log_prob(value)
            log_liks[site].append(ll_i)

    # Stack
    log_liks = {k: jnp.stack(v) for k, v in log_liks.items()}

    # Reshape back to (chains, samples, ...)
    if has_chains:
        num_samples = total_samples // num_chains
        log_liks = {
            k: v.reshape(num_chains, num_samples, *v.shape[1:])
            for k, v in log_liks.items()
        }

    if len(log_liks) == 1:
        return list(log_liks.values())[0]
    return log_liks
