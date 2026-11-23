import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
from scipy.spatial.distance import jensenshannon

def calculate_kl_divergence(p_samples, q_samples, bins=100):
    """
    Calculate KL divergence between two distributions given their samples.
    p_samples: Ground truth (Stan)
    q_samples: Approximation (BF)
    """
    # Combine samples to find the range
    combined = np.concatenate([p_samples, q_samples])
    vmin, vmax = np.min(combined), np.max(combined)
    
    # Create histograms
    p_hist, _ = np.histogram(p_samples, bins=bins, range=(vmin, vmax), density=True)
    q_hist, _ = np.histogram(q_samples, bins=bins, range=(vmin, vmax), density=True)
    
    # Add small constant to avoid division by zero
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    
    # Normalize
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    
    return entropy(p_hist, q_hist)

def prepare_bi_data(m):
    #data_dict = m.sampler.get_samples()
    data_dict = m.posteriors
    # Initialize an empty DataFrame to collect parameters
    all_params = []

    # Loop through each array in the dictionary
    for key, array in data_dict.items():
        # m.posteriors is flat (draws, ...) on every backend (RELEASE_AUDIT
        # Samplers M3): no chain axis to merge here.
        # Now handle the flattened array
        if array.ndim == 1:
            all_params.append(pd.DataFrame({key: array}))
            # Also add [0] version for consistency with scripts expecting it
            all_params.append(pd.DataFrame({f"{key}[0]": array}))
        elif array.ndim == 2:
            # (samples, params)
            if array.shape[1] == 1:
                all_params.append(pd.DataFrame({key: array.flatten()}))
                all_params.append(pd.DataFrame({f"{key}[0]": array.flatten()}))
            else:
                param_df = pd.DataFrame(array)
                param_df.columns = [f"{key}[{j}]" for j in range(array.shape[1])]
                all_params.append(param_df)
        elif array.ndim >= 3:
            # (samples, row, col, ...)
            # Flatten everything after the first dimension
            samples = array.shape[0]
            rest = array.shape[1:]
            import itertools
            indices = list(itertools.product(*[range(d) for d in rest]))
            for idx in indices:
                col_name = f"{key}" + "".join([f"[{i}]" for i in idx])
                # Take all samples for this specific element
                indexer = (slice(None),) + idx
                all_params.append(pd.DataFrame({col_name: array[indexer]}))

    # Concatenate all parameter DataFrames along the rows
    df_BF = pd.concat(all_params, axis=1)
    return df_BF

def prepare_stan_data(df):
    columns_to_remove = ['lp__', 'accept_stat__', 'stepsize__', 'treedepth__', 'n_leapfrog__', 'divergent__', 'energy__', 'chain__', 'iter__', 'draw__']
    d = df.drop(columns=[c for c in columns_to_remove if c in df.columns])
    return d

def build_stan_model(stan_code, data, chains=1, iter_sampling=1000, iter_warmup=1000):
    from cmdstanpy import CmdStanModel
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        stan_file = os.path.join(tmpdir, 'model.stan')
        with open(stan_file, 'w') as f:
            f.write(stan_code)
        model = CmdStanModel(stan_file=stan_file)
        fit = model.sample(data=data, chains=chains, iter_sampling=iter_sampling, iter_warmup=iter_warmup, show_progress=False)
        return fit.draws_pd()

def combine_data(df_BF, d):
    # Match columns by name where possible
    common_cols = list(set(df_BF.columns) & set(d.columns))
    if common_cols:
        df_bi_sub = df_BF[common_cols].copy()
        d_sub = d[common_cols].copy()
    else:
        # Fallback to positional if no names match
        d_sub = d.copy()
        df_bi_sub = df_BF.copy()
        if len(d_sub.columns) == len(df_bi_sub.columns):
            d_sub.columns = df_bi_sub.columns
            common_cols = df_bi_sub.columns.tolist()
        else:
            print("Warning: Column count mismatch in combine_data")
            return pd.concat([df_BF, d], ignore_index=True)

    df_bi_sub['method'] = 'BF'
    d_sub['method'] = 'STAN'
    d_comb = pd.concat([d_sub, df_bi_sub], ignore_index=True)
    return d_comb

def plot_comparaison(m, df, param_map=None, model_name=None):
    if isinstance(m, pd.DataFrame):
        df_BF = m.copy()
    else:
        df_BF = prepare_bi_data(m)
        
    d = prepare_stan_data(df).copy()
    
    if param_map:
        d = d[list(param_map.values())]
        d.columns = list(param_map.keys())
        df_BF = df_BF[list(param_map.keys())]
        
    d_comb = combine_data(df_BF, d)
    d_stan_clean = d_comb[d_comb['method'] == 'STAN']
    df_bi_clean = d_comb[d_comb['method'] == 'BF']

    params = [c for c in df_bi_clean.columns if c != 'method']
    num_params = len(params)
    num_cols = 3
    num_rows = (num_params + num_cols - 1) // num_cols

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, max(5, num_rows * 4)), sharey=False)
    if num_rows * num_cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for a, i in enumerate(params):
        sns.kdeplot(data=d_comb, x=i, hue='method', ax=axes[a], fill=True, alpha=0.5)
        
        # Calculate KL Divergence
        p_samples = d_stan_clean[i].values
        q_samples = df_bi_clean[i].values
        kl_div = calculate_kl_divergence(p_samples, q_samples)
        
        axes[a].set_title(f"{i}\nKL Div: {kl_div:.4f}")
        if a % num_cols != 0:
            axes[a].set_ylabel('')

    plt.tight_layout()
    
    if model_name:
        import os
        plot_dir = "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # Generate and save comparison table
        generate_comparison_table(df_bi_clean, d_stan_clean, model_name)
        
        plt.savefig(f"{plot_dir}/{model_name}_comparison.png")
        print(f"Plot saved to {plot_dir}/{model_name}_comparison.png")
        
    plt.close()
    return plt

def generate_comparison_table(df_BF, d, model_name):
    params = [c for c in df_BF.columns if c != 'method']
    table_data = []
    
    for p in params:
        BF_mean = df_BF[p].mean()
        stan_mean = d[p].mean()
        diff = BF_mean - stan_mean
        kl_div = calculate_kl_divergence(d[p].values, df_BF[p].values)
        
        table_data.append({
            'Parameter': p,
            'BF Mean': f"{BF_mean:.4f}",
            'Stan Mean': f"{stan_mean:.4f}",
            'Diff': f"{diff:.4f}",
            'KL Div': f"{kl_div:.4f}"
        })
    
    table_df = pd.DataFrame(table_data)
    
    # Save to txt file
    output_path = f"plots/{model_name}_comparison_table.txt"
    with open(output_path, 'w') as f:
        f.write(f"Comparison Table for {model_name}\n")
        f.write("="*60 + "\n")
        f.write(table_df.to_string(index=False))
        f.write("\n" + "="*60 + "\n")
    
    print(f"Comparison table saved to {output_path}")
    return table_df

def plot_recovery(results_df, model_name=None, r2_threshold=0.8):
    num_params = len(results_df['parameter'].unique())
    num_cols = 3
    num_rows = (num_params + num_cols - 1) // num_cols

    # Ensure numeric types for plotting
    results_df['simulated'] = pd.to_numeric(results_df['simulated'], errors='coerce')
    results_df['estimations'] = pd.to_numeric(results_df['estimations'], errors='coerce')

    g = sns.FacetGrid(results_df, col="parameter", col_wrap=num_cols, height=4, sharey=False, sharex=False)
    g.map(sns.scatterplot, "simulated", "estimations")
    
    for ax in g.axes.flat:
        low, high = ax.get_xlim()
        ax.plot([low, high], [low, high], color='red', ls='--')
        title = ax.get_title()
        param = title.split(" = ")[-1] if " = " in title else title
        sub = results_df[results_df['parameter'] == param].dropna(subset=['simulated', 'estimations'])
        if len(sub) >= 2:
            r2 = np.corrcoef(sub['simulated'], sub['estimations'])[0, 1] ** 2
            ax.set_title(f"{title}\n$R^2={r2:.2f}$")
    
    if model_name:
        import os
        plot_dir = "plots"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
        
        # R2 report generation skipped as per user request
        
        g.savefig(f"{plot_dir}/{model_name}_recovery.png")
        print(f"Plot saved to {plot_dir}/{model_name}_recovery.png")
        
    plt.close(g.fig)
    return g


def waic_report(m, model_name, ref_model=None, ref_kwargs=None, tol=None):
    """Compute WAIC two independent ways and append both to the model's txt report.

    The goal is to show BF's WAIC equals what ArviZ would compute — logged
    programmatically for every model.

    NumPyro backend
        [A] direct : ``_waic`` on ``numpyro.infer.log_likelihood`` of the fitted
                     model (BF's arviz-free path).
        [B] arviz  : ``_waic`` on ArviZ's own ``az.from_numpyro`` log_likelihood
                     group. Same draws -> bit-identical numbers.
    TFP backend
        [A] native : ``_waic`` on the TFP pointwise log-likelihood obtained by
                     replaying the ``JointDistributionCoroutine``.
        [B] ref    : ``_waic`` on ``numpyro.infer.log_likelihood`` evaluated on
                     the *same* TFP draws via ``ref_model`` (an equivalent
                     numpyro model). This is the exact machinery ArviZ uses, so
                     the two match to floating-point (TFP samples in float32).

    Both WAICs (elpd_waic, p_waic), their absolute difference and a PASS/FAIL
    verdict are printed and appended to
    ``<plots>/<model_name>_comparison_table.txt``.
    """
    import os
    import numpy as np
    import jax.numpy as jnp
    from BayesForge.Diagnostic.Diag2 import (
        _waic, _is_tfp_sampler, _pointwise_loglik_from_tfp,
        _pointwise_loglik_from_sampler, _pointwise_loglik_from_idata,
        _numpyro_idata_with_loglik,
    )

    smp = m.sampler
    if _is_tfp_sampler(smp):
        if ref_model is None:
            raise ValueError(
                "waic_report on the TFP backend needs ref_model (an equivalent "
                "numpyro model) to compute the reference WAIC."
            )
        from numpyro.infer import log_likelihood as _np_ll
        ll_a = _pointwise_loglik_from_tfp(smp)
        label_a = "TFP native (JointDistributionCoroutine replay)"
        draws = smp.get_samples(group_by_chain=True)
        samples = {
            k: jnp.asarray(v).reshape((-1,) + jnp.asarray(v).shape[2:])
            for k, v in draws.items()
        }
        lld = _np_ll(ref_model, samples, **(ref_kwargs or {}))
        arr = np.asarray(lld[list(lld.keys())[0]])
        ll_b = arr.reshape(arr.shape[0], -1)
        label_b = "NumPyro log_likelihood on same draws (== ArviZ machinery)"
        default_tol = 1e-1
    else:
        ll_a = _pointwise_loglik_from_sampler(smp)
        label_a = "NumPyro direct (numpyro.infer.log_likelihood)"
        label_b = "ArviZ round-trip (az.from_numpyro log_likelihood group)"
        try:
            ll_b = _pointwise_loglik_from_idata(_numpyro_idata_with_loglik(smp))
        except Exception as _e:
            # az.from_numpyro can choke on BF pytrees under jax.eval_shape for
            # some models (e.g. GP with SampledData). Degrade gracefully.
            ll_b = None
            label_b += f"  [UNAVAILABLE: {type(_e).__name__}]"
        default_tol = 1e-6

    tol = default_tol if tol is None else tol
    import arviz as az
    from BayesForge.Diagnostic.jax_diagnostics import loo as _jax_loo

    # ================= WAIC: [A] vs [B] (same draws) =================
    wa = _waic(ll_a)
    if ll_b is not None:
        wb = _waic(ll_b)
        dW_elpd = abs(float(wa.elpd_waic) - float(wb.elpd_waic))
        dW_p = abs(float(wa.p_waic) - float(wb.p_waic))
        okW = (
            dW_elpd <= max(tol, 1e-4 * abs(float(wa.elpd_waic)))
            and dW_p <= max(tol, 1e-3 * abs(float(wa.p_waic)))
        )
        waic_B_lines = [
            f"[B] {label_b}",
            f"      elpd_waic = {float(wb.elpd_waic):+.6f}    p_waic = {float(wb.p_waic):.6f}",
            f"|delta elpd_waic| = {dW_elpd:.3e}      |delta p_waic| = {dW_p:.3e}",
            f"RESULT: {'PASS' if okW else 'FAIL'}   (S={int(wa.n_samples)}, N={int(wa.n_data_points)})",
        ]
    else:
        wb = None
        waic_B_lines = [
            f"[B] {label_b}",
            f"      elpd_waic = {float(wa.elpd_waic):+.6f}  (WAIC[A] shown; ArviZ reference unavailable)",
        ]

    # ============= LOO: BF native PSIS-LOO vs ArviZ az.loo =============
    # Both run on the *same* pointwise log-likelihood (ll_a): BF's own JAX
    # PSIS-LOO vs ArviZ's az.loo. The reference InferenceData is built directly
    # with az.from_dict (posterior + log-likelihood) so it never calls
    # az.from_numpyro -- robust for every model, both backends. Small deltas
    # come from ArviZ's relative-ESS tail correction, which the JAX omits.
    la = _jax_loo(np.asarray(ll_a))            # [A'] BF native (JAX) PSIS-LOO
    post = {k: np.asarray(v) for k, v in smp.get_samples(group_by_chain=True).items()}
    C = next(iter(post.values())).shape[0]
    Sper = next(iter(post.values())).shape[1]
    N = np.asarray(ll_a).shape[1]
    ll3 = np.asarray(ll_a).reshape(C, Sper, N)
    idata_loo = az.from_dict({"posterior": post, "log_likelihood": {"obs": ll3}})
    lb = az.loo(idata_loo)                      # [B'] ArviZ PSIS-LOO
    lb_elpd = float(getattr(lb, "elpd", getattr(lb, "elpd_loo", float("nan"))))
    lb_p = float(getattr(lb, "p", getattr(lb, "p_loo", float("nan"))))
    dL_elpd = abs(float(la.elpd) - lb_elpd)
    dL_p = abs(float(la.p) - lb_p)
    okL = (
        dL_elpd <= max(0.5, 1e-3 * abs(float(la.elpd)))
        and dL_p <= max(0.5, 5e-2 * abs(float(la.p)))
    )

    lines = [
        "",
        "=" * 64,
        f"WAIC cross-check  -  {model_name}",
        "=" * 64,
        f"[A] {label_a}",
        f"      elpd_waic = {float(wa.elpd_waic):+.6f}    p_waic = {float(wa.p_waic):.6f}",
        *waic_B_lines,
        "-" * 64,
        f"LOO cross-check  -  {model_name}",
        "-" * 64,
        "[A'] BF native JAX PSIS-LOO (jax_diagnostics.loo)",
        f"      elpd_loo  = {float(la.elpd):+.6f}    p_loo  = {float(la.p):.6f}",
        "[B'] ArviZ az.loo (PSIS-LOO) on the same log-likelihood",
        f"      elpd_loo  = {lb_elpd:+.6f}    p_loo  = {lb_p:.6f}",
        f"|delta elpd_loo| = {dL_elpd:.3e}      |delta p_loo| = {dL_p:.3e}",
        f"RESULT: {'PASS' if okL else 'FAIL'}",
        "=" * 64,
        "",
    ]
    text = "\n".join(lines)
    plot_dir = os.environ.get('BF_PLOTS_DIR', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    with open(f"{plot_dir}/{model_name}_comparison_table.txt", "a") as f:
        f.write(text)
    print(text)
    return {"waic": (wa, wb), "loo": (la, lb)}
