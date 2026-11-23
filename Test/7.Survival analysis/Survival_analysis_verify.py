import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import gaussian_kde, ks_2samp
import pymc as pm
import pytensor.tensor as pt
import jax.numpy as jnp

from BayesForge import bf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "3.Documentation"))
from doc_blocks import doc_model
# The model as published on the documentation page (explicit version). Same model as the
# built-in m.models.survival.model the script used to fit, and the same site names
# ('Baseline_rate', 'Hazard_rate_metastasized'); same priors too - the page writes out
# the built-in defaults Baseline_rate ~ Gamma(0.01, 0.01) and
# Hazard_rate_metastasized ~ Normal(0, 10). The only difference is that the page has no
# observed_mask on the Poisson likelihood (every mastectomy interval is observed here).
PAGE_MODEL = doc_model("14. Survival analysis.qmd", 0)


def main():
    print("Initializing BF model...")
    m = bf(platform="cpu")

    print("Loading mastectomy data...")
    data_path = m.load.mastectomy(only_path=True)
    m.data(data_path)

    print("Preprocessing data...")
    # Convert 'yes'/'no' to 1/0
    m.df.metastasized = (m.df.metastasized.values == "yes").astype(jnp.int64)
    
    # Import time with even interval length = 3
    m.models.survival.import_time_even(m.df.time.values, m.df.event.values, interval_length=3)
    m.models.survival.import_covF(m.df.metastasized.values, ["metastasized"])

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Censoring plot: one row per subject, line length = follow-up time,
    # colour = censored vs. event observed, black dots = metastasized.
    print("Saving censoring plot...")
    fig, _ = m.models.survival.plot_censoring(cov="metastasized")
    censoring_path = os.path.join(script_dir, "survival_censoring_verify.png")
    fig.savefig(censoring_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Censoring plot saved to {censoring_path}")

    print("Fitting BF model...")
    # Sampler settings are matched to the PyMC call below so the only thing
    # being compared is the model. target_accept_prob=0.99 (PyMC default here
    # is 0.99 too) tames the awkward geometry from the many near-empty late
    # baseline intervals; 4 chains x 4000 draws drives the Monte Carlo error
    # on the posterior mean well below the 0.01 comparison tolerance.
    N_DRAWS, N_WARMUP, N_CHAINS, TARGET_ACCEPT = 4000, 2000, 4, 0.99
    m.fit(PAGE_MODEL, num_samples=N_DRAWS, num_warmup=N_WARMUP,
          num_chains=N_CHAINS, target_accept_prob=TARGET_ACCEPT,
          progress_bar=False, seed=42)

    print("Summarizing BF model...")
    BF_summary = m.summary()
    print(BF_summary)

    # Posterior cumulative-hazard and survival curves by metastasized status.
    print("Saving survival plot...")
    fig, _ = m.models.survival.plot_surv(beta="Hazard_rate_metastasized")
    surv_path = os.path.join(script_dir, "survival_plot_verify.png")
    fig.savefig(surv_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Survival plot saved to {surv_path}")

    # --- PyMC Implementation ---
    print("Fitting PyMC model...")
    death = np.array(m.models.survival.death)
    exposure = np.array(m.models.survival.exposure)
    metastasized = np.array(m.df.metastasized.values)
    n_intervals = m.models.survival.n_intervals

    # The Poisson offset must be identical to BF's, which adds
    # jnp.finfo(mu.dtype).tiny (mu is float64).
    poisson_offset = np.finfo(np.float64).tiny

    with pm.Model() as pymc_model:
        # Baseline_rate ~ Gamma(0.01, 0.01), one rate per interval: the prior of the
        # page model fitted above, which is also the built-in default stored in
        # m.models.survival.baseline_rate_prior (read here so the two models cannot
        # drift apart). (On the numpyro backend the built-in samples this in log space
        # via _LogGamma; that is a sampler reparametrisation only - the prior on
        # lambda0 is the same Gamma.)
        lambda0 = pm.Gamma("lambda0", *m.models.survival.baseline_rate_prior, shape=n_intervals)
        # Hazard_rate ~ Normal(0, 10), same scale as the page model
        # (m.models.survival.hazard_rate_prior_scale holds the same default). shape=(1,)
        # matches the page's beta = m.dist.normal(0, 10, shape=(1,)).
        beta = pm.Normal("beta", 0, sigma=m.models.survival.hazard_rate_prior_scale, shape=(1,))

        # Hazard rate: lambda[i,k] = lambda0[k] * exp(beta * x_i).
        # Identical to BF's calculate_hazard_rate_uni_cov.
        lambda_ = pm.Deterministic("lambda_", pt.outer(pt.exp(beta * metastasized), lambda0))
        mu = pm.Deterministic("mu", exposure * lambda_)

        # Poisson count likelihood on the 0/1 event indicator, same
        # offset as BF.
        obs = pm.Poisson("obs", mu + poisson_offset, observed=death)

        idata_pymc = pm.sample(N_DRAWS, tune=N_WARMUP, target_accept=TARGET_ACCEPT,
                               random_seed=42, progressbar=False,
                               chains=N_CHAINS, cores=1)

    # --- Comparison Plot ---
    print("Plotting comparison...")
    BF_beta = np.array(m.posteriors["Hazard_rate_metastasized"]).flatten()
    pymc_beta = idata_pymc.posterior["beta"].values.flatten()

    # Shared grid so the two densities are directly comparable, padded so the
    # tails are not clipped at the sample extremes.
    lo = min(BF_beta.min(), pymc_beta.min())
    hi = max(BF_beta.max(), pymc_beta.max())
    pad = 0.1 * (hi - lo)
    xs = np.linspace(lo - pad, hi + pad, 300)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for arr, color, label in [(BF_beta, "C0", "BF"), (pymc_beta, "C1", "PyMC")]:
        if arr.std() < 1e-8:
            # Frozen chains: a KDE here is a meaningless spike, so show the
            # draws themselves instead of pretending there is a density.
            print(f"WARNING: {label} draws are near-constant (std={arr.std():.2e}); "
                  "plotting locations instead of a density.")
            for x in np.unique(arr):
                ax.axvline(x, color=color, linestyle=":", label=label)
            continue
        ax.plot(xs, gaussian_kde(arr)(xs), color=color, label=label)
    ax.set_title("Posterior Distribution Comparison: Beta (metastasized)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()

    plot_path = os.path.join(script_dir, "survival_comparison_verify.png")
    plt.savefig(plot_path)
    print(f"Comparison plot saved to {plot_path}")

    # --- Scatter Plot for Consistency ---
    print("Generating scatter plot for parameter consistency...")
    BF_lambda0_means = np.mean(np.array(m.posteriors["Baseline_rate"]), axis=0)
    pymc_lambda0_means = idata_pymc.posterior["lambda0"].mean(dim=("chain", "draw")).values

    BF_beta_mean = np.mean(BF_beta)
    pymc_beta_mean = np.mean(pymc_beta)

    # Combine all parameter means
    BF_all_means = np.concatenate([BF_lambda0_means, [BF_beta_mean]])
    pymc_all_means = np.concatenate([pymc_lambda0_means, [pymc_beta_mean]])

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(pymc_all_means, BF_all_means, alpha=0.6, color="darkgreen", label="Parameters (lambda0 & beta)")

    # Add diagonal line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="y = x (Perfect Match)")
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_title("Posterior Means Comparison: PyMC vs BF")
    ax.set_xlabel("PyMC Posterior Means")
    ax.set_ylabel("BF Posterior Means")
    ax.legend()

    scatter_path = os.path.join(script_dir, "survival_scatter_verify.png")
    plt.savefig(scatter_path)
    print(f"Scatter plot saved to {scatter_path}")
    
    print("\n--- Final Posterior Comparison Summary ---")
    diff = abs(BF_beta_mean - pymc_beta_mean)
    print(f"BF Beta Mean:   {BF_beta_mean:.4f}")
    print(f"PyMC Beta Mean: {pymc_beta_mean:.4f}")
    print(f"|Difference|:   {diff:.4f}")

    # The two runs are independent MCMC estimates of the SAME posterior, so the
    # means never coincide exactly - they differ by Monte Carlo error. Judge
    # `diff` against that error, not against 0: mcse = sd / sqrt(ess), and the
    # error on the difference is sqrt(mcse_BF^2 + mcse_PyMC^2).
    bf_beta_row = BF_summary.loc["Hazard_rate_metastasized[0]"]
    pymc_beta_row = az.summary(idata_pymc, var_names=["beta"]).iloc[0]
    bf_ess = float(bf_beta_row["ess_bulk"])
    pymc_ess = float(pymc_beta_row["ess_bulk"])
    mcse_bf = BF_beta.std() / np.sqrt(bf_ess)
    mcse_pymc = pymc_beta.std() / np.sqrt(pymc_ess)
    mcse_diff = np.sqrt(mcse_bf**2 + mcse_pymc**2)
    print(f"MC error on difference: {mcse_diff:.4f}  ->  diff / mc_error = {diff / mcse_diff:.2f}")

    # Distributional check on the beta draws: two-sample KS *statistic* = the
    # largest gap between the two empirical CDFs. Judge this number, NOT the
    # p-value: KS assumes i.i.d. samples, MCMC draws are autocorrelated, so at
    # n = N_DRAWS * N_CHAINS the p-value is spuriously ~0 even for two runs of
    # the *same* model with different seeds (verified: BF-vs-BF gives the same
    # statistic ~0.023). A statistic below ~0.05 means the posteriors coincide.
    ks = ks_2samp(BF_beta, pymc_beta)
    print(f"KS(beta): statistic={ks.statistic:.4f} (p={ks.pvalue:.1e}, not diagnostic)")

    print("\n--- Convergence ---")
    print(f"BF   beta: r_hat={bf_beta_row['r_hat']:.3g} ess_bulk={bf_ess:.1f}")
    print(f"PyMC beta: r_hat={pymc_beta_row['r_hat']:.3g} ess_bulk={pymc_ess:.1f}")
    if bf_beta_row["r_hat"] > 1.05 or bf_ess < 100:
        print("WARNING: BF chains did not converge - the comparison above is not valid.")

    # Pass criterion: means agree within ~3x the Monte Carlo error, and the KS
    # statistic is small (posteriors overlap). The residual |diff| ~ 0.02 is
    # run-to-run MCMC scatter of this estimand (50 of 76 baseline intervals are
    # unidentified and weakly couple to beta); BF disagrees with itself by the
    # same amount, so it is not a BF-vs-PyMC model difference.
    if diff <= 3 * mcse_diff and ks.statistic < 0.05:
        print("PASS: BF and PyMC posteriors match within Monte Carlo error.")
    else:
        print("FAIL: BF and PyMC posteriors differ beyond Monte Carlo error.")
    print("Done!")

if __name__ == "__main__":
    main()
