"""Shared pieces of the g-and-k parameter-recovery study.

DGP -> inference -> recovery: for each of BF_NSIM simulated datasets, true
parameters are drawn over a range, a dataset is simulated from the g-and-k
distribution, the parameters are estimated with a simulation-based approach,
and the posterior means are compared with the true values (scatter plot,
R^2 and RMSE per parameter).
"""
import gc
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PARAMS = ["A", "B", "g", "k"]
N_OBS = 500                      # observations per dataset

# Range of the true parameter values explored by the DGP
TRUE_RANGES = {"A": (1.0, 5.0), "B": (0.5, 2.0), "g": (0.5, 3.0), "k": (0.1, 0.8)}


def g_and_k(z, A, B, g, k):
    """g-and-k quantile function applied to standard normal draws z."""
    return A + B * (1 + 0.8 * jnp.tanh(g * z / 2)) * (1 + z**2) ** k * z


def make_model(m):
    """g-and-k simulator written as a BF model (Uniform(0, 10) priors)."""
    def model(N):
        A = m.dist.uniform(0, 10, name="A")
        B = m.dist.uniform(0, 10, name="B")
        g = m.dist.uniform(0, 10, name="g")
        k = m.dist.uniform(0, 10, name="k")
        z = m.dist.normal(0, 1, shape=(N,), name="z")
        m.dist.track("y", g_and_k(z, A, B, g, k))
    return model


def simulate_datasets(m, nsim, n_obs=N_OBS):
    """DGP: true parameters over TRUE_RANGES and one dataset per parameter set."""
    truths = {p: np.asarray(m.dist.uniform(lo, hi, sample=True, shape=(nsim,)))
              for p, (lo, hi) in TRUE_RANGES.items()}
    z = m.dist.normal(0, 1, sample=True, shape=(nsim, n_obs))
    y = g_and_k(z, *[jnp.asarray(truths[p])[:, None] for p in PARAMS])
    return truths, np.asarray(y)


def recovery_rows(i, truths, posterior):
    return [{"sim": i, "parameter": p, "simulated": float(truths[p][i]),
             "estimations": float(np.mean(posterior[p]))} for p in PARAMS]


def free_jax_memory():
    """Drop JAX's compiled-executable cache between fits (keeps RSS flat)."""
    jax.clear_caches()
    gc.collect()


def recovery_metrics(results_df):
    """R^2 (squared correlation) and RMSE of estimations vs simulated values."""
    rows = []
    for p, sub in results_df.groupby("parameter", sort=False):
        sub = sub.dropna(subset=["simulated", "estimations"])
        r2 = np.corrcoef(sub["simulated"], sub["estimations"])[0, 1] ** 2
        rmse = np.sqrt(np.mean((sub["estimations"] - sub["simulated"]) ** 2))
        rows.append({"parameter": p, "R2": r2, "RMSE": rmse, "n": len(sub)})
    return pd.DataFrame(rows)


def plot_recovery(results_df, model_name=None):
    """Scatter of estimations vs simulated values per parameter, with R^2 and RMSE."""
    results_df = results_df.copy()
    results_df["simulated"] = pd.to_numeric(results_df["simulated"], errors="coerce")
    results_df["estimations"] = pd.to_numeric(results_df["estimations"], errors="coerce")
    metrics = recovery_metrics(results_df).set_index("parameter")

    g = sns.FacetGrid(results_df, col="parameter", col_wrap=4, height=3.5,
                      sharey=False, sharex=False)
    g.map(sns.scatterplot, "simulated", "estimations")
    for ax in g.axes.flat:
        low, high = ax.get_xlim()
        ax.plot([low, high], [low, high], color="red", ls="--")
        title = ax.get_title()
        param = title.split(" = ")[-1] if " = " in title else title
        if param in metrics.index:
            r2, rmse = metrics.loc[param, "R2"], metrics.loc[param, "RMSE"]
            ax.set_title(f"{title}\n$R^2={r2:.2f}$, RMSE$={rmse:.3f}$")

    if model_name:
        plot_dir = os.environ.get("BF_PLOTS_DIR", "plots")
        os.makedirs(plot_dir, exist_ok=True)
        g.savefig(f"{plot_dir}/{model_name}_recovery.png")
        results_df.to_csv(f"{plot_dir}/{model_name}_recovery.csv", index=False)
        metrics.to_csv(f"{plot_dir}/{model_name}_metrics.csv")
        print(f"Plot saved to {plot_dir}/{model_name}_recovery.png")
        print(metrics.round(3).to_string())
    plt.close(g.fig)
    return metrics
