# %%
# =============================================================================
# 04_builtin_function.py  —  DGP → Estimation → Validation
#
# Pipeline:
#   1. DGP  : simulate nested data with known true parameters
#   2. Fit  : non-centered  &  centered  built-in nested_varying_effects
#   3. Plot : scatter (true vs posterior mean) per parameter set, R² in each panel
# =============================================================================

import datetime
import os
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
from BayesForge import bf

# ── 0.  device ────────────────────────────────────────────────────────────────
m = bf(platform="cpu")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "3.Documentation"))
from doc_blocks import doc_model

# The models as published on the documentation page: the two built-in ones
# (m.effects.nested_varying_effects, non-centered and centered) and the two written out by
# hand (non-centered with an LKJ Cholesky factor, centered with a correlation matrix).
# Same priors as the local copies they replace: sigma ~ exponential(1);
# global_intercept ~ normal(5, 2), global_beta ~ normal(-1, 1) and
# sigma_region / sigma_group ~ exponential(1) shape (2,) with corr ~ lkj(2, 2) /
# L ~ lkj_cholesky(2, 2) in the hand-written models; the built-in models keep the library
# defaults for the global mean (normal(0, 2) / normal(0, 1)), as the old copies did.
# The page's hand-written non-centered model names its Cholesky factors 'L_region' /
# 'L_group' (the built-in uses 'L_corr_region' / 'L_corr_group') and samples the global
# mean as scalars; the lookups below accept both.
PAGE_MODEL_BUILTIN_NC = doc_model(
    "17. Nested varying effects.qmd", 0, name="model_nested_builtin"
)
PAGE_MODEL_BUILTIN_C = doc_model(
    "17. Nested varying effects.qmd", 0, name="model_nested_builtin_centered"
)
PAGE_MODEL_RAW_NC = doc_model(
    "17. Nested varying effects.qmd", 0, name="model_nested_noncentered"
)
PAGE_MODEL_RAW_C = doc_model(
    "17. Nested varying effects.qmd", 0, name="model_nested"
)
# =============================================================================
# 1.  DATA GENERATING PROCESS (DGP)
# =============================================================================
np.random.seed(42)

N_regions = 5
N_groups_per_region = 4
N_groups = N_regions * N_groups_per_region
N_obs_per_group = 5
N = N_groups * N_obs_per_group

# ── true hyperparameters ──
mu_a_region = 5.0
mu_b_region = -1.0
sigma_a_region = 1.0
sigma_b_region = 0.5
rho_region = -0.5

sigma_a_group = 0.5
sigma_b_group = 0.2
rho_group = 0.3
sigma_obs = 0.5

# ── region-level effects ──
mu_region = np.array([mu_a_region, mu_b_region])
sigmas_reg = np.array([sigma_a_region, sigma_b_region])
Rho_reg = np.array([[1, rho_region], [rho_region, 1]])
Sigma_reg = np.outer(sigmas_reg, sigmas_reg) * Rho_reg
region_effects = np.random.multivariate_normal(mu_region, Sigma_reg, size=N_regions)
# shape: (N_regions, 2)  → col 0 = intercepts, col 1 = slopes

# ── group-level effects nested in regions ──
group_to_region = np.repeat(np.arange(N_regions), N_groups_per_region)
sigmas_grp = np.array([sigma_a_group, sigma_b_group])
Rho_grp = np.array([[1, rho_group], [rho_group, 1]])
Sigma_grp = np.outer(sigmas_grp, sigmas_grp) * Rho_grp

group_effects = np.zeros((N_groups, 2))
for g in range(N_groups):
    reg = group_to_region[g]
    group_effects[g] = np.random.multivariate_normal(region_effects[reg], Sigma_grp)
# shape: (N_groups, 2)  → col 0 = intercepts, col 1 = slopes

# ── observations ──
group_id = np.repeat(np.arange(N_groups), N_obs_per_group)
x = np.random.normal(0, 1, size=N).astype(np.float32)
a_g = group_effects[:, 0]
b_g = group_effects[:, 1]
mu_obs = a_g[group_id] + b_g[group_id] * x
y = np.random.normal(mu_obs, sigma_obs).astype(np.float32)

# obs-level region index (derived from group membership)
region_id = group_to_region[group_id]

# ── pack for model ──
m.data_on_model = {
    "group_id":  group_id.astype(np.int32),
    "region_id": region_id.astype(np.int32),
    "x":         x,
    "y":         y,
    "N_regions": N_regions,
    "N_groups":  N_groups,
}

print(f"DGP complete — {N} obs | {N_groups} groups | {N_regions} regions")
print(f"  True mu_a_region = {mu_a_region:.2f}  mu_b_region = {mu_b_region:.2f}")
print(
    f"  True sigma_a_region = {sigma_a_region:.2f}  sigma_b_region = {sigma_b_region:.2f}"
)
print(f"  True sigma_obs = {sigma_obs:.2f}")

# ground-truth dict used for log and quality checks
true_params = {
    "mu_a_region": mu_a_region,
    "mu_b_region": mu_b_region,
    "sigma_a_region": sigma_a_region,
    "sigma_b_region": sigma_b_region,
    "rho_region": rho_region,
    "sigma_a_group": sigma_a_group,
    "sigma_b_group": sigma_b_group,
    "rho_group": rho_group,
    "sigma_obs": sigma_obs,
}

# =============================================================================
# 2.  MODEL DEFINITIONS  (the four models of the documentation page, see above)
# =============================================================================


def _chol(posteriors, level):
    """Cholesky factor of a level's correlation matrix.

    Named 'L_<level>' by the page's hand-written non-centered model and
    'L_corr_<level>' by the built-in one.
    """
    key = f"L_{level}" if f"L_{level}" in posteriors else f"L_corr_{level}"
    return posteriors[key]


# =============================================================================
# 3.  RECONSTRUCTION OF GROUP/REGION EFFECTS FROM POSTERIORS
#     (non-centered model stores z_* latents; we decode them here)
# =============================================================================


def reconstruct_noncentered(posteriors):
    """
    Decode group- and region-level effects from the non-centered parameterisation.

    Posterior shapes after concatenating chains:
      global_intercept : (S, 1) or (S,)
      global_beta      : (S, 1) or (S,)
      sigma_region     : (S, 2)
      L_region / L_corr_region : (S, 2, 2)
      z_region         : (S, 2, N_regions)
      sigma_group      : (S, 2)
      L_group / L_corr_group   : (S, 2, 2)
      z_group          : (S, 2, N_groups)

    Returns dicts mapping name → posterior-mean array.
    """
    gi = posteriors["global_intercept"]  # (S, 1) or (S,)
    gb = posteriors["global_beta"]  # (S, 1) or (S,)
    mu_global = np.stack([np.ravel(gi), np.ravel(gb)], axis=1)  # (S, 2)

    s_reg = posteriors["sigma_region"]  # (S, 2)
    L_reg = _chol(posteriors, "region")  # (S, 2, 2)
    z_reg = posteriors["z_region"]  # (S, 2, N_regions)

    s_grp = posteriors["sigma_group"]  # (S, 2)
    L_grp = _chol(posteriors, "group")  # (S, 2, 2)
    z_grp = posteriors["z_group"]  # (S, 2, N_groups)

    S = gi.shape[0]

    # region effects  (S, N_regions, 2)
    # formula matches library: ((sigma[:,None] * L_corr) @ z).T
    reg_eff = np.zeros((S, N_regions, 2))
    for s in range(S):
        dev = ((s_reg[s, :, None] * L_reg[s]) @ z_reg[s]).T  # (N_regions, 2)
        reg_eff[s] = mu_global[s] + dev

    # group effects  (S, N_groups, 2)
    grp_eff = np.zeros((S, N_groups, 2))
    for s in range(S):
        dev_g = ((s_grp[s, :, None] * L_grp[s]) @ z_grp[s]).T  # (N_groups, 2)
        mu_g = reg_eff[s][group_to_region]
        grp_eff[s] = mu_g + dev_g

    return {
        "region_intercept": reg_eff[:, :, 0].mean(0),  # (N_regions,)
        "region_slope": reg_eff[:, :, 1].mean(0),
        "group_intercept": grp_eff[:, :, 0].mean(0),  # (N_groups,)
        "group_slope": grp_eff[:, :, 1].mean(0),
    }


def reconstruct_centered(posteriors):
    """
    For the centered model, region_effects and group_effects are sampled
    directly as MVN, so they live in the posteriors.
    Names (from Effects.py): 'region_effects'  (S, N_regions, 2)
                             'group_effects'   (S, N_groups,  2)
    """
    reg = posteriors["region_effects"]  # (S, N_regions, 2)
    grp = posteriors["group_effects"]  # (S, N_groups,  2)
    return {
        "region_intercept": reg[:, :, 0].mean(0),
        "region_slope": reg[:, :, 1].mean(0),
        "group_intercept": grp[:, :, 0].mean(0),
        "group_slope": grp[:, :, 1].mean(0),
    }


# =============================================================================
# 4.  SCATTER PLOT HELPER
# =============================================================================


def r2(true, pred):
    """R² = 1 - SS_res / SS_tot"""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def scatter_panel(ax, true_vals, est_vals, label, color):
    """Scatter true vs estimated, identity line, R² annotation."""
    ax.scatter(
        true_vals,
        est_vals,
        color=color,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.5,
        s=60,
        zorder=3,
    )

    lo = min(true_vals.min(), est_vals.min())
    hi = max(true_vals.max(), est_vals.max())
    pad = (hi - lo) * 0.12
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        "k--",
        lw=1.2,
        alpha=0.55,
        zorder=2,
        label="identity",
    )
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)

    rv = r2(true_vals, est_vals)
    ax.set_xlabel("True value", fontsize=9)
    ax.set_ylabel("Posterior mean", fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.text(
        0.05,
        0.93,
        f"$R^2$ = {rv:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )
    ax.tick_params(labelsize=8)
    ax.set_aspect("equal", "box")
    ax.grid(True, lw=0.4, alpha=0.4)


def _build_log_rows(model_title, posteriors, eff):
    """Compare posterior means of scalar hyperparameters to DGP true values."""
    rows = []

    def add(param, estimated):
        true = true_params.get(param)
        if true is None:
            return
        rows.append(
            dict(
                model=model_title,
                parameter=param,
                estimated=estimated,
                true=true,
                diff=estimated - true,
            )
        )

    if "sigma" in posteriors:
        add("sigma_obs", float(posteriors["sigma"].mean()))

    if "global_intercept" in posteriors:
        add("mu_a_region", float(posteriors["global_intercept"].mean()))
    elif "region_effects" in posteriors:
        add("mu_a_region", float(posteriors["region_effects"][:, :, 0].mean()))

    if "global_beta" in posteriors:
        add("mu_b_region", float(posteriors["global_beta"].mean()))
    elif "region_effects" in posteriors:
        add("mu_b_region", float(posteriors["region_effects"][:, :, 1].mean()))

    if "sigma_region" in posteriors:
        sr = posteriors["sigma_region"].mean(0)
        add("sigma_a_region", float(sr[0]))
        add("sigma_b_region", float(sr[1]))

    if "sigma_group" in posteriors:
        sg = posteriors["sigma_group"].mean(0)
        add("sigma_a_group", float(sg[0]))
        add("sigma_b_group", float(sg[1]))

    if "L_region" in posteriors or "L_corr_region" in posteriors:
        L = _chol(posteriors, "region").mean(0)
        add("rho_region", float((L @ L.T)[0, 1]))

    if "L_group" in posteriors or "L_corr_group" in posteriors:
        L = _chol(posteriors, "group").mean(0)
        add("rho_group", float((L @ L.T)[0, 1]))

    # grand means of random effects
    rows.append(
        dict(
            model=model_title,
            parameter="mean(region_intercept)",
            estimated=float(eff["region_intercept"].mean()),
            true=true_params["mu_a_region"],
            diff=float(eff["region_intercept"].mean()) - true_params["mu_a_region"],
        )
    )
    rows.append(
        dict(
            model=model_title,
            parameter="mean(region_slope)",
            estimated=float(eff["region_slope"].mean()),
            true=true_params["mu_b_region"],
            diff=float(eff["region_slope"].mean()) - true_params["mu_b_region"],
        )
    )
    return rows


def _write_log(log_rows, filepath):
    """Write parameter recovery table to a text file."""
    w = (30, 25, 12, 12, 12)
    hdr = (
        f"{'Model':<{w[0]}} {'Parameter':<{w[1]}} "
        f"{'Estimated':>{w[2]}} {'True':>{w[3]}} {'Difference':>{w[4]}}"
    )
    sep = "-" * len(hdr)

    with open(filepath, "w") as f:
        f.write("DGP-Estimation Parameter Recovery Log\n")
        f.write(
            f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )
        f.write(sep + "\n")
        f.write(hdr + "\n")
        f.write(sep + "\n")
        prev = None
        for r in log_rows:
            if r["model"] != prev:
                if prev is not None:
                    f.write("\n")
                prev = r["model"]
            f.write(
                f"{r['model']:<{w[0]}} {r['parameter']:<{w[1]}} "
                f"{r['estimated']:>{w[2]}.4f} {r['true']:>{w[3]}.4f} "
                f"{r['diff']:>+{w[4]}.4f}\n"
            )
        f.write(sep + "\n")
    print(f"Log saved → {filepath}")


# =============================================================================
# 5.  FIT & PLOT LOOP
# =============================================================================

configs = [
    dict(title="Non-Centered (Built-in)", model_fn=PAGE_MODEL_BUILTIN_NC, reconstruct_fn=reconstruct_noncentered, color="#4C72B0"),
    dict(title="Centered (Built-in)",     model_fn=PAGE_MODEL_BUILTIN_C,  reconstruct_fn=reconstruct_centered,    color="#DD8452"),
    dict(title="Non-Centered (Raw)",      model_fn=PAGE_MODEL_RAW_NC,     reconstruct_fn=reconstruct_noncentered, color="#55A868"),
    dict(title="Centered (Raw)",          model_fn=PAGE_MODEL_RAW_C,      reconstruct_fn=reconstruct_centered,    color="#C44E52"),
]

# True values to compare against
true_region_intercepts = region_effects[:, 0]
true_region_slopes = region_effects[:, 1]
true_group_intercepts = group_effects[:, 0]
true_group_slopes = group_effects[:, 1]

panel_specs = [
    ("Region Intercepts\n(α_region)", "region_intercept", true_region_intercepts),
    ("Region Slopes\n(β_region)", "region_slope", true_region_slopes),
    ("Group Intercepts\n(α_group)", "group_intercept", true_group_intercepts),
    ("Group Slopes\n(β_group)", "group_slope", true_group_slopes),
]

fig, axes = plt.subplots(
    nrows=len(configs),
    ncols=4,
    figsize=(16, 4.5 * len(configs)),
    squeeze=False,
)
fig.suptitle(
    "DGP → Estimation Validation\nNested Varying Effects (Built-in)",
    fontsize=14,
    fontweight="bold",
    y=1.01,
)

summary_rows = []
all_log_rows = []

for row, cfg in enumerate(configs):
    print(f"\n{'='*60}")
    print(f"Fitting: {cfg['title']}")
    print("=" * 60)

    m.fit(
        cfg["model_fn"],
        num_samples=1000,
        num_warmup=1000,
        num_chains=2,
        progress_bar=True,
    )

    tab = m.summary()
    print(tab.to_string())

    posteriors = m.posteriors
    eff = cfg["reconstruct_fn"](posteriors)

    all_log_rows.extend(_build_log_rows(cfg["title"], posteriors, eff))

    for col, (panel_label, key, true_vals) in enumerate(panel_specs):
        ax = axes[row][col]
        est_vals = eff[key]
        scatter_panel(
            ax,
            true_vals,
            est_vals,
            label=f"{cfg['title']}\n{panel_label}",
            color=cfg["color"],
        )
        rv = r2(true_vals, est_vals)
        summary_rows.append(
            {
                "Model": cfg["title"],
                "Parameter": panel_label.replace("\n", " "),
                "N": len(true_vals),
                "R²": f"{rv:.3f}",
            }
        )

# ── row labels ──
for row, cfg in enumerate(configs):
    axes[row][0].set_ylabel(f"{cfg['title']}\n\nPosterior mean", fontsize=9, labelpad=6)

plt.tight_layout()
out_path = "dgp_estimation_validation.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")

# =============================================================================
# 6.  SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 55)
print("  DGP-Estimation Validation Summary")
print("=" * 55)
header = f"{'Model':<28} {'Parameter':<26} {'N':>4} {'R²':>7}"
print(header)
print("-" * 55)
for r in summary_rows:
    print(f"{r['Model']:<28} {r['Parameter']:<26} {r['N']:>4} {r['R²']:>7}")
print("=" * 55)

# =============================================================================
# 7.  PARAMETER RECOVERY LOG
# =============================================================================
_write_log(all_log_rows, "log.txt")

# %%
