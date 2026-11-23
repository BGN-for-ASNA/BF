# %%
"""Posterior Predictive Check (PPC) demos for BF.

Covers all ppc_* functions in BF.Diagnostic.ppc:
  Distributions : ppc_density, ppc_hist, ppc_boxplot
  Statistics    : ppc_stat, ppc_stat_2d
  Intervals     : ppc_intervals, ppc_ribbon
  Errors        : ppc_error_hist
  Scatter       : ppc_scatter
  Discrete      : ppc_rootogram, ppc_bars
  LOO           : ppc_loo_intervals
"""
import plotly.io as pio
pio.renderers.default = "json"
from BayesForge import bf
import jax.numpy as jnp
import numpy as np

# =============================================================================
# Gaussian model setup
# =============================================================================
m = bf(platform="cpu")
data_path = m.load.howell1(only_path=True)
m.data(data_path, sep=";")
m.df = m.df[m.df.age > 18]
m.scale(data=["weight"])

def model(weight, height):
    a = m.dist.normal(178, 20, name="a")
    b = m.dist.log_normal(0, 1, name="b")
    s = m.dist.uniform(0, 50, name="s")
    m.dist.normal(a + b * weight, s, obs=height, shape=(weight.shape[0],))

m.fit(model, num_samples=500, num_chains=4, progress_bar=False)

# %%
# Convenience: get yrep matrix once (S x n_obs)
yrep = m.diag.get_yrep()
y    = np.asarray(m.data_on_model["height"])
print("yrep shape:", yrep.shape, "  y shape:", y.shape)

# =============================================================================
# PPC — Distributions
# =============================================================================
# %%
# KDE overlay: y (black line) vs 50 yrep draws (blue lines)
m.diag.ppc_density().show()

# %%
# Histogram small multiples
m.diag.ppc_hist(n=9).show()

# %%
# Boxplot overlay
m.diag.ppc_boxplot(n=25).show()

# =============================================================================
# PPC — Statistics
# =============================================================================
# %%
# Distribution of mean(yrep) vs mean(y); Bayesian p-value shown in title
m.diag.ppc_stat(stat="mean").show()

# %%
# SD
m.diag.ppc_stat(stat="sd").show()

# %%
# Custom statistic: 90th percentile
m.diag.ppc_stat(stat=lambda x: np.percentile(x, 90),
                title="PPC: 90th percentile").show()

# %%
# 2D: mean vs sd scatter
m.diag.ppc_stat_2d(stat1="mean", stat2="sd").show()

# =============================================================================
# PPC — Intervals
# =============================================================================
# %%
# Predictive intervals vs observed, indexed by weight
weight_obs = np.asarray(m.data_on_model["weight"])
m.diag.ppc_intervals(x=weight_obs).show()

# %%
# Ribbon (same but sorted by x)
m.diag.ppc_ribbon(x=weight_obs).show()

# =============================================================================
# PPC — Errors
# =============================================================================
# %%
# Distribution of residuals y - E[yrep]
m.diag.ppc_error_hist().show()

# =============================================================================
# PPC — Scatterplots
# =============================================================================
# %%
m.diag.ppc_scatter(n_reps=9).show()

# =============================================================================
# PPC — LOO
# =============================================================================
# %%
# LOO-style intervals sorted by observed y
m.diag.ppc_loo_intervals().show()

# =============================================================================
# PPC — Discrete (Poisson example)
# =============================================================================
# %%
rng = np.random.default_rng(42)
counts_obs = rng.poisson(lam=3, size=200)

m_pois = bf(platform="cpu")

def model_pois(x, y):
    lam = m_pois.dist.exponential(1 / 3, name="lam")
    m_pois.dist.poisson(lam * jnp.ones_like(x), obs=y, name="y")

m_pois.fit(
    model_pois,
    obs=dict(x=jnp.ones(len(counts_obs)), y=jnp.array(counts_obs)),
    num_samples=300, num_chains=2, progress_bar=False,
)

# %%
yrep_pois = m_pois.diag.get_yrep()
from BayesForge.Diagnostic.ppc import ppc_rootogram, ppc_bars
ppc_rootogram(counts_obs, yrep_pois).show()
ppc_bars(counts_obs, yrep_pois).show()
