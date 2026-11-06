# %%
"""Sensitivity analysis and advanced posterior checks for BF.

Covers:
  1. Prior sensitivity — compare posteriors from models with different priors
  2. LOO influence     — Pareto-k plot to find influential observations
  3. Calibration       — empirical vs nominal coverage
  4. Divergence/energy — HMC health check
  5. Multimodality     — mode detection per parameter
"""
from BayesForge import bf
import jax.numpy as jnp
import numpy as np

# =============================================================================
# Shared setup: Gaussian linear regression
# =============================================================================
m = bf(platform="cpu")
data_path = m.load.howell1(only_path=True)
m.data(data_path, sep=";")
m.df = m.df[m.df.age > 18]
m.scale(data=["weight"])


def model_weak(weight, height):
    a = m.dist.normal(178, 20, name="a")
    b = m.dist.log_normal(0, 1, name="b")
    s = m.dist.uniform(0, 50, name="s")
    m.dist.normal(a + b * weight, s, obs=height, shape=(weight.shape[0],))


m.fit(model_weak, num_samples=500, num_chains=4, progress_bar=False)

# =============================================================================
# 1. Prior sensitivity: refit with tighter prior on b
# =============================================================================
# %%
m2 = bf(platform="cpu")
m2.data(data_path, sep=";")
m2.df = m2.df[m2.df.age > 18]
m2.scale(data=["weight"])


def model_tight(weight, height):
    a = m2.dist.normal(178, 5, name="a")  # tighter prior on a
    b = m2.dist.log_normal(0, 0.5, name="b")  # tighter prior on b
    s = m2.dist.uniform(0, 50, name="s")
    m2.dist.normal(a + b * weight, s, obs=height, shape=(weight.shape[0],))


m2.fit(model_tight, num_samples=500, num_chains=4, progress_bar=False)

# %%
from BayesForge.Diagnostic.sensitivity import prior_sensitivity_plot

fig = prior_sensitivity_plot(
    [m.posteriors, m2.posteriors],
    labels=["Weak prior", "Tight prior"],
    param_names=["a", "b", "s"],
)
fig.show()

# %%
# Via m.diag (compares against another posteriors dict directly)
m.diag.prior_sensitivity(
    [m.posteriors, m2.posteriors],
    labels=["Weak", "Tight"],
    param_names=["a", "b"],
).show()

# =============================================================================
# 2. LOO influence: Pareto-k per observation
# =============================================================================
# %%
# Auto-computes log-likelihood from m.model + m.data_on_model
m.diag.influence(
    y=np.asarray(m.data_on_model["height"]),  # optional: adds y to hover
).show()

# =============================================================================
# 3. Calibration check
# =============================================================================
# %%
m.diag.calibration(levels=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95]).show()

# =============================================================================
# 4. HMC divergences and energy
# =============================================================================
# %%
m.diag.divergence_energy().show()

# =============================================================================
# 5. Multimodality check
# =============================================================================
# %%
m.diag.multimodality(param_names=["a", "b", "s"]).show()

# =============================================================================
# Standalone imports (use without m.diag)
# =============================================================================
# %%
from BayesForge.Diagnostic.ppc import get_yrep
from BayesForge.Diagnostic.sensitivity import calibration_plot, influence_plot

yrep = get_yrep(m)
y = np.asarray(m.data_on_model["height"])

calibration_plot(y, yrep).show()
influence_plot(m=m).show()

# %%

# %%
# Declared prior variations, fitted in parallel (one-at-a-time)
if __name__ == "__main__":
    from BayesForge.Diagnostic.sensitivity import prior_sensitivity, _prior_scenarios

    sc = _prior_scenarios({"b_sd": [0.1, 0.5], "a_sd": [5]})
    assert [s["label"] for s in sc] == ["baseline", "b_sd=0.1", "b_sd=0.5", "a_sd=5"]
    assert len(_prior_scenarios({"b_sd": [1, 2], "a_sd": [5, 6]}, mode="grid")) == 4
