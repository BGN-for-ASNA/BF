# %%
"""Regression overlay plot demo — Gaussian and logistic models."""
from BayesForge import bf
import jax
import jax.numpy as jnp

# =============================================================================
# Gaussian model (Howell height ~ weight)
# =============================================================================
m = bf(platform="cpu")
data_path = m.load.howell1(only_path=True)
m.data(data_path, sep=";")
m.df = m.df[m.df.age > 18]
m.scale(data=["weight"])


def model_gaussian(weight, height):
    a = m.dist.normal(178, 20, name="a")
    b = m.dist.log_normal(0, 1, name="b")
    s = m.dist.uniform(0, 50, name="s")
    m.dist.normal(a + b * weight, s, obs=height, shape=(weight.shape[0],))


m.fit(model_gaussian, num_samples=500, num_chains=4, progress_bar=False)

# %%
# plot_regression is in development and not exposed on m.diag:
# call it from its module (auto-detects x and y from data)
from BayesForge.Diagnostic.regression_plot import plot_regression

fig = plot_regression(m, "weight", n=25)
fig.show()

# %%
# --- with axis labels ---
import numpy as np

fig2 = plot_regression(
    m,
    "weight",
    n=20,
    x_label="Weight (scaled)",
    y_label="Height (cm)",
)
fig2.show()

# =============================================================================
# Logistic regression demo (bioassay-style simulated data)
# =============================================================================
rng = np.random.default_rng(7)
n_obs = 80
x_data = rng.uniform(-3, 3, n_obs)
true_p = jax.nn.sigmoid(0.5 + 1.2 * x_data)
y_data = rng.binomial(1, np.asarray(true_p), n_obs).astype(float)

m2 = bf(platform="cpu")


def model_logistic(x, y):
    a = m2.dist.normal(0, 10, name="a")
    b = m2.dist.normal(0, 10, name="b")
    p = jax.nn.sigmoid(a + b * x)
    m2.dist.bernoulli(p, obs=y, name="y")


m2.fit(
    model_logistic,
    obs=dict(x=jnp.array(x_data), y=jnp.array(y_data)),
    num_samples=500,
    num_chains=4,
    progress_bar=False,
)

# %%
# For logistic, posterior lines from m.sample() are 0/1 draws.
# Mean line = P(y=1|x), which is the regression curve we want.
fig3 = plot_regression(m2, "x", n=30, x_label="x", y_label="P(y=1 | x)")
fig3.show()

# %%
