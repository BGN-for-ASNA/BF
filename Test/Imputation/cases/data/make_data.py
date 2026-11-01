"""Simulate the MAR imputation case data (same DGP as ../../0.Code.py)."""

from pathlib import Path

from jax import numpy as jnp, random
from jax.scipy.special import expit
import pandas as pd

simkeys = random.split(random.PRNGKey(0), 10)
nsim = 5000
b_A = 0.25; b_B = 0.25; s_Y = 0.25
Z = random.normal(simkeys[0], (nsim,))
A = random.bernoulli(simkeys[1], expit(Z))
B = random.bernoulli(simkeys[2], expit(Z))
Y = A * b_A + B * b_B + s_Y * random.normal(simkeys[3], (nsim,))
A_obs = random.bernoulli(simkeys[4], expit(3 * (Y - Y.mean())))
Aobs = jnp.where(A_obs, A, -1)        # -1 = missing

df = pd.DataFrame({
    "A": jnp.asarray(Aobs, dtype=jnp.int32),
    "B": jnp.asarray(B, dtype=jnp.int32),
    "Y": jnp.asarray(Y),
})
df.to_csv(Path(__file__).parent / "imputation_mar.csv", index=False)
