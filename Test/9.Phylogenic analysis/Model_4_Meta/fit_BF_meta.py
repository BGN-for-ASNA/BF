# %%
import os
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from BayesForge import bf

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "3.Documentation"))
from doc_blocks import doc_model

# The model as published on the documentation page.
# Priors differ from the script's previous copy: the page uses half_normal(10)
# for sd_obs and sd_phylo (the script used a left-truncated student_t(3, 0, 10));
# the intercept prior normal(0, 10) and the standard-normal z-vectors are the same.
# The page also sizes the phylogenetic z-vector from L (L.shape[0]) instead of a
# hardcoded 200, and names the effects z_p / z_o (previously z_phylo / z_obs).
PAGE_MODEL = doc_model("18. Phylogenetic analysis.qmd", 3, name="model")

# Initialize BF with float64
jax.config.update("jax_enable_x64", True)
m = bf("cpu")

# Load data
url = "https://paul-buerkner.github.io/data/data_effect.txt"
data_fisher = pd.read_table(url, sep="\s+", header=0, names=["Zr", "N", "phylo"])

data_fisher["obs_idx"] = np.arange(len(data_fisher))
data_fisher["se"] = np.sqrt(1.0 / (data_fisher["N"].values - 3.0))

# Load Cholesky of A
L_df = pd.read_csv("L_meta.csv")
L = L_df.values
matrix_species = L_df.columns.tolist()
species_to_idx = {sp: i for i, sp in enumerate(matrix_species)}

data_fisher["phylo_idx"] = data_fisher["phylo"].map(species_to_idx)

# Fixed SE
data_fisher["se"] = np.sqrt(1.0 / (data_fisher["N"] - 3.0))

# Prepare data for BF
m.data_on_model = {
    "y": jnp.array(data_fisher["Zr"].values, dtype=jnp.float64),
    "se": jnp.array(data_fisher["se"].values, dtype=jnp.float64),
    "phylo_idx": jnp.array(data_fisher["phylo_idx"].values, dtype=jnp.int32),
    "obs_idx": jnp.array(data_fisher["obs_idx"].values, dtype=jnp.int32),
    "L": jnp.array(L, dtype=jnp.float64),
}


# Fit model
print("Fitting BF meta-analysis model...")
m.fit(PAGE_MODEL, num_samples=3000, num_warmup=2000, num_chains=2)

# Summary
print(m.summary())

# Posteriors
post = m.posteriors
params_of_interest = ["Intercept", "sd_obs", "sd_phylo"]
post_df = pd.DataFrame({k: np.array(post[k]).flatten() for k in params_of_interest})
post_df.to_csv("BF_post_meta.csv", index=False)

# %%
