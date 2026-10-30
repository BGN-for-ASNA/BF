"""Build the case data for the phylogenetic cases from Model_1..Model_5."""
from pathlib import Path

import pandas as pd

here = Path(__file__).parent
src = here / ".." / ".." / "Model_1_Simple"

data_simple = pd.read_table(src / "data_simple.txt", sep=r"\s+")
mean_cofactor = data_simple["cofactor"].mean()
data_simple["cofactor_centered"] = data_simple["cofactor"] - mean_cofactor

L_df = pd.read_csv(src / "L_simple.csv")
data_simple["phylo_idx"] = pd.Categorical(data_simple["phylo"], categories=L_df.columns).codes

data_simple[["phen", "cofactor_centered", "phylo_idx"]].to_csv(
    here / "phylo_simple.csv", index=False
)
pd.DataFrame(L_df.values).to_csv(here / "phylo_L.csv", header=False, index=False)

# --- 2.Poisson.md ---
src2 = here / ".." / ".." / "Model_2_Poisson"
data_pois = pd.read_table(src2 / "data_pois.csv", sep=",")
data_pois["phen"] = data_pois["phen_pois"].astype(int)
data_pois["cofactor_centered"] = data_pois["cofactor"] - data_pois["cofactor"].mean()
data_pois["phylo_idx"] = pd.Categorical(data_pois["phylo"], categories=L_df.columns).codes
data_pois["obs_idx"] = range(len(data_pois))
data_pois[["phen", "cofactor_centered", "phylo_idx", "obs_idx"]].to_csv(
    here / "phylo_pois.csv", index=False
)

# --- 3.Repeated.md ---
src3 = here / ".." / ".." / "Model_3_Repeated"
data_repeat = pd.read_table(src3 / "data_repeat.txt", sep=r"\s+")
data_repeat["spec_mean_cf"] = data_repeat.groupby("species")["cofactor"].transform("mean")
data_repeat["spec_mean_cf_centered"] = (
    data_repeat["spec_mean_cf"] - data_repeat["spec_mean_cf"].mean()
)
data_repeat["within_spec_cf"] = data_repeat["cofactor"] - data_repeat["spec_mean_cf"]
data_repeat["phylo_idx"] = pd.Categorical(data_repeat["phylo"], categories=L_df.columns).codes
data_repeat["species_idx"] = pd.Categorical(
    data_repeat["species"], categories=L_df.columns
).codes
data_repeat[
    ["phen", "spec_mean_cf_centered", "within_spec_cf", "phylo_idx", "species_idx"]
].to_csv(here / "phylo_repeat.csv", index=False)

# --- phylo_meta.csv (shared metadata) ---
data_effect = pd.read_table(here / ".." / ".." / "data_effect.txt", sep=r"\s+")
data_effect["se"] = (1.0 / (data_effect["N"] - 3.0)) ** 0.5
data_effect["phylo_idx"] = pd.Categorical(data_effect["phylo"], categories=L_df.columns).codes
data_effect["obs_idx"] = range(len(data_effect))
data_effect[["Zr", "se", "phylo_idx", "obs_idx"]].to_csv(
    here / "phylo_meta.csv", index=False
)

# --- 5.Slopes.md ---
src5 = here / ".." / ".." / "Model_5_Slopes"
data_slopes = pd.read_table(src5 / "data_slopes.txt", sep=r"\s+")
data_slopes["x_centered"] = data_slopes["x"] - data_slopes["x"].mean()
data_slopes["phylo_idx"] = data_slopes["phylo"].str[3:].astype(int) - 1
data_slopes[["y", "x_centered", "phylo_idx"]].to_csv(
    here / "phylo_slopes.csv", index=False
)

L_slopes = pd.read_csv(src5 / "L_slopes.csv")
pd.DataFrame(L_slopes.values).to_csv(here / "phylo_L_slopes.csv", header=False, index=False)
