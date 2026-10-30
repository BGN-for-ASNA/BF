"""Build the case data for 1.Simple.md from Model_1_Simple."""
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
