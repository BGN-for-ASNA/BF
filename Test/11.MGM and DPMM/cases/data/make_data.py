"""Simulate the GMM/DPMM case data (same make_blobs call as Documentation/24. GMM.qmd) and freeze the k-means centroids."""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.datasets import make_blobs
from BayesForge import bf

here = Path(__file__).parent
data, _ = make_blobs(n_samples=500, centers=8, cluster_std=0.8, center_box=(-10, 10), random_state=101)
pd.DataFrame(data, columns=["x1", "x2"]).to_csv(here / "blobs.csv", index=False, float_format="%.17g")
m = bf(platform="cpu")
m.ml.KMEANS(data, n_clusters=8)
c = np.asarray(m.ml.results["centroids"])
order = np.lexsort((c[:, 1], c[:, 0]))      # sort by x1 then x2 so the order is stable
pd.DataFrame(c[order], columns=["x1", "x2"]).to_csv(here / "gmm_centroids.csv", index=False, float_format="%.17g")
