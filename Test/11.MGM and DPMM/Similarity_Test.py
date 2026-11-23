import os
import sys
from unittest.mock import MagicMock

# Mock IPython for environments where it is not installed
mock_ipython = MagicMock()
sys.modules["IPython"] = mock_ipython
sys.modules["IPython.display"] = mock_ipython

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

from BayesForge import bf
from BayesForge.Models.GMM import predict_gmm

SEEDS = [61755, 62809, 14920, 83712, 49201, 75019, 31940, 92831, 58102, 27491]
N_SAMPLES = 400
N_FEATURES = 2
T = 10


def run_similarity_test():
    """Evaluate BF vs. Sklearn clustering similarity across the 10 validation simulations."""
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")

    with open(log_path, "w", encoding="utf-8") as log_file:
        def log_msg(msg):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        log_msg("=== MGM (GMM) & DPMM Similarity Test Results (10 Simulations) ===")
        m = bf(platform="cpu")

        rows = []
        header = (
            f"{'Sim':>3} | {'Seed':>5} | {'TrueK':>5} | {'Std':>4} | "
            f"{'BF_K':>4} | {'SK_K':>4} | "
            f"{'DPMM_ARI(BF/SK)':>15} | {'BF_DPMM(True)':>13} | {'SK_DPMM(True)':>13} | "
            f"{'GMM_ARI(BF/SK)':>14} | {'BF_GMM(True)':>12} | {'SK_GMM(True)':>12}"
        )
        sep = "-" * len(header)
        log_msg(header)
        log_msg(sep)

        for i, rng_seed in enumerate(SEEDS, 1):
            np.random.seed(rng_seed)
            n_centers = int(np.random.randint(3, 8))
            cluster_std = float(np.round(np.random.uniform(0.6, 1.1), 2))

            data, true_labels = make_blobs(
                n_samples=N_SAMPLES,
                n_features=N_FEATURES,
                centers=n_centers,
                cluster_std=cluster_std,
                center_box=(-10.0, 10.0),
                random_state=rng_seed,
            )

            # --- DPMM Evaluation ---
            m.data_on_model = dict(data=data, T=T)
            m.fit(
                m.models.dpmm,
                num_chains=2,
                num_samples=250,
                num_warmup=250,
                progress_bar=False,
            )
            _, _, _, _, bf_dpmm_labels = m.models.dpmm.predict(
                data, m.sampler, threshold=0.5, linkage_method="complete"
            )
            bgm = BayesianGaussianMixture(
                n_components=T,
                weight_concentration_prior_type="dirichlet_process",
                random_state=rng_seed,
                max_iter=500,
            )
            sklearn_dpmm_labels = bgm.fit_predict(data)

            n_clusters_bf_dpmm = len(np.unique(bf_dpmm_labels))
            n_clusters_sk_dpmm = len(np.unique(sklearn_dpmm_labels))
            ari_dpmm_vs_sklearn = adjusted_rand_score(bf_dpmm_labels, sklearn_dpmm_labels)
            ari_bf_dpmm_vs_true = adjusted_rand_score(true_labels, bf_dpmm_labels)
            ari_sk_dpmm_vs_true = adjusted_rand_score(true_labels, sklearn_dpmm_labels)

            # --- GMM Evaluation ---
            K = n_centers
            kmeans = KMeans(n_clusters=K, random_state=rng_seed, n_init=10).fit(data)
            m.data_on_model = dict(data=data, K=K, initial_means=kmeans.cluster_centers_)
            m.fit(
                m.models.available["gmm"],
                num_chains=2,
                num_samples=250,
                num_warmup=250,
                progress_bar=False,
            )
            _, _, _, bf_gmm_labels = predict_gmm(data, m.sampler)
            gmm = GaussianMixture(n_components=K, random_state=rng_seed)
            sklearn_gmm_labels = gmm.fit_predict(data)

            ari_gmm_vs_sklearn = adjusted_rand_score(bf_gmm_labels, sklearn_gmm_labels)
            ari_bf_gmm_vs_true = adjusted_rand_score(true_labels, bf_gmm_labels)
            ari_sk_gmm_vs_true = adjusted_rand_score(true_labels, sklearn_gmm_labels)

            rows.append({
                "sim": i,
                "seed": rng_seed,
                "true_k": n_centers,
                "std": cluster_std,
                "bf_k": n_clusters_bf_dpmm,
                "sk_k": n_clusters_sk_dpmm,
                "dpmm_bf_sk": ari_dpmm_vs_sklearn,
                "dpmm_bf_true": ari_bf_dpmm_vs_true,
                "dpmm_sk_true": ari_sk_dpmm_vs_true,
                "gmm_bf_sk": ari_gmm_vs_sklearn,
                "gmm_bf_true": ari_bf_gmm_vs_true,
                "gmm_sk_true": ari_sk_gmm_vs_true,
            })

            log_msg(
                f"{i:>3} | {rng_seed:>5} | {n_centers:>5} | {cluster_std:>4.2f} | "
                f"{n_clusters_bf_dpmm:>4} | {n_clusters_sk_dpmm:>4} | "
                f"{ari_dpmm_vs_sklearn:>15.4f} | {ari_bf_dpmm_vs_true:>13.4f} | {ari_sk_dpmm_vs_true:>13.4f} | "
                f"{ari_gmm_vs_sklearn:>14.4f} | {ari_bf_gmm_vs_true:>12.4f} | {ari_sk_gmm_vs_true:>12.4f}"
            )

        log_msg(sep)
        log_msg("\n=== Aggregate Summary (10 Simulations) ===")
        for label, key in [
            ("DPMM ARI (BF vs Sklearn)", "dpmm_bf_sk"),
            ("DPMM ARI (BF vs True)   ", "dpmm_bf_true"),
            ("DPMM ARI (SK vs True)   ", "dpmm_sk_true"),
            ("GMM  ARI (BF vs Sklearn)", "gmm_bf_sk"),
            ("GMM  ARI (BF vs True)   ", "gmm_bf_true"),
            ("GMM  ARI (SK vs True)   ", "gmm_sk_true"),
        ]:
            vals = np.array([r[key] for r in rows], dtype=float)
            log_msg(
                f"{label}: mean={vals.mean():.4f} | std={vals.std():.4f} | "
                f"min={vals.min():.4f} | max={vals.max():.4f}"
            )

        mean_dpmm = float(np.mean([r["dpmm_bf_sk"] for r in rows]))
        mean_gmm = float(np.mean([r["gmm_bf_sk"] for r in rows]))
        if mean_dpmm > 0.8 and mean_gmm > 0.8:
            log_msg("SUCCESS: High similarity detected between BF and Sklearn across the 10 simulations!")
        else:
            log_msg("WARNING: Low similarity detected. Please check model hyperparameters.")


if __name__ == "__main__":
    run_similarity_test()
