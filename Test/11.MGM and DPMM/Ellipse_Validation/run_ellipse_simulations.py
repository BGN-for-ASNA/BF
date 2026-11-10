"""Simulation et validation visuelle des ellipses du DPMM sur 10 datasets aléatoires.

Chaque itération génère :
- Un nombre aléatoire de clusters (ex: 3 à 8).
- Des séparations et dispersions variées avec seeds distinctes.
- L'ajustement du modèle DPMM.
- La sauvegarde des figures dans plots/ avec titre clair (vrais clusters vs détectés).
- Reprend automatiquement là où elle s'est arrêtée si une simulation a déjà été générée.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from BayesForge import bf

HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(HERE, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

N_SIMULATIONS = 10
N_SAMPLES = 400
N_FEATURES = 2
NUM_WARMUP = 250
NUM_MCMC_SAMPLES = 250

SEEDS = [61755, 62809, 14920, 83712, 49201, 75019, 31940, 92831, 58102, 27491]


def run_ellipse_simulations(n_sims=N_SIMULATIONS, out_dir=PLOTS_DIR):
    m = bf(platform="cpu")
    summary_path = os.path.join(out_dir, "simulation_summary.csv")
    
    if os.path.exists(summary_path):
        results = pd.read_csv(summary_path).to_dict("records")
        done_sims = {r["sim"] for r in results}
    else:
        results = []
        done_sims = set()

    print(f"=== Lancement de {n_sims} simulations aléatoires pour validation des ellipses DPMM ===")
    print(f"Dossier de sortie : {out_dir}\n")

    for i in range(1, n_sims + 1):
        if i in done_sims:
            print(f"[{i:02d}/{n_sims:02d}] Déjà calculé, passage à la suivante.")
            continue

        rng_seed = SEEDS[i - 1]
        np.random.seed(rng_seed)
        n_centers = int(np.random.randint(3, 8))
        cluster_std = float(np.round(np.random.uniform(0.6, 1.1), 2))

        X, y_true = make_blobs(
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            centers=n_centers,
            cluster_std=cluster_std,
            center_box=(-10.0, 10.0),
            random_state=rng_seed,
        )

        print(f"[{i:02d}/{n_sims:02d}] Seed={rng_seed} | Vrais clusters={n_centers} | std={cluster_std}...")

        m.data_on_model = {"data": X}
        m.fit(
            m.models.dpmm,
            num_samples=NUM_MCMC_SAMPLES,
            num_warmup=NUM_WARMUP,
            num_chains=2,
            progress_bar=False,
        )

        w, mu, sigma, Lcorr, labels = m.models.dpmm.predict(
            X, m.sampler, threshold=0.5, linkage_method="complete"
        )
        n_detected = len(np.unique(labels))

        title = (
            f"Sim {i:02d} — Vrais={n_centers} clusters | Détectés={n_detected} clusters\n"
            f"(seed={rng_seed}, std={cluster_std}, n={N_SAMPLES})"
        )
        fig, ax = m.plot(
            X,
            m.sampler,
            title=title,
            threshold=0.5,
            linkage_method="complete",
            figsize=(11, 8),
        )

        plot_path = os.path.join(out_dir, f"sim_{i:02d}_vrais_{n_centers}_detectes_{n_detected}.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close("all")

        results.append({
            "sim": i,
            "seed": rng_seed,
            "true_centers": n_centers,
            "detected_clusters": n_detected,
            "cluster_std": cluster_std,
            "plot_path": plot_path,
        })
        print(f"       -> Plot sauvé : {os.path.basename(plot_path)}")
        pd.DataFrame(results).to_csv(summary_path, index=False)

    print(f"\nTerminé ! 10/10 simulations sauvegardées dans {out_dir}")
    return pd.DataFrame(results)


if __name__ == "__main__":
    run_ellipse_simulations()
