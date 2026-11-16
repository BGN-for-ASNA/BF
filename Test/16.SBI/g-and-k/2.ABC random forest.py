from Utils import *
from BayesForge import bf
import os
import pandas as pd

model_name = "2.ABC random forest"

print(f"Running BF parameter recovery for {model_name} (ABC-RF, g-and-k)")
m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
model = make_model(m)

# Candidate summary statistics ------------------------------------------------
probs = jnp.array([0.01, 0.05, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.95, 0.99])
stat_names = [f"q{p:g}" for p in probs.tolist()] + [
    "mean", "sd", "skewness", "kurtosis", "IQR", "bowley_skew", "moors_kurt", "min", "max"]

def stats(y):
    q = jnp.quantile(y, probs)
    z = (y - y.mean()) / y.std()
    iqr = q[7] - q[3]
    return jnp.concatenate([q, jnp.stack([
        y.mean(), y.std(), jnp.mean(z**3), jnp.mean(z**4), iqr,
        (q[7] + q[3] - 2 * q[5]) / iqr, (q[8] - q[6] + q[4] - q[2]) / iqr,
        y.min(), y.max()])])

# DGP ------------------------------------------------
nsim = int(os.environ.get("BF_NSIM", 20))
truths, y = simulate_datasets(m, nsim)

# Inference: one ABC-RF run per dataset ------------------------------------------------
results = []
for i in range(nsim):
    m.abc_rf(model, data={"N": N_OBS}, observed={"y": y[i]}, params=PARAMS,
             summary_stats=stats, stat_names=stat_names, num_simulations=10_000,
             progress_bar=False, seed=i)
    results += recovery_rows(i, truths, m.posteriors)
    print(f"sim {i + 1}/{nsim} done")
    free_jax_memory()

# Recovery ------------------------------------------------
plot_recovery(pd.DataFrame(results), model_name=model_name)
