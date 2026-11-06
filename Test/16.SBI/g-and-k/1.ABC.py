from Utils import *
from BayesForge import bf
import os
import pandas as pd

model_name = "1.ABC"

print(f"Running BF parameter recovery for {model_name} (ABC-SMC, g-and-k)")
m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
model = make_model(m)

# DGP ------------------------------------------------
nsim = int(os.environ.get("BF_NSIM", 20))
truths, y = simulate_datasets(m, nsim)

# Inference: one ABC-SMC run per dataset ------------------------------------------------
results = []
for i in range(nsim):
    m.abc(model, data={"N": N_OBS}, observed={"y": y[i]}, params=PARAMS,
          summary_stats="robust", method="smc", num_simulations=200_000,
          progress_bar=False, seed=i)
    results += recovery_rows(i, truths, m.posteriors)
    print(f"sim {i + 1}/{nsim} done")
    free_jax_memory()

# Recovery ------------------------------------------------
plot_recovery(pd.DataFrame(results), model_name=model_name)
