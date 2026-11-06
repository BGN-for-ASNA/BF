from Utils import *
from BayesForge import bf
import os
import pandas as pd

model_name = "4.Bayesian NPE"

print(f"Running BF parameter recovery for {model_name} (Bayesian NPE, g-and-k)")
m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
model = make_model(m)

# DGP ------------------------------------------------
nsim = int(os.environ.get("BF_NSIM", 20))
truths, y = simulate_datasets(m, nsim)

# Inference: small simulation budget, train once and evaluate every dataset -----------
m.npe(model, data={"N": N_OBS}, observed={"y": y[0]}, params=PARAMS,
      summary_stats="robust", num_simulations=1_000, bayesian=True,
      progress_bar=False, seed=0)

results = []
for i in range(nsim):
    results += recovery_rows(i, truths, m.npe_posterior(y[i], seed=i))

# Recovery ------------------------------------------------
plot_recovery(pd.DataFrame(results), model_name=model_name)

# Calibration over the prior ------------------------------------------------
cov = m.sbc_coverage(n_test=200, levels=[0.5, 0.8, 0.95], plot=False)
print("Expected coverage (nominal vs empirical):")
print(cov.round(3).to_string(index=False))
