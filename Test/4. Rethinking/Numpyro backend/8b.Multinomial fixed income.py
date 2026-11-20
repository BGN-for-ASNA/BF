from Utils import *
from BayesForge import bf
import pandas as pd
import os
from pathlib import Path
import numpy as np
import jax.numpy as jnp
import jax
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "3.Documentation"))
from doc_blocks import doc_model
# The model as published on the documentation page. Priors are identical to the previous local
# copy: alpha ~ normal(0, 1) and beta ~ normal(0, 1), both shape (2,). The page hardcodes
# total_count=20 in the multinomial likelihood, which is the trial count of this data set
# (n_trials below), so the likelihood is the same.
PAGE_MODEL = doc_model("11. Multinomial model.qmd", 0)

model_name = "8b.Multinomial fixed income"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data (same model as 11. Multinomial model.qmd) -------------------
# Simulated by make_multinomial_fixed_income() in ../cases/data/make_data.py,
# shared with the cross-language case '../cases/8b.Multinomial fixed income.md'.
df_data = pd.read_csv(Path(__file__).resolve().parent.parent / 'cases' / 'data' / f'{model_name}.csv')
X = df_data['X'].to_numpy()
Y = df_data[['Y0', 'Y1', 'Y2']].to_numpy(dtype=int)
N = len(X)
n_trials = int(Y[0].sum())
np.random.seed(42)  # parameter recovery below draws from the numpy global state

m.data_on_model = dict(X=jnp.array(X), Y=jnp.array(Y))

model_BF = PAGE_MODEL

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N;
    int<lower=2> K;
    vector[N] X;
    array[N, K] int Y;
}
parameters {
    vector[K-1] alpha;
    vector[K-1] beta;
}
model {
    alpha ~ normal(0, 1);
    beta ~ normal(0, 1);
    for (n in 1:N) {
        vector[K] phi;
        phi[1] = alpha[1] + beta[1] * X[n];
        phi[2] = alpha[2] + beta[2] * X[n];
        phi[3] = 0;
        Y[n] ~ multinomial(softmax(phi));
    }
}
"""
data_stan = {
    'N': N,
    'K': 3,
    'X': X.tolist(),
    'Y': Y.astype(int).tolist(),
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'alpha[0]': 'alpha[1]',
    'alpha[1]': 'alpha[2]',
    'beta[0]': 'beta[1]',
    'beta[1]': 'beta[2]',
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery (DGP-IF pipeline) --------------------
def estimate(X_sim, alpha_true, beta_true):
    N_sim = len(X_sim)
    phi = np.column_stack([
        alpha_true[0] + beta_true[0] * X_sim,
        alpha_true[1] + beta_true[1] * X_sim,
        np.zeros(N_sim),
    ])
    theta = np.asarray(jax.nn.softmax(jnp.array(phi), axis=1))
    Y_sim = np.array([np.random.multinomial(n_trials, theta[i]) for i in range(N_sim)])

    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {
        'X': jnp.array(X_sim),
        'Y': jnp.array(Y_sim),
    }
    model_rec = PAGE_MODEL

    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    free_jax_memory()
    return s.iloc[:, 0]

def param_recovery(X_sim, alpha_sims, beta_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(X_sim, alpha_sims[i], beta_sims[i])
        results.append({'sim': i, 'parameter': 'alpha[0]', 'simulated': float(alpha_sims[i, 0]), 'estimations': float(est['alpha[0]'])})
        results.append({'sim': i, 'parameter': 'alpha[1]', 'simulated': float(alpha_sims[i, 1]), 'estimations': float(est['alpha[1]'])})
        results.append({'sim': i, 'parameter': 'beta[0]', 'simulated': float(beta_sims[i, 0]), 'estimations': float(est['beta[0]'])})
        results.append({'sim': i, 'parameter': 'beta[1]', 'simulated': float(beta_sims[i, 1]), 'estimations': float(est['beta[1]'])})

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
# Drawn from the page priors: alpha ~ normal(0, 1) shape (2,), beta ~ normal(0, 1) shape (2,)
alpha_sims = np.random.normal(0, 1.0, size=(nsim_test, 2))
beta_sims = np.random.normal(0, 1.0, size=(nsim_test, 2))
X_rec = np.random.normal(0, 1, size=200)

recovery_results = param_recovery(X_rec, alpha_sims, beta_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
