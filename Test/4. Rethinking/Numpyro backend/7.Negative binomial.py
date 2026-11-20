from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "3.Documentation"))
from doc_blocks import doc_model
# The model as published on the documentation page (Gamma-Poisson / negative
# binomial). Same model and priors as the script's old copy: a ~ normal(0, 1),
# b ~ normal(0, 1), phi ~ exponential(1), Lambda ~ gamma(mu*phi, phi),
# y ~ poisson(Lambda).
PAGE_MODEL = doc_model("9. Gamma-Poisson.qmd", 0)

model_name = "7.Negative binomial"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Loading (same dataset & model as 9. Gamma-Poisson.qmd) --------
data_path = m.load.sim_gamma_poisson(only_path=True)
m.data(data_path, sep=',')
m.data_to_model(['log_days', 'monastery', 'y'])

model_BF = PAGE_MODEL

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N;
    array[N] int y;
    vector[N] log_days;
    vector[N] monastery;
}
parameters {
    real a;
    real b;
    real<lower=0> phi;
}
model {
    vector[N] mu;
    a ~ normal(0, 1);
    b ~ normal(0, 1);
    phi ~ exponential(1);
    mu = exp(log_days + a + b * monastery);
    y ~ neg_binomial(mu * phi, phi);
}
"""
data_stan = {
    'N': len(m.df),
    'y': m.df['y'].values.astype(int).tolist(),
    'log_days': m.df['log_days'].values.astype(float).tolist(),
    'monastery': m.df['monastery'].values.astype(float).tolist(),
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a[0]': 'a',
    'b[0]': 'b',
    'phi[0]': 'phi',
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery (DGP-IF pipeline) --------------------
def estimate(log_days, monastery, a_true, b_true, phi_true):
    mu_sim = np.exp(log_days + a_true + b_true * monastery)
    lambda_sim = np.random.gamma(shape=mu_sim * phi_true, scale=1.0 / phi_true)
    y_sim = np.random.poisson(lambda_sim)

    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {
        'log_days': jnp.array(log_days),
        'monastery': jnp.array(monastery),
        'y': jnp.array(y_sim),
    }
    model_rec = PAGE_MODEL

    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    free_jax_memory()
    return s.iloc[:, 0]

def param_recovery(log_days, monastery, a_sims, b_sims, phi_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(log_days, monastery, a_sims[i], b_sims[i], phi_sims[i])
        results.append({'sim': i, 'parameter': 'a[0]', 'simulated': float(a_sims[i]), 'estimations': float(est['a[0]'])})
        results.append({'sim': i, 'parameter': 'b[0]', 'simulated': float(b_sims[i]), 'estimations': float(est['b[0]'])})
        results.append({'sim': i, 'parameter': 'phi[0]', 'simulated': float(phi_sims[i]), 'estimations': float(est['phi[0]'])})

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
np.random.seed(42)
nsim_test = int(os.environ.get("BF_NSIM", 10))
# Simulated truths drawn from the page priors: a ~ normal(0, 1), b ~ normal(0, 1),
# phi ~ exponential(1)
a_sims = np.random.normal(0, 1, nsim_test)
b_sims = np.random.normal(0, 1, nsim_test)
phi_sims = np.random.exponential(1.0, nsim_test)

# Use an exposure schedule with sufficient expected counts so overdispersion phi is well-identified
log_days_sub = np.concatenate([np.repeat(np.log(7.0), 250), np.repeat(np.log(14.0), 250)])
monastery_sub = np.concatenate([np.zeros(250), np.ones(250)])

recovery_results = param_recovery(log_days_sub, monastery_sub, a_sims, b_sims, phi_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
