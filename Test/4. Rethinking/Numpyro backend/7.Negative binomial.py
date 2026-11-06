from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "7.Negative binomial"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Loading (same dataset & model as 9. Gamma-Poisson.qmd) --------
data_path = m.load.sim_gamma_poisson(only_path=True)
m.data(data_path, sep=',')
m.data_to_model(['log_days', 'monastery', 'y'])

def model_BF(log_days, monastery, y):
    a = m.dist.normal(0, 1, name='a', shape=(1,))
    b = m.dist.normal(0, 1, name='b', shape=(1,))
    phi = m.dist.exponential(1, name='phi', shape=(1,))
    mu = jnp.exp(log_days + a + b * monastery)
    Lambda = m.dist.gamma(concentration=mu * phi, rate=phi, name='Lambda')
    m.dist.poisson(rate=Lambda, obs=y)

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
    def model_rec(log_days, monastery, y):
        a = m_rec.dist.normal(0, 1, name='a', shape=(1,))
        b = m_rec.dist.normal(0, 1, name='b', shape=(1,))
        phi = m_rec.dist.exponential(1, name='phi', shape=(1,))
        mu = jnp.exp(log_days + a + b * monastery)
        Lambda = m_rec.dist.gamma(concentration=mu * phi, rate=phi, name='Lambda')
        m_rec.dist.poisson(rate=Lambda, obs=y)

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
a_sims = np.random.uniform(0.2, 1.2, nsim_test)
b_sims = np.random.uniform(-0.8, 0.8, nsim_test)
phi_sims = np.random.uniform(0.3, 2.5, nsim_test)

# Use an exposure schedule with sufficient expected counts so overdispersion phi is well-identified
log_days_sub = np.concatenate([np.repeat(np.log(7.0), 250), np.repeat(np.log(14.0), 250)])
monastery_sub = np.concatenate([np.zeros(250), np.ones(250)])

recovery_results = param_recovery(log_days_sub, monastery_sub, a_sims, b_sims, phi_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
