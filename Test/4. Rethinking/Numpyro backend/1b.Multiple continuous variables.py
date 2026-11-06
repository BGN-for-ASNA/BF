from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "1b.Multiple continuous variables"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Import data ------------------------------------------------
data_path = m.load.howell1(only_path=True)
m.data(data_path, sep=';')
m.df = m.df[m.df.age > 18]
m.scale(['weight', 'age'])

# 2. Define model ------------------------------------------------
def model_BF(height, weight, age):
    alpha = m.dist.normal(178, 20, name='alpha')
    beta1 = m.dist.normal(0, 0.5, name='beta1')
    beta2 = m.dist.normal(0, 0.5, name='beta2')
    sigma = m.dist.uniform(0, 50, name='sigma')
    m.dist.normal(alpha + beta1 * weight + beta2 * age, sigma, obs=height)

# 3. Run sampler ------------------------------------------------
m.fit(model_BF, num_samples=1000, num_warmup=1000)
m.summary()

# 4. Run Stan ------------------------------------------------
print('Running Stan')
stan_code = """
data {
  int<lower=0> N;
  vector[N] height;
  vector[N] weight;
  vector[N] age;
}
parameters {
  real alpha;
  real beta1;
  real beta2;
  real<lower=0, upper=50> sigma;
}
model {
  vector[N] mu;
  alpha ~ normal(178, 20);
  beta1 ~ normal(0, 0.5);
  beta2 ~ normal(0, 0.5);
  sigma ~ uniform(0, 50);
  mu = alpha + beta1 * weight + beta2 * age;
  height ~ normal(mu, sigma);
}
"""
data = {
    'N': len(m.df),
    'height': m.df.height.values,
    'weight': m.df.weight.values,
    'age': m.df.age.values,
}

stan_df = build_stan_model(stan_code, data=data, chains=4)

print('Posterior distributions comparison')
param_map = {
    'alpha': 'alpha',
    'beta1': 'beta1',
    'beta2': 'beta2',
    'sigma': 'sigma',
}
plot_comparaison(m, stan_df, param_map=param_map, model_name=model_name)

# 5. Parameter Recovery (DGP-IF pipeline) ------------------------------------------------
print('Running Parameters recovery')
def model_rec(height, weight, age):
    alpha = m.dist.normal(0, 1, name='alpha')
    beta1 = m.dist.normal(0, 0.5, name='beta1')
    beta2 = m.dist.normal(0, 0.5, name='beta2')
    sigma = m.dist.exponential(1, name='sigma')
    m.dist.normal(alpha + beta1 * weight + beta2 * age, sigma, obs=height)

def simulate_height(weight, age, alpha, beta1, beta2, sigma):
    weight_scaled = (weight - weight.mean()) / weight.std()
    age_scaled = (age - age.mean()) / age.std()
    height = m.dist.normal(alpha + beta1 * weight_scaled + beta2 * age_scaled, sigma, sample=True)
    return weight_scaled, age_scaled, height

def estimate(weight, age, alpha, beta1, beta2, sigma):
    weight_scaled, age_scaled, height = simulate_height(weight, age, alpha, beta1, beta2, sigma)
    m_rec = bf(print_devices_found=False)
    m_rec.df = pd.DataFrame({"weight": weight_scaled, "age": age_scaled, "height": height})
    m_rec.data_to_model(['weight', 'age', 'height'])
    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    sum_df = m_rec.summary()
    free_jax_memory()
    return sum_df.iloc[:, 0]

def param_recovery(weight_data, age_data, alpha_sims, beta1_sims, beta2_sims, sigma_sims, nsim):
    results = []
    for i in range(nsim):
        estimations = estimate(
            weight_data[i, :], age_data[i, :],
            alpha_sims[i, :], beta1_sims[i, :], beta2_sims[i, :], sigma_sims[i, :]
        )
        for param, sims in [('alpha', alpha_sims), ('beta1', beta1_sims), ('beta2', beta2_sims), ('sigma', sigma_sims)]:
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': float(sims[i, 0]),
                'estimations': float(estimations[param]),
            })

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

N = 200
nsim = int(os.environ.get("BF_NSIM", 10))
alpha_sims = np.random.normal(0, 1, size=(nsim, 1))
beta1_sims = np.random.normal(0, 0.5, size=(nsim, 1))
beta2_sims = np.random.normal(0, 0.5, size=(nsim, 1))
sigma_sims = np.random.exponential(1, size=(nsim, 1))
weight_data = np.random.normal(80, 30, size=(nsim, N))
age_data = np.random.normal(40, 15, size=(nsim, N))

res = param_recovery(weight_data, age_data, alpha_sims, beta1_sims, beta2_sims, sigma_sims, nsim=nsim)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
