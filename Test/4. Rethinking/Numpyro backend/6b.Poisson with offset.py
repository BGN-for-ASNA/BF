from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "6b.Poisson with offset"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Simulation (same DGP as 8. Poisson mode with offset.qmd) --------
np.random.seed(42)
N = 100
population = np.random.normal(0, 1, size=N)
cid = np.random.binomial(1, 0.5, size=N)
hours = np.random.uniform(1, 10, size=N)

true_a = np.array([2.5, 3.5])
true_b = np.array([-0.1, 0.2])

l = hours * np.exp(true_a[cid] + true_b[cid] * population)
total_tools = np.random.poisson(l)

m.data_on_model = dict(
    cid=jnp.array(cid, dtype=jnp.int32),
    population=jnp.array(population),
    hours=jnp.array(hours),
    total_tools=jnp.array(total_tools, dtype=jnp.int32),
)

def model_BF(cid, population, hours, total_tools):
    a = m.dist.normal(3, 0.5, shape=(2,), name='a')
    b = m.dist.normal(0, 0.2, shape=(2,), name='b')
    l = hours * jnp.exp(a[cid] + b[cid] * population)
    m.dist.poisson(l, obs=total_tools)

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N;
    array[N] int total_tools;
    vector[N] population;
    vector[N] hours;
    array[N] int cid;
}
parameters {
    vector[2] a;
    vector[2] b;
}
model {
    vector[N] lambda;
    a ~ normal(3, 0.5);
    b ~ normal(0, 0.2);
    for (i in 1:N) {
        lambda[i] = hours[i] * exp(a[cid[i]] + b[cid[i]] * population[i]);
    }
    total_tools ~ poisson(lambda);
}
"""
data_stan = {
    'N': N,
    'total_tools': total_tools.astype(int).tolist(),
    'population': population.astype(float).tolist(),
    'hours': hours.astype(float).tolist(),
    'cid': (cid + 1).astype(int).tolist(),
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a[0]': 'a[1]',
    'a[1]': 'a[2]',
    'b[0]': 'b[1]',
    'b[1]': 'b[2]',
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery (DGP-IF pipeline) --------------------
def estimate(cid, population, hours, a_true, b_true):
    lambda_sim = hours * np.exp(a_true[cid] + b_true[cid] * population)
    total_tools_sim = np.random.poisson(lambda_sim)

    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {
        'cid': jnp.array(cid, dtype=jnp.int32),
        'population': jnp.array(population),
        'hours': jnp.array(hours),
        'total_tools': jnp.array(total_tools_sim, dtype=jnp.int32),
    }
    def model_rec(cid, population, hours, total_tools):
        a = m_rec.dist.normal(3, 0.5, shape=(2,), name='a')
        b = m_rec.dist.normal(0, 0.2, shape=(2,), name='b')
        l = hours * jnp.exp(a[cid] + b[cid] * population)
        m_rec.dist.poisson(l, obs=total_tools)

    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    free_jax_memory()
    return s.iloc[:, 0]

def param_recovery(cid, population, hours, a_sims, b_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(cid, population, hours, a_sims[i], b_sims[i])
        for j in range(2):
            results.append({'sim': i, 'parameter': f'a[{j}]', 'simulated': float(a_sims[i, j]), 'estimations': float(est[f'a[{j}]'])})
            results.append({'sim': i, 'parameter': f'b[{j}]', 'simulated': float(b_sims[i, j]), 'estimations': float(est[f'b[{j}]'])})

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
a_sims = np.random.normal(3, 0.5, size=(nsim_test, 2))
b_sims = np.random.normal(0, 0.2, size=(nsim_test, 2))

recovery_results = param_recovery(cid, population, hours, a_sims, b_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)

