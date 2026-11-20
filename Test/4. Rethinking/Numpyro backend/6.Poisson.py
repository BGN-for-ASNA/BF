from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "6.Poisson"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Preparation ----------------------------------------
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "BayesForge", "Resources")) + os.sep
df = pd.read_csv(data_path + 'Kline.csv', sep=';')
# Log-transform population to reduce outlier impact (standard for Kline dataset)
df['log_pop'] = np.log(df['population'])
df['population_scaled'] = (df['log_pop'] - df['log_pop'].mean()) / df['log_pop'].std()
df['cid'] = (df['contact'] == "high").astype(int)

m.df = df
m.data_to_model(['cid', 'population_scaled', 'total_tools'])

def model_BF(cid, population_scaled, total_tools):
    a = m.dist.normal(3, 0.5, shape=(2,), name='a')
    b = m.dist.normal(0, 0.2, shape=(2,), name='b')
    lambda_ = jnp.exp(a[cid] + b[cid] * population_scaled)
    m.dist.poisson(lambda_, obs=total_tools)

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
        lambda[i] = a[cid[i]] + b[cid[i]] * population[i];
    }
    total_tools ~ poisson_log(lambda);
}
"""
data_stan = {
    'N': len(df),
    'total_tools': df['total_tools'].values.astype(int).tolist(),
    'population': df['population_scaled'].values.astype(float).tolist(),
    'cid': (df['cid'].values + 1).astype(int).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a[0]': 'a[1]',
    'a[1]': 'a[2]',
    'b[0]': 'b[1]',
    'b[1]': 'b[2]'
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(cid, population, a_true, b_true):
    lambda_sim = np.exp(a_true[cid] + b_true[cid] * population)
    total_tools_sim = np.random.poisson(lambda_sim)
    
    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {
        'cid': jnp.array(cid),
        'population_scaled': jnp.array(population),
        'total_tools': jnp.array(total_tools_sim)
    }
    def model_rec(cid, population_scaled, total_tools):
        a = m_rec.dist.normal(3, 0.5, shape=(2,))
        b = m_rec.dist.normal(0, 0.5, shape=(2,))
        lambda_ = jnp.exp(a[cid] + b[cid] * population_scaled)
        m_rec.dist.poisson(lambda_, obs=total_tools)
        
    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(cid, population, a_sims, b_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(cid, population, a_sims[i], b_sims[i])
        for j in range(2):
            results.append({'sim': i, 'parameter': f'a[{j}]', 'simulated': float(a_sims[i,j]), 'estimations': float(est[f'a[{j}]'])})
            results.append({'sim': i, 'parameter': f'b[{j}]', 'simulated': float(b_sims[i,j]), 'estimations': float(est[f'b[{j}]'])})
            
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
a_sims = np.random.normal(3, 0.5, size=(nsim_test, 2))
b_sims = np.random.normal(0, 0.4, size=(nsim_test, 2))

# Tile even more observations for stronger signal (100x tiling = 1000 obs)
cid_extended = np.tile(df.cid.values, 100)
pop_extended = np.tile(df.population_scaled.values, 100)
recovery_results = param_recovery(cid_extended, pop_extended, a_sims, b_sims, nsim=nsim_test)


# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
