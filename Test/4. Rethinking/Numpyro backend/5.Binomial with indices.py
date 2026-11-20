from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "5.Binomial with indices"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Preparation ----------------------------------------
data_path = m.load.chimpanzees(only_path=True)
m.data(data_path, sep=';')
m.df['treatment'] = m.df.prosoc_left + 2 * m.df.condition
m.df['actor'] = m.df['actor'] - 1  # 0-indexed

m.data_to_model(['actor', 'treatment', 'pulled_left'])

def model_BF(actor, treatment, pulled_left):
    a = m.dist.normal(0, 1.5, shape=(7,), name='a')
    b = m.dist.normal(0, 0.5, shape=(4,), name='b')
    p = a[actor] + b[treatment]
    m.dist.binomial(total_count=1, logits=p, obs=pulled_left)

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=0> N;
    int<lower=0> N_actors;
    int<lower=0> N_treatments;
    array[N] int pulled_left;
    array[N] int treatment;
    array[N] int actor;
}
parameters {
    vector[N_actors] a;
    vector[N_treatments] b;
}
model {
    vector[N] p;
    a ~ normal(0, 1.5);
    b ~ normal(0, 0.5);
    for (i in 1:N) {
        p[i] = a[actor[i]] + b[treatment[i]];
    }
    pulled_left ~ binomial_logit(1, p);
}
"""
data_stan = {
    'N': len(m.df),
    'N_actors': 7,
    'N_treatments': 4,
    'pulled_left': m.df.pulled_left.values.astype(int).tolist(),
    'treatment': (m.df.treatment.values + 1).astype(int).tolist(),  # 1-indexed for Stan
    'actor': (m.df.actor.values + 1).astype(int).tolist(),          # 1-indexed for Stan
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a[0]': 'a[1]', 'a[1]': 'a[2]', 'a[2]': 'a[3]', 'a[3]': 'a[4]',
    'a[4]': 'a[5]', 'a[5]': 'a[6]', 'a[6]': 'a[7]',
    'b[0]': 'b[1]', 'b[1]': 'b[2]', 'b[2]': 'b[3]', 'b[3]': 'b[4]',
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery (DGP-IF pipeline) --------------------
def estimate(actor, treatment, a_true, b_true):
    logits = a_true[actor] + b_true[treatment]
    pulled_left_sim = np.random.binomial(1, jax.nn.sigmoid(logits))

    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {
        'actor': jnp.array(actor),
        'treatment': jnp.array(treatment),
        'pulled_left': jnp.array(pulled_left_sim),
    }
    def model_rec(actor, treatment, pulled_left):
        a = m_rec.dist.normal(0, 1.5, shape=(7,), name='a')
        b = m_rec.dist.normal(0, 0.5, shape=(4,), name='b')
        p = a[actor] + b[treatment]
        m_rec.dist.binomial(total_count=1, logits=p, obs=pulled_left)

    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    free_jax_memory()
    return s.iloc[:, 0]

def param_recovery(actor, treatment, a_sims, b_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(actor, treatment, a_sims[i], b_sims[i])
        for j in range(7):
            results.append({'sim': i, 'parameter': f'a[{j}]', 'simulated': float(a_sims[i, j]), 'estimations': float(est[f'a[{j}]'])})
        for k in range(4):
            results.append({'sim': i, 'parameter': f'b[{k}]', 'simulated': float(b_sims[i, k]), 'estimations': float(est[f'b[{k}]'])})

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
a_sims = np.random.normal(0, 1.5, size=(nsim_test, 7))
b_sims = np.random.normal(0, 0.5, size=(nsim_test, 4))

recovery_results = param_recovery(m.df.actor.values, m.df.treatment.values, a_sims, b_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
