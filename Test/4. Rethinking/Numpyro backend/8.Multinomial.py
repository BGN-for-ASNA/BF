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
# The model as published on the documentation page. The categorical page is the page of this
# model (one career per person); the multinomial page models Y with 20 trials per observation.
# Priors are identical to the previous local copy: a ~ normal(0, 1) shape (2,), b ~ half_normal(1.0).
PAGE_MODEL = doc_model("10. Categorical model.qmd", 0)

model_name = "8.Multinomial"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Simulation ----------------------------------------
# To identify the 'b' parameter in a multinomial model, the covariate (income) 
# must vary across observations. If all observations see the same incomes, 
# 'b' is confounded with the intercepts 'a'.
N = 1000
np.random.seed(1)
# Each observation sees 3 careers with different incomes
income_sim = np.random.uniform(1, 10, size=(N, 3))

a_true_gen = np.array([0.5, -0.2]) # a3 is pivot 0
b_true_gen = 0.4

# Calculate probabilities for each observation
# scores: (N, 3)
s1 = a_true_gen[0] + b_true_gen * income_sim[:, 0]
s2 = a_true_gen[1] + b_true_gen * income_sim[:, 1]
s3 = b_true_gen * income_sim[:, 2]
scores = np.stack([s1, s2, s3], axis=1)
p_true = jax.nn.softmax(scores, axis=1)

career = np.array([np.random.choice([0, 1, 2], p=p) for p in p_true])
df = pd.DataFrame({
    'career': career,
    'inc0': income_sim[:, 0],
    'inc1': income_sim[:, 1],
    'inc2': income_sim[:, 2]
})

m.df = df
m.data_to_model(['career', 'inc0', 'inc1', 'inc2'])

model_BF = PAGE_MODEL

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int N;
    array[N] int career;
    vector[N] inc0;
    vector[N] inc1;
    vector[N] inc2;
}
parameters {
    vector[2] a;
    real<lower=0> b;
}
model {
    a ~ normal(0, 1);
    b ~ normal(0, 1);
    for (n in 1:N) {
        vector[3] s;
        s[1] = a[1] + b * inc0[n];
        s[2] = a[2] + b * inc1[n];
        s[3] = b * inc2[n];
        career[n] ~ categorical_logit(s);
    }
}
"""
data_stan = {
    'N': N,
    'career': (df.career.values + 1).astype(int).tolist(),
    'inc0': df.inc0.values.tolist(),
    'inc1': df.inc1.values.tolist(),
    'inc2': df.inc2.values.tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a[0]': 'a[1]',
    'a[1]': 'a[2]',
    'b': 'b'
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(income_sim, a_true, b_true):
    s1 = a_true[0] + b_true * income_sim[:, 0]
    s2 = a_true[1] + b_true * income_sim[:, 1]
    s3 = b_true * income_sim[:, 2]
    scores = np.stack([s1, s2, s3], axis=1)
    p = jax.nn.softmax(scores, axis=1)
    career_sim = np.array([np.random.choice([0, 1, 2], p=pp) for pp in p])
    
    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {
        'career': jnp.array(career_sim),
        'inc0': jnp.array(income_sim[:, 0]),
        'inc1': jnp.array(income_sim[:, 1]),
        'inc2': jnp.array(income_sim[:, 2])
    }
    model_rec = PAGE_MODEL
    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(income_sim, a_sims, b_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(income_sim, a_sims[i], b_sims[i])
        results.append({'sim': i, 'parameter': 'a[0]', 'simulated': float(a_sims[i,0]), 'estimations': float(est['a[0]'])})
        results.append({'sim': i, 'parameter': 'a[1]', 'simulated': float(a_sims[i,1]), 'estimations': float(est['a[1]'])})
        results.append({'sim': i, 'parameter': 'b', 'simulated': float(b_sims[i, 0]), 'estimations': float(est['b'])})
            
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 100)) 
a_sims = np.random.normal(0, 1, size=(nsim_test, 2))
b_sims = np.abs(np.random.normal(0, 1.0, size=(nsim_test, 1)))

# Generate varied income for recovery test
income_recovery = np.random.uniform(1, 10, size=(1000, 3))
recovery_results = param_recovery(income_recovery, a_sims, b_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
