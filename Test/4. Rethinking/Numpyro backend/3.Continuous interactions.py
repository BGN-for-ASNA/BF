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
# The model as published on the documentation page. Same likelihood and same
# parameters (sigma, bws, bs, bw, a) and the same priors as the previous local
# copy, for both the main fit and the recovery fit
# (sigma ~ exponential(1), bws/bs/bw ~ normal(0,0.25), a ~ normal(0.5,0.25)).
# The page's argument order is (blooms_scaled, shade_scaled, water_scaled).
PAGE_MODEL = doc_model("3. Interaction between continuous variables.qmd", 0)

model_name = "3.Continuous interactions"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Loading -------------------------------------------
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "BayesForge", "Resources")) + os.sep
df = pd.read_csv(data_path + 'tulips.csv', sep=';')
df['blooms_scaled'] = df.blooms / df.blooms.max()
df['water_scaled'] = df.water - df.water.mean()
df['shade_scaled'] = df.shade - df.shade.mean()

m.df = df
m.data_to_model(['blooms_scaled', 'water_scaled', 'shade_scaled'])

model_BF = PAGE_MODEL

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N;
    vector[N] blooms;
    vector[N] water;
    vector[N] shade;
}
parameters {
    real a;
    real bw;
    real bs;
    real bws;
    real<lower=0> sigma;
}
model {
    vector[N] mu;
    sigma ~ exponential(1);
    bws ~ normal(0, 0.25);
    bs ~ normal(0, 0.25);
    bw ~ normal(0, 0.25);
    a ~ normal(0.5, 0.25);
    mu = a + bw*water + bs*shade + bws*water.*shade;
    blooms ~ normal(mu, sigma);
}
"""
data_stan = {
    'N': len(df),
    'blooms': df['blooms_scaled'].values,
    'water': df['water_scaled'].values,
    'shade': df['shade_scaled'].values
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a': 'a',
    'bw': 'bw',
    'bs': 'bs',
    'bws': 'bws',
    'sigma': 'sigma'
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def sim_blooms(water, shade, sigma, bws, bs, bw, a):
    mu = a + bw*water + bs*shade + bws*water*shade
    return m.dist.normal(mu, sigma, sample=True)

def estimate(water, shade, sigma, bws, bs, bw, a):
    blooms_sim = sim_blooms(water, shade, sigma, bws, bs, bw, a)
    m_rec = bf(print_devices_found=False)
    m_rec.df = pd.DataFrame({"water_scaled": water, "shade_scaled": shade,
                             "blooms_scaled": blooms_sim})

    # Page model; same priors as the old local recovery model.
    model_rec = PAGE_MODEL

    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(water, shade, sigma_true, bws_true, bs_true, bw_true, a_true, nsim):
    results = []
    
    for i in range(nsim):
        est = estimate(water, shade, sigma_true[i], bws_true[i], bs_true[i], bw_true[i], a_true[i])
        for param in ['a', 'bs', 'bw', 'bws', 'sigma']:
            true_val = a_true[i,0] if param == 'a' else (bs_true[i,0] if param == 'bs' else (bw_true[i,0] if param == 'bw' else (bws_true[i,0] if param == 'bws' else sigma_true[i,0])))
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': true_val,
                'estimations': est[param]
            })
        
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
sigma_sim = np.random.exponential(1, size=(nsim_test, 1))
bws_sim = np.random.normal(0, 0.25, size=(nsim_test, 1))
bs_sim = np.random.normal(0, 0.25, size=(nsim_test, 1))
bw_sim = np.random.normal(0, 0.25, size=(nsim_test, 1))
a_sim = np.random.normal(0.5, 0.25, size=(nsim_test, 1))

water_vals = df.water_scaled.values
shade_vals = df.shade_scaled.values

recovery_results = param_recovery(water_vals, shade_vals, sigma_sim, bws_sim, bs_sim, bw_sim, a_sim, nsim=nsim_test)


# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
