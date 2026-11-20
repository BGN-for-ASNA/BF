from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "2.Categorical variable"

print(f'Running BF for {model_name}')
m = bf(platform='cpu', backend='tfp')

# import data ------------------------------------------------
data_path = m.load.milk(only_path = True)
m.data(data_path, sep=';') 
# index() only names the column it creates (RELEASE_AUDIT Data M5): rename the
# dotted milk column explicitly so it can be a model argument.
m.df = m.df.rename(columns={'kcal.per.g': 'kcal_per_g'})
m.index(["clade"])
m.scale(['kcal_per_g'])

def model_BF(kcal_per_g, index_clade):
    a = yield m.dist.normal(0, 0.5, shape=(4,), name='a')
    s = yield m.dist.exponential( 1, name='s')    
    mu = a[index_clade]
    yield m.dist.normal(mu, s, obs=kcal_per_g)

m.data_to_model(['kcal_per_g', "index_clade"])
m.fit(model_BF, num_samples=1000, num_warmup=1000) 
m.summary()

print('Running Stan')
stan_code = """
data {
    int<lower=1> N;
    int<lower=1> K;
    vector[N] kcal;
    array[N] int clade;
}
parameters {
    vector[K] a;
    real<lower=0> s;
}
model {
    s ~ exponential(1);
    a ~ normal(0, 0.5);
    kcal ~ normal(a[clade], s);
}
"""
data = {
    'N': len(m.df),
    'K': 4,
    'kcal': m.df.kcal_per_g.values,
    'clade': (m.df.index_clade.values + 1).astype(int)
}

stan_df = build_stan_model(stan_code, data = data, chains=4)

print('Posterior distributions comparison')
param_map = {
    'a[0]': 'a[1]',
    'a[1]': 'a[2]',
    'a[2]': 'a[3]',
    'a[3]': 'a[4]',
    's[0]': 's'
}
BF_df = prepare_bi_data(m)
plot_comparaison(BF_df, stan_df, param_map=param_map, model_name=model_name)

print('Running Parameters recovery')
def model_rec(kcal_per_g, index_clade):
    a = yield m.dist.normal(0, 0.5, shape=(4,), name='a')
    s = yield m.dist.exponential( 1, name='s')    
    mu = a[index_clade]
    yield m.dist.normal(mu, s, obs=kcal_per_g)

def simulate_data(index_clade, a, s):    
    kcal_per_g = m.dist.normal(a[index_clade], s, sample = True)
    return kcal_per_g

def estimate(index_clade, a, s):
    kcal_per_g = simulate_data(index_clade, a, s)
    m_rec = bf(print_devices_found=False, backend='tfp')
    m_rec.df = pd.DataFrame({"kcal_per_g": kcal_per_g, "index_clade": index_clade})
    m_rec.data_to_model(['kcal_per_g', 'index_clade'])
    m_rec.fit(model_rec, num_samples=500, num_warmup=500, progress_bar=False) 
    summary = m_rec.summary()
    return summary.iloc[:,0]

def param_recovery(index_clade, a_sim, s_sim, nsim):
    results = []
    for i in range(nsim):
        estimations = estimate(index_clade, a_sim[i], s_sim[i])
        # a has 4 components
        for j in range(4):
            results.append({
                'sim': i,
                'parameter': f'a[{j}]',
                'simulated': a_sim[i,j],
                'estimations': estimations[f'a[{j}]']
            })
        results.append({
            'sim': i,
            'parameter': 's',
            'simulated': float(s_sim[i]),
            'estimations': float(estimations['s'])
        })

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)    
    return df_res

N = 500
nsim = int(os.getenv('BF_NSIM', 20)) 
a_sim = np.random.normal(0, 0.5, size=(nsim, 4))
s_sim = np.random.exponential(1, size=(nsim,))
index_clade = np.random.choice([0,1,2,3], size=N)

res = param_recovery(index_clade, a_sim, s_sim, nsim = nsim)



# --- WAIC & LOO cross-check: native TFP vs NumPyro/ArviZ reference (same draws) ---
# `waic_ref` is a numpyro-mode transcription of the fitted TFP model (identical
# latent site names), used to evaluate the same posterior draws through
# numpyro.infer.log_likelihood -- the exact machinery ArviZ uses.
m_ref = bf(platform='cpu', print_devices_found=False)

def waic_ref(kcal_per_g, index_clade):
    a = m_ref.dist.normal(0, 0.5, shape=(4,))
    s = m_ref.dist.exponential(1)
    mu = a[index_clade]
    m_ref.dist.normal(mu, s, obs=kcal_per_g)
waic_report(m, model_name, ref_model=waic_ref, ref_kwargs={'kcal_per_g': m.data_on_model['kcal_per_g'], 'index_clade': m.data_on_model['index_clade']})
