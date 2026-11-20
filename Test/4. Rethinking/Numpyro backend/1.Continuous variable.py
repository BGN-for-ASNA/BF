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
# parameters as the previous local copy; priors identical for the main fit
# (a ~ normal(178,20), b ~ log_normal(0,1), s ~ uniform(0,50)).
# The recovery model differs: this page uses a ~ normal(178,20),
# b ~ log_normal(0,1), s ~ uniform(0,50) where the old local recovery model used
# a ~ normal(0,1), b ~ normal(0,1), s ~ exponential(1).
PAGE_MODEL = doc_model("1. Linear Regression for continuous variable.qmd", 0)

model_name = "1.Continuous variable"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# import data ------------------------------------------------
data_path = m.load.howell1(only_path = True)
m.data(data_path, sep=';') 
m.df = m.df[m.df.age > 18]
m.scale(['weight'])

# define model ------------------------------------------------
model_BF = PAGE_MODEL

# Run sampler ------------------------------------------------
m.fit(model_BF) 
m.summary()

print('Running Stan')
stan_code = """
data{
  int<lower=0> N;
  vector[N] height;
  vector[N] weight;
}
parameters{
  real a;
  real<lower=0> b;
  real<lower=0,upper=50> s;
}
model{
  vector[N] mu;
  s ~ uniform( 0 , 50 );
  b ~ lognormal( 0 , 1 );
  a ~ normal( 178 , 20 );
  for ( i in 1:N ) {
    mu[i] = a + b* weight[i] ;
  }
  height ~ normal( mu , s );  
}
"""
data = {
  'N': len(m.df),
  'height': m.df.height.values,
  'weight': m.df.weight.values,
}

stan_df = build_stan_model(stan_code, data = data, chains=4)

print('Posterior distributions comparison')
plot_comparaison(m, stan_df, model_name=model_name)

print('Running Parameters recovery')
# Page priors (a ~ normal(178,20), b ~ log_normal(0,1), s ~ uniform(0,50)) replace
# the old recovery priors (a ~ normal(0,1), b ~ normal(0,1), s ~ exponential(1)).
model_rec = PAGE_MODEL

def simulate_height(weight, a, b, s):    
    weight_scaled = (weight - weight.mean())/weight.std()
    height = m.dist.normal( a + b * weight_scaled , s, sample = True)
    return weight_scaled, height

def estimate(weight, a, b, s):
    weight_scaled, height = simulate_height(weight, a, b, s)
    m_rec = bf(print_devices_found=False)
    m_rec.df = pd.DataFrame({"weight": weight_scaled, "height": height})
    m_rec.data_to_model(['weight', 'height'])
    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False) 
    sum_df = m_rec.summary()
    return sum_df.iloc[:,0]

def param_recovery(weight_data, a_sims, b_sims, s_sims, nsim):
    results = []
    for i in range(nsim):
        estimations = estimate(weight_data[i,:], a_sims[i,:], b_sims[i,:], s_sims[i,:])
        for param in ['a', 'b', 's']:
            true_val = a_sims[i,0] if param == 'a' else (b_sims[i,0] if param == 'b' else s_sims[i,0])
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': float(true_val),
                'estimations': float(estimations[param])
            })
    
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

N = 200
nsim = int(os.environ.get("BF_NSIM", 10))
a_sims = np.random.normal(178, 20, size=(nsim, 1))
b_sims = np.random.lognormal(0, 1, size=(nsim, 1))
s_sims = np.random.uniform(0, 50, size=(nsim, 1))
weight_data = np.random.normal(80, 30, size=(nsim, N))

res = param_recovery(weight_data, a_sims, b_sims, s_sims, nsim = nsim)


# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
