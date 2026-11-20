#%%
from Utils import *
from BayesForge import bf, BF
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
import numpyro
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "3.Documentation"))
from doc_blocks import doc_model
# The model as published on the documentation page.
# Priors identical to the previous local copy (a,b,g ~ exponential(1), etasq ~ exponential(2),
# rhosq ~ exponential(0.5)); the page has no 'a_over_g'/'k' deterministic sites.
PAGE_MODEL = doc_model("19. Gaussian processes.qmd", 0)

model_name = "13.Gaussian processes"

print(f'Running BF for {model_name}')
# Setup device------------------------------------------------
m = bf(platform='cpu')

# Import Data & Data Manipulation ------------------------------------------------
# Import
data_path = m.load.kline2(only_path=True)
m.data(data_path, sep=';') 

from importlib.resources import files
data_path2 = files('BayesForge.Resources') / 'islandsDistMatrix.csv'
islandsDistMatrix = pd.read_csv(data_path2, index_col=0)

m.data_to_model(['total_tools', 'population'])
m.data_on_model["society"] = jnp.arange(0, 10)
m.data_on_model["Dmat"] = islandsDistMatrix.values

model_BF = PAGE_MODEL

#%%
# Run sampler ------------------------------------------------
print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=2000, target_accept_prob=0.9, dense_mass=True)
print("BF Summary:")
print(m.summary())

#%%
# 3. Stan Model 

stan_code = """
functions{
  matrix cov_GPL2(matrix x, real sq_alpha, real sq_rho, real delta) {
    int N = dims(x)[1];
    matrix[N, N] K;
    for (i in 1:(N-1)) {
      K[i, i] = sq_alpha + delta;
      for (j in (i + 1):N) {
        K[i, j] = sq_alpha * exp(-sq_rho * square(x[i,j]) );
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = sq_alpha + delta;
    return K;
  }
}

data{
  array[10] int T;
  array[10] int society;
  array[10] int P;
  matrix[10,10] Dmat;
}

parameters{
  real<lower=0> a;
  real<lower=0> b;
  real<lower=0> etasq;
  real<lower=0> g;
  real<lower=0> rhosq;
  vector[10] k;
}

model{
  vector[10] lambda;
  matrix[10,10] SIGMA;
  rhosq ~ exponential(0.5);
  etasq ~ exponential(2);
  a ~ exponential(1);
  b ~ exponential(1);
  g ~ exponential(1);

  SIGMA = cov_GPL2(Dmat, etasq, rhosq, 0.01);
  k ~ multi_normal(rep_vector(0,10), SIGMA);
  for (i in 1:10) {
    lambda[i] = (a * P[i]^b / g) * exp(k[society[i]]);
  }
  T ~ poisson(lambda);
}
"""
data = {
    'T'      : m.df["total_tools"].values.astype(int),
    'P'      : m.df["population"].values.astype(int),
    'society': np.array(m.data_on_model['society'] + 1).astype(int),
    'Dmat'   : np.array(islandsDistMatrix),
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data= data, chains=4)

#%%
# 4. Comparison
param_map = {'a': 'a', 'b': 'b', 'g': 'g', 'etasq': 'etasq', 'rhosq': 'rhosq'}
BF_df = prepare_bi_data(m)
plot_comparaison(BF_df, df_stan, param_map, model_name=model_name)

#%%
# 5. Parameter Recovery Analysis
def param_recovery(Dm, P_vals, a_sims, b_sims, g_sims, etasq_sims, rhosq_sims, society, nsim):
    results = []
    model_rec = PAGE_MODEL
    # Keep original 10 islands
    Dm_rec = Dm
    P_rec = P_vals
    
    for i in range(nsim):
        a_true = a_sims[i]
        b_true = b_sims[i]
        g_true = g_sims[i]
        etasq_true = etasq_sims[i]
        rhosq_true = rhosq_sims[i]
        
        n = Dm_rec.shape[0]
        K_sim = etasq_true * jnp.exp(-rhosq_true * jnp.square(Dm_rec))
        K_sim = K_sim.at[jnp.diag_indices(n)].add(0.01)
        k_sim = m.dist.multivariate_normal(jnp.zeros(n), K_sim, sample=True)
    
        lambda_sim = a_true * P_rec**b_true / g_true * jnp.exp(k_sim[society])
        total_tools_sim = m.dist.poisson(lambda_sim, sample=True)
    
        m.data_on_model = {
            'population': jnp.array(P_rec, dtype=jnp.float32),
            'total_tools': total_tools_sim,
            'Dmat': jnp.array(Dm_rec, dtype=jnp.float32),
            'society': society,
        }
        
        m.fit(model_rec, num_samples=1000, num_warmup=2000, progress_bar=False,
              target_accept_prob=0.9, dense_mass=True, shard=False)
        est = m.summary().iloc[:, 0]
        
        param_map_rec = {
            'a':       (a_true,                   est['a']),
            'b':       (b_true,                   est['b']),
            'g':       (g_true,                   est['g']),
            'etasq':   (etasq_true,               est['etasq']),
            'rhosq':   (rhosq_true,               est['rhosq']),
            'a_over_g':(a_true / g_true,       est['a'] / est['g']),
        }
        for param, (true_val, est_val) in param_map_rec.items():
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': float(np.squeeze(true_val)),
                'estimations': float(est_val),
            })

        # Each fit compiles a new executable; free it before the next iteration
        free_jax_memory()

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("\nRunning Parameter Recovery...")
nsim = int(os.environ.get("BF_NSIM", 5))
# Sample true params from the page model's priors, untruncated:
# a, b, g ~ exponential(1); etasq ~ exponential(2); rhosq ~ exponential(0.5)
a_sims     = np.random.exponential(1.0, size=(nsim,))
g_sims     = np.random.exponential(1.0, size=(nsim,))
b_sims     = np.random.exponential(1.0, size=(nsim,))
etasq_sims = np.random.exponential(0.5,  # rate 2 -> numpy scale 1/2
             size=(nsim,))
rhosq_sims = np.random.exponential(2.0,  # rate 0.5 -> numpy scale 2
             size=(nsim,))

Dm     = m.data_on_model['Dmat']
P_vals = m.data_on_model['population']
society= m.data_on_model["society"]
res = param_recovery(Dm, P_vals, a_sims, b_sims, g_sims, etasq_sims, rhosq_sims, society, nsim = nsim)



# %%


# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
