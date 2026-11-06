from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "15.Measuring error"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Loading (same dataset & model as 20. Measuring error.qmd) --------
data_path = m.load.WaffleDivorce(only_path=True)
m.data(data_path, sep=';')
m.scale(['MedianAgeMarriage', 'Marriage'])
dat = dict(
    D_obs=m.z_score(m.df['Divorce'].values),
    D_sd=jnp.array(m.df['Divorce SE'].values / m.df['Divorce'].std()),
    A=jnp.array(m.df['MedianAgeMarriage'].values),
    M=jnp.array(m.df['Marriage'].values),
    N=m.df.shape[0],
)
m.data_on_model = dat

def model_BF(D_obs, D_sd, A, N, M):
    a = m.dist.normal(0, 0.2, name='a')
    beta = m.dist.normal(0, 0.5, name='beta')
    eta = m.dist.normal(0, 0.5, name='eta')
    s = m.dist.exponential(1, name='s')
    mu = a + beta * A + eta * M
    D_true = m.dist.normal(mu, s, name='D_true')
    m.dist.normal(D_true, D_sd, obs=D_obs)

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N;
    vector[N] D_obs;
    vector<lower=0>[N] D_sd;
    vector[N] A;
    vector[N] M;
}
parameters {
    real a;
    real beta;
    real eta;
    real<lower=0> s;
    vector[N] D_true;
}
model {
    vector[N] mu;
    a ~ normal(0, 0.2);
    beta ~ normal(0, 0.5);
    eta ~ normal(0, 0.5);
    s ~ exponential(1);
    mu = a + beta * A + eta * M;
    D_true ~ normal(mu, s);
    D_obs ~ normal(D_true, D_sd);
}
"""
data_stan = {
    'N': int(dat['N']),
    'D_obs': np.asarray(dat['D_obs'], dtype=float).tolist(),
    'D_sd': np.asarray(dat['D_sd'], dtype=float).tolist(),
    'A': np.asarray(dat['A'], dtype=float).tolist(),
    'M': np.asarray(dat['M'], dtype=float).tolist(),
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a': 'a',
    'beta': 'beta',
    'eta': 'eta',
    's': 's',
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery (DGP-IF pipeline) --------------------
def estimate(A_vec, M_vec, D_sd_vec, a_true, beta_true, eta_true, s_true):
    mu_true = a_true + beta_true * A_vec + eta_true * M_vec
    D_true_sim = np.random.normal(mu_true, s_true)
    D_obs_sim = np.random.normal(D_true_sim, D_sd_vec)

    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {
        'D_obs': jnp.array(D_obs_sim),
        'D_sd': jnp.array(D_sd_vec),
        'A': jnp.array(A_vec),
        'M': jnp.array(M_vec),
        'N': len(A_vec),
    }
    def model_rec(D_obs, D_sd, A, N, M):
        a = m_rec.dist.normal(0, 0.2, name='a')
        beta = m_rec.dist.normal(0, 0.5, name='beta')
        eta = m_rec.dist.normal(0, 0.5, name='eta')
        s = m_rec.dist.exponential(1, name='s')
        mu = a + beta * A + eta * M
        D_true = m_rec.dist.normal(mu, s, name='D_true')
        m_rec.dist.normal(D_true, D_sd, obs=D_obs)

    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    summ = m_rec.summary()
    free_jax_memory()
    return summ.iloc[:, 0]

def param_recovery(A_vec, M_vec, D_sd_vec, a_sims, beta_sims, eta_sims, s_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(A_vec, M_vec, D_sd_vec, a_sims[i], beta_sims[i], eta_sims[i], s_sims[i])
        for param, sims in [('a', a_sims), ('beta', beta_sims), ('eta', eta_sims), ('s', s_sims)]:
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': float(sims[i]),
                'estimations': float(est[param]),
            })

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
a_sims = np.random.normal(0, 0.2, size=nsim_test)
beta_sims = np.random.normal(0, 0.5, size=nsim_test)
eta_sims = np.random.normal(0, 0.5, size=nsim_test)
s_sims = np.random.uniform(0.3, 1.2, size=nsim_test)

A_ext = np.tile(np.asarray(dat['A']), 4)
M_ext = np.tile(np.asarray(dat['M']), 4)
D_sd_ext = np.tile(np.asarray(dat['D_sd']), 4)

recovery_results = param_recovery(A_ext, M_ext, D_sd_ext, a_sims, beta_sims, eta_sims, s_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)

