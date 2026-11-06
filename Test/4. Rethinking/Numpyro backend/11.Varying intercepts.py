from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "11.Varying intercepts"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Loading -------------------------------------------
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "BayesForge", "Resources")) + os.sep
df = pd.read_csv(data_path + 'reedfrogs.csv', sep=';')
df["tank"] = np.arange(len(df))

m.df = df
m.data_on_model = {
    'tank': jnp.array(df.tank.values),
    'surv': jnp.array(df.surv.values),
    'density': jnp.array(df.density.values)
}

def model_BF(tank, surv, density):
    sigma = m.dist.exponential(1.0)
    a_bar = m.dist.normal(0.0, 1.5)
    alpha = m.dist.normal(a_bar, sigma, shape=(48,))
    p = alpha[tank]
    m.dist.binomial(total_count=density, logits=p, obs=surv)

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N_obs;
    array[N_obs] int S;
    array[N_obs] int D;
    array[N_obs] int tank;
}
parameters {
    real a_bar;
    vector[48] a;
    real<lower=0> sigma;
}
model {
    vector[N_obs] p;
    sigma ~ exponential(1);
    a_bar ~ normal(0, 1.5);
    a ~ normal(a_bar, sigma);
    for (i in 1:N_obs) {
        p[i] = a[tank[i]];
    }
    S ~ binomial_logit(D, p);
}
"""
data_stan = {
    'N_obs': len(df),
    'S': df['surv'].values.astype(int).tolist(),
    'D': df['density'].values.astype(int).tolist(),
    'tank': (df['tank'].values + 1).astype(int).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'a_bar': 'a_bar',
    'sigma': 'sigma'
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def param_recovery(tank, density, a_bar_sims, sigma_sims, alpha_sims, nsim):
    results = []
    
    m_rec = bf(print_devices_found=False)
    
    def model_rec(tank, surv, density):
        sigma = m_rec.dist.exponential(1.0)
        a_bar = m_rec.dist.normal(0.0, 2.0)
        alpha = m_rec.dist.normal(a_bar, sigma, shape=(48,))
        p = alpha[tank]
        m_rec.dist.binomial(total_count=density, logits=p, obs=surv)

    for i in range(nsim):
        a_bar_true = a_bar_sims[i]
        sigma_true = sigma_sims[i]
        alpha_true = alpha_sims[i]
        
        p_true = alpha_true[tank]
        surv_sim = np.random.binomial(density, jax.nn.sigmoid(p_true))
        
        m_rec.data_on_model = {
            'tank': jnp.array(tank),
            'surv': jnp.array(surv_sim),
            'density': jnp.array(density)
        }
        
        m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
        est = m_rec.summary().iloc[:, 0]
        
        for param in ['a_bar', 'sigma', 'alpha[0]']:
            if param == 'a_bar':
                true_val = a_bar_true
            elif param == 'sigma':
                true_val = sigma_true
            else:
                true_val = alpha_true[0]
                
            results.append({
                'sim': i,
                'parameter': param,
                'simulated': float(np.squeeze(true_val)),
                'estimations': float(est[param])
            })

        # Each fit compiles a new executable; free it before the next iteration
        free_jax_memory()

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
a_bar_sims = np.random.normal(0, 2.0, (nsim_test, 1))
sigma_sims = np.random.exponential(1.0, (nsim_test, 1))
alpha_sims = np.random.normal(a_bar_sims, sigma_sims, (nsim_test, 48))

recovery_results = param_recovery(df.tank.values, df.density.values, a_bar_sims, sigma_sims, alpha_sims, nsim=nsim_test)


# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
