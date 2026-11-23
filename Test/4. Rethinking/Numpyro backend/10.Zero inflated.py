from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.special import expit
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "3.Documentation"))
from doc_blocks import doc_model
# The model as published on the documentation page. Priors are identical to the previous local
# copy: al ~ normal(1, 0.5), ap ~ normal(-1.5, 1) (the page uses m.link.inv_logit where the
# script used expit, and names the sites 'al' and 'ap').
PAGE_MODEL = doc_model("13. Zero inflated.qmd", 0)

model_name = "10.Zero inflated"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Simulation -----------------------------------------
# Simulate production of manuscripts
prob_drink = 0.2  # 20% of days
rate_work = 1     # average 1 manuscript per day
N = 365

np.random.seed(42)
drink = np.random.binomial(1, prob_drink, N)
y = (1 - drink) * np.random.poisson(rate_work, N)
df = pd.DataFrame({'y': y})

m.df = df
m.data_on_model = {'y': jnp.array(y)}

model_BF = PAGE_MODEL

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int N;
    array[N] int y;
}
parameters {
    real al;
    real ap;
}
model {
    real p;
    real lambda;
    al ~ normal(1, 0.5);
    ap ~ normal(-1.5, 1);
    
    lambda = exp(al);
    p = inv_logit(ap);
    
    for (n in 1:N) {
        if (y[n] == 0)
            target += log_mix(p, 0, poisson_lpmf(0 | lambda));
        else
            target += log1m(p) + poisson_lpmf(y[n] | lambda);
    }
}
"""
data_stan = {
    'N': N,
    'y': y.astype(int).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
plot_comparaison(m, df_stan, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(y_sim):
    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {'y': jnp.array(y_sim)}
    model_rec = PAGE_MODEL
    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(y_true, ap_sims, al_sims, nsim):
    results = []
    for i in range(nsim):
        # We simulate new y for each run based on true parameters
        p_true = expit(ap_sims[i])
        lambda_true = np.exp(al_sims[i])
        drink_sim = np.random.binomial(1, p_true, N)
        y_sim = (1 - drink_sim) * np.random.poisson(lambda_true, N)
        
        est = estimate(y_sim)
        results.append({'sim': i, 'parameter': 'ap', 'simulated': ap_sims[i], 'estimations': est['ap']})
        results.append({'sim': i, 'parameter': 'al', 'simulated': al_sims[i], 'estimations': est['al']})
            
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
# Drawn from the page priors: ap ~ normal(-1.5, 1), al ~ normal(1, 0.5)
ap_sims = np.random.normal(-1.5, 1.0, nsim_test)
al_sims = np.random.normal(1.0, 0.5, nsim_test)

recovery_results = param_recovery(y, ap_sims, al_sims, nsim=nsim_test)


# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
