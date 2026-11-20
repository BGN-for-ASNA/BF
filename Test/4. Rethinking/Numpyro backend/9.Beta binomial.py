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
# The model as published on the documentation page. Priors are identical to the previous local
# copy: phi ~ exponential(1), alpha ~ normal(0, 1.5) shape (2,), theta = phi + 2.
PAGE_MODEL = doc_model("6. Beta binomial model.qmd", 0)

model_name = "9.Beta binomial"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Loading -------------------------------------------
data_path = m.load.ucbadmit(only_path=True)
m.data(data_path, sep=';')
m.df["gid"] = (m.df["applicant_gender"] != "male").astype(int)

m.data_on_model = {
    'gid': jnp.array(m.df.gid.values),
    'applications': jnp.array(m.df.applications.values),
    'admit': jnp.array(m.df.admit.values),
}

model_BF = PAGE_MODEL

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int N;
    array[N] int admit;
    array[N] int applications;
    array[N] int gid;
}
parameters {
    vector[2] a;
    real<lower=0> phi;
}
transformed parameters {
    real theta;
    theta = phi + 2;
}
model {
    vector[N] pbar;
    phi ~ exponential(1);
    a ~ normal(0, 1.5);
    for (i in 1:N) {
        pbar[i] = inv_logit(a[gid[i]]);
    }
    // Added 1e-5 to prevent 0 concentration parameters during warmup
    admit ~ beta_binomial(applications, pbar * theta + 1e-5, (1 - pbar) * theta + 1e-5);
}
"""
data_stan = {
    'N': len(m.df),
    'admit': m.df.admit.values.astype(int).tolist(),
    'applications': m.df.applications.values.astype(int).tolist(),
    'gid': (m.df.gid.values + 1).astype(int).tolist(),
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'alpha[0]': 'a[1]',
    'alpha[1]': 'a[2]',
    'phi': 'phi',
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def param_recovery(gid, applications, a_sims, phi_sims, nsim):
    results = []

    m_rec = bf(print_devices_found=False)

    model_rec = PAGE_MODEL

    for i in range(nsim):
        a_true = a_sims[i]
        phi_true = phi_sims[i]

        theta_true = float(np.squeeze(phi_true)) + 2.0
        pbar_true = jax.nn.sigmoid(a_true[gid])

        # Beta-binomial simulation
        p = np.random.beta(pbar_true * theta_true, (1.0 - pbar_true) * theta_true)
        admit_sim = np.random.binomial(applications, p)

        m_rec.data_on_model = {
            'gid': jnp.array(gid),
            'applications': jnp.array(applications),
            'admit': jnp.array(admit_sim),
        }

        m_rec.fit(model_rec, num_samples=1000, progress_bar=False, shard=False)
        s = m_rec.summary()
        est = s.iloc[:, 0]

        results.append({'sim': i, 'parameter': 'alpha[0]', 'simulated': float(a_sims[i, 0]), 'estimations': float(est['alpha[0]'])})
        results.append({'sim': i, 'parameter': 'alpha[1]', 'simulated': float(a_sims[i, 1]), 'estimations': float(est['alpha[1]'])})
        results.append({'sim': i, 'parameter': 'phi', 'simulated': float(phi_sims[i, 0]), 'estimations': float(est['phi'])})

        free_jax_memory()

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
# Drawn from the page priors: alpha ~ normal(0, 1.5) shape (2,), phi ~ exponential(1)
a_sims = np.random.normal(0, 1.5, size=(nsim_test, 2))
phi_sims = np.random.exponential(1.0, size=(nsim_test, 1))

# Tile data 200x for massive signal (2400 rows)
gid_ext = np.tile(m.df.gid.values, 200)
apps_ext = np.tile(m.df.applications.values, 200)
recovery_results = param_recovery(gid_ext, apps_ext, a_sims, phi_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
