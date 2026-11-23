from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "9.Beta binomial"

print(f'Running BF for {model_name}')
m = bf(platform='cpu', backend='tfp')

# 1. Data Loading -------------------------------------------
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "BayesForge", "Resources")) + os.sep
df = pd.read_csv(data_path + 'UCBadmit.csv', sep=';')
df["gid"] = (df["applicant.gender"] != "male").astype(int)

m.df = df
m.data_on_model = {
    'gid': jnp.array(df.gid.values, dtype=jnp.int32),
    'applications': jnp.array(df.applications.values, dtype=jnp.float32), 
    'admit': jnp.array(df.admit.values, dtype=jnp.float32)
}

def model_BF(gid, applications, admit):
    # Uniform prior for phi
    phi = yield m.dist.uniform(0.0, 20.0, name='phi')
    alpha = yield m.dist.normal(0.0, 1.5, shape=(2,), name='alpha')
    theta = phi + 2.0
    pbar = jax.nn.sigmoid(alpha[gid])
    concentration1 = pbar * theta
    concentration0 = (1.0 - pbar) * theta
    yield m.dist.beta_binomial(total_count=applications, concentration1=concentration1, concentration0=concentration0, obs=admit)

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
    real<lower=0, upper=20> phi;
}
transformed parameters {
    real theta;
    theta = phi + 2;
}
model {
    vector[N] pbar;
    phi ~ uniform(0, 20);
    a ~ normal(0, 1.5);
    for (i in 1:N) {
        pbar[i] = inv_logit(a[gid[i]]);
    }
    admit ~ beta_binomial(applications, pbar * theta, (1 - pbar) * theta);
}
"""
data_stan = {
    'N': len(df),
    'admit': df.admit.values.astype(int).tolist(),
    'applications': df.applications.values.astype(int).tolist(),
    'gid': (df.gid.values + 1).astype(int).tolist()
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'alpha[0]': 'a[1]',
    'alpha[1]': 'a[2]',
    'phi': 'phi'
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery --------------------------------------
def estimate(gid, applications, a_true, phi_true):
    theta = float(np.squeeze(phi_true)) + 2.0
    pbar = jax.nn.sigmoid(a_true[gid])
    # Beta-binomial simulation
    p = np.random.beta(pbar * theta, (1.0 - pbar) * theta)
    admit_sim = np.random.binomial(applications.astype(int), p).astype(float)
    
    m_rec = bf(print_devices_found=False, backend='tfp')
    m_rec.data_on_model = {
        'gid': jnp.array(gid, dtype=jnp.int32),
        'applications': jnp.array(applications, dtype=jnp.float32),
        'admit': jnp.array(admit_sim, dtype=jnp.float32)
    }
    def model_rec(gid, applications, admit):
        phi = yield m_rec.dist.uniform(0.0, 20.0, name='phi')
        alpha = yield m_rec.dist.normal(0.0, 1.5, shape=(2,), name='alpha')
        theta = phi + 2.0
        pbar = jax.nn.sigmoid(alpha[gid])
        yield m_rec.dist.beta_binomial(total_count=applications, concentration1=pbar*theta, concentration0=(1-pbar)*theta, obs=admit)
        
    m_rec.fit(model_rec, num_samples=1000, num_warmup=1000, progress_bar=False) 
    s = m_rec.summary()
    return s.iloc[:, 0]

def param_recovery(gid, applications, a_sims, phi_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(gid, applications, a_sims[i], phi_sims[i])
        results.append({'sim': i, 'parameter': 'alpha[0]', 'simulated': float(a_sims[i,0]), 'estimations': float(est['alpha[0]'])})
        results.append({'sim': i, 'parameter': 'alpha[1]', 'simulated': float(a_sims[i,1]), 'estimations': float(est['alpha[1]'])})
        results.append({'sim': i, 'parameter': 'phi', 'simulated': float(phi_sims[i, 0]), 'estimations': float(est['phi'])})
            
    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.getenv('BF_NSIM', 10))
a_sims = np.random.normal(0, 1.5, size=(nsim_test, 2))
phi_sims = np.random.uniform(1.0, 15.0, size=(nsim_test, 1))

# Tile data 200x for massive signal (2400 rows)
gid_ext = np.tile(df.gid.values, 200)
apps_ext = np.tile(df.applications.values, 200).astype(float)
recovery_results = param_recovery(gid_ext, apps_ext, a_sims, phi_sims, nsim=nsim_test)


# --- WAIC & LOO cross-check: native TFP vs NumPyro/ArviZ reference (same draws) ---
# `waic_ref` is a numpyro-mode transcription of the fitted TFP model (identical
# latent site names), used to evaluate the same posterior draws through
# numpyro.infer.log_likelihood -- the exact machinery ArviZ uses.
m_ref = bf(platform='cpu', print_devices_found=False)

def waic_ref(gid, applications, admit):
    phi = m_ref.dist.uniform(0.0, 20.0)
    alpha = m_ref.dist.normal(0.0, 1.5, shape=(2,))
    theta = phi + 2.0
    pbar = jax.nn.sigmoid(alpha[gid])
    m_ref.dist.beta_binomial(total_count=applications, concentration1=pbar*theta, concentration0=(1.0-pbar)*theta, obs=admit)
waic_report(m, model_name, ref_model=waic_ref, ref_kwargs={'gid': m.data_on_model['gid'], 'applications': m.data_on_model['applications'], 'admit': m.data_on_model['admit']})
