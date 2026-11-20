from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax
from jax.nn import softmax

model_name = "14.Dirichlet"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Simulation (same DGP & model as 12. Dirichlet model.qmd) --------
np.random.seed(42)
N = 200
X = np.random.normal(0, 1, size=N)

true_alpha = np.array([0.5, -0.5])
true_beta = np.array([1.0, 0.5])
true_kappa = 10.0

phi = np.zeros((N, 3))
phi[:, 0] = true_alpha[0] + true_beta[0] * X
phi[:, 1] = true_alpha[1] + true_beta[1] * X
phi[:, 2] = 0.0

theta = np.asarray(softmax(phi, axis=1))
Y = np.zeros((N, 3))
for i in range(N):
    Y[i, :] = np.random.dirichlet(theta[i, :] * true_kappa)

m.data_on_model = dict(X=jnp.array(X), Y=jnp.array(Y))

def model_BF(X, Y):
    alpha = m.dist.normal(0, 5, shape=(2,), name="alpha")
    beta = m.dist.normal(0, 5, shape=(2,), name="beta")
    kappa = m.dist.exponential(1.0, name="kappa")

    s1 = alpha[0] + beta[0] * X
    s2 = alpha[1] + beta[1] * X
    s3 = jnp.zeros_like(s1)

    phi = jnp.stack([s1, s2, s3], axis=1)
    theta = jax.nn.softmax(phi, axis=1)

    m.dist.dirichlet(theta * kappa, obs=Y)

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N;
    vector[N] X;
    array[N] simplex[3] Y;
}
parameters {
    vector[2] alpha;
    vector[2] beta;
    real<lower=0> kappa;
}
model {
    alpha ~ normal(0, 5);
    beta ~ normal(0, 5);
    kappa ~ exponential(1.0);
    for (n in 1:N) {
        vector[3] s;
        vector[3] theta;
        s[1] = alpha[1] + beta[1] * X[n];
        s[2] = alpha[2] + beta[2] * X[n];
        s[3] = 0.0;
        theta = softmax(s);
        Y[n] ~ dirichlet(theta * kappa);
    }
}
"""
data_stan = {
    'N': N,
    'X': X.astype(float).tolist(),
    'Y': Y.astype(float).tolist(),
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'alpha[0]': 'alpha[1]',
    'alpha[1]': 'alpha[2]',
    'beta[0]': 'beta[1]',
    'beta[1]': 'beta[2]',
    'kappa': 'kappa',
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery (DGP-IF pipeline) --------------------
def estimate(X_vec, alpha_true, beta_true, kappa_true):
    N_sim = len(X_vec)
    phi_sim = np.zeros((N_sim, 3))
    phi_sim[:, 0] = alpha_true[0] + beta_true[0] * X_vec
    phi_sim[:, 1] = alpha_true[1] + beta_true[1] * X_vec
    phi_sim[:, 2] = 0.0
    theta_sim = np.asarray(softmax(phi_sim, axis=1))
    Y_sim = np.zeros((N_sim, 3))
    for k in range(N_sim):
        Y_sim[k, :] = np.random.dirichlet(np.clip(theta_sim[k, :] * kappa_true, 1e-3, None))
        Y_sim[k, :] = np.clip(Y_sim[k, :], 1e-6, 1.0 - 1e-6)
        Y_sim[k, :] /= Y_sim[k, :].sum()

    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {'X': jnp.array(X_vec), 'Y': jnp.array(Y_sim)}
    def model_rec(X, Y):
        alpha = m_rec.dist.normal(0, 5, shape=(2,), name="alpha")
        beta = m_rec.dist.normal(0, 5, shape=(2,), name="beta")
        kappa = m_rec.dist.exponential(1.0, name="kappa")
        s1 = alpha[0] + beta[0] * X
        s2 = alpha[1] + beta[1] * X
        s3 = jnp.zeros_like(s1)
        phi = jnp.stack([s1, s2, s3], axis=1)
        theta = jax.nn.softmax(phi, axis=1)
        m_rec.dist.dirichlet(theta * kappa, obs=Y)

    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    free_jax_memory()
    return s.iloc[:, 0]

def param_recovery(X_vec, alpha_sims, beta_sims, kappa_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(X_vec, alpha_sims[i], beta_sims[i], kappa_sims[i])
        for j in range(2):
            results.append({'sim': i, 'parameter': f'alpha[{j}]', 'simulated': float(alpha_sims[i, j]), 'estimations': float(est[f'alpha[{j}]'])})
            results.append({'sim': i, 'parameter': f'beta[{j}]', 'simulated': float(beta_sims[i, j]), 'estimations': float(est[f'beta[{j}]'])})
        results.append({'sim': i, 'parameter': 'kappa', 'simulated': float(kappa_sims[i]), 'estimations': float(est['kappa'])})

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
alpha_sims = np.random.normal(0, 1.0, size=(nsim_test, 2))
beta_sims = np.random.normal(0, 1.0, size=(nsim_test, 2))
kappa_sims = np.random.uniform(5.0, 15.0, size=(nsim_test,))

recovery_results = param_recovery(X, alpha_sims, beta_sims, kappa_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)

