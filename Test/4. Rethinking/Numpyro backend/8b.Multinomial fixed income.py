from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import jax.numpy as jnp
import jax

model_name = "8b.Multinomial fixed income"

print(f'Running BF for {model_name}')
m = bf(platform='cpu')

# 1. Data Simulation (same model as 11. Multinomial model.qmd) --------
np.random.seed(42)
N = 200
n_trials = 20
X = np.random.normal(0, 1, size=N)

true_alpha = np.array([0.5, -0.5])
true_beta = np.array([1.0, 0.5])

phi_sim = np.column_stack([
    true_alpha[0] + true_beta[0] * X,
    true_alpha[1] + true_beta[1] * X,
    np.zeros(N),
])
theta_sim = np.asarray(jax.nn.softmax(jnp.array(phi_sim), axis=1))
Y = np.array([np.random.multinomial(n_trials, theta_sim[i]) for i in range(N)])

m.data_on_model = dict(X=jnp.array(X), Y=jnp.array(Y))

def model_BF(X, Y):
    alpha = m.dist.normal(0, 1, shape=(2,), name='alpha')
    beta = m.dist.normal(0, 1, shape=(2,), name='beta')
    s1 = alpha[0] + beta[0] * X
    s2 = alpha[1] + beta[1] * X
    s3 = jnp.zeros_like(s1)
    phi = jnp.stack([s1, s2, s3], axis=1)
    theta = jax.nn.softmax(phi, axis=1)
    m.dist.multinomial(total_count=n_trials, probs=theta, obs=Y)

print("Fitting BF model...")
m.fit(model_BF, num_samples=1000, num_warmup=1000)
print("BF Summary:")
print(m.summary())

# 2. STAN Model ----------------------------------------------
stan_code = """
data {
    int<lower=1> N;
    int<lower=2> K;
    vector[N] X;
    array[N, K] int Y;
}
parameters {
    vector[K-1] alpha;
    vector[K-1] beta;
}
model {
    alpha ~ normal(0, 1);
    beta ~ normal(0, 1);
    for (n in 1:N) {
        vector[K] phi;
        phi[1] = alpha[1] + beta[1] * X[n];
        phi[2] = alpha[2] + beta[2] * X[n];
        phi[3] = 0;
        Y[n] ~ multinomial(softmax(phi));
    }
}
"""
data_stan = {
    'N': N,
    'K': 3,
    'X': X.tolist(),
    'Y': Y.astype(int).tolist(),
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=data_stan, chains=4)

# 3. Output Comparison ---------------------------------------
param_map = {
    'alpha[0]': 'alpha[1]',
    'alpha[1]': 'alpha[2]',
    'beta[0]': 'beta[1]',
    'beta[1]': 'beta[2]',
}
plot_comparaison(m, df_stan, param_map=param_map, model_name=model_name)

# 4. Parameter Recovery (DGP-IF pipeline) --------------------
def estimate(X_sim, alpha_true, beta_true):
    N_sim = len(X_sim)
    phi = np.column_stack([
        alpha_true[0] + beta_true[0] * X_sim,
        alpha_true[1] + beta_true[1] * X_sim,
        np.zeros(N_sim),
    ])
    theta = np.asarray(jax.nn.softmax(jnp.array(phi), axis=1))
    Y_sim = np.array([np.random.multinomial(n_trials, theta[i]) for i in range(N_sim)])

    m_rec = bf(print_devices_found=False)
    m_rec.data_on_model = {
        'X': jnp.array(X_sim),
        'Y': jnp.array(Y_sim),
    }
    def model_rec(X, Y):
        alpha = m_rec.dist.normal(0, 1, shape=(2,), name='alpha')
        beta = m_rec.dist.normal(0, 1, shape=(2,), name='beta')
        s1 = alpha[0] + beta[0] * X
        s2 = alpha[1] + beta[1] * X
        s3 = jnp.zeros_like(s1)
        phi = jnp.stack([s1, s2, s3], axis=1)
        theta = jax.nn.softmax(phi, axis=1)
        m_rec.dist.multinomial(total_count=n_trials, probs=theta, obs=Y)

    m_rec.fit(model_rec, num_samples=500, progress_bar=False, shard=False)
    s = m_rec.summary()
    free_jax_memory()
    return s.iloc[:, 0]

def param_recovery(X_sim, alpha_sims, beta_sims, nsim):
    results = []
    for i in range(nsim):
        est = estimate(X_sim, alpha_sims[i], beta_sims[i])
        results.append({'sim': i, 'parameter': 'alpha[0]', 'simulated': float(alpha_sims[i, 0]), 'estimations': float(est['alpha[0]'])})
        results.append({'sim': i, 'parameter': 'alpha[1]', 'simulated': float(alpha_sims[i, 1]), 'estimations': float(est['alpha[1]'])})
        results.append({'sim': i, 'parameter': 'beta[0]', 'simulated': float(beta_sims[i, 0]), 'estimations': float(est['beta[0]'])})
        results.append({'sim': i, 'parameter': 'beta[1]', 'simulated': float(beta_sims[i, 1]), 'estimations': float(est['beta[1]'])})

    df_res = pd.DataFrame(results)
    plot_recovery(df_res, model_name=model_name)
    return df_res

print("Running Parameter Recovery...")
nsim_test = int(os.environ.get("BF_NSIM", 10))
alpha_sims = np.random.normal(0, 0.8, size=(nsim_test, 2))
beta_sims = np.random.normal(0, 0.8, size=(nsim_test, 2))
X_rec = np.random.normal(0, 1, size=200)

recovery_results = param_recovery(X_rec, alpha_sims, beta_sims, nsim=nsim_test)

# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
