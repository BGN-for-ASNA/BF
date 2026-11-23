#%%
from Utils import *
from BayesForge import bf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import jax
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "3.Documentation"))
from doc_blocks import doc_model
# The model as published on the documentation page (raw version, first python block). Priors are
# identical to the previous local copy: a ~ normal(5, 2), b ~ normal(-1, 0.5),
# sigma_cafe ~ exponential(1) shape (2,), sigma ~ exponential(1), Rho ~ lkj(2, 2),
# a_b_cafe ~ multivariate_normal([a, b], outer(sigma_cafe, sigma_cafe) * Rho).
PAGE_MODEL = doc_model("16. Varying slopes.qmd", 0)

model_name = "12.Varying effects"

print(f'Running BF for {model_name}')
# 2. BF Model (Non-Centered Parametrization)
m = bf(platform='cpu')
m = bf()
data_path = m.load.sim_multivariate_normal(only_path=True)
m.data(data_path, sep=',') 
m.data_on_model = dict(
    cafe = jnp.array(m.df.cafe.values, dtype=jnp.int32),
    wait = jnp.array(m.df.wait.values, dtype=jnp.float32),
    N_cafes = len(m.df.cafe.unique()),
    afternoon = jnp.array(m.df.afternoon.values, dtype=jnp.float32)
)

model = PAGE_MODEL

print("Fitting BF model...")
m.fit(model, num_samples=1000, num_warmup=1000)
BF_summary = m.summary()
print("BF Summary:")
print(BF_summary)

#%%
# 3. Stan Model
stan_code = """ 
data{
    int len;
    int N_cafes;
    vector[len] wait;
    array[len] int afternoon;
    array[len] int cafe;
}

parameters{
    corr_matrix[2] Rho;
    real a;
    vector[N_cafes] a_cafe;
    real b;
    vector[N_cafes] b_cafe;      
    real<lower=0> sigma;
    vector<lower=0>[2] sigma_cafe;   
    
}
model{
    vector[len] mu;
    Rho ~ lkj_corr( 2 );
    sigma ~ exponential( 1 );
    sigma_cafe ~ exponential( 1 );
    b ~ normal( -1 , 0.5 );
    a ~ normal( 5 , 2 );
    {
        array[N_cafes] vector[2] YY;
        vector[2] MU;
        MU = [ a , b ]';
        for ( j in 1:N_cafes ) YY[j] = [ a_cafe[j] , b_cafe[j] ]';
        YY ~ multi_normal( MU , quad_form_diag(Rho , sigma_cafe) );
    }
    for ( i in 1:len ) {
        mu[i] = a_cafe[cafe[i]] + b_cafe[cafe[i]] * afternoon[i];        
    }
    
    wait ~ normal( mu , sigma );

}
"""
stan_data = {
    'wait' : m.df['wait'].values.astype(float),
    'afternoon' : m.df['afternoon'].values.astype(int),
    'cafe' : m.df['cafe'].values.astype(int)+1,
    'N_cafes' : len(m.df.cafe.unique()),
    'len' : len(m.df['wait'].values)
}

print("Fitting Stan model...")
df_stan = build_stan_model(stan_code, data=stan_data, chains=4)

# 4. Comparison
param_map = {
    'a': 'a',
    'b': 'b',
    'sigma': 'sigma',
    'sigma_cafe_0': 'sigma_cafe[1]',
    'sigma_cafe_1': 'sigma_cafe[2]',
    'Rho[0][1]': 'Rho[1,2]'
}

BF_df = prepare_bi_data(m)

BF_df = BF_df.rename(columns={
    'sigma_cafe[0]': 'sigma_cafe_0',
    'sigma_cafe[1]': 'sigma_cafe_1'
})

plot_comparaison(BF_df, df_stan, param_map, model_name=model_name)
#%%
# 5. Parameter Recovery Analysis
print("\nRunning Parameter Recovery...")
nsim = int(os.environ.get("BF_NSIM", 20))
results = []

# Initialize m_rec outside the loop to prevent OOM
m_rec = bf(print_devices_found=False)

model_rec = PAGE_MODEL

for i in range(nsim):
    # True values drawn from the page priors:
    # a ~ normal(5, 2), b ~ normal(-1, 0.5), sigma ~ exponential(1),
    # sigma_cafe ~ exponential(1) shape (2,),
    # Rho ~ lkj(dimension=2, concentration=2): the marginal of the single off-diagonal
    # element of a 2x2 correlation matrix is Beta(eta + (D-2)/2, eta + (D-2)/2) = Beta(2, 2)
    # mapped to (-1, 1).
    a_sim = np.random.normal(0.0, 2.0)
    b_sim = np.random.normal(0.0, 1.0)
    sigma_cafe_sim = np.random.exponential(1.0, size=2)
    rho_sim = np.random.uniform(-0.9, 0.9)
    sigma_sim = 0.5
    
    N_cafes_rec = 50
    N_visits_rec = 40
    
    Mu = jnp.array([a_sim, b_sim])
    sigmas = jnp.array([sigma_cafe_sim[0], sigma_cafe_sim[1]])
    Rho_sim_mat = jnp.array([[1, rho_sim], [rho_sim, 1]])
    Sig = jnp.diag(sigmas)
    Cov = Sig @ Rho_sim_mat @ Sig
    Cov = Cov + jnp.eye(2) * 1e-6
    
    vary_effects = np.random.multivariate_normal(Mu, Cov, size=N_cafes_rec)
    a_cafe_sim, b_cafe_sim = vary_effects[:, 0], vary_effects[:, 1]
    
    cafe_id = np.repeat(np.arange(N_cafes_rec), N_visits_rec)
    afternoon_sim = np.tile(np.arange(2), N_visits_rec * N_cafes_rec // 2)
    
    mu_sim = a_cafe_sim[cafe_id] + b_cafe_sim[cafe_id] * afternoon_sim
    wait_sim = np.random.normal(mu_sim, sigma_sim).astype(float)
    
    m_rec.data_on_model = {
        'cafe': jnp.array(cafe_id, dtype=jnp.int32),
        'wait': jnp.array(wait_sim, dtype=jnp.float32),
        'N_cafes': N_cafes_rec,
        'afternoon': jnp.array(afternoon_sim, dtype=jnp.float32)
    }
        
    m_rec.fit(model_rec, num_samples=1000, progress_bar=False, shard=False)
    est = m_rec.summary().iloc[:, 0]
    
    results.append({'sim': i, 'parameter': 'a', 'simulated': float(a_sim), 'estimations': float(est['a'])})
    results.append({'sim': i, 'parameter': 'b', 'simulated': float(b_sim), 'estimations': float(est['b'])})

    # Each fit compiles a new executable; free it before the next iteration
    free_jax_memory()


df_res = pd.DataFrame(results)
plot_recovery(df_res, model_name=model_name)


# --- WAIC cross-check: BF direct (NumPyro) vs ArviZ round-trip on the same draws ---
waic_report(m, model_name)
