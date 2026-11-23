"""Julia vs Python libm: run plain, then with LD_PRELOAD of Julia's libopenlibm (see Test/CROSS_LANGUAGE.md)."""
from BayesForge import bf
import jax.numpy as jnp

m = bf(platform='cpu')
m.data_on_model = dict(y=jnp.array([0.1, 0.5, -0.3, 1.2, 0.8]))

def model(y):
    alpha = m.dist.normal(0, 1, name='alpha')
    s = m.dist.exponential(1, name='s')
    m.dist.normal(alpha, s, obs=y, name='y')

m.fit(model, num_samples=200, num_warmup=200, num_chains=1, seed=0, progress_bar=False)
print(m.summary(round_to=8).loc[['alpha', 's'], 'mean'].tolist())
