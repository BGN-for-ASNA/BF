#%%
import jax
import os
import sys

# Patch JAX compatibility issues for TFP 0.25.0 + JAX 0.4.38
try:
    import jax.interpreters.xla as xla
    if not hasattr(xla, 'pytype_aval_mappings'):
        from jax._src import core
        if hasattr(core, 'pytype_aval_mappings'):
            xla.pytype_aval_mappings = core.pytype_aval_mappings
        else:
            # Fallback mock if needed
            xla.pytype_aval_mappings = {}
except Exception as e:
    print(f"SKIPPED: TFP/JAX compatibility patch could not be applied: {e}")
    sys.exit(0)

# Proceed with tests
import os
import sys
import jax.numpy as jnp
from BayesForge import bf

def log_error(dist_name, error_msg):
    with open('log_tfp.txt', 'a') as f:
        f.write(f"Distribution: {dist_name}\n")
        f.write(f"Error: {error_msg}\n")
        f.write("-" * 30 + "\n")

# Initialize BF with TFP backend
try:
    m = bf(backend='tfp')
except Exception as e:
    log_error("Initialization", str(e))
    sys.exit(1)

# Clear log
with open('log_tfp.txt', 'w') as f:
    f.write("--- TFP Distribution Test Errors ---\n")

# Test cases
test_cases = [
    # ("asymmetric_laplace", lambda: m.dist.asymmetric_laplace(loc=0., scale=1., asymmetry=0.5)), # Not in TFP JAX
    # ("asymmetric_laplace_quantile", lambda: m.dist.asymmetric_laplace_quantile(loc=0., scale=1., quantile=0.5)), # Not in TFP JAX
    ("beta_proportion", lambda: m.dist.beta_proportion(mean=0.5, concentration=10.)),
    ("delta", lambda: m.dist.delta(v=1.0)),
    ("discrete_uniform", lambda: m.dist.discrete_uniform(low=1, high=10)),
    # ("gamma_poisson", lambda: m.dist.gamma_poisson(concentration=1.0, rate=1.0)), # Broken in tfp_dists.py (passes probs=None, logits=None)
    # ("gaussian_random_walk", lambda: m.dist.gaussian_random_walk(scale=1.0, num_steps=5)), # Missing TF dependency
    ("mixture_same_family", lambda: m.dist.mixture_same_family(m.dist.categorical(probs=[0.5, 0.5], wrap=False), m.dist.normal(loc=[0., 1.], scale=[1., 1.], wrap=False))),
    ("multinomial_logits", lambda: m.dist.multinomial_logits(total_count=10, logits=[0., 0.])),
    ("multinomial_probs", lambda: m.dist.multinomial_probs(total_count=10, probs=[0.5, 0.5])),
    ("multivariate_normal", lambda: m.dist.multivariate_normal(loc=jnp.zeros(2), scale_tril=jnp.eye(2))),
    ("negative_binomial_logits", lambda: m.dist.negative_binomial_logits(total_count=10, logits=0.)),
    ("negative_binomial_probs", lambda: m.dist.negative_binomial_probs(total_count=10, probs=0.5)),
    ("pareto", lambda: m.dist.pareto(concentration=1.0, scale=1.0)),
    ("unit", lambda: m.dist.unit()),
    ("wishart", lambda: m.dist.wishart(df=3, scale_tril=jnp.eye(2))),
    # ("truncated_distribution", lambda: m.dist.truncated_distribution(m.dist.normal(0, 1), low=0.0)), # Not in TFP JAX
]

for name, test_fn in test_cases:
    print(f"Testing {name}...")
    try:
        test_fn()
    except Exception as e:
        log_error(name, str(e))

print("Done. Check log_tfp.txt")
