import warnings

import numpyro
from numpyro.infer import SVI, Trace_ELBO, Predictive
import jax

class SVI_BF:
    def __init__(self, model, guide, optim, loss=None, num_steps=1000, num_samples=1000, **kwargs):
        # Minor 13: no shared mutable default -- each instance builds its own
        # Trace_ELBO when none is given.
        if loss is None:
            loss = Trace_ELBO()
        self.svi = SVI(model, guide, optim, loss, **kwargs)
        self.model = model
        self.guide = guide
        self.result = None
        self.num_chains = 1
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.thinning = 1
        self.num_warmup = 0
        self.sampler = self
        self._args = ()
        self._kwargs = {}
        self.losses = None

    def run(self, rng_key, *args, num_steps=None, **kwargs):
        """Optimise the guide for ``num_steps`` steps (default: the
        constructor's ``num_steps``).

        Minor 12: ``num_steps`` is keyword-only. Any further positional
        arguments are forwarded to the model and guide, so ``run(key, y)``
        passes ``y`` to the model instead of swallowing it as ``num_steps``.
        """
        if num_steps is None:
            num_steps = self.num_steps
        self.last_kwargs = kwargs
        self._args = args
        self._kwargs = kwargs
        self.result = self.svi.run(rng_key, num_steps, *args, **kwargs)
        self.losses = self.result.losses
    
    def get_samples(self, num_samples=None, seed=0, group_by_chain=False):
        """Draw posterior samples from the fitted guide.

        Args:
            num_samples (int, optional): Number of draws; updates the
                instance default when given.
            seed (int): PRNG seed for the draw. Minor 14: the default is 0,
                so repeated calls without an explicit seed return the SAME
                draws; ``bf.svi()`` always passes one.
            group_by_chain (bool): Add a leading chain dimension of size 1.
        """
        if self.result is None:
            raise RuntimeError("Call run() before get_samples().")
        if num_samples is not None:
            self.num_samples = num_samples
        
        rng_key = jax.random.PRNGKey(seed)
        
        # 1. Get latent samples from the guide
        # We try to use sample_posterior if available (standard for AutoGuides)
        try:
            samples = self.guide.sample_posterior(rng_key, self.result.params, sample_shape=(self.num_samples,))
        except (NotImplementedError, AttributeError) as e:
            warnings.warn(f"guide.sample_posterior failed ({type(e).__name__}: {e}); "
                          "falling back to Predictive(model, guide).", UserWarning, stacklevel=2)
            # Fallback to Predictive if sample_posterior fails or is not available
            predictive = Predictive(self.model, guide=self.guide, params=self.result.params, num_samples=self.num_samples)
            samples = predictive(rng_key, *self._args, **self._kwargs)
            if group_by_chain:
                return {k: v[None, ...] for k, v in samples.items()}
            return samples

        # 2. Get other sites (like 'obs' or deterministic sites) from the model
        # We use the latent samples we just got
        predictive = Predictive(self.model, posterior_samples=samples)
        model_samples = predictive(rng_key, *self._args, **self._kwargs)
        
        # 3. Merge them
        samples.update(model_samples)
        
        # 4. Filter out observed sites
        from numpyro.handlers import trace, seed
        tr = trace(seed(self.model, rng_key)).get_trace(*self._args, **self._kwargs)
        obs_sites = {name for name, site in tr.items() if site["type"] == "sample" and site.get("is_observed", False)}
        samples = {k: v for k, v in samples.items() if k not in obs_sites}

        if group_by_chain:
            # Add a chain dimension of size 1
            return {k: v[None, ...] for k, v in samples.items()}
        return samples

    def get_extra_fields(self, group_by_chain=False, **kwargs):
        return {}

def svi_numpyro(model, guide, optim, loss=None, num_steps=1000, num_samples=1000, **kwargs):
    return SVI_BF(model, guide, optim, loss, num_steps=num_steps, num_samples=num_samples, **kwargs)
