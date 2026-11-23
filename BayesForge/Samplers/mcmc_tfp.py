from jax import jit
import numbers
import warnings
import tensorflow_probability 
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates.jax.distributions import JointDistributionCoroutine
from tensorflow_probability.substrates import jax as tfp
tfb = tfp.bijectors
import jax.numpy as jnp
import inspect
import re
from BayesForge.Samplers.Model_handler import model_handler 

class mcmc(model_handler):
    def __init__(self, *args, **kwargs):
        # Call super() without specifying the class name in a multiple inheritance context
        super().__init__(*args, **kwargs)

    def convert_to_tensor(self, model, vars):
        self.tensor = JointDistributionCoroutine(lambda: model(**vars))
        return self.tensor

    def init_Model(self, model,  init = None, bijectors = None, seed=0, num_obs=1):
        # M2: `seed` may be a PRNG key (derived from the user's seed in run) or an int.
        init_key = seed if not isinstance(seed, (int, float)) else jax.random.PRNGKey(int(seed))
        self.init_key = init_key
        self.get_model_args(model)
        self.get_model_var(model)
        self.get_model_distributions(model)
        self.tensor = self.convert_to_tensor(model, self.vars) 

        if init is None:
            init_params = self.tensor.sample(seed = init_key)
            # M1: drop the trailing observed sites (works for num_obs=0 too).
            self.init_params = list(init_params)[:len(init_params) - num_obs]
        else:
            self.init_params = init

        if bijectors is None:
            # M5: take each latent distribution's own support bijector.
            dists, _ = self.tensor.sample_distributions(seed=init_key)
            latent = list(dists)[:len(dists) - num_obs]
            self.bijectors = []
            for d in latent:
                b = d.experimental_default_event_space_bijector()
                self.bijectors.append(tfb.Identity() if b is None else b)
        else:
            self.bijectors = bijectors

    @staticmethod
    def trace_fn(_, pkr):
        return (
            pkr.inner_results.inner_results.target_log_prob,
            pkr.inner_results.inner_results.leapfrogs_taken,
            pkr.inner_results.inner_results.has_divergence,
            pkr.inner_results.inner_results.energy,
            pkr.inner_results.inner_results.log_accept_ratio
        )

    def NUTS(self, model, obs, n_chains = 1, target_log_prob_fn = None,
         num_results = 500, num_burnin_steps=500, num_steps_between_results=0,
         parallel_iterations = 10, seed=0, name=None,
         target_accept_prob=0.8, step_size=1e-3):
        """Sample ``n_chains`` NUTS chains from ``target_log_prob_fn``.

        Args:
            model, obs: **ignored (deprecated).** Kept only for backward
                compatibility of the public signature; the target distribution
                is the ``target_log_prob_fn`` closure built over the joint
                distribution in :meth:`run`.
            n_chains (int): Number of chains, must be >= 1. When it exceeds
                the number of local devices, the chains are sampled in
                sequential ``pmap`` batches (a warning is emitted).

        Returns:
            mcmc: ``self`` (minor 8), so ``self.sampler = self.NUTS(...)``
            stores a usable sampler object and :meth:`run` returns it,
            consistent with ``SVI_BF.sampler = self``.
        """
        # Minor 11: a nonsensical chain count must fail here, not inside pmap.
        if (isinstance(n_chains, bool) or not isinstance(n_chains, numbers.Integral)
                or n_chains < 1):
            raise ValueError(f"n_chains must be an int >= 1, got {n_chains!r}.")

        @jit
        def run_chain(key):
            inner_kernel = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn,
                step_size= step_size
            )

            kernel = tensorflow_probability.substrates.jax.mcmc.TransformedTransitionKernel(
                    inner_kernel=inner_kernel,
                    bijector=self.bijectors
            )

            hmc  = tfp.mcmc.DualAveragingStepSizeAdaptation(
                kernel,
                target_accept_prob=target_accept_prob,
                num_adaptation_steps=int(0.8*num_burnin_steps),
                step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
                      inner_results=pkr.inner_results._replace(step_size=new_step_size)
                  ),
                step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
                log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
            )

            return tfp.mcmc.sample_chain(num_results = num_results,
                                         num_steps_between_results = num_steps_between_results,
                                         current_state= self.init_params,
                                         kernel=hmc,
                                         trace_fn=self.trace_fn,
                                         num_burnin_steps=num_burnin_steps,
                                         parallel_iterations = parallel_iterations,
                                         seed=key)

        if isinstance(seed, (int, float)):
            master_key = jax.random.PRNGKey(int(seed))
        else:
            master_key = seed

        Ndevices = jax.local_device_count(backend=None)
        if(n_chains > Ndevices):
            # Minor 11: correct, but the batches run one after the other.
            runs = int(jnp.ceil(n_chains/Ndevices))
            warnings.warn(
                f"TFP sampler: n_chains={n_chains} exceeds the {Ndevices} local "
                f"device(s); the chains are sampled in {runs} sequential pmap "
                "batches (slower than a single batched run).",
                UserWarning, stacklevel=2)
            run_keys = jax.random.split(master_key, runs)
            all_samples = []
            all_stats = []
            for run in range(runs):
                rng_keys = jax.random.split(run_keys[run], Ndevices)
                res, stats = jax.pmap(run_chain)(rng_keys)
                all_samples.append(res)
                all_stats.append(stats)
            
            # Stack across runs to get (n_chains, num_results, ...)
            num_params = len(all_samples[0])
            self.posterior = []
            for p in range(num_params):
                self.posterior.append(jnp.concatenate([run_res[p] for run_res in all_samples], axis=0))
            
            self.sample_stats = []
            num_stats = len(all_stats[0])
            for s in range(num_stats):
                self.sample_stats.append(jnp.concatenate([run_stats[s] for run_stats in all_stats], axis=0))

        else:
            rng_keys = jax.random.split(master_key, n_chains)
            res, stats = jax.pmap(run_chain)(rng_keys)
            # Minor 10: same (list) type as the multi-run branch above, so
            # consumers (Diag/Diag2, get_samples) see one consistent type.
            self.posterior = list(res)
            self.sample_stats = list(stats)

        return self

    def get_samples(self, group_by_chain=False):
        """
        Get posterior samples from the fitted model.

        Same contract as the numpyro backend: posterior arrays have shape
        (n_chains, num_results, ...); by default the chains are flattened into
        a single leading dimension (n_chains * num_results, ...).

        Args:
            group_by_chain (bool): If True, keep the chain dimension
                (n_chains, num_results, ...). If False (default), flatten to
                (n_chains * num_results, ...).
        Returns:
            dict: Dictionary of parameter names and their posterior samples.
        """
        var_names = list(self.model_info.keys())
        if len(var_names) != len(self.posterior):
            raise RuntimeError(
                f"TFP sampler: {len(self.posterior)} sampled sites but "
                f"{len(var_names)} parameter names ({var_names}).")
        trace = {}
        for name, samp in zip(var_names, self.posterior):
            if not group_by_chain:
                samp = samp.reshape((-1,) + samp.shape[2:])
            trace[name] = samp
        return trace

    def run(self, seed=0, model = None,  obs = None, n_chains = 1, init = None, bijectors = None, target_log_prob_fn = None,
         num_results = 500, num_burnin_steps=500, num_steps_between_results=0,
         parallel_iterations = 10, name=None, target_accept_prob=0.8, step_size=1e-3, **kwargs):
        
        if model is None:
            model = self.model
        
        if kwargs:
            if self.data_on_model is None:
                self.data_on_model = {}
            self.data_on_model.update(kwargs)

        obs_from_source = obs is None
        if obs is None:
            obs = self.find_obs_in_model(model)
        
        if isinstance(obs, str):
            obs = [obs]

        if not obs:
            raise ValueError("TFP backend: the model has no observed site (obs=...); nothing to condition on.")

        if obs_from_source:
            # M1: the joint distribution appends the observed sites last, so
            # the model's yields must put every obs= site after the latents.
            self.check_obs_last(self.yield_sites(model))

        self.obs_names = obs
        
        # Verify all obs exist in data_on_model
        for name in obs:
            if name not in self.data_on_model:
                raise KeyError(f"Observed variable '{name}' not found in data_on_model. "
                               f"Available keys: {list(self.data_on_model.keys())}")
        
        obs_tuple = tuple(self.data_on_model[name] for name in obs)

        # Handle seed - ensure it's a PRNGKey, then derive a distinct key for
        # the initialisation draw (M2: the user's seed must reach init_Model).
        if isinstance(seed, (int, float)):
            key = jax.random.PRNGKey(int(seed))
        else:
            key = seed

        init_key_seed, key = jax.random.split(key)

        self.init_Model(model, init = init, bijectors = bijectors, seed=init_key_seed, num_obs=len(obs)) 

        if target_log_prob_fn == None:
            def target_log_prob(*params):
                return self.tensor.log_prob(params + obs_tuple)
        else:
            user_target_log_prob_fn = target_log_prob_fn
            def target_log_prob(*params):
                return user_target_log_prob_fn(params + obs_tuple)

        self.sampler = self.NUTS(model = self.tensor, obs = obs_tuple,  n_chains = n_chains,  target_log_prob_fn = target_log_prob,
        num_results = num_results, num_burnin_steps=num_burnin_steps, num_steps_between_results=num_steps_between_results,
        parallel_iterations = parallel_iterations, seed=key, name=name,
        target_accept_prob=target_accept_prob, step_size=step_size)

        return self.sampler

