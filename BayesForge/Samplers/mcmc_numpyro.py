import numbers
import warnings

import numpyro
from numpyro.infer import MCMC, NUTS, HMC, Predictive
from numpyro.infer import MixedHMC, DiscreteHMCGibbs, HMCGibbs
from numpyro.handlers import condition
import jax

#: Sampler names accepted by ``m.fit(sampler=...)``.
SAMPLERS = ("nuts", "mixed_hmc", "discrete_gibbs", "hmc_gibbs")


def build_kernel(
    sampler,
    model,
    nuts_kwargs,
    gibbs_fn=None,
    gibbs_sites=None,
    num_discrete_updates=None,
):
    """Build the NumPyro MCMC kernel selected by ``sampler``.

    * ``"nuts"`` -- plain NUTS (default, continuous models and models
      whose discrete latents are exactly enumerated).
    * ``"mixed_hmc"`` -- :class:`MixedHMC` (HMC for the continuous sites,
      Gibbs for the discrete ones). Requires discrete latent sites and an
      HMC -- not NUTS -- inner kernel, which is built here from the same
      tuning arguments.
    * ``"discrete_gibbs"`` -- :class:`DiscreteHMCGibbs`: NUTS for the
      continuous sites, automatic Gibbs updates for the discrete ones.
    * ``"hmc_gibbs"`` -- :class:`HMCGibbs`: NUTS combined with a
      user-supplied ``gibbs_fn`` over ``gibbs_sites``.
    """
    if sampler == "nuts":
        return NUTS(model, **nuts_kwargs)
    if sampler == "mixed_hmc":
        hmc_kwargs = {
            k: v for k, v in nuts_kwargs.items() if k != "max_tree_depth"
        }
        if hmc_kwargs.get("trajectory_length") is None:
            # HMC (unlike NUTS) rejects trajectory_length=None; fall back
            # to its 2π default.
            hmc_kwargs.pop("trajectory_length", None)
        inner = HMC(model, **hmc_kwargs)
        return MixedHMC(inner, num_discrete_updates=num_discrete_updates)
    inner = NUTS(model, **nuts_kwargs)
    if sampler == "discrete_gibbs":
        return DiscreteHMCGibbs(inner)
    if sampler == "hmc_gibbs":
        if gibbs_fn is None or gibbs_sites is None:
            raise ValueError(
                'sampler="hmc_gibbs" requires both `gibbs_fn` and '
                "`gibbs_sites` (or use sampler='discrete_gibbs' for "
                "automatic discrete Gibbs updates)."
            )
        return HMCGibbs(inner, gibbs_fn=gibbs_fn, gibbs_sites=list(gibbs_sites))
    raise ValueError(f"Unknown sampler {sampler!r}. Choose from {SAMPLERS}.")


def mcmc_numpyro(
    model = None,
    potential_fn=None,
    kinetic_fn=None,
    step_size=1.0,
    inverse_mass_matrix=None,
    adapt_step_size=True,
    adapt_mass_matrix=True,
    dense_mass=False,
    target_accept_prob=0.8,
    trajectory_length=None,
    max_tree_depth=10,
    init_strategy= numpyro.infer.init_to_uniform,
    find_heuristic_step_size=False,
    forward_mode_differentiation=False,
    regularize_mass_matrix=True,

    num_warmup = 500,
    num_samples = 500,
    num_chains=1,
    thinning=1,
    postprocess_fn=None,
    chain_method="parallel",
    progress_bar=True,
    jit_model_args=False,
    seed=None,
    sampler="nuts",
    gibbs_fn=None,
    gibbs_sites=None,
    num_discrete_updates=None,
):
    """Return a numpyro MCMC object (kernel + sampling settings).

    ``seed`` is unused: the PRNG key goes to the returned object's ``.run(key)``.

    Raises early (before any kernel is built) when ``num_warmup < 0``,
    ``num_samples < 1`` or ``num_chains < 1``, or when one of them is not an
    int; warns when ``num_chains > 1`` with ``chain_method="parallel"`` on
    fewer devices than chains (numpyro falls back to sequential chains).
    """
    # Minor 6: validate the run counts here instead of letting numpyro fail
    # (or silently fall back) deep inside .run().
    for name, value, minimum in (("num_warmup", num_warmup, 0),
                                 ("num_samples", num_samples, 1),
                                 ("num_chains", num_chains, 1)):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(f"mcmc_numpyro: {name} must be an int >= {minimum}, "
                            f"got {value!r}.")
        if value < minimum:
            raise ValueError(f"mcmc_numpyro: {name} must be >= {minimum}, "
                             f"got {value}.")
    if seed is not None:
        warnings.warn("mcmc_numpyro(seed=...) has no effect: pass the PRNG key to "
                      ".run(key) (bf.fit takes seed=).", UserWarning, stacklevel=2)
    if num_chains > 1 and chain_method == "parallel":
        n_devices = jax.local_device_count()
        if n_devices < num_chains:
            warnings.warn(
                f"mcmc_numpyro: num_chains={num_chains} with "
                f"chain_method='parallel' but only {n_devices} device(s) "
                "visible; numpyro will fall back to sequential chains.",
                UserWarning, stacklevel=2)
    nuts_kwargs = dict(
        potential_fn=potential_fn,
        kinetic_fn=kinetic_fn,
        step_size=step_size,
        inverse_mass_matrix=inverse_mass_matrix,
        adapt_step_size=adapt_step_size,
        adapt_mass_matrix=adapt_mass_matrix,
        dense_mass=dense_mass,
        target_accept_prob=target_accept_prob,
        trajectory_length=trajectory_length,
        max_tree_depth=max_tree_depth,
        init_strategy=init_strategy,
        find_heuristic_step_size=find_heuristic_step_size,
        forward_mode_differentiation=forward_mode_differentiation,
        regularize_mass_matrix=regularize_mass_matrix,
    )
    kernel = build_kernel(
        sampler,
        model,
        nuts_kwargs,
        gibbs_fn=gibbs_fn,
        gibbs_sites=gibbs_sites,
        num_discrete_updates=num_discrete_updates,
    )
    return MCMC(
                kernel,
                num_warmup = num_warmup,
                num_samples = num_samples,
                num_chains=num_chains,
                thinning=thinning,
                postprocess_fn=postprocess_fn,
                chain_method=chain_method,
                progress_bar=progress_bar,
                jit_model_args=jit_model_args
            )

