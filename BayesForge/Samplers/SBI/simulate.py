"""Shared helpers for simulation-based inference (ABC, NPE).

A BF model is a numpyro program, so it already *is* a simulator: removing the
observations from its observed sites and running it forward draws
``(theta, x) ~ p(theta) p(x | theta)``. The helpers below turn a BF model into
such a simulator, flatten parameter dictionaries to matrices (and back), map
parameters to an unconstrained space, and compute summary statistics.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from numpyro import handlers
from numpyro.distributions import biject_to
from numpyro.infer import Predictive
from numpyro.primitives import Messenger


# Model -> simulator -------------------------------------------------------------------
class _uncondition(Messenger):
    """Turn every observed sample site into a latent one, so it is simulated.

    Must sit *inside* ``seed`` (it is the innermost handler when the wrapped
    model is passed to ``Predictive``), so ``seed`` sees ``value=None`` and
    hands the site an rng key.
    """

    def process_message(self, msg):
        if msg["type"] == "sample" and msg["is_observed"]:
            # A likelihood like normal(mu, 1, obs=y) with scalar mu has batch
            # shape () and relies on broadcasting against y; to simulate a whole
            # dataset, expand it to the shape of the observations.
            fn = msg["fn"]
            obs_shape = jnp.shape(msg["value"])
            batch = obs_shape[:len(obs_shape) - len(fn.event_shape)]
            if tuple(fn.batch_shape) != tuple(batch):
                msg["fn"] = fn.expand(jnp.broadcast_shapes(fn.batch_shape, batch))
            msg["value"] = None
            msg["is_observed"] = False
            msg["infer"] = dict(msg["infer"], _sbi_observed=True)


def _summary_quantiles(x, probs=(0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)):
    return jnp.quantile(x, jnp.asarray(probs))


def _summary_moments(x):
    mu = jnp.mean(x)
    sd = jnp.std(x)
    z = (x - mu) / (sd + 1e-12)
    return jnp.stack([mu, sd, jnp.mean(z**3), jnp.mean(z**4)])


def _summary_robust(x):
    """Robust quantile-based stats (Drovandi & Pettitt 2011): location, scale,
    skewness and kurtosis from octiles -- the usual choice for the g-and-k."""
    e = _summary_quantiles(x)
    loc = e[3]
    scale = e[5] - e[1]
    skew = (e[5] + e[1] - 2 * e[3]) / scale
    kurt = (e[6] - e[4] + e[2] - e[0]) / scale
    return jnp.stack([loc, scale, skew, kurt])


SUMMARY_STATS = {
    "identity": lambda x: x,
    "moments": _summary_moments,
    "quantiles": _summary_quantiles,
    "robust": _summary_robust,
}


def resolve_summary_stats(summary_stats):
    if summary_stats is None:
        return SUMMARY_STATS["identity"]
    if callable(summary_stats):
        return summary_stats
    if summary_stats in SUMMARY_STATS:
        return SUMMARY_STATS[summary_stats]
    raise ValueError(
        f"Unknown summary_stats {summary_stats!r}. "
        f"Use a callable or one of {list(SUMMARY_STATS)}."
    )


class Simulator:
    """Wrap a BF model as a simulator for (theta, x) pairs.

    Args:
        model: BF / numpyro model function.
        data: dict of keyword arguments for the model (covariates and observed
            data). Observed values are read from here.
        params: names of the parameter sites to infer. Defaults to every latent
            sample site. Other latent sites (e.g. per-observation noise in a
            simulator) are treated as nuisance and integrated out by simulation.
        observed: optional dict ``{site_name: value}`` of observed data. Use it
            when the simulator output is a ``track``/deterministic site instead
            of a sample site with ``obs=``. Defaults to the observed sample
            sites found in the model trace.
        summary_stats: callable or name ("identity", "moments", "quantiles",
            "robust") applied to the flattened simulated data of one draw.
    """

    def __init__(self, model, data=None, params=None, observed=None,
                 summary_stats="identity", batch_size=10_000):
        self.model = model
        self.data = {} if data is None else dict(data)
        self.batch_size = int(batch_size)
        self.stat_fn = resolve_summary_stats(summary_stats)

        tr = handlers.trace(handlers.seed(model, 0)).get_trace(**self.data)
        obs_sites = [n for n, s in tr.items()
                     if s["type"] == "sample" and s["is_observed"]]
        latent = [n for n, s in tr.items()
                  if s["type"] == "sample" and not s["is_observed"]]

        if observed is None:
            if not obs_sites:
                raise ValueError(
                    "No observed site found. Either declare a likelihood with "
                    "obs=... or pass observed={'site_name': value}."
                )
            observed = {n: tr[n]["value"] for n in obs_sites}
        self.observed_names = list(observed)
        for n in self.observed_names:
            if n not in tr:
                raise ValueError(f"Observed site {n!r} not found in the model.")
        self.x_obs_dict = {n: jnp.asarray(v) for n, v in observed.items()}

        self.param_names = list(params) if params is not None else latent
        for n in self.param_names:
            if n not in tr or tr[n]["type"] != "sample":
                raise ValueError(f"Parameter {n!r} is not a sample site of the model.")
            if tr[n]["fn"].support.is_discrete:
                raise ValueError(
                    f"Parameter {n!r} is discrete; simulation-based inference "
                    "here supports continuous parameters only."
                )

        # Parameter layout (constrained and unconstrained) -----------------------
        self.transforms = {}
        self.shapes = {}
        self.u_shapes = {}
        self.supports = {}
        for n in self.param_names:
            site = tr[n]
            t = biject_to(site["fn"].support)
            v = jnp.asarray(site["value"])
            self.transforms[n] = t
            self.supports[n] = site["fn"].support
            self.shapes[n] = v.shape
            self.u_shapes[n] = jnp.shape(t.inv(v))
        self.u_sizes = {n: int(np.prod(s)) for n, s in self.u_shapes.items()}
        self.dim = sum(self.u_sizes.values())

        self.sim_model = _uncondition(model)
        self._prior_fn = None
        self._cond_fn = None
        self.n_simulations = 0

        self.s_obs = self.stats(self.flatten_x(
            {n: v[None] for n, v in self.x_obs_dict.items()}))[0]

    # --- shapes ----------------------------------------------------------------
    def flatten_x(self, xdict):
        """dict of (n, ...) arrays -> (n, d) matrix of concatenated data."""
        return jnp.concatenate(
            [jnp.reshape(xdict[n], (xdict[n].shape[0], -1)) for n in self.observed_names],
            axis=1,
        )

    def stats(self, x):
        return jax.vmap(self.stat_fn)(x).reshape(x.shape[0], -1)

    def to_unconstrained(self, theta):
        """dict of (n, ...) constrained arrays -> (n, D) unconstrained matrix."""
        cols = []
        for n in self.param_names:
            u = self.transforms[n].inv(theta[n])
            cols.append(jnp.reshape(u, (u.shape[0], -1)))
        return jnp.concatenate(cols, axis=1)

    def to_constrained(self, u):
        """(n, D) unconstrained matrix -> dict of constrained arrays."""
        out, i = {}, 0
        for n in self.param_names:
            k = self.u_sizes[n]
            un = jnp.reshape(u[:, i:i + k], (u.shape[0],) + self.u_shapes[n])
            out[n] = self.transforms[n](un)
            i += k
        return out

    # --- simulation ------------------------------------------------------------
    def _x_from(self, samples):
        return self.flatten_x({n: samples[n] for n in self.observed_names})

    def simulate_prior(self, rng_key, n):
        """Draw n (theta, x) pairs from the prior predictive.

        Returns (theta dict, x matrix (n, d), s matrix (n, k))."""
        b = min(self.batch_size, n)
        if self._prior_fn is None or self._prior_fn[0] != b:
            pred = Predictive(self.sim_model, num_samples=b,
                              return_sites=self.param_names + self.observed_names,
                              parallel=True)
            data = self.data
            self._prior_fn = (b, jax.jit(lambda k: pred(k, **data)))
        fn = self._prior_fn[1]
        thetas, xs, ss = [], [], []
        for key in jax.random.split(rng_key, math.ceil(n / b)):
            out = fn(key)
            x = self._x_from(out)
            thetas.append({k: out[k] for k in self.param_names})
            xs.append(x)
            ss.append(self.stats(x))
        theta = {k: jnp.concatenate([t[k] for t in thetas])[:n] for k in self.param_names}
        x = jnp.concatenate(xs)[:n]
        s = jnp.concatenate(ss)[:n]
        self.n_simulations += n
        return theta, x, s

    def simulate_given(self, rng_key, u):
        """Simulate data for given unconstrained parameters u (n, D).

        Returns (x matrix, s matrix)."""
        n = u.shape[0]
        # Bucket the batch to a power of two so jit compiles only a few shapes.
        b = min(self.batch_size, 1 << max(int(n - 1).bit_length(), 0))
        if self._cond_fn is None:
            self._cond_fn = {}
        if b not in self._cond_fn:
            data = self.data
            names = self.observed_names

            def run(k, post):
                pred = Predictive(self.sim_model, posterior_samples=post,
                                  return_sites=names, parallel=True)
                return pred(k, **data)

            self._cond_fn[b] = jax.jit(run)
        fn = self._cond_fn[b]
        pad = (-n) % b
        if pad:
            u = jnp.concatenate([u, jnp.repeat(u[-1:], pad, axis=0)])
        xs, ss = [], []
        keys = jax.random.split(rng_key, u.shape[0] // b)
        for j, key in enumerate(keys):
            theta = self.to_constrained(u[j * b:(j + 1) * b])
            x = self._x_from(fn(key, theta))
            xs.append(x)
            ss.append(self.stats(x))
        self.n_simulations += n
        return jnp.concatenate(xs)[:n], jnp.concatenate(ss)[:n]

    def prior_log_density_u(self, u):
        """Prior log density of unconstrained parameters (includes the Jacobian).

        Parameter priors that depend on other *parameters* are handled (all
        parameters are substituted); returns -inf outside the support."""
        if getattr(self, "_logp_fn", None) is None:
            data = self.data
            key = jax.random.PRNGKey(0)

            def one(ui):
                theta = {k: v[0] for k, v in self.to_constrained(ui[None]).items()}
                tr = handlers.trace(
                    handlers.seed(handlers.substitute(self.model, data=theta), key)
                ).get_trace(**data)
                lp = 0.0
                for n in self.param_names:
                    un = self.transforms[n].inv(theta[n])
                    lp = lp + jnp.sum(tr[n]["fn"].log_prob(theta[n]))
                    lp = lp + jnp.sum(self.transforms[n].log_abs_det_jacobian(un, theta[n]))
                return lp

            self._logp_fn = jax.jit(jax.vmap(one))
        lp = self._logp_fn(u)
        return jnp.where(jnp.isfinite(lp), lp, -jnp.inf)

    def observed_stats(self, x_obs=None):
        """Summary statistics of observed data (default: the data given at init).

        x_obs can be a dict ``{site: value}`` or an array for a single observed site."""
        if x_obs is None:
            return self.s_obs
        if not isinstance(x_obs, dict):
            if len(self.observed_names) != 1:
                raise ValueError("Pass a dict {site: value} when the model has "
                                 "several observed sites.")
            x_obs = {self.observed_names[0]: x_obs}
        x = self.flatten_x({n: jnp.asarray(x_obs[n])[None] for n in self.observed_names})
        return self.stats(x)[0]


def weighted_resample(rng_key, u, weights, n):
    idx = jax.random.choice(rng_key, u.shape[0], (n,), p=weights / weights.sum())
    return u[idx]


class SBIPosterior:
    """Duck-typed sampler shared by ABC and NPE results.

    Implements the contract BF's diagnostics rely on (see SVI_BF):
    ``get_samples(group_by_chain=...)``, ``get_extra_fields``, ``num_chains``.
    ``az_from_dict`` tells ``diag.to_az`` to build the arviz trace with
    ``az.from_dict`` (there is no numpyro MCMC object behind it).
    """

    az_from_dict = True

    def __init__(self):
        self.num_warmup = 0
        self.thinning = 1
        self.num_samples = 0
        self.posterior = None
        self.extra = {}

    @property
    def num_chains(self):
        # Draws are independent, so they can be split into 4 pseudo-chains;
        # arviz then reports R-hat/ESS (expected ~1 and ~n) without warnings.
        n = next(iter(self.posterior.values())).shape[0] if self.posterior else 0
        return 4 if n >= 16 and n % 4 == 0 else 1

    def get_samples(self, group_by_chain=False, **kwargs):
        if group_by_chain:
            c = self.num_chains
            return {k: v.reshape((c, -1) + v.shape[1:]) for k, v in self.posterior.items()}
        return dict(self.posterior)

    def get_extra_fields(self, group_by_chain=False, **kwargs):
        return dict(self.extra)
