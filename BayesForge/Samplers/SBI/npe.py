"""Neural Posterior Estimation (NPE) and Bayesian NPE.

NPE (Papamakarios & Murray 2016; Greenberg et al. 2019) trains a conditional
density estimator q_phi(theta | s(x)) on pairs (theta, x) simulated from the
prior predictive, by minimising E[-log q_phi(theta | s(x))]. Once trained, the
posterior for *any* dataset is one forward pass away (amortisation).

Bayesian NPE (Delaunoy et al. 2024) puts a prior on the network weights phi and
approximates p(phi | simulations) with variational inference, so the posterior
estimate is a mixture over plausible networks:
    q(theta | x) = E_{p(phi | D)}[q_phi(theta | x)].
This counteracts the over-confidence of NPE when the simulation budget is small.

Two density estimators, both pure JAX (no extra dependency):
* ``mdn``  -- mixture density network, K full-covariance Gaussians.
* ``flow`` -- conditional affine-coupling normalizing flow (RealNVP).

Parameters are modelled in the unconstrained space of their support and mapped
back, so ``sigma > 0`` etc. hold by construction.
"""

import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from jax.scipy.special import logsumexp

from BayesForge.BNN.bnn import bnn
from BayesForge.Samplers.SBI.simulate import Simulator, SBIPosterior

# Network layers come from BF's BNN module, so NPE and Bayesian NPE share the layer
# and activation implementations with m.bnn.
_BNN = bnn()


def _check_activation(name):
    if not callable(getattr(_BNN, name, None)) or name not in _BNN._activation_map:
        raise ValueError(f"Unknown activation {name!r}. Available: {_BNN.available_activations()}")
    return name


# MLP ------------------------------------------------------------------------------------
def _mlp_init(key, sizes):
    params = []
    for k, (n_in, n_out) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append({"W": jax.random.normal(k, (n_in, n_out)) * math.sqrt(1.0 / n_in),
                       "b": jnp.zeros(n_out)})
    return params


def _mlp(params, x, act, last_act=False):
    """Stack of bnn.layer_linear layers; `act` is a bnn activation name."""
    for i, p in enumerate(params):
        hidden = i < len(params) - 1 or last_act
        x = _BNN.layer_linear(x, p["W"], activation=act if hidden else None,
                              bias=p["b"], verbose=False)
    return x


# Density estimators -----------------------------------------------------------------------
class MDN:
    """Mixture of K full-covariance Gaussians whose parameters are an MLP of the context."""

    def __init__(self, dim, ctx_dim, hidden=(64, 64), n_components=5, activation="gelu"):
        self.D, self.K = dim, n_components
        self.sizes = (ctx_dim,) + tuple(hidden) + (bnn.mixture_normal_size(self.K, dim),)
        self.act = _check_activation(activation)

    def init(self, key):
        p = _mlp_init(key, self.sizes)
        p[-1]["W"] = p[-1]["W"] * 0.1  # start close to a standard normal mixture
        return {"net": p}

    def dist(self, params, ctx):
        """bnn layers -> BF mixture of K multivariate Normals (bnn.layer_mixture_normal)."""
        out = _mlp(params["net"], ctx, self.act)
        return _BNN.layer_mixture_normal(out, self.K, self.D, diag_offset=0.5413)

    def log_prob(self, params, u, ctx):
        return self.dist(params, ctx).log_prob(u)

    def sample(self, params, key, ctx, n):
        return self.dist(params, ctx).sample(key, (n,))


class CouplingFlow:
    """Conditional RealNVP: affine coupling layers with alternating masks.

    Each layer maps u -> m*u + (1-m)*(u*exp(s) + t) with (s, t) = NN([m*u, h]),
    where h is an embedding of the context. Base distribution: N(0, I).
    """

    def __init__(self, dim, ctx_dim, hidden=(64, 64), n_layers=6, activation="gelu"):
        self.D = dim
        self.act = _check_activation(activation)
        self.emb = hidden[-1]
        self.emb_sizes = (ctx_dim,) + tuple(hidden)
        self.layer_sizes = (dim + self.emb,) + tuple(hidden) + (2 * dim,)
        masks = []
        for i in range(n_layers):
            m = (jnp.arange(dim) + i) % 2
            masks.append(m.astype(jnp.float32))
        self.masks = masks

    def init(self, key):
        k_e, *k_l = jax.random.split(key, len(self.masks) + 1)
        layers = []
        for k in k_l:
            p = _mlp_init(k, self.layer_sizes)
            p[-1]["W"] = p[-1]["W"] * 0.0  # identity at initialisation
            layers.append(p)
        return {"emb": _mlp_init(k_e, self.emb_sizes), "layers": layers}

    def _st(self, p, m, u, h):
        out = _mlp(p, jnp.concatenate([u * m, h], axis=-1), self.act)
        s = 2.0 * jnp.tanh(out[..., :self.D] / 2.0) * (1 - m)
        t = out[..., self.D:] * (1 - m)
        return s, t

    def log_prob(self, params, u, ctx):
        h = _mlp(params["emb"], ctx, self.act, last_act=True)
        h = jnp.broadcast_to(h, u.shape[:-1] + h.shape[-1:])
        logdet = 0.0
        z = u
        for p, m in zip(params["layers"], self.masks):
            s, t = self._st(p, m, z, h)
            z = z * jnp.exp(s) + t
            logdet = logdet + jnp.sum(s, axis=-1)
        base = jnp.sum(-0.5 * z**2 - 0.5 * jnp.log(2 * jnp.pi), axis=-1)
        return base + logdet

    def sample(self, params, key, ctx, n):
        h = _mlp(params["emb"], ctx, self.act, last_act=True)
        h = jnp.broadcast_to(h, (n,) + h.shape[-1:])
        z = jax.random.normal(key, (n, self.D))
        for p, m in reversed(list(zip(params["layers"], self.masks))):
            s, t = self._st(p, m, z, h)  # depends only on the unchanged (masked) part
            z = (z - t) * jnp.exp(-s)
        return z


# NPE driver --------------------------------------------------------------------------------
class NPE_BF(SBIPosterior):
    """Train a (Bayesian) neural posterior estimator. Behaves like a BF sampler.

    Attributes after ``run``:
        posterior: posterior draws for the observed data (constrained space).
        losses / val_losses: training curves (per evaluation chunk).
        elbo: SVI losses of the Bayesian variant.
    """

    def __init__(self, simulator: Simulator, estimator="mdn", bayesian=False,
                 num_simulations=10_000, hidden=(64, 64), n_components=5, n_layers=6,
                 activation="gelu", lr=1e-3, batch_size=256, max_steps=20_000,
                 val_frac=0.1, patience=20, eval_every=200, prior_scale=1.0,
                 svi_steps=5_000, num_weight_samples=100, num_samples=1000,
                 progress_bar=True):
        super().__init__()
        self.sim = simulator
        self.estimator_name = estimator
        self.bayesian = bool(bayesian)
        self.num_simulations = int(num_simulations)
        self.hidden = tuple(hidden)
        self.n_components = n_components
        self.n_layers = n_layers
        self.activation = activation
        self.lr = lr
        self.batch_size = int(batch_size)
        self.max_steps = int(max_steps)
        self.val_frac = val_frac
        self.patience = patience
        self.eval_every = int(eval_every)
        self.prior_scale = prior_scale
        self.svi_steps = int(svi_steps)
        self.num_weight_samples = int(num_weight_samples)
        self.num_samples = int(num_samples)
        self.progress_bar = progress_bar

    def _log(self, msg):
        if self.progress_bar:
            print(msg)

    # --- training ---------------------------------------------------------------------
    def run(self, rng_key):
        t0 = time.time()
        k_sim, k_init, k_train, k_svi, k_post = jax.random.split(rng_key, 5)
        sim = self.sim
        theta, _, s = sim.simulate_prior(k_sim, self.num_simulations)
        u = sim.to_unconstrained(theta)
        ok = jnp.all(jnp.isfinite(u), axis=1) & jnp.all(jnp.isfinite(s), axis=1)
        u, s = u[ok], s[ok]

        self.u_mean, self.u_std = u.mean(0), u.std(0) + 1e-8
        self.s_mean, self.s_std = s.mean(0), s.std(0) + 1e-8
        un = (u - self.u_mean) / self.u_std
        sn = (s - self.s_mean) / self.s_std

        D, S = un.shape[1], sn.shape[1]
        if self.estimator_name == "mdn":
            self.est = MDN(D, S, self.hidden, self.n_components, self.activation)
        elif self.estimator_name == "flow":
            self.est = CouplingFlow(D, S, self.hidden, self.n_layers, self.activation)
        else:
            raise ValueError("estimator must be 'mdn' or 'flow'.")

        n_val = max(int(self.val_frac * un.shape[0]), 1)
        perm = jax.random.permutation(k_init, un.shape[0])
        val, tr = perm[:n_val], perm[n_val:]
        self.params = self._train(k_train, self.est.init(k_init), un[tr], sn[tr], un[val], sn[val])
        self._log(f"NPE ({self.estimator_name}): {un.shape[0]} simulations, "
                  f"best validation NLL = {min(self.val_losses):.3f}")
        if self.bayesian:
            self._fit_bayesian(k_svi, un, sn)

        self.posterior = self.sample_posterior(k_post, None, self.num_samples)
        self.extra = {}
        self.n_simulations = sim.n_simulations
        self.runtime = time.time() - t0
        return self

    def _train(self, key, params, u_tr, s_tr, u_val, s_val):
        opt = numpyro.optim.Adam(self.lr)
        state = opt.init(params)
        est, bs, n = self.est, min(self.batch_size, u_tr.shape[0]), u_tr.shape[0]

        def nll(p, u, c):
            return -jnp.mean(est.log_prob(p, u, c))

        @jax.jit
        def chunk(state, key):
            def step(state, k):
                idx = jax.random.randint(k, (bs,), 0, n)
                l, g = jax.value_and_grad(nll)(opt.get_params(state), u_tr[idx], s_tr[idx])
                return opt.update(g, state), l
            state, ls = jax.lax.scan(step, state, jax.random.split(key, self.eval_every))
            return state, ls.mean(), nll(opt.get_params(state), u_val, s_val)

        self.losses, self.val_losses = [], []
        best, best_params, wait = jnp.inf, params, 0
        for key_c in jax.random.split(key, max(self.max_steps // self.eval_every, 1)):
            state, l_tr, l_val = chunk(state, key_c)
            self.losses.append(float(l_tr))
            self.val_losses.append(float(l_val))
            if l_val < best - 1e-4:
                best, best_params, wait = l_val, opt.get_params(state), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        return best_params

    def _fit_bayesian(self, key, un, sn):
        """Mean-field VI over the network weights (Bayesian NPE)."""
        from numpyro.infer import SVI, Trace_ELBO
        from numpyro.infer.autoguide import AutoNormal
        from numpyro.infer.initialization import init_to_value

        leaves, treedef = jax.tree_util.tree_flatten(self.params)
        names = [f"phi_{i}" for i in range(len(leaves))]
        # Weight prior N(0, prior_scale / sqrt(fan_in)); biases N(0, prior_scale).
        scales = [self.prior_scale / math.sqrt(l.shape[0]) if l.ndim == 2 else self.prior_scale
                  for l in leaves]
        est, n, bs = self.est, un.shape[0], min(self.batch_size, un.shape[0])

        def model(u, c):
            phi = [_BNN.dist.normal(0.0, sc, shape=l.shape, event=l.ndim, name=nm)
                   for nm, sc, l in zip(names, scales, leaves)]
            params = jax.tree_util.tree_unflatten(treedef, phi)
            with numpyro.plate("sims", n, subsample_size=bs) as idx:
                numpyro.factor("loglik", est.log_prob(params, u[idx], c[idx]))

        guide = AutoNormal(model, init_loc_fn=init_to_value(values=dict(zip(names, leaves))),
                           init_scale=1e-3)
        svi = SVI(model, guide, numpyro.optim.Adam(self.lr), Trace_ELBO())
        res = svi.run(key, self.svi_steps, un, sn, progress_bar=self.progress_bar)
        self.elbo = res.losses
        k = jax.random.fold_in(key, 1)
        draws = guide.sample_posterior(k, res.params, sample_shape=(self.num_weight_samples,))
        # Stack the M weight draws into one pytree with a leading M axis.
        self.params_bayes = jax.tree_util.tree_unflatten(treedef, [draws[nm] for nm in names])
        self._log(f"Bayesian NPE: {self.num_weight_samples} network draws from the "
                  f"variational weight posterior (final ELBO loss = {float(res.losses[-1]):.1f})")

    # --- posterior --------------------------------------------------------------------
    def _ctx(self, s):
        return (s - self.s_mean) / self.s_std

    def log_q_u(self, u, s):
        """log q(u | s) for raw unconstrained parameters u (n, D) and stats s (n, S) or (S,)."""
        un = (u - self.u_mean) / self.u_std
        c = jnp.broadcast_to(self._ctx(s), (un.shape[0],) + self.s_mean.shape)
        jac = -jnp.sum(jnp.log(self.u_std))
        if self.bayesian:
            lp = jax.vmap(lambda p: self.est.log_prob(p, un, c))(self.params_bayes)
            return logsumexp(lp, axis=0) - jnp.log(lp.shape[0]) + jac
        return self.est.log_prob(self.params, un, c) + jac

    def sample_u(self, key, s, n):
        """n draws of raw unconstrained parameters given stats s (S,)."""
        c = self._ctx(s)
        if self.bayesian:
            M = self.num_weight_samples
            per = math.ceil(n / M)
            keys = jax.random.split(key, M)
            un = jax.vmap(lambda p, k: self.est.sample(p, k, c, per))(self.params_bayes, keys)
            un = un.reshape(-1, un.shape[-1])
            un = jax.random.permutation(jax.random.fold_in(key, 7), un)[:n]
        else:
            un = self.est.sample(self.params, key, c, n)
        return un * self.u_std + self.u_mean

    def sample_posterior(self, key, x_obs=None, num_samples=1000):
        """Posterior draws (constrained) for x_obs; default: the data given at fit time."""
        s = self.sim.observed_stats(x_obs)
        return self.sim.to_constrained(self.sample_u(key, s, num_samples))

    # --- calibration ------------------------------------------------------------------
    def coverage(self, key, n_test=200, n_post=500, levels=None):
        """Expected coverage of highest-posterior-density regions (Hermans et al. 2022).

        For each test pair (theta*, x*) drawn from the prior predictive, compute
        the smallest HPD level whose region contains theta*. A calibrated
        estimator gives empirical coverage = nominal level; below the diagonal
        means over-confident posteriors.

        Returns a pandas DataFrame with columns ``nominal`` and ``empirical``.
        """
        import pandas as pd
        if levels is None:
            levels = np.linspace(0.01, 0.99, 50)
        k_sim, k_post = jax.random.split(key)
        theta, _, s = self.sim.simulate_prior(k_sim, n_test)
        u_star = self.sim.to_unconstrained(theta)

        def rank(k, u0, s0):
            us = self.sample_u(k, s0, n_post)
            lq = self.log_q_u(us, s0)
            lq0 = self.log_q_u(u0[None], s0)[0]
            return jnp.mean(lq > lq0)  # mass of the smallest HPD region containing theta*

        ranks = np.asarray(jax.lax.map(lambda a: rank(*a),
                                       (jax.random.split(k_post, n_test), u_star, s)))
        emp = [(ranks <= a).mean() for a in levels]
        self.coverage_table = pd.DataFrame({"nominal": levels, "empirical": emp})
        return self.coverage_table

    # --- plots ------------------------------------------------------------------------
    def plot_loss(self, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        steps = np.arange(1, len(self.losses) + 1) * self.eval_every
        ax.plot(steps, self.losses, label="train")
        ax.plot(steps, self.val_losses, label="validation")
        ax.set_xlabel("Step")
        ax.set_ylabel("Negative log-likelihood")
        ax.legend()
        return ax
