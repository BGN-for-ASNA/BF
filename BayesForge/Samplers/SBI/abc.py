"""Approximate Bayesian Computation (ABC).

Two algorithms, both driven by a BF model used as a simulator
(:class:`BayesForge.Samplers.SBI.simulate.Simulator`):

* ``rejection`` -- Pritchard et al. (1999) / Beaumont et al. (2002): simulate
  from the prior predictive, keep the parameters whose simulated summary
  statistics fall closest to the observed ones. Optional local-linear
  regression adjustment (Beaumont et al. 2002).
* ``smc`` -- ABC population Monte Carlo (Beaumont et al. 2009): a sequence of
  shrinking tolerances; particles are resampled, perturbed with a Gaussian
  kernel and re-weighted by prior / proposal density.

Perturbations are made in the unconstrained parameter space, so supports
(e.g. ``sigma > 0`` or ``Uniform(0, 10)``) are always respected.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from BayesForge.Samplers.SBI.simulate import Simulator, SBIPosterior, weighted_resample


def _scale(s):
    """Robust per-statistic scale: MAD, falling back to sd when MAD is 0."""
    med = jnp.median(s, axis=0)
    mad = 1.4826 * jnp.median(jnp.abs(s - med), axis=0)
    sd = jnp.std(s, axis=0)
    sc = jnp.where(mad > 0, mad, sd)
    return jnp.where(sc > 0, sc, 1.0)


def _make_distance(distance, s_ref):
    """Return d(s, s_obs) -> (n,) given reference simulations to set scales."""
    if callable(distance):
        return lambda s, s_obs: jax.vmap(lambda si: distance(si, s_obs))(s)
    if distance == "euclidean":
        sc = _scale(s_ref)
        return lambda s, s_obs: jnp.sqrt(jnp.sum(((s - s_obs) / sc) ** 2, axis=1))
    if distance == "mahalanobis":
        cov = jnp.atleast_2d(jnp.cov(s_ref, rowvar=False))
        prec = jnp.linalg.pinv(cov)
        return lambda s, s_obs: jnp.sqrt(jnp.einsum("ni,ij,nj->n", s - s_obs, prec, s - s_obs))
    raise ValueError("distance must be 'euclidean', 'mahalanobis' or a callable.")


def _regression_adjust(u, s, s_obs, d, eps):
    """Local-linear regression adjustment (Beaumont et al. 2002), in u-space."""
    w = 1.0 - (d / eps) ** 2  # Epanechnikov kernel
    w = jnp.clip(w, 0.0, None)
    X = jnp.concatenate([jnp.ones((s.shape[0], 1)), s - s_obs], axis=1)
    Xw = X * w[:, None]
    beta = jnp.linalg.lstsq(Xw.T @ X + 1e-8 * jnp.eye(X.shape[1]), Xw.T @ u)[0]
    return u - (s - s_obs) @ beta[1:]


class ABC_BF(SBIPosterior):
    """Result object + driver for ABC. Behaves like a BF sampler.

    Attributes after ``run``:
        posterior: dict of posterior draws (constrained space).
        extra: {"distance", "weights"} for the accepted particles.
        history: list of dicts per round (epsilon, acceptance, n_simulations).
    """

    def __init__(self, simulator: Simulator, method="rejection", num_simulations=100_000,
                 quantile=0.01, epsilon=None, distance="euclidean", regression_adjust=False,
                 num_samples=1000, num_particles=None, num_rounds=10, alpha=0.5,
                 min_acceptance=0.01, proposal_batch=None, progress_bar=True):
        super().__init__()
        if method not in ("rejection", "smc"):
            raise ValueError("method must be 'rejection' or 'smc'.")
        self.sim = simulator
        self.method = method
        self.num_simulations = int(num_simulations)
        self.quantile = quantile
        self.epsilon = epsilon
        self.distance = distance
        self.regression_adjust = regression_adjust
        self.num_samples = int(num_samples)
        self.num_particles = int(num_particles or num_samples)
        self.num_rounds = int(num_rounds)
        self.alpha = float(alpha)
        self.min_acceptance = float(min_acceptance)
        self.proposal_batch = proposal_batch
        self.progress_bar = progress_bar
        self.history = []

    # -----------------------------------------------------------------------------
    def run(self, rng_key):
        t0 = time.time()
        if self.method == "rejection":
            self._run_rejection(rng_key)
        else:
            self._run_smc(rng_key)
        self.runtime = time.time() - t0
        self.n_simulations = self.sim.n_simulations
        return self

    def _log(self, msg):
        if self.progress_bar:
            print(msg)

    def _finish(self, rng_key, u, weights, d):
        self.particles = u
        self.weights = weights / weights.sum()
        u_post = weighted_resample(rng_key, u, self.weights, self.num_samples)
        self.posterior = self.sim.to_constrained(u_post)
        self.extra = {"distance": d, "weights": self.weights}

    # --- rejection ------------------------------------------------------------------
    def _run_rejection(self, rng_key):
        k_sim, k_res = jax.random.split(rng_key)
        sim = self.sim
        theta, _, s = sim.simulate_prior(k_sim, self.num_simulations)
        dist = _make_distance(self.distance, s)
        d = dist(s, sim.s_obs)
        if self.epsilon is not None:
            keep = jnp.where(d <= self.epsilon)[0]
            eps = float(self.epsilon)
        else:
            k = max(int(np.ceil(self.quantile * self.num_simulations)), 2)
            keep = jnp.argsort(d)[:k]
            eps = float(d[keep[-1]])
        if keep.shape[0] < 2:
            raise RuntimeError(
                f"Only {keep.shape[0]} simulation(s) within epsilon={eps:.3g}. "
                "Increase epsilon / quantile or num_simulations.")
        u = sim.to_unconstrained({k: v[keep] for k, v in theta.items()})
        if self.regression_adjust:
            u = _regression_adjust(u, s[keep], sim.s_obs, d[keep], eps * (1 + 1e-6))
        self.epsilon_final = eps
        self.history.append({"round": 0, "epsilon": eps,
                             "acceptance": keep.shape[0] / self.num_simulations,
                             "n_simulations": self.num_simulations})
        self._log(f"ABC rejection: {self.num_simulations} simulations, "
                  f"{keep.shape[0]} accepted, epsilon = {eps:.4g}")
        self._finish(k_res, u, jnp.ones(u.shape[0]), d[keep])

    # --- SMC (ABC-PMC) --------------------------------------------------------------
    def _run_smc(self, rng_key):
        sim = self.sim
        N = self.num_particles
        budget = self.num_simulations
        n0 = max(int(np.ceil(N / self.quantile)), N) if self.quantile else 10 * N
        n0 = min(n0, max(budget // 4, N))  # keep most of the budget for the SMC rounds
        B = int(self.proposal_batch or max(N, 1000))

        # Round 0: rejection from the prior.
        rng_key, k = jax.random.split(rng_key)
        theta, _, s = sim.simulate_prior(k, n0)
        dist = _make_distance(self.distance, s)
        d = dist(s, sim.s_obs)
        keep = jnp.argsort(d)[:N]
        u, d, w = sim.to_unconstrained({k_: v[keep] for k_, v in theta.items()}), d[keep], jnp.ones(N) / N
        eps = float(d.max())
        self.history.append({"round": 0, "epsilon": eps, "acceptance": N / n0, "n_simulations": n0})
        self._log(f"ABC-SMC round 0: epsilon = {eps:.4g}, acceptance = {N / n0:.3f}")
        used = n0

        # Fixed-shape jitted kernels: proposals are (B, D) and particles (N, D) every
        # round, so each compiles once (dynamic boolean indexing is done in numpy).
        @jax.jit
        def propose(key, u, w, L):
            k_a, k_p = jax.random.split(key)
            anc = jax.random.choice(k_a, u.shape[0], (B,), p=w)
            return u[anc] + jax.random.normal(k_p, (B, u.shape[1])) @ L.T

        @jax.jit
        def log_kernel_mixture(u_new, u, w, prec, logdet):
            diff = u_new[:, None, :] - u[None, :, :]
            log_k = -0.5 * (jnp.einsum("nmi,ij,nmj->nm", diff, prec, diff)
                            + logdet + u.shape[1] * jnp.log(2 * jnp.pi))
            return logsumexp(log_k + jnp.log(w)[None, :], axis=1)

        for r in range(1, self.num_rounds):
            if self.epsilon is not None and eps <= self.epsilon:
                break
            eps_t = float(jnp.quantile(d, self.alpha))
            if self.epsilon is not None:
                eps_t = max(eps_t, float(self.epsilon))
            # Gaussian kernel, twice the weighted covariance (Beaumont et al. 2009).
            mean = jnp.sum(w[:, None] * u, axis=0)
            cov = 2.0 * jnp.atleast_2d((w[:, None] * (u - mean)).T @ (u - mean))
            cov = cov + 1e-9 * jnp.eye(u.shape[1])
            L = jnp.linalg.cholesky(cov)
            prec = jnp.linalg.inv(cov)
            logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

            acc_u, acc_d, n_acc, n_prop = [], [], 0, 0
            while n_acc < N and used + n_prop < budget:
                rng_key, k_p, k_s = jax.random.split(rng_key, 3)
                u_prop = propose(k_p, u, w, L)
                ok = np.isfinite(np.asarray(sim.prior_log_density_u(u_prop)))
                _, s_prop = sim.simulate_given(k_s, u_prop)
                d_prop = np.asarray(dist(s_prop, sim.s_obs))
                n_prop += int(ok.sum())
                a = ok & (d_prop <= eps_t)
                acc_u.append(np.asarray(u_prop)[a])
                acc_d.append(d_prop[a])
                n_acc += int(a.sum())
            used += n_prop
            acceptance = n_acc / max(n_prop, 1)
            if n_acc < N:
                self._log(f"ABC-SMC round {r}: simulation budget exhausted "
                          f"({n_acc}/{N} particles); stopping at previous round.")
                break
            u_new = jnp.asarray(np.concatenate(acc_u)[:N])
            d_new = jnp.asarray(np.concatenate(acc_d)[:N])
            # Importance weights: prior / kernel mixture.
            log_q = log_kernel_mixture(u_new, u, w, prec, logdet)
            log_w = sim.prior_log_density_u(u_new) - log_q
            w_new = jnp.exp(log_w - log_w.max())
            u, d, w, eps = u_new, d_new, w_new / w_new.sum(), eps_t
            ess = float(1.0 / jnp.sum(w**2))
            self.history.append({"round": r, "epsilon": eps, "acceptance": acceptance,
                                 "n_simulations": n_prop, "ess": ess})
            self._log(f"ABC-SMC round {r}: epsilon = {eps:.4g}, "
                      f"acceptance = {acceptance:.3f}, ESS = {ess:.0f}")
            if acceptance < self.min_acceptance or used >= budget:
                break

        self.epsilon_final = eps
        rng_key, k = jax.random.split(rng_key)
        self._finish(k, u, w, d)

    # --- plots ----------------------------------------------------------------------
    def plot_history(self, ax=None):
        """Tolerance and cumulative simulations per SMC round."""
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        rounds = [h["round"] for h in self.history]
        ax.plot(rounds, [h["epsilon"] for h in self.history], "o-")
        ax.set_xlabel("Round")
        ax.set_ylabel("Tolerance $\\epsilon$")
        ax2 = ax.twinx()
        ax2.plot(rounds, np.cumsum([h["n_simulations"] for h in self.history]), "s--", color="grey")
        ax2.set_ylabel("Cumulative simulations", color="grey")
        return ax
