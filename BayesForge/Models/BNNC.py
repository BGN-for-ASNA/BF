import jax.numpy as jnp
import jax
import numpyro
import jax.scipy.stats as stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
from BayesForge.Utils.dev_warning import warn_dev
from BayesForge.BNN.bnn import _prior_scale

# Floor added inside log() terms (weights, gating probabilities) so a 0 in
# float32 gives log(tiny) instead of -inf/NaN. Used by model, predict, plot.
_LOG_EPS = 1e-10

class bnnc:
    def __init__(self, parent):
        """
        Bayesian Neural Network Clustering (BNNC) Class.
        
        Args:
            parent: The BF parent instance.
        """
        self.__name__ = 'bnnc' 
        self.parent = parent

    def __call__(self, data, K=11, D_H1=10, empirical_bayes=False, store_theta=False,
                 prior_scale=None):
        """
        Makes the class instance callable. 
        Redirects the call to self.model().
        """
        return self.model(data, K=K, D_H1=D_H1, empirical_bayes=empirical_bayes,
                          store_theta=store_theta, prior_scale=prior_scale)

    def model(self, data, K=11, D_H1=10, empirical_bayes=False, store_theta=False,
              prior_scale=None):
        """
        BNNC marginal mixture model definition.

        Two prior sets, selected by ``empirical_bayes``. With
        ``empirical_bayes=True`` the priors are data-driven: mu ~ N(mean, 2 *
        std) and sigma ~ HalfNormal(2 * std) per feature, with a Dirichlet(1/K)
        global weight — they follow the data scale. With
        ``empirical_bayes=False`` (the default) the priors are fixed:
        mu ~ N(0, 5), sigma ~ Exponential(1), Dirichlet(0.05) — these assume
        standardised features, so the model warns when the data do not look
        standardised (mean far from 0 or spread far from 1).

        ``store_theta`` adds a ``numpyro.deterministic('theta', theta)`` site
        holding the per-point gating weights (N, K). It is off by default:
        the site is stored for every draw and costs gigabytes for N ~ 1e4;
        ``_theta_draws`` recomputes theta from the weight draws on demand.

        ``prior_scale`` (MAJ-5, opt-in) sets the std of the w1 weight prior:
        None (default) keeps the historical N(0, 1); "fan_in" uses
        N(0, 1/sqrt(fan_in)) with fan_in = D_X (first dimension of w1); a
        float is used as the scale directly.
        """
        if jnp.ndim(data) != 2:
            raise ValueError("bnnc: data must be 2-D (N, D)")
        if data.shape[0] < 1:
            raise ValueError("bnnc: data must contain at least one sample (N >= 1)")
        if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 1:
            raise ValueError("bnnc: K must be an integer >= 1")
        warn_dev("bnnc")
        m = self.parent
        N, D_X = data.shape
        self._model_kwargs = dict(K=K, D_H1=D_H1, empirical_bayes=empirical_bayes,
                                  prior_scale=prior_scale)

        if not empirical_bayes:
            # The fixed priors assume standardised features; warn on raw-scale
            # data. Under a JAX tracer the concrete conversion fails -> skip.
            try:
                m_ = np.asarray(jnp.mean(data, 0))
                s_ = np.asarray(jnp.std(data, 0))
                if np.any(np.abs(m_) > 3) or np.any(s_ > 5) or np.any(s_ < 0.2):
                    warnings.warn(
                        "bnnc: the default priors (mu ~ N(0, 5), sigma ~ Exponential(1)) "
                        "assume standardised features; standardise the data or pass "
                        "empirical_bayes=True.",
                        UserWarning, stacklevel=2)
            except Exception:
                pass
        
        if empirical_bayes:
            # Data-driven priors
            data_mean = jnp.mean(data, axis=0)
            data_std = jnp.std(data, axis=0) * 2.0
            sigma_base = data_std
            alpha_bnnc = 1.0 / K
        else:
            # Fixed uninformative priors
            data_mean = jnp.zeros(D_X)
            data_std = jnp.ones(D_X) * 5.0
            sigma_base = 1.0
            alpha_bnnc = 0.05
        
        # --- BNN Gating Network ---
        w1 = m.bnn.layer_linear(
            data, 
            dist=m.dist.normal(0, _prior_scale(D_X, prior_scale), name='w1_weight', shape=(D_X, D_H1)),
            activation='tanh'
        )
        
        w2 = m.bnn.layer_linear(
            w1,
            dist=m.dist.normal(0, 0.05, name='w2_weight', shape=(D_H1, K))
        )
        
        # Global parsimony prior: Dirichlet prior on global cluster weights
        # Alpha < 1 acts as a sparse prior to suppress inactive clusters
        pi = numpyro.sample('global_pi', numpyro.distributions.Dirichlet(jnp.ones(K) * alpha_bnnc))
        
        logits = w2 + jnp.log(pi + _LOG_EPS)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        theta = jnp.exp(log_p)
        if store_theta:
            numpyro.deterministic('theta', theta)
        
        # --- Mixture Components ---
        mu = m.dist.normal(data_mean, data_std, name='mu', shape=(K, D_X))
        
        if empirical_bayes:
            sigma = m.dist.half_normal(sigma_base, name='sigma', shape=(K, D_X))
        else:
            sigma = m.dist.exponential(sigma_base, name='sigma', shape=(K, D_X))
        
        # --- Marginalized Gaussian Mixture ---
        data_exp = jnp.expand_dims(data, axis=1) # (N, 1, D_X)
        mu_exp = jnp.expand_dims(mu, axis=0)     # (1, K, D_X)
        sigma_exp = jnp.expand_dims(sigma, axis=0) # (1, K, D_X)
        
        log_pdf_clusters = jnp.sum(stats.norm.logpdf(data_exp, loc=mu_exp, scale=sigma_exp), axis=-1) # (N, K)
        weighted_log_pdf = log_p + log_pdf_clusters # (N, K)
        
        total_log_likelihood = jax.scipy.special.logsumexp(weighted_log_pdf, axis=-1)
        numpyro.factor('mixture_likelihood', jnp.sum(total_log_likelihood))

    def _theta_draws(self, data, posterior_samples, key=None):
        """Per-point gating weights theta (S, N, K) recomputed from the posterior
        draws of the network weights and global_pi (theta is not stored by the fit)."""
        from numpyro.infer import Predictive
        if 'theta' in posterior_samples:
            return posterior_samples['theta']
        kw = dict(getattr(self, '_model_kwargs', {}))
        kw['store_theta'] = True
        pred = Predictive(self.model, posterior_samples=posterior_samples, return_sites=['theta'])
        key = jax.random.PRNGKey(0) if key is None else key
        with warnings.catch_warnings():
            # Re-running the model on a plot grid must not repeat the B3
            # "standardise the data" warning meant for the fitted data.
            warnings.simplefilter("ignore", UserWarning)
            return pred(key, jnp.asarray(data), **kw)['theta']

    def get_cluster_probs(self, data_pts, theta_prob, mu_p, sigma_p):
        """
        Compute normalized posterior cluster membership probabilities for data points.

        Parameters:
        - data_pts: Data array (shape: [N, D]).
        - theta_prob: Cluster mixture weights (shape: [K] or [N_draws, K]).
        - mu_p: Cluster component means (shape: [K, D]).
        - sigma_p: Cluster component standard deviations (shape: [K, D]).

        Returns:
        - Array of normalized posterior cluster assignment probabilities (shape: [N, K]).
        """
        X_e = jnp.expand_dims(data_pts, axis=1) # (N, 1, D_X)
        mu_e = jnp.expand_dims(mu_p, axis=0) # (1, K, D_X)
        sig_e = jnp.expand_dims(sigma_p, axis=0) # (1, K, D_X)
        
        log_liks = jnp.sum(stats.norm.logpdf(X_e, loc=mu_e, scale=sig_e), axis=-1)
        log_probs = jnp.log(theta_prob + _LOG_EPS) + log_liks
        norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
        return norm_probs

    def predict(self, data, sampler, verbose=False):
        """
        Performs Consensus Clustering.

        Args:
            data: Input data array of shape ``(N, D)``.
            sampler: Fitted BayesForge sampler instance.
            verbose (bool, optional): If True, print the number of clusters
                found by the consensus cut. Defaults to False (silent).
        """
        warn_dev("bnnc")
        posterior_samples = sampler.get_samples(group_by_chain=False)

        # Cap draws for the O(S * N^2) consensus step below: the full chain
        # length OOMs and kills the kernel. An evenly-spaced subsample of a few
        # hundred draws gives the same consensus clustering. The subsample is
        # applied to every stored array FIRST; theta is not stored by the fit
        # (B1) and is recomputed only for the draws that are actually used.
        MAX_DRAWS = 300
        S = posterior_samples['mu'].shape[0]
        if S > MAX_DRAWS:
            idx = np.linspace(0, S - 1, MAX_DRAWS).astype(int)
            posterior_samples = {k: v[idx] for k, v in posterior_samples.items()}

        theta_samps = self._theta_draws(data, posterior_samples)
        mu_samps = posterior_samples['mu']
        sigma_samps = posterior_samples['sigma']

        cluster_probs = jax.vmap(self.get_cluster_probs, in_axes=(None, 0, 0, 0))(
            data, theta_samps, mu_samps, sigma_samps
        )

        similarity_matrix = (cluster_probs @ cluster_probs.transpose(0, 2, 1)).mean(axis=0)

        distance_matrix = 1 - similarity_matrix
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        distance_matrix = distance_matrix.at[jnp.diag_indices_from(distance_matrix)].set(0.0)
        distance_matrix = jnp.clip(distance_matrix, min=0.0)
        
        condensed_dist = squareform(distance_matrix)
        Z = linkage(condensed_dist, 'average')
        
        final_labels = fcluster(Z, t=0.5, criterion='distance')
        if verbose:
            print(f"Model found {len(np.unique(final_labels))} clusters.")

        return theta_samps, mu_samps, sigma_samps, final_labels

    def plot(self, data, sampler, figsize=(10, 8), point_size=30):
        """
        Posterior predictive plot: a filled contour of the expected cluster
        index over the feature plane, with the data coloured by their
        argmax cluster.

        The predictive at a point x is  p(k | x) ∝ theta_k(x) * N(x | mu_k, sigma_k),
        averaged over posterior draws of the gating network (w1_weight,
        w2_weight), global_pi, mu and sigma. theta_k(x) = softmax(w2(x) +
        log pi) is the network's per-point gating (not the global pi_k);
        it is recomputed for the grid and for the data points with
        ``_theta_draws`` on the subsampled draws, exactly like ``predict``.
        The contour shows sum_k k * p(k | x); the colour bar is that expected index.
        """
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("bnnc.plot: plot requires 2-D data with D=2 "
                             f"features (got shape {data.shape}).")
        if data.shape[0] < 2:
            raise ValueError("bnnc.plot: plot requires at least 2 data points "
                             f"(got {data.shape[0]}).")
        warn_dev("bnnc")

        ps = sampler.get_samples(group_by_chain=False)

        # Cap draws (grid PDF is O(S * n_grid * K)).
        MAX_DRAWS = 200
        S = np.asarray(ps['mu']).shape[0]
        if S > MAX_DRAWS:
            idx = np.linspace(0, S - 1, MAX_DRAWS).astype(int)
            ps = {k: v[idx] for k, v in ps.items()}

        mu_s = np.asarray(ps['mu'], dtype=np.float32)      # (S, K, D)
        sig_s = np.asarray(ps['sigma'], dtype=np.float32)  # (S, K, D)
        K = mu_s.shape[1]
        data = np.asarray(data)

        def post_pred(pts, theta):
            """Posterior-mean p(k | pts): (n_pts, K). theta: (S, n_pts, K)."""
            X = pts[:, None, None, :]                       # (n,1,1,D)
            mu = mu_s[None, :, :, :]                        # (1,S,K,D)
            sg = sig_s[None, :, :, :]
            logn = -0.5 * (((X - mu) / sg) ** 2) - np.log(sg) - 0.5 * np.log(2 * np.pi)
            logp = logn.sum(-1) + np.log(theta.transpose(1, 0, 2) + _LOG_EPS)   # (n,S,K)
            logp -= logp.max(-1, keepdims=True)
            p = np.exp(logp)
            p /= p.sum(-1, keepdims=True)
            return p.mean(1)                                # (n,K)

        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.column_stack([xx.ravel(), yy.ravel()])

        # Chunk the grid predictive (B4): peak memory stays
        # chunk * S * K * D * 4 bytes instead of n_grid * S * K * D * 4.
        CHUNK = 2000
        grid_p = np.concatenate([
            post_pred(np.asarray(grid[i:i + CHUNK], dtype=np.float32),
                      np.asarray(self._theta_draws(grid[i:i + CHUNK], ps),
                                 dtype=np.float32))
            for i in range(0, grid.shape[0], CHUNK)
        ])
        idx_grid = np.arange(K)
        Z = (grid_p * idx_grid).sum(-1).reshape(xx.shape)   # expected cluster index
        data_lbl = post_pred(
            np.asarray(data, dtype=np.float32),
            np.asarray(self._theta_draws(data, ps), dtype=np.float32)).argmax(-1)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=figsize)
        cf = ax.contourf(xx, yy, Z, levels=14, cmap='viridis', alpha=0.85)
        fig.colorbar(cf, ax=ax)
        ax.scatter(data[:, 0], data[:, 1], c=data_lbl, cmap='viridis',
                   vmin=0, vmax=K - 1, s=point_size, edgecolor='k', linewidth=0.4)

        ax.set_title("Posterior Predictive Mean", fontsize=16)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

        fig.tight_layout()
        return fig, ax