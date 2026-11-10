import jax.numpy as jnp
import jax
import numpyro
import jax.scipy.stats as stats
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from BayesForge.Utils.dev_warning import warn_dev

class bnnc:
    def __init__(self, parent):
        """
        Bayesian Neural Network Clustering (BNNC) Class.
        
        Args:
            parent: The BF parent instance.
        """
        self.__name__ = 'bnnc' 
        self.parent = parent

    def __call__(self, data, K=11, D_H1=10, empirical_bayes=False):
        """
        Makes the class instance callable. 
        Redirects the call to self.model().
        """
        return self.model(data, K=K, D_H1=D_H1, empirical_bayes=empirical_bayes)

    def model(self, data, K=11, D_H1=10, empirical_bayes=False):
        """
        BNNC marginal mixture model definition.
        """
        warn_dev("bnnc")
        m = self.parent
        N, D_X = data.shape
        
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
            dist=m.dist.normal(0, 1, name='w1_weight', shape=(D_X, D_H1)),
            activation='tanh'
        )
        
        w2 = m.bnn.layer_linear(
            w1,
            dist=m.dist.normal(0, 0.05, name='w2_weight', shape=(D_H1, K))
        )
        
        # Global parsimony prior: Dirichlet prior on global cluster weights
        # Alpha < 1 acts as a sparse prior to suppress inactive clusters
        pi = numpyro.sample('global_pi', numpyro.distributions.Dirichlet(jnp.ones(K) * alpha_bnnc))
        
        logits = w2 + jnp.log(pi + 1e-10)
        log_p = jax.nn.log_softmax(logits, axis=-1)
        theta = jnp.exp(log_p)
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
        log_probs = jnp.log(theta_prob + 1e-8) + log_liks
        norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
        return norm_probs

    def predict(self, data, sampler):
        """
        Performs Consensus Clustering.
        """
        warn_dev("bnnc")
        posterior_samples = sampler.get_samples(group_by_chain=False)

        theta_samps = posterior_samples['theta']
        mu_samps = posterior_samples['mu']
        sigma_samps = posterior_samples['sigma']

        # Cap draws for the O(S * N^2) consensus step below: the full chain
        # length OOMs and kills the kernel. An evenly-spaced subsample of a few
        # hundred draws gives the same consensus clustering.
        MAX_DRAWS = 300
        S = theta_samps.shape[0]
        if S > MAX_DRAWS:
            idx = np.linspace(0, S - 1, MAX_DRAWS).astype(int)
            theta_samps = theta_samps[idx]
            mu_samps = mu_samps[idx]
            sigma_samps = sigma_samps[idx]

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
        print(f"Model found {len(np.unique(final_labels))} clusters.")

        return theta_samps, mu_samps, sigma_samps, final_labels

    def plot(self, data, sampler, figsize=(10, 8), point_size=30):
        """
        Posterior predictive plot: a filled contour of the expected cluster
        index over the feature plane, with the data coloured by their
        argmax cluster.

        The predictive at a point x is  p(k | x) ∝ pi_k * N(x | mu_k, sigma_k),
        averaged over posterior draws of (global_pi, mu, sigma). The contour
        shows sum_k k * p(k | x); the colour bar is that expected index.
        """
        warn_dev("bnnc")

        ps = sampler.get_samples(group_by_chain=False)
        pi_s = np.asarray(ps['global_pi'])   # (S, K)
        mu_s = np.asarray(ps['mu'])          # (S, K, D)
        sig_s = np.asarray(ps['sigma'])      # (S, K, D)

        # Cap draws (grid PDF is O(S * n_grid * K)).
        MAX_DRAWS = 200
        S = pi_s.shape[0]
        if S > MAX_DRAWS:
            idx = np.linspace(0, S - 1, MAX_DRAWS).astype(int)
            pi_s, mu_s, sig_s = pi_s[idx], mu_s[idx], sig_s[idx]

        K = pi_s.shape[1]
        data = np.asarray(data)

        def post_pred(pts):
            """Posterior-mean p(k | pts): (n_pts, K)."""
            X = pts[:, None, None, :]                       # (n,1,1,D)
            mu = mu_s[None, :, :, :]                        # (1,S,K,D)
            sg = sig_s[None, :, :, :]
            logn = -0.5 * (((X - mu) / sg) ** 2) - np.log(sg) - 0.5 * np.log(2 * np.pi)
            logp = logn.sum(-1) + np.log(pi_s[None] + 1e-10)   # (n,S,K)
            logp -= logp.max(-1, keepdims=True)
            p = np.exp(logp)
            p /= p.sum(-1, keepdims=True)
            return p.mean(1)                                # (n,K)

        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.column_stack([xx.ravel(), yy.ravel()])

        grid_p = post_pred(grid)
        idx_grid = np.arange(K)
        Z = (grid_p * idx_grid).sum(-1).reshape(xx.shape)   # expected cluster index
        data_lbl = post_pred(data).argmax(-1)

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