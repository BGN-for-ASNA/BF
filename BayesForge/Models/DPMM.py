from BayesForge.Distributions.np_dists import UnifiedDist as dist
from BayesForge.Utils.dev_warning import warn_dev
import jax.numpy as jnp
import numpyro.distributions as Dist
import numpyro
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

import seaborn as sns
import numpy as np
import jax

import numpyro

import jax.numpy as jnp

dist = dist()


class dpmm:
    def __init__(self,parent):
        """
        Dirichlet Process Mixture Model (DPMM) Class.
        
        Args:
            distribution_handler: Instance of UnifiedDist (or similar wrapper). 
                                  Defaults to UnifiedDist() if None.
            T (int): Truncation level (max number of clusters).
            method (str): 'marginal' (integrates out z, for NUTS) or 
                          'latent' (explicit z, requires MixedHMC/Gibbs).
        """
        self.__name__ = 'dpmm' 
        self.parent = parent

    def __call__(self, data, T=10, method='marginal', alpha=None):
        """
        Makes the class instance callable. 
        Redirects the call to self.model().
        """

        return self.model(data, T=T, method=method, alpha=alpha)
    
    @staticmethod   
    def mix_weights(beta):
        """
        Mixture weights (stick-breaking) for DPMM.
        The stick-breaking weights are used to sample the mixture component assignments.
        Args:
            beta: Stick-breaking weights (T-1 components)
        Returns:
            w: Mixture weights (T components)
        Reference:
            https://pyro.ai/examples/dirichlet_process_mixture.html
        """
        beta1m_cumprod = jnp.cumprod(1.0 - beta, axis=-1)
        padded_beta = jnp.pad(beta, (0, 1), constant_values=1.0)
        padded_cumprod = jnp.pad(beta1m_cumprod, (1, 0), constant_values=1.0)
        return padded_beta * padded_cumprod

    def dpmm_latent(self,data, T=10,  alpha = None):
        """
        Latent Variable formulation: Explicitly samples 'z'.
        Requires a sampler that supports discrete variables (e.g., MixedHMC or DiscreteHMCGibbs).
        """
        warn_dev("dpmm")
        N, D = data.shape  # Number of features
        data_mean = jnp.mean(data, axis=0)
        data_std = jnp.std(data, axis=0)*2

        # 1) stick-breaking weights

        if alpha is None:
            alpha = dist.gamma(1.0, 10.0,name='alpha')

        with numpyro.plate("beta_plate", T - 1):
            beta = numpyro.sample('beta', Dist.Beta(1, alpha))

        w = numpyro.deterministic("w",dpmm.mix_weights(beta))


        # 2) component parameters
        with numpyro.plate("components", T):
            mu = dist.multivariate_normal(loc=data_mean, covariance_matrix=data_std*jnp.eye(D),name='mu')# shape (T, D)     

            sigma = dist.log_normal(0.0, 1.0,shape=(D,),event=1,name='sigma')# shape (T, D)
            Lcorr = dist.lkj_cholesky(dimension=D, concentration=1.0,name='Lcorr')# shape (T, D, D)

            scale_tril = sigma[..., None] * Lcorr  # shape (T, D, D)

        # 3) Latent cluster assignments for each data point
        with numpyro.plate("data", N):
            # Sample the assignment for each data point
            z = numpyro.sample("z", Dist.Categorical(w)) # shape (N,)  

            numpyro.sample(
                "obs",
                Dist.MultivariateNormal(loc=mu[z], scale_tril=scale_tril[z]),
                obs=data
            )  

    def dpmm_marginal(self,data, T=10,  alpha = None):
        """
        Marginalized formulation: Integrates out 'z'.
        Standard formulation for NUTS/HMC samplers.
        """

        D = data.shape[1]
        # 1) stick-breaking weights
        if alpha is None:
            alpha = dist.gamma(1.0, 15.0,name='alpha')

        beta = dist.beta(1, alpha,name='beta',shape=(T-1,))
        w = numpyro.deterministic("w",dpmm.mix_weights(beta))

        # 2) component parameters
        data_mean = jnp.mean(data, axis=0)
        with numpyro.plate("components", T):
            mu = dist.multivariate_normal(loc=data_mean, covariance_matrix=5.0*jnp.eye(D),name='mu')# shape (T, D)        
            sigma = dist.half_cauchy(1,shape=(D,),event=1,name='sigma')# shape (T, D)
            Lcorr = dist.lkj_cholesky(dimension=D, concentration=1.0,name='Lcorr')# shape (T, D, D)

            scale_tril = sigma[..., None] * Lcorr  # shape (T, D, D)

        # 3) marginal mixture over obs
        dist.mixture_same_family(
            mixing_distribution=dist.categorical_probs(w,name='cat', create_obj=True),
            component_distribution=dist.multivariate_normal(loc=mu, scale_tril=scale_tril,name='mvn', create_obj=True),
            name="obs",  
            obs=data   
        )

    def model(self,data, T=10, method='marginal', alpha = None):
        """
        Wrapper function for DPMM.

        Args:
            data: Input data array (N, D)
            T: Truncation level (max number of clusters)
            method: 'marginal' (default because it is faster, for NUTS) or 'latent' (explicit z, requires MixedHMC/Gibbs)
            alpha: Stick-breaking parameter (default: None)
        """
        if method == 'marginal':
            return self.dpmm_marginal(data, T, alpha = alpha)
        elif method == 'latent':
            return self.dpmm_latent(data, T, alpha = alpha)
        else:
            raise ValueError("Method must be 'marginal' or 'latent'")

    def get_cluster_probs(self, data, w, mu, sigma, Lcorr):
        """
        Get the cluster probabilities for each data point and each sample.
        
        Args:
            data: Data array (N, D).
            w: Mixture weights (N_samples, N_components).
            mu: Cluster means (N_samples, N_components, D).
            sigma: Scale (variance) parameters (N_samples, N_components, D).
            Lcorr: Cholesky correlation matrices (N_samples, N_components, D, D).
        
        Returns:
            Cluster probabilities (N_samples, N_data, N_components).
        """
        # Construct the lower triangular Cholesky factor of the covariance matrix
        # Combines standard deviations (sigma) with the correlation structure (Lcorr)
        scale_tril = sigma[..., None] * Lcorr
        # Compute the Log-Likelihood of the data under a Multivariate Normal distribution
        # Shape becomes (N_data, N_components)
        log_liks = Dist.MultivariateNormal(mu, scale_tril=scale_tril).log_prob(data[:, None, :])
        # Calculate unnormalized log posterior probabilities: log(weight) + log(Likelihood)
        log_probs = jnp.log(w) + log_liks
        # Normalize probabilities (Softmax) using LogSumExp trick for numerical stability
        # Result is P(z=k | data, parameters)
        norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
        return norm_probs
    
    def proportion_of_data_assigned_to_cluster(self):
        """
        Plots the proportion of data assigned to clusters, sorted by size.
        
        Fixes the Label Switching issue in boxplots by visualizing 
        "1st Largest Cluster", "2nd Largest", etc., instead of "Index 0", "Index 1".
        """
        # 1. Access Data
        posteriors = self.parent.posteriors
        w = posteriors['w']
        mu = posteriors['mu']
        sigma = posteriors['sigma']
        Lcorr = posteriors['Lcorr']

        # 2. Flatten Chains if necessary (Robustness)
        # If shape is (Chains, Samples, ...), flatten to (Total_Samples, ...)
        if w.ndim == 3: 
            w = w.reshape(-1, *w.shape[2:])
            mu = mu.reshape(-1, *mu.shape[2:])
            sigma = sigma.reshape(-1, *sigma.shape[2:])
            Lcorr = Lcorr.reshape(-1, *Lcorr.shape[2:])
        
        # 3. Compute Proportions per Sample
        # Shape: (Total_Samples, N_data, N_components)
        cluster_probs = jax.vmap(self.get_cluster_probs, in_axes=(None, 0, 0, 0, 0))(
            self.parent.data_on_model['data'], 
            w, mu, sigma, Lcorr
        )
        
        # Average over data points to get global proportion per component
        # Shape: (Total_Samples, N_components)
        cluster_proportions = np.array(cluster_probs).mean(axis=1)

        # === SORT BY SIZE ===
        # We sort the proportions for each sample descending (largest to smallest).
        # This aligns the "big" clusters across chains, regardless of their ID.
        sorted_proportions = np.sort(cluster_proportions, axis=1)[:, ::-1]

        # 4. Create the Plot
        N_max_groups = sorted_proportions.shape[1]
        indices = np.arange(N_max_groups)

        plt.figure(figsize=(15, 6))

        # Plot the SORTED proportions
        plt.boxplot(sorted_proportions, positions=indices, patch_artist=True, 
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red', linewidth=1.5),
                    flierprops=dict(marker='o', markersize=2, alpha=0.5))

        # 5. Formatting
        plt.xlabel("Cluster Rank (Ordered by Size)", fontsize=12)
        plt.ylabel("Proportion of Data Assigned", fontsize=12)
        plt.title("Posterior Distribution of Cluster Sizes (Sorted by Rank)", fontsize=14)

        # Update x-ticks to reflect Ranks, not IDs
        rank_labels = [f"#{i+1}" for i in indices]
        plt.xticks(indices, rank_labels)
        
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1.05) 

        plt.show()
    
    def predict(self, data, sampler):
        """
        Performs Consensus Clustering. 
        Robust to Label Switching across multiple chains.
        """
        # 1. Flatten chains: (N_chains, N_samples) -> (Total_samples)
        posterior_samples = sampler.get_samples(group_by_chain=False)

        w_samps = posterior_samples['w']
        mu_samps = posterior_samples['mu']
        Lcorr_samps = posterior_samples['Lcorr']
        sigma_samps = posterior_samples['sigma']

        # Cap the draws fed to the vmapped consensus / density steps below:
        # they cost O(S * N_data^2) and O(S * N_grid * T) memory, which OOMs
        # (kills the kernel) at the full chain length. An evenly-spaced
        # subsample of a few hundred draws gives the same posterior-predictive.
        MAX_DRAWS = 300
        S = w_samps.shape[0]
        if S > MAX_DRAWS:
            idx = np.linspace(0, S - 1, MAX_DRAWS).astype(int)
            w_samps = w_samps[idx]
            mu_samps = mu_samps[idx]
            Lcorr_samps = Lcorr_samps[idx]
            sigma_samps = sigma_samps[idx]

        # 2. Calculate Soft Assignments for every sample
        # shape: (Total_Samples, N_data, N_components)
        cluster_probs = jax.vmap(self.get_cluster_probs, in_axes=(None, 0, 0, 0, 0))(
            data, w_samps, mu_samps, sigma_samps, Lcorr_samps
        )

        # 3. Consensus Matrix (Posterior Similarity)
        # Averaging the adjacency matrix handles label switching automatically.
        similarity_matrix = (cluster_probs @ cluster_probs.transpose(0, 2, 1)).mean(axis=0)

        # 4. Hierarchical Clustering
        distance_matrix = 1 - similarity_matrix
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        distance_matrix = distance_matrix.at[jnp.diag_indices_from(distance_matrix)].set(0.0)
        distance_matrix = jnp.clip(distance_matrix, min=0.0)
        
        condensed_dist = squareform(distance_matrix)
        Z = linkage(condensed_dist, 'average')
        
        # Cut tree
        final_labels = fcluster(Z, t=0.5, criterion='distance')
        print(f"Model found {len(np.unique(final_labels))} clusters.")

        # NOTE: We return the RAW samples, not the means. 
        # Averaging parameters (mu_mean) destroys information if chains swapped labels.
        return w_samps, mu_samps, sigma_samps, Lcorr_samps, final_labels

    def plot(self, data, sampler, figsize=(10, 8), point_size=10):
        """
        Plots the Posterior Predictive Density.
        Correctly handles multiple chains by averaging Densities, not Parameters.
        """
        # Get raw samples and labels
        w_samps, mu_samps, sigma_samps, Lcorr_samps, final_labels = self.predict(data, sampler)

        # 1. Create Grid
        x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
        y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
        xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, 100),
                              jnp.linspace(y_min, y_max, 100))
        grid_points = jnp.c_[xx.ravel(), yy.ravel()]

        # 2. Define function to compute PDF for ONE MCMC sample
        def get_pdf_for_one_sample(pts, w, mu, sigma, Lcorr):
            # Reconstruct Covariance
            scale_tril = sigma[..., None] * Lcorr
            
            # Compute log_prob for all points x all components: (N_grid, T)
            log_probs = Dist.MultivariateNormal(loc=mu, scale_tril=scale_tril).log_prob(pts[:, None, :])
            
            # Weighted sum (log-sum-exp trick for stability, though simple exp sum is fine here)
            # p(x) = sum_k w_k * N(x | mu_k, cov_k)
            weighted_probs = jnp.exp(log_probs) * w
            return jnp.sum(weighted_probs, axis=-1)

        if self.parent.num_chains > 1:
            print("Computing density across all chains (this might take a moment)...")

        
        # 3. Vectorize over all MCMC samples (batching to avoid memory issues)
        # We process the samples in batches if you have huge chains, but here we map directly.
        # vmap over axis 0 of w, mu, sigma, Lcorr
        all_pdfs = jax.vmap(get_pdf_for_one_sample, in_axes=(None, 0, 0, 0, 0))(
            grid_points, w_samps, mu_samps, sigma_samps, Lcorr_samps
        )

        # 4. Average the PDF values (Posterior Predictive)
        # This is label-switching invariant!
        avg_pdf = jnp.mean(all_pdfs, axis=0)
        Z = avg_pdf.reshape(xx.shape)

        # 5. Plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#f0f0f0') 
        ax.set_facecolor('#f0f0f0')

        # Colors
        unique_labels = np.unique(final_labels)
        n_clusters = len(unique_labels)
        palette = sns.color_palette("viridis", n_colors=n_clusters) 
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
        point_colors = [color_map[l] for l in final_labels]

        ax.scatter(data[:, 0], data[:, 1], c=point_colors, s=point_size, alpha=0.9, edgecolor='white', linewidth=0.3)

        contour_color = 'navy'
        contour = ax.contour(xx, yy, Z, levels=10, colors=contour_color, linewidths=0.8)
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

        ax.set_title("DPMM Posterior Predictive Density (Multi-Chain Robust)", fontsize=16)
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle=':', color='gray', alpha=0.6)

        fig.tight_layout()
        return fig, ax