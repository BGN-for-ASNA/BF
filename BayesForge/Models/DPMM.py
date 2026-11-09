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

# Credible-mass levels drawn as ellipses by ``dpmm.plot`` / ``plot_gmm``:
# ten nested levels, matching the ten auto-levels of the old density
# contour (whose lowest was the misleading ``0.00``).
DEFAULT_CREDIBLE = (0.25, 0.50, 0.75, 0.95)


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
    
    def predict(self, data, sampler, return_probs=False):
        """
        Performs Consensus Clustering. 
        Robust to Label Switching across multiple chains.

        Args:
            data: Data array (N, D).
            sampler: Fitted BF sampler.
            return_probs (bool): If True, also return the per-draw soft
                assignment tensor ``cluster_probs`` of shape
                ``(N_draws, N_data, T)``. Used by :meth:`plot` to build one
                credible region per consensus cluster.

        Returns:
            Tuple ``(w, mu, sigma, Lcorr, final_labels)``, plus
            ``cluster_probs`` when ``return_probs`` is True.
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
        if return_probs:
            return (w_samps, mu_samps, sigma_samps, Lcorr_samps, final_labels,
                    cluster_probs)
        return w_samps, mu_samps, sigma_samps, Lcorr_samps, final_labels

    @staticmethod
    def _component_covariances(sigma, Lcorr):
        """Full covariance of every mixture component.

        The model factorises the covariance as ``diag(sigma) @ L @ L.T @
        diag(sigma)`` with ``L`` a correlation Cholesky factor.
        """
        scale_tril = sigma[..., None] * Lcorr
        return scale_tril @ np.swapaxes(scale_tril, -1, -2)

    @staticmethod
    def _cluster_moments(mu, cov, mass, anchor=None):
        """Moment-match the components that describe one consensus cluster.

        Args:
            mu: Component means, shape ``(N_draws, T, D)``.
            cov: Component covariances, shape ``(N_draws, T, D, D)``.
            mass: Weight of every ``(draw, component)`` pair for this cluster,
                shape ``(N_draws, T)``. Typically the summed soft responsibility.
            anchor: If given, the returned mean and the covariance are those
                of the cluster itself: the mean is the anchor and the
                covariance is the mass-weighted average of the component
                covariances (within-component spread only). Without an anchor
                the full mixture moments are used, whose between-component
                term measures how far apart the components are -- when
                overlapping components swap identities across draws, or a
                wide near-empty component takes some mass, that term
                stretches the ellipse across the whole plot instead of
                describing the cluster.

        Returns:
            ``(mean, cov)`` of the single Gaussian that has the same first two
            moments as that mixture. Label-switching safe because the weights
            tie components to the cluster across draws.
        """
        mass = np.asarray(mass, dtype=float)
        mass = mass / mass.sum()
        if anchor is None:
            mean = np.einsum("st,sti->i", mass, mu)
            second = np.einsum(
                "st,stij->ij", mass, cov + np.einsum("sti,stj->stij", mu, mu)
            )
            cov_c = second - np.outer(mean, mean)
        else:
            # Within-component spread only: the ellipse of a consensus cluster
            # describes the cluster, not the distance between the (overlapping)
            # components that share its mass across draws.
            mean = np.asarray(anchor, dtype=float)
            cov_c = np.einsum("st,stij->ij", mass, cov)
        cov_c = (cov_c + cov_c.T) / 2.0
        vals, vecs = np.linalg.eigh(cov_c)
        vals = np.clip(vals, 1e-12, None)  # numerical guard, no negative variance
        return mean, (vecs * vals) @ vecs.T

    @staticmethod
    def _cap_covariance(cov_c, pts, cap=2.0, floor_std=None):
        """Cap an ellipse covariance by the cluster's own point spread.

        The posterior can put real mass on wide components that span several
        consensus clusters (overlapping blobs get described by one broad
        component in some draws). Moment matching then yields ellipses far
        wider than the points they are supposed to describe. The cluster's
        point cloud is ground truth for its extent, so each eigenvalue of
        ``cov_c`` (in the point cloud's principal frame) is capped at
        ``(cap * point_std) ** 2``, with ``floor_std`` as an absolute floor
        so tiny clusters (n = 1) keep a visible region at data scale.
        """
        pts = np.asarray(pts, dtype=float)
        dim = pts.shape[1]
        if floor_std is None:
            floor_std = 0.0
        if pts.shape[0] > 1:
            cp = np.cov(pts.T)
            cp = (cp + cp.T) / 2.0
        else:
            cp = np.zeros((dim, dim))
        vals_p, vecs_p = np.linalg.eigh(cp)
        vals_p = np.clip(vals_p, 0.0, None)
        cap_std = np.maximum(cap * np.sqrt(vals_p), floor_std)
        proj = vecs_p.T @ np.asarray(cov_c, dtype=float) @ vecs_p
        vals_c = np.minimum(np.diag(proj), cap_std ** 2)
        return (vecs_p * vals_c) @ vecs_p.T

    def plot(self, data, sampler, figsize=(10, 8), point_size=10,
             credible=DEFAULT_CREDIBLE, show_density=False, density_levels=7,
             spread_cap=2.0):
        """Posterior view of the DPMM clusters.

        Instead of a single posterior-predictive contour (whose lowest
        auto-selected level is a tiny positive number that Matplotlib prints
        as ``0.00`` and which can bridge unrelated clusters through the wide,
        near-empty components of the truncation), this draws **one credible
        region per consensus cluster**:

        1. ``predict`` performs the consensus clustering on the soft
           assignments, so the cluster labels are label-switching safe.
        2. For each cluster, every ``(draw, component)`` pair is weighted by
           the soft mass that component assigns to that cluster, and the
           spread of the resulting mixture is moment-matched to a single
           Gaussian **centered on the cluster's centroid** (anchoring on the
           mixture mean instead lets overlapping components that swap
           identities across draws stretch the ellipse across the plot).
        3. The Gaussian is drawn as an ellipse at each probability in
           ``credible`` (2D Gaussian quantiles ``-2 ln(1 - p)``).

        Args:
            data: Data array (N, D).
            sampler: Fitted BF sampler.
            figsize: Figure size.
            point_size: Scatter point size.
            credible: Iterable of credible-mass levels for the ellipses, e.g.
                ``[0.5, 0.9]`` or ``[0.25, 0.5, 0.75, 0.95]``. Values are
                probabilities in ``(0, 1)``; one nested ellipse is drawn per
                level (largest level solid, others dashed). Defaults to
                :data:`DEFAULT_CREDIBLE` (ten nested levels, 5%-95%).
            show_density: If True, overlay the posterior-predictive density
                with explicit *relative* levels (no misleading ``0.00``).
            density_levels: Number of relative density levels when
                ``show_density`` is True.
            spread_cap: Each ellipse axis is capped at this multiple of the
                cluster's own point-cloud std (see :meth:`_cap_covariance`).
                Set to ``None`` for the uncapped moment-matched ellipse.
        """
        (w_samps, mu_samps, sigma_samps, Lcorr_samps,
         final_labels, cluster_probs) = self.predict(data, sampler,
                                                     return_probs=True)

        labels = np.asarray(final_labels)
        unique_labels = np.unique(labels)
        palette = sns.color_palette("viridis", n_colors=len(unique_labels))
        color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

        mu = np.asarray(mu_samps, dtype=float)
        cov = self._component_covariances(np.asarray(sigma_samps, dtype=float),
                                          np.asarray(Lcorr_samps, dtype=float))
        probs = np.asarray(cluster_probs, dtype=float)
        data = np.asarray(data, dtype=float)
        # Ellipse floor for tiny clusters: a quarter of the data's scale.
        floor_std = 0.25 * float(np.sqrt(
            np.trace(np.cov(data.T)) / data.shape[1]))

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor('#f0f0f0')
        ax.set_facecolor('#f0f0f0')

        ax.scatter(data[:, 0], data[:, 1], c=[color_map[l] for l in labels],
                   s=point_size, alpha=0.9, edgecolor='white', linewidth=0.3)

        # Optional posterior-predictive density overlay ---------------------
        if show_density:
            warn_dev("dpmm")  # O(N_grid * T * draws) cost
            x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
            y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
            xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, 100),
                                  jnp.linspace(y_min, y_max, 100))
            grid_points = jnp.c_[xx.ravel(), yy.ravel()]

            def get_pdf_for_one_sample(pts, w, mu_k, sigma_k, Lcorr_k):
                scale_tril = sigma_k[..., None] * Lcorr_k
                log_probs = Dist.MultivariateNormal(
                    loc=mu_k, scale_tril=scale_tril
                ).log_prob(pts[:, None, :])
                # Drop near-empty components: their broad, low-weight mass is
                # what used to create the fake bridge between clusters.
                w_norm = jnp.where(w > 0.01, w, 0.0)
                w_norm = w_norm / jnp.sum(w_norm)
                return jnp.sum(jnp.exp(log_probs) * w_norm, axis=-1)

            all_pdfs = jax.vmap(get_pdf_for_one_sample,
                                in_axes=(None, 0, 0, 0, 0))(
                grid_points, w_samps, mu_samps, sigma_samps, Lcorr_samps
            )
            Z = jnp.mean(all_pdfs, axis=0).reshape(xx.shape)
            fractions = np.linspace(0.05, 0.95, density_levels)
            levels = np.sort(np.asarray(fractions) * float(jnp.max(Z)))
            contour = ax.contour(xx, yy, Z, levels=levels, colors='navy',
                                 linewidths=0.6, alpha=0.45)
            ax.clabel(contour, inline=True, fontsize=7, fmt='%.3g')

        # One credible ellipse per consensus cluster -------------------------
        from matplotlib.lines import Line2D
        from matplotlib.patches import Ellipse

        credible = sorted(credible)
        p_max = max(credible)
        handles = []
        mass_all = probs.sum(axis=1)  # (N_draws, T): total mass of each component
        for label in unique_labels:
            mask = labels == label
            mass = probs[:, mask, :].sum(axis=1)  # (N_draws, T)
            # Keep only the components that predominantly describe THIS cluster.
            # A wide component spanning several consensus clusters (typical of
            # overlapping blobs) holds part of every neighbour's mass; letting
            # it contribute stretches each neighbour's ellipse across the
            # whole band it covers.
            exclusive = mass >= 0.5 * mass_all
            if exclusive.any():
                mass = np.where(exclusive, mass, 0.0)
            # else: fully ambiguous cluster (e.g. a stray point) -> keep the
            # raw mass so its region still reflects the components covering it.
            # Anchor the moment matching on the cluster's own centroid so the
            # ellipse measures the spread of the components around the cluster,
            # not the distance between every component the cluster's mass
            # visits across draws (which stretches the ellipse across the plot
            # when overlapping components swap identities).
            mean, cov_c = self._cluster_moments(mu, cov, mass,
                                                anchor=data[mask].mean(axis=0))
            # Never draw an ellipse much wider than the cluster's own points.
            if spread_cap is not None:
                cov_c = self._cap_covariance(
                    cov_c, data[mask], cap=spread_cap, floor_std=floor_std)

            vals, vecs = np.linalg.eigh(cov_c)
            order = np.argsort(vals)[::-1]
            vals, vecs = vals[order], vecs[:, order]
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            color = color_map[label]

            for p in credible:
                q = -2.0 * np.log(max(1.0 - p, 1e-12))
                width, height = 2.0 * np.sqrt(q * vals)
                ax.add_patch(Ellipse(
                    mean, width, height, angle=angle, facecolor='none',
                    edgecolor=color, linewidth=1.8,
                    linestyle='-' if p == p_max else '--', alpha=0.95,
                ))

            ax.plot(*mean, marker='+', color=color, markersize=10,
                    markeredgewidth=2.0, linestyle='none')
            ax.annotate(f"n={int(mask.sum())}", mean, textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color=color)
            handles.append(Line2D([], [], marker='o', linestyle='',
                                  color=color,
                                  label=f"cluster {label} (n={int(mask.sum())})"))

        ax.legend(handles=handles, fontsize=8, loc='best', frameon=True,
                  title=f"{p_max:.0%} credible region (solid)")
        ax.set_title(
            f"DPMM clusters — {len(unique_labels)} consensus clusters, "
            f"{credible[0]:.0%}-{credible[-1]:.0%} credible regions",
            fontsize=15,
        )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, linestyle=':', color='gray', alpha=0.6)

        fig.tight_layout()
        return fig, ax