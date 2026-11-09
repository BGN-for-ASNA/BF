import jax
import jax.numpy as jnp
from jax import random, vmap
from BayesForge.Distributions.np_dists import UnifiedDist as dist
import numpyro.distributions as Dist
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
import seaborn as sns # Ensure seaborn is imported for palettes
import matplotlib.pyplot as plt
from BayesForge.Utils.dev_warning import warn_dev
from BayesForge.Models.DPMM import dpmm as _dpmm, DEFAULT_CREDIBLE

dist = dist()
def gmm(data, K, initial_means=None):
    """
    Gaussian Mixture Model with a fixed number of clusters K.
    Parameters:
    - data: Input data points (shape: [N, D] where N is the number of samples and D is the number of features).
    - K: The exact number of clusters.
    - initial_means: Prior means for the clusters (shape: [K, D]). If None
      (default) they are initialised with a K-means fit on ``data``; this
      anchors ``mu`` near plausible values and breaks the mixture's label
      symmetry.
    Returns:
    - A model that defines the GMM with K clusters.
    This model assumes that the data is generated from a mixture of K Gaussian distributions.
    The model estimates the means, covariances, and mixture weights for each cluster.
    The number of clusters K is fixed and must be specified in advance.
    """
    D = data.shape[1]  # Number of features

    if initial_means is None:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(np.asarray(data))
        initial_means = jnp.asarray(km.cluster_centers_)
    alpha_prior = 0.5 * jnp.ones(K)
    w = dist.dirichlet(concentration=alpha_prior, name='weights') 

    with dist.plate("components", K): # Use fixed K
        mu = dist.multivariate_normal(loc=initial_means, covariance_matrix=0.1*jnp.eye(D), name='mu')        
        sigma = dist.half_cauchy(1, shape=(D,), event=1, name='sigma')
        Lcorr = dist.lkj_cholesky(dimension=D, concentration=1.0, name='Lcorr')

        scale_tril = sigma[..., None] * Lcorr

    ## 3) marginal mixture over obs (this part remains almost identical)
    #with numpyro.plate('data', len(data)):
    #    assignment = numpyro.sample('assignment', dist.Categorical(w),infer={"enumerate": "parallel"}) 
    #    numpyro.sample('obs', dist.MultivariateNormal(mu[assignment,:][1], sigma[assignment][1]*jnp.eye(D)), obs=data)
    #    
    dist.mixture_same_family(
        mixing_distribution=dist.categorical(probs=w, create_obj=True),
        component_distribution=dist.multivariate_normal(loc=mu, scale_tril=scale_tril, create_obj=True),
        name="obs",
        obs=data
    )

def predict_gmm(data, sampler, return_probs=False):

    
    """ Predicts the GMM density contours based on posterior samples and final labels.

    This function processes the posterior samples from a Bayesian Gaussian Mixture
    Model (GMM) to derive a single, representative clustering and the
    corresponding GMM parameters. The approach is based on two main steps:
    calculating posterior point estimates and performing co-clustering analysis
    via a Posterior Similarity Matrix (PSM).

    Methodology:
    1.  **Posterior Point Estimates**: The function first computes the posterior
        mean of the GMM parameters (weights, means, and covariance components)
        by averaging over all samples obtained from the MCMC sampler. This
        provides a single set of parameters that represents the central
        tendency of the posterior distribution.

    2.  **Co-clustering via Posterior Similarity Matrix (PSM)**: To obtain a
        single, robust clustering, this function uses a co-clustering approach
        that leverages the entire posterior distribution.
        a. A similarity matrix is constructed where the entry (i, j) represents
           the posterior probability that data point `i` and data point `j`
           belong to the same cluster. This is calculated by averaging their
           co-assignment probabilities across all posterior samples.
        b. This similarity matrix is converted into a distance matrix (1 - similarity).
        c. Agglomerative hierarchical clustering ('average' linkage) is then
           applied to this distance matrix.
        d. The resulting hierarchy is cut at a specified distance threshold to
           produce a final, single set of cluster labels for the data.

    This method avoids issues like label switching inherent in MCMC and produces
    a stable clustering that summarizes the relationships discovered in the
    full Bayesian posterior.

    Parameters:
    - data (jnp.ndarray): The input data points used for the model, with
      shape [N, D] where N is the number of samples and D is the number
      of features.
    - sampler (BF.sampler): The fitted MCMC sampler object from which
      posterior samples can be extracted.

    Returns:
    - tuple: A tuple containing four elements:
        - post_mean_w (jnp.ndarray): The posterior mean of the mixture weights.
        - post_mean_mu (jnp.ndarray): The posterior mean of the cluster means.
        - post_mean_cov (jnp.ndarray): The reconstructed posterior mean of the
          cluster covariance matrices.
        - final_labels (jnp.ndarray): The final, single-partition cluster
          labels for each data point.
        Plus ``(w_samps, mu_samps, sigma_samps, Lcorr_samps, cluster_probs)``
        when ``return_probs`` is True (raw draws + per-draw soft assignments,
        used by :func:`plot_gmm` for the credible ellipses).

    Reference:
    * https://www.pure.ed.ac.uk/ws/portalfiles/portal/80251963/1508378464.pdf
    * https://arxiv.org/abs/2108.11753
    * https://en.wikipedia.org/wiki/Mixture_model
    * https://projecteuclid.org/journals/bayesian-analysis/volume-4/issue-2/Improved-criteria-for-clustering-based-on-the-posterior-similarity-matrix/10.1214/09-BA414.pdf
    * https://www.pure.ed.ac.uk/ws/portalfiles/portal/80251963/1508378464.pdf
    * https://pmc.ncbi.nlm.nih.gov/articles/PMC10441802/


    """
    warn_dev("gmm")
    # 1. Calculate posterior mean of all model parameters
    posterior_samples = sampler.get_samples()
    w_samps = posterior_samples['weights']
    mu_samps = posterior_samples['mu']
    Lcorr_samps = posterior_samples['Lcorr']
    sigma_samps = posterior_samples['sigma']

    post_mean_w = jnp.mean(w_samps, axis=0)
    post_mean_mu =jnp.mean(mu_samps, axis=0)
    post_mean_sigma = jnp.mean(sigma_samps, axis=0)
    post_mean_Lcorr = jnp.mean(Lcorr_samps, axis=0)

    # Reconstruct the full covariance matrices
    post_mean_scale_tril = post_mean_sigma[..., None] * post_mean_Lcorr
    post_mean_cov = post_mean_scale_tril @ jnp.transpose(post_mean_scale_tril, (0, 2, 1))

    # ... (The entire co-clustering block to get final_labels) ...
    def get_cluster_probs(data, w, mu, sigma, Lcorr):
        scale_tril = sigma[..., None] * Lcorr
        log_liks = Dist.MultivariateNormal(mu, scale_tril=scale_tril).log_prob(data[:, None, :])
        log_probs = jnp.log(w) + log_liks
        norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
        return norm_probs

    cluster_probs = jax.vmap(get_cluster_probs, in_axes=(None, 0, 0, 0, 0))(
        data, w_samps, mu_samps, sigma_samps, Lcorr_samps
    )
    similarity_matrix = (cluster_probs @ cluster_probs.transpose(0, 2, 1)).mean(axis=0)
    similarity_matrix_np = similarity_matrix
    distance_matrix = 1 - similarity_matrix_np
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix = distance_matrix.at[jnp.diag_indices_from(distance_matrix)].set(0.0)  # Set diagonal to 0
    distance_matrix = jnp.clip(distance_matrix, min=0.0)
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, 'average')
    distance_threshold = 0.5 
    final_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    if return_probs:
        return (post_mean_w, post_mean_mu, post_mean_cov, final_labels,
                w_samps, mu_samps, sigma_samps, Lcorr_samps, cluster_probs)
    return post_mean_w, post_mean_mu, post_mean_cov, final_labels

def plot_gmm(data, sampler, figsize=(10, 8), point_size=15,
             credible=DEFAULT_CREDIBLE, show_density=False,
             density_levels=7, spread_cap=2.0):
    """Posterior view of the GMM clusters.

    Same visualization as :meth:`dpmm.plot`: one credible ellipse per
    consensus cluster (labels made switching-safe by the posterior
    similarity matrix), each axis capped at ``spread_cap`` times the
    cluster's own point spread. Replaces the old density contour whose
    lowest auto-selected level printed as a misleading ``0.00``.

    Args:
        data: Data array (N, D).
        sampler: Fitted BF sampler.
        figsize: Figure size.
        point_size: Scatter point size.
        credible: Iterable of credible-mass levels for the ellipses, e.g.
            ``[0.5, 0.9]`` or ``[0.25, 0.5, 0.75, 0.95]``. One nested ellipse
            per level (largest solid, others dashed). Defaults to
            :data:`DEFAULT_CREDIBLE` (ten nested levels, 5%-95%).
        show_density: If True, overlay the posterior-predictive density
            with explicit *relative* levels (no misleading ``0.00``).
        density_levels: Number of relative density levels when
            ``show_density`` is True.
        spread_cap: Ellipse axis cap as a multiple of the cluster's point
            std (see :meth:`dpmm._cap_covariance`). None to disable.
    """
    warn_dev("gmm")
    (post_mean_w, post_mean_mu, post_mean_cov, final_labels,
     w_samps, mu_samps, sigma_samps, Lcorr_samps,
     cluster_probs) = predict_gmm(data, sampler, return_probs=True)

    labels = np.asarray(final_labels)
    unique_labels = np.unique(labels)
    palette = sns.color_palette("viridis", n_colors=len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    mu = np.asarray(mu_samps, dtype=float)
    cov = _dpmm._component_covariances(np.asarray(sigma_samps, dtype=float),
                                       np.asarray(Lcorr_samps, dtype=float))
    probs = np.asarray(cluster_probs, dtype=float)
    data = np.asarray(data, dtype=float)
    floor_std = 0.25 * float(np.sqrt(
        np.trace(np.cov(data.T)) / data.shape[1]))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#f0f0f0')
    ax.set_facecolor('#f0f0f0')

    ax.scatter(data[:, 0], data[:, 1], c=[color_map[l] for l in labels],
               s=point_size, alpha=0.9, edgecolor='white', linewidth=0.3)

    # Optional posterior-predictive density overlay --------------------------
    if show_density:
        x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
        y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
        xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, 100),
                              jnp.linspace(y_min, y_max, 100))
        grid_points = jnp.c_[xx.ravel(), yy.ravel()]

        # Subsample draws like dpmm.predict: the vmap costs O(grid * K * draws).
        w_np = np.asarray(w_samps)
        if w_np.shape[0] > 300:
            idx = np.linspace(0, w_np.shape[0] - 1, 300).astype(int)
            w_samps, mu_d = w_np[idx], mu[idx]
            sigma_d = np.asarray(sigma_samps, dtype=float)[idx]
            Lcorr_d = np.asarray(Lcorr_samps, dtype=float)[idx]
        else:
            w_samps, mu_d = w_np, mu
            sigma_d = np.asarray(sigma_samps, dtype=float)
            Lcorr_d = np.asarray(Lcorr_samps, dtype=float)

        def get_pdf_for_one_sample(pts, w, mu_k, sigma_k, Lcorr_k):
            scale_tril = sigma_k[..., None] * Lcorr_k
            log_probs = Dist.MultivariateNormal(
                loc=mu_k, scale_tril=scale_tril
            ).log_prob(pts[:, None, :])
            # Drop near-empty components: their broad, low-weight mass is
            # what creates the fake bridge between clusters.
            w_norm = jnp.where(w > 0.01, w, 0.0)
            w_norm = w_norm / jnp.sum(w_norm)
            return jnp.sum(jnp.exp(log_probs) * w_norm, axis=-1)

        all_pdfs = jax.vmap(get_pdf_for_one_sample,
                            in_axes=(None, 0, 0, 0, 0))(
            grid_points, w_samps, mu_d, sigma_d, Lcorr_d
        )
        Z = jnp.mean(all_pdfs, axis=0).reshape(xx.shape)
        fractions = np.linspace(0.05, 0.95, density_levels)
        levels = np.sort(np.asarray(fractions) * float(jnp.max(Z)))
        contour = ax.contour(xx, yy, Z, levels=levels, colors='navy',
                             linewidths=0.6, alpha=0.45)
        ax.clabel(contour, inline=True, fontsize=7, fmt='%.3g')

    # One credible ellipse per consensus cluster ------------------------------
    from matplotlib.lines import Line2D
    from matplotlib.patches import Ellipse

    credible = sorted(credible)
    p_max = max(credible)
    handles = []
    mass_all = probs.sum(axis=1)  # (N_draws, K)
    for label in unique_labels:
        mask = labels == label
        mass = probs[:, mask, :].sum(axis=1)  # (N_draws, K)
        # Keep only components that predominantly describe THIS cluster
        # (see dpmm.plot for the rationale).
        exclusive = mass >= 0.5 * mass_all
        if exclusive.any():
            mass = np.where(exclusive, mass, 0.0)
        mean, cov_c = _dpmm._cluster_moments(mu, cov, mass,
                                             anchor=data[mask].mean(axis=0))
        if spread_cap is not None:
            cov_c = _dpmm._cap_covariance(cov_c, data[mask], cap=spread_cap,
                                          floor_std=floor_std)

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
        f"GMM clusters — {len(unique_labels)} consensus clusters, "
        f"{credible[0]:.0%}-{credible[-1]:.0%} credible regions",
        fontsize=16,
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle=':', color='gray', alpha=0.6)

    fig.tight_layout()
    return fig, ax