import jax
import jax.numpy as jnp
from jax import random, vmap
from BayesForge.Distributions.np_dists import UnifiedDist as dist
import numpyro.distributions as Dist
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
import numpy as np
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
      anchors ``mu`` near plausible values; it does NOT remove label switching
      in MCMC, which is handled after the fit by the posterior similarity
      matrix (co-clustering) in :func:`predict_gmm`.
    Returns:
    - A model that defines the GMM with K clusters.
    This model assumes that the data is generated from a mixture of K Gaussian distributions.
    The model estimates the means, covariances, and mixture weights for each cluster.
    The number of clusters K is fixed and must be specified in advance.
    The priors scale with the per-feature standard deviation of the data, so
    the model is not tied to standardised inputs.
    """
    if jnp.ndim(data) != 2:
        raise ValueError("gmm: data must be 2-D (N, D)")
    N = data.shape[0]
    if N < 1:
        raise ValueError("gmm: data must contain at least one sample (N >= 1)")
    if isinstance(K, bool) or not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("gmm: K must be an integer >= 1")
    if K > N:
        raise ValueError("gmm: K must not exceed the number of samples (K <= N)")
    D = data.shape[1]  # Number of features

    if initial_means is None:
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(np.asarray(data))
        initial_means = jnp.asarray(km.cluster_centers_)
    # Scale-aware priors: they scale with the per-feature standard deviation
    # of the data (identical to the old ones when the data is standardised).
    data_std = jnp.std(jnp.asarray(data), axis=0)
    data_std = jnp.where(data_std > 0, data_std, 1.0)

    alpha_prior = 0.5 * jnp.ones(K)
    w = dist.dirichlet(concentration=alpha_prior, name='weights') 

    with dist.plate("components", K): # Use fixed K
        mu = dist.multivariate_normal(loc=initial_means, covariance_matrix=jnp.diag((0.1 ** 0.5 * data_std) ** 2), name='mu')        
        sigma = dist.half_cauchy(data_std, event=1, name='sigma')
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

def _relabel_draws(mu_samps, *others):
    """Align the component labels of every draw to a pivot draw.

    Label switching permutes components between draws, so a plain mean of
    ``mu`` across draws mixes clusters. Each draw is permuted with the
    assignment (Hungarian algorithm on squared distances between means) that
    best matches the pivot draw (the draw whose means are closest to the
    elementwise median), then all arrays are permuted the same way.
    """
    from scipy.optimize import linear_sum_assignment
    mu = np.asarray(mu_samps)
    med = np.median(mu, axis=0)
    pivot = mu[np.argmin(((mu - med) ** 2).sum(axis=(1, 2)))]
    perms = np.empty(mu.shape[:2], dtype=int)
    for s in range(mu.shape[0]):
        cost = ((mu[s][:, None, :] - pivot[None, :, :]) ** 2).sum(-1)
        rows, cols = linear_sum_assignment(cost)
        perm = np.empty_like(cols)
        perm[cols] = rows
        perms[s] = perm
    take = lambda a: np.take_along_axis(
        np.asarray(a), perms.reshape(perms.shape + (1,) * (np.asarray(a).ndim - 2)), axis=1)
    return (take(mu),) + tuple(take(a) for a in others)

_MAX_PSM_DRAWS = 300


def _psm_draw_index(S):
    """Evenly spaced draw indices used for the soft assignments, or None.

    cluster_probs costs O(S * N_data * K) memory and the posterior similarity
    matrix O(S * N_data^2), which runs out of memory at full chain length; a
    few hundred evenly spaced draws give the same matrix (as in DPMM).
    """
    if S <= _MAX_PSM_DRAWS:
        return None
    return np.linspace(0, S - 1, _MAX_PSM_DRAWS).astype(int)


def predict_gmm(data, sampler, return_probs=False, distance_threshold=0.5):

    
    """ Predicts the GMM density contours based on posterior samples and final labels.

    This function processes the posterior samples from a Bayesian Gaussian Mixture
    Model (GMM) to derive a single, representative clustering and the
    corresponding GMM parameters. The approach is based on two main steps:
    calculating posterior point estimates and performing co-clustering analysis
    via a Posterior Similarity Matrix (PSM).

    Methodology:
    1.  **Posterior Point Estimates**: Every draw is first relabelled to a
        pivot draw (:func:`_relabel_draws`) to undo label switching, then the
        posterior mean of the weights and means is computed over the aligned
        samples (posterior mean after relabelling each draw to a pivot draw);
        the covariance is the mean of the per-draw covariances, because the
        mean of Cholesky factors is not the Cholesky of the mean covariance.
        This provides a single set of parameters that represents the central
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
    - distance_threshold (float): Cophenetic distance in ``[0, 1]`` used to
      cut the hierarchical tree (criterion ``'distance'``). A smaller value
      yields more fine-grained clusters, a larger one merges nearby
      clusters. Defaults to 0.5.

    Returns:
    - tuple: A tuple containing four elements:
        - post_mean_w (jnp.ndarray): The posterior mean of the mixture weights.
        - post_mean_mu (jnp.ndarray): The posterior mean of the cluster means.
        - post_mean_cov (jnp.ndarray): The reconstructed posterior mean of the
          cluster covariance matrices.
        - final_labels (jnp.ndarray): The final, single-partition cluster
          labels for each data point.
        Plus ``(w_samps, mu_samps, sigma_samps, Lcorr_samps, cluster_probs)``
        when ``return_probs`` is True (raw, unaligned draws + per-draw soft
        assignments, used by :func:`plot_gmm` for the credible ellipses).
        ``cluster_probs`` is computed on the subsample of at most 300 draws
        used for the PSM, so its leading dimension is ``min(S, 300)``.

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

    # Undo label switching before averaging: a plain mean of mu mixes the
    # clusters of permuted draws, and the mean of Cholesky factors is not the
    # Cholesky of the mean covariance, so the covariance is averaged per draw.
    mu_al, w_al, sigma_al, Lcorr_al = _relabel_draws(
        mu_samps, w_samps, sigma_samps, Lcorr_samps)

    post_mean_w = jnp.mean(w_al, axis=0)
    post_mean_mu = jnp.mean(mu_al, axis=0)
    scale_tril_al = sigma_al[..., None] * Lcorr_al
    post_mean_cov = jnp.mean(
        scale_tril_al @ jnp.swapaxes(scale_tril_al, -1, -2), axis=0)

    # ... (The entire co-clustering block to get final_labels) ...
    def get_cluster_probs(data, w, mu, sigma, Lcorr):
        scale_tril = sigma[..., None] * Lcorr
        log_liks = Dist.MultivariateNormal(mu, scale_tril=scale_tril).log_prob(data[:, None, :])
        log_probs = jnp.log(jnp.clip(w, jnp.finfo(jnp.result_type(w, jnp.float32)).tiny)) + log_liks
        norm_probs = jnp.exp(log_probs - jax.scipy.special.logsumexp(log_probs, axis=-1, keepdims=True))
        return norm_probs

    # Cap the draws fed to the vmapped soft-assignment step: cluster_probs
    # costs O(S * N_data * K) memory and the PSM product O(S * N_data^2),
    # which OOMs at full chain length. An evenly-spaced subsample of a few
    # hundred draws gives the same posterior similarity matrix (as in DPMM).
    S = w_samps.shape[0]
    idx = _psm_draw_index(S)
    if idx is not None:
        w_psm, mu_psm = w_samps[idx], mu_samps[idx]
        sigma_psm, Lcorr_psm = sigma_samps[idx], Lcorr_samps[idx]
    else:
        w_psm, mu_psm, sigma_psm, Lcorr_psm = (
            w_samps, mu_samps, sigma_samps, Lcorr_samps)

    cluster_probs = jax.vmap(get_cluster_probs, in_axes=(None, 0, 0, 0, 0))(
        data, w_psm, mu_psm, sigma_psm, Lcorr_psm
    )
    similarity_matrix = (cluster_probs @ cluster_probs.transpose(0, 2, 1)).mean(axis=0)
    similarity_matrix_np = similarity_matrix
    distance_matrix = 1 - similarity_matrix_np
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix = distance_matrix.at[jnp.diag_indices_from(distance_matrix)].set(0.0)  # Set diagonal to 0
    distance_matrix = jnp.clip(distance_matrix, min=0.0)
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, 'average')
    final_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    if return_probs:
        return (post_mean_w, post_mean_mu, post_mean_cov, final_labels,
                w_samps, mu_samps, sigma_samps, Lcorr_samps, cluster_probs)
    return post_mean_w, post_mean_mu, post_mean_cov, final_labels

def plot_gmm(data, sampler, figsize=(10, 8), point_size=15,
             credible=DEFAULT_CREDIBLE, show_density=False,
             density_levels=7, spread_cap=2.0, distance_threshold=0.5):
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
            :data:`DEFAULT_CREDIBLE` (four nested levels, 25% to 95%).
        show_density: If True, overlay the posterior-predictive density
            with explicit *relative* levels (no misleading ``0.00``).
        density_levels: Number of relative density levels when
            ``show_density`` is True.
        spread_cap: Ellipse axis cap as a multiple of the cluster's point
            std (see :meth:`dpmm._cap_covariance`). None to disable.
        distance_threshold: Cophenetic distance in ``[0, 1]`` used to cut
            the consensus-clustering tree in :func:`predict_gmm`. Defaults
            to 0.5.
    """
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("plot_gmm: plot requires 2-D data with D=2 features "
                         f"(got shape {data.shape}).")
    if data.shape[0] < 2:
        raise ValueError("plot_gmm: plot requires at least 2 data points "
                         f"(got {data.shape[0]}).")
    warn_dev("gmm")
    (post_mean_w, post_mean_mu, post_mean_cov, final_labels,
     w_samps, mu_samps, sigma_samps, Lcorr_samps,
     cluster_probs) = predict_gmm(data, sampler, return_probs=True,
                                  distance_threshold=distance_threshold)

    labels = np.asarray(final_labels)
    unique_labels = np.unique(labels)
    palette = sns.color_palette("viridis", n_colors=len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}

    # cluster_probs is computed on the draws picked by _psm_draw_index; use
    # the same draws of mu/cov so every (draw, component) mass matches its
    # component.
    idx = _psm_draw_index(np.asarray(mu_samps).shape[0])
    if idx is not None:
        mu_samps, sigma_samps, Lcorr_samps = (
            mu_samps[idx], sigma_samps[idx], Lcorr_samps[idx])
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