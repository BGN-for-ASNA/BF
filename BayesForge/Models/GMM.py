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

def predict_gmm(data,sampler):

    
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

    return post_mean_w, post_mean_mu, post_mean_cov, final_labels

def plot_gmm(data,sampler, figsize = (10, 8)):
    warn_dev("gmm")
    post_mean_w, post_mean_mu, post_mean_cov, final_labels = predict_gmm(data,sampler)
    # 2. Set up a grid of points to evaluate the GMM density
    x_min, x_max = data[:, 0].min() - 2, data[:, 0].max() + 2
    y_min, y_max = data[:, 1].min() - 2, data[:, 1].max() + 2
    xx, yy = jnp.meshgrid(jnp.linspace(x_min, x_max, 150),
                         jnp.linspace(y_min, y_max, 150))
    grid_points = jnp.c_[xx.ravel(), yy.ravel()]

    # 3. Calculate the PDF of the GMM on the grid
    num_components = post_mean_mu.shape[0]
    gmm_pdf = jnp.zeros(grid_points.shape[0])

    for k in range(num_components):
        # Get parameters for the k-th component
        weight = post_mean_w[k]
        mean = post_mean_mu[k]
        cov = post_mean_cov[k]

        # Calculate the PDF of this component and add its weighted value to the total
        component_pdf = multivariate_normal(mean=mean, cov=cov).pdf(grid_points)
        gmm_pdf += weight * component_pdf

    # Reshape the PDF values to match the grid shape
    Z = gmm_pdf.reshape(xx.shape)

    # 4. Create the plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#f0f0f0') 
    ax.set_facecolor('#f0f0f0')

    # === FIX IS HERE ===
    # Dynamically create a color palette based on the number of clusters found
    unique_labels = jnp.unique(final_labels)
    n_clusters = len(unique_labels)
    # Using 'viridis' to match your first plot, but 'tab10' or 'Set2' are also good
    palette = sns.color_palette("viridis", n_colors=n_clusters) 

    # Create a mapping from each cluster label to its assigned color
    unique_labels = np.unique(final_labels)
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    # Create a list of colors for each data point corresponding to its cluster
    point_colors = [color_map[l] for l in final_labels]
    # === END OF FIX ===

    # Plot the data points using the dynamically generated colors
    ax.scatter(data[:, 0], data[:, 1], c=point_colors, s=15, alpha=0.9, edgecolor='white', linewidth=0.3)

    # Plot the density contours
    # Using a different colormap for the contours (e.g., 'Blues' or 'Reds') can look nice
    # to distinguish them from the points. Here we'll use a single color for simplicity.
    contour_color = 'navy'
    contour = ax.contour(xx, yy, Z, levels=10, colors=contour_color, linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')

    # Final styling touches
    ax.set_title("GMM Probability Density Contours", fontsize=16)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.grid(True, linestyle=':', color='gray', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    return fig, ax