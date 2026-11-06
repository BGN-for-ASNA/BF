import jax as jax
import jax.numpy as jnp
from jax import jit
from jax import vmap
import numpyro
from BayesForge.Distributions.np_dists import UnifiedDist as dist
from BayesForge.Utils.dev_warning import warn_dev

dist = dist()


# Random factors related functions --------------------------------------------
@jit
def jax_LinearOperatorDiag(s, cov):
    def multiply_with_s(a):
        return jnp.multiply(a, s)

    vectorized_multiply = vmap(multiply_with_s)
    return jnp.transpose(vectorized_multiply(cov))


def _parent_ids(child_obs_ids, parent_obs_ids, n_child):
    """Derive parent index for each child unit via JAX scatter.

    For strictly nested designs: every observation in child group g shares
    the same parent, so scattering parent_obs_ids keyed by child_obs_ids
    gives the correct parent for each child unit regardless of write order.

    Fully JAX-traceable — no Python-level loops over traced values.
    """
    parent_obs_ids = jnp.asarray(parent_obs_ids)
    return jnp.zeros(n_child, dtype=parent_obs_ids.dtype).at[child_obs_ids].set(parent_obs_ids)


class effects:
    """Class for handling specific factor operations in JAX, including diagonal matrix multiplication and random factor centered generation."""

    def __init__(self) -> None:
        pass

    @staticmethod
    @jit
    def diag_pre_multiply(v, m):
        """Dot product of a diagonal matrix with a matrix.

        Args:
            v (1D jax array): Vector of diagonal elements.
            m (2D jax array): Matrix to be multiplied.

        Returns:
            2D jax array: Dot product of a diagonal matrix with a matrix.
        """
        return jnp.matmul(jnp.diag(v), m)

    @staticmethod
    @jit
    def random_centered(sigma, cor_mat, offset_mat):
        """Generate the centered matrix of random factors

        Args:
            sigma (vector): Prior, vector of length N
            cor_mat (2D array): correlation matrix, cholesky_factor_corr of dim N, N
            offset_mat (2D array): matrix of offsets, matrix of dim N*k

        Returns:
            2D array: Random factors, shape (k, N).
        """
        # return jnp.dot(factors.diag_pre_multiply(sigma, cor_mat), offset_mat).T
        return (effects.diag_pre_multiply(sigma, cor_mat) @ offset_mat).T

    @staticmethod
    def phylo_effect(L, sd, name="z_phylo", sample=False):
        """Phylogenetic random effect, non-centred: ``u = sd * L z`` with ``z ~ Normal(0, 1)``.

        ``L`` is the Cholesky factor of the phylogenetic covariance matrix, so
        ``u ~ MultivariateNormal(0, sd**2 * L L^T)``. Passing a stack of trees
        (shape (T, N, N)) gives one independent effect per tree, which fits the
        model across a posterior sample of phylogenies in a single vectorised run.

        Args:
            L (jax.Array): Cholesky factor(s), shape (N, N) or (T, N, N).
            sd (float or jax.Array): Phylogenetic standard deviation; shape (T,)
                for one value per tree.
            name (str): Name of the standard-normal site ``z``. Defaults to "z_phylo".
            sample (bool): Draw values directly instead of defining a model site.

        Returns:
            jax.Array: Phylogenetic effects, shape (N,) or (T, N).
        """
        z = dist.normal(0, 1, name=name, shape=L.shape[:-1], sample=sample)
        u = jnp.einsum("...ij,...j->...i", L, z)
        return u * jnp.expand_dims(jnp.asarray(sd), -1)

    @staticmethod
    def varying_effects(
        N_vars,
        N_group,
        group_id,
        group_name="age",
        alpha_bar=None,
        beta_bar=None,
        sd_intercept=None,
        sd_beta=None,
        corr=None,
        centered=False,
        sample=False,
    ):
        """
        Models a varying intercept and varying slope for a continuous predictor.

        This uses a "centered" parameterization to sample a correlated
        [intercept, slope] pair for each group from a Multivariate Normal distribution.

        Args:
            N_vars (int): The total number of covariates for the model.
            N_group (int): The total number of unique groups (e.g., 7 subjects).
            group_id (array): An array of integer indices for the group of each observation.
            group_name (str, optional): A descriptive name for the grouping factor.
            alpha_bar (array or None): Mean for the intercept dimension. Pass jnp.zeros(1)
                to produce zero-centered random effects when fixed effects are specified
                externally. Defaults to None (samples global_intercept ~ N(0,1)).
            beta_bar (array or None): Mean for the slope dimensions. Pass jnp.zeros(N_vars)
                to produce zero-centered random effects. Defaults to None (samples
                global_beta ~ N(0,1)).
            sd_intercept (distribution, optional): The hyperprior for the standard deviation of the intercepts.
            sd_beta (distribution, optional): The hyperprior for the standard deviation of the slopes.
            corr (distribution, optional): The prior for the correlation matrix.
            centered (bool, optional): Whether to use a centered parameterization. Defaults to False.
            sample (bool, optional): Whether to sample from the posterior. Defaults to False.

        Returns:
            tuple: A tuple containing two arrays:
                - varying_intercepts (array): The intercept for each observation.
                - varying_slopes (array): The slope for each observation.
        """
        # 1. Priors.
        if alpha_bar is None:
            alpha_bar = dist.normal(
                0, 2, name="global_intercept", sample=sample, shape=(1,)
            )

        if beta_bar is None:
            beta_bar = dist.normal(
                0, 1, name="global_beta", sample=sample, shape=(N_vars,)
            )

        # 2. Hyperpriors.
        if sd_intercept is None:
            sd_intercept = dist.exponential(
                1, shape=(1,), name=f"{group_name}_sd_intercept", sample=sample
            )

        if sd_beta is None:
            sd_beta = dist.exponential(
                1, shape=(N_vars,), name=f"{group_name}_sd_beta", sample=sample
            )

        mu = jnp.concat([alpha_bar, beta_bar])
        sigma = jnp.concat([sd_intercept, sd_beta])

        if centered == False:
            if corr is None:
                L_corr = dist.lkj_cholesky(
                    dimension=(N_vars + 1),
                    concentration=2,
                    name=f"{group_name}_L_corr",
                    sample=sample,
                )
            else:
                L_corr = corr  # use caller-supplied Cholesky factor directly

            z = dist.normal(
                0, 1, name=f"{group_name}_z", shape=(N_vars + 1, N_group), sample=sample
            )

            effects = (L_corr @ z).T * sigma + mu

            params_for_obs = effects[group_id]
            varying_intercepts = params_for_obs[:, 0]
            varying_slopes = params_for_obs[:, 1:]
            if N_vars == 1:
                varying_slopes = varying_slopes[:, 0]
            return varying_intercepts, varying_slopes

        else:
            if corr is None:
                corr = dist.lkj(
                    dimension=(N_vars + 1),
                    concentration=2,
                    name=f"corr_{group_name}",
                    sample=sample,
                )
            cov = jnp.diag(sigma) @ corr @ jnp.diag(sigma)
            # cov = jnp.outer(sigma, sigma) * corr
            group_params = dist.multivariate_normal(
                mu, cov, shape=(N_group,), name=f"{group_name}_mvn", sample=sample
            )

            tmp = group_params[group_id]
            varying_slopes = tmp[:, 1:]
            if N_vars == 1:
                varying_slopes = varying_slopes[:, 0]
            return tmp[:, 0], varying_slopes

    @staticmethod
    def varying_intercept_slope(
        N_group,
        group,
        global_intercept,
        global_slope,
        group_name="group",
        group_std=None,
        L_corr=None,
        sample=False,
    ):
        """
        Models a varying intercept and varying slope for a continuous predictor.

        This uses a "centered" parameterization to sample a correlated
        [intercept, slope] pair for each group from a Multivariate Normal distribution.

        Args:
            N_groups (int): The total number of unique groups (e.g., 20 schools).
            group_idx (array): An array of integer indices for the group of each observation.
            global_intercept_mean (float or distribution): The hyperprior for the mean of the intercepts.
            global_slope_mean (float or distribution): The hyperprior for the mean of the slopes.
            group_name (str, optional): A descriptive name for the grouping factor.
            group_std (distribution, optional): Prior for the standard deviations of the
                intercept and slope. Must have shape (2,). Defaults to Exponential(1).
            L_corr (distribution, optional): Prior for the Cholesky factor of the 2x2
                correlation matrix between the intercept and slope. Defaults to LKJ(2, 2).

        Returns:
            tuple: A tuple containing two arrays:
                - varying_intercepts (array): The intercept for each observation.
                - varying_slopes (array): The slope for each observation.
        """
        warn_dev("effects")
        # 1. Handle default hyperpriors.
        if group_std is None:
            group_std = dist.exponential(
                1, shape=(2,), name=f"{group_name}_std", sample=sample
            )

        if L_corr is None:
            L_corr = dist.lkj_cholesky(
                2, 2, name=f"L_corr_{group_name}", sample=sample
            )  # N_vars*N_vars

        cov = jnp.outer(group_std, group_std) * L_corr
        mu = jnp.stack([global_intercept, global_slope])

        group_params = dist.multivariate_normal(
            mu, cov, shape=(N_group,), name=f"{group_name}_params", sample=sample
        )
        varying_intercept, varying_slope = group_params[:, 0], group_params[:, 1]
        varying_intercept = group_params[:, 0][group]
        varying_slope = group_params[:, 1][group]
        return varying_intercept, varying_slope

    @staticmethod
    def varying_slope(
        category_idx,
        grouping_idx,
        num_categories,
        num_groups,
        L_Rho=None,
        sigma=None,
        group_name="group",
        sample=False,
    ):
        """Generates hierarchical varying effects using a non-centered parameterization.

        This is a general-purpose component for building hierarchical models. It models
        a set of related effects (e.g., slopes or intercepts) that vary by a grouping
        factor. The effects for each group are "pooled" together, meaning the model
        learns a shared distribution for them, allowing information to be shared
        among groups.

        This is useful for modeling:
        - The effect of different treatments ('categories') varying across subjects ('groups').
        - The performance of students ('groups') across different school subjects ('categories').
        - The changing approval rating ('category') of a policy over time in different countries ('groups').

        Args:
            category_idx (array): An array of integer indices for the category of the
                effect for each observation. For example, if you have 4 treatments,
                this would be an array of numbers from 0 to 3.
            grouping_idx (array): An array of integer indices for the grouping factor
                for each observation (e.g., subject ID, school ID, country ID).
            num_categories (int): The total number of unique categories for the effects
                (e.g., 4 treatments). This defines the dimensionality of the effects.
            num_groups (int): The total number of unique groups (e.g., 7 subjects).
            L_Rho (distribution, optional): The prior for the Cholesky factor of the
                correlation matrix between the effects. If None, defaults to a weakly
                informative LKJCholesky(num_categories, 2) prior.
            sigma (distribution, optional): The prior for the standard deviations (scales)
                of the effects for each category. If None, defaults to a weakly
                informative Exponential(1) prior.
            group_name (str, optional): A descriptive name for the grouping factor, used
                to create unique parameter names in the model trace. Defaults to 'group'.

        Returns:
            array: An array containing the specific varying effect for each
                   corresponding observation in the input arrays.
        """

        # 1. Define default hyperpriors if not provided by the user.
        # These priors define the multivariate distribution from which the
        # individual group effects are drawn.
        if L_Rho is None:
            # The LKJ Cholesky prior is a distribution over correlation matrices.
            # An eta > 1 (here, 2) encourages weaker correlations.
            L_Rho = dist.lkj_cholesky(
                num_categories, 2, name=f"L_Rho_{group_name}", sample=sample
            )

        if sigma is None:
            # The Exponential prior ensures the standard deviations are positive.
            sigma = dist.exponential(
                1, name=f"sigma_{group_name}", shape=(num_categories,), sample=sample
            )

        # 2. Implement the non-centered parameterization for efficiency.
        # We sample uncorrelated standard normal values and then transform them.
        # This improves the geometry of the posterior for the sampler. [1, 2, 3, 4, 5]
        z = dist.normal(
            0,
            1,
            name=f"z_{group_name}",
            shape=(num_categories, num_groups),
            sample=sample,
        )

        # 3. Transform the standard normal samples into correlated varying effects.
        # This operation combines the scale (sigma), correlation (L_Rho), and the
        # base samples (z). The result 'effects' is a matrix of shape
        # (num_groups, num_categories).
        effects = ((sigma[..., None] * L_Rho) @ z).T

        # 4. Return the specific effect for the category and group of each observation.
        # This uses advanced indexing to pick the correct value from the 'effects' matrix
        # for each row of data.
        return effects[grouping_idx, category_idx]

    @staticmethod
    def varying_intercept(
        N_groups, group_id, a_bar=None, sigma=None, group_name="group", sample=False
    ):
        """
        Varying intercepts model using a non-centered parameterization.

        Args:
            N_groups (int): The total number of unique groups.
            group_id (array): An array of integer indices specifying the group for each observation.
            a_bar (distribution, optional): The hyperprior for the mean of the group-level intercepts.
            sigma (distribution, optional): The hyperprior for the standard deviation of the group-level intercepts.

        Returns:
            array: An array of the appropriate group-level intercepts for each observation.
        """
        # 1. Handle default hyperpriors.
        if a_bar is None:
            a_bar = dist.normal(
                0.0, 2.0, name=f"global_intercept_{group_name}", sample=sample
            )
        if sigma is None:
            sigma = dist.exponential(1.0, name=f"sd_{group_name}", sample=sample)

        # 2. Define the distribution for the group-level intercepts ('alpha')
        # There is one alpha for each unique group.
        alpha = dist.normal(
            a_bar,
            sigma,
            shape=(N_groups,),
            name=f"intercept_{group_name}",
            sample=sample,
        )

        # 3. Return the specific intercept corresponding to the group of each observation
        return alpha[group_id]

    @staticmethod
    def varying_intercept_slope_non_centered(
        N_groups,
        group_idx,
        global_intercept_mean,
        global_slope_mean,
        group_name="group",
        sigma=None,
        L_Rho=None,
        sample=False,
    ):
        """
        Models a varying intercept and varying slope using a NON-CENTERED parameterization.

        Args:
            N_groups (int): The total number of unique groups.
            group_idx (array): An array of integer indices for the group of each observation.
            global_intercept_mean (float or distribution): The hyperprior for the mean of the intercepts.
            global_slope_mean (float or distribution): The hyperprior for the mean of the slopes.
            group_name (str, optional): A descriptive name for the grouping factor.
            sigma (distribution, optional): Prior for the standard deviations of the
                intercept and slope. Must have shape (2,). Defaults to Exponential(1).
            L_Rho (distribution, optional): Prior for the Cholesky factor of the 2x2
                correlation matrix. Defaults to LKJCholesky(2, 2).

        Returns:
            tuple: A tuple containing two arrays:
                - varying_intercepts (array): The intercept for each observation.
                - varying_slopes (array): The slope for each observation.
        """
        warn_dev("effects")
        # 1. Handle default hyperpriors for scale (sigma) and correlation (L_Rho).
        if sigma is None:
            sigma = dist.exponential(
                1, shape=(2,), name=f"{group_name}_sigma", sample=sample
            )
        if L_Rho is None:
            L_Rho = dist.lkj_cholesky(2, 2, name=f"L_Rho_{group_name}", sample=sample)

        # 2. Sample the uncorrelated, standard normal "offsets".
        z = dist.normal(
            0, 1, name=f"z_{group_name}", shape=(2, N_groups), sample=sample
        )

        # 3. Transform the offsets into correlated deviations from zero.
        effects = ((sigma[..., None] * L_Rho) @ z).T

        # 4. Construct the final parameters by adding the global means.
        mean_vector = jnp.stack([global_intercept_mean, global_slope_mean])
        params_per_group = mean_vector + effects

        # 5. Select the correct parameter pair for each observation.
        params_for_obs = params_per_group[group_idx]

        # 6. Unpack and return the final intercepts and slopes.
        varying_intercepts = params_for_obs[:, 0]
        varying_slopes = params_for_obs[:, 1]
        return varying_intercepts, varying_slopes

    @staticmethod
    def varying_slope_non_centered(
        category_idx,
        grouping_idx,
        num_categories,
        num_groups,
        L_Rho=None,
        sigma=None,
        group_name="group",
        sample=False,
    ):
        """Generates hierarchical varying effects for CATEGORIES using a non-centered parameterization."""

        warn_dev("effects")
        # 1. Define default hyperpriors if not provided by the user.
        if L_Rho is None:
            L_Rho = m.dist.lkj_cholesky(
                num_categories, 2, name=f"L_Rho_{group_name}", sample=sample
            )
        if sigma is None:
            sigma = m.dist.exponential(
                1, name=f"sigma_{group_name}", shape=(num_categories,), sample=sample
            )

        # 2. Sample uncorrelated standard normal values ("offsets").
        z = m.dist.normal(
            0,
            1,
            name=f"z_{group_name}",
            shape=(num_categories, num_groups),
            sample=sample,
        )

        # 3. Transform the offsets into correlated varying effects.
        effects = ((sigma[..., None] * L_Rho) @ z).T

        # 4. Return the specific effect for the category and group of each observation.
        return effects[grouping_idx, category_idx]

    @staticmethod
    def varying_intercept_non_centered(
        N_groups, group_idx, a_bar=None, sigma=None, group_name="group", sample=False
    ):
        """
        Varying intercepts model using a NON-CENTERED parameterization.

        Args:
            N_groups (int): The total number of unique groups.
            group_idx (array): An array of integer indices specifying the group for each observation.
            a_bar (distribution, optional): The hyperprior for the mean of the group-level intercepts.
            sigma (distribution, optional): The hyperprior for the standard deviation of the group-level intercepts.

        Returns:
            array: An array of the appropriate group-level intercepts for each observation.
        """
        warn_dev("effects")
        # 1. Handle default hyperpriors for the mean (a_bar) and scale (sigma).
        if a_bar is None:
            a_bar = dist.normal(
                0.0, 2.0, name=f"global_intercept_{group_name}", sample=sample
            )
        if sigma is None:
            sigma = dist.exponential(1.0, name=f"sd_{group_name}", sample=sample)

        # 2. Sample the standard normal "offsets", one for each unique group.
        z = dist.normal(
            0.0, 1.0, name=f"z_intercept_{group_name}", shape=(N_groups,), sample=sample
        )

        # 3. Construct the intercepts by scaling and shifting the offsets.
        alpha = a_bar + z * sigma

        # 4. Return the specific intercept corresponding to the group of each observation.
        return alpha[group_idx]

    @staticmethod
    def nested_varying_effects(
        N_vars,
        names,
        N_groups,
        group_ids,
        mu=None,
        sigma=None,
        L_corr=None,
        corr=None,
        centered=False,
        sample=False,
    ):
        """
        Nested varying effects model for arbitrary hierarchy depth and variable count.

        N_vars is the number of modelled variables (intercept + slopes) and is independent
        of the number of levels. You can have 4 variables across 2 levels, or 2 variables
        across 5 levels — they are separate dimensions.

        Each level shares the same N_vars because effects at child levels are deviations
        from the parent on the same set of variables:
            child_effect = parent_effect[parent_id] + deviation   # both shape (N_vars,)

        Parent structure is derived automatically from group_ids: for each unit g at
        level i, its parent = group_ids[i-1] at any observation where group_ids[i] == g.
        This works for balanced and unbalanced nested designs with no extra arguments.

        Args:
            N_vars (int): Number of variables per level (intercept + number of slopes).
                Independent of the number of levels — e.g., N_vars=4 with 2 levels is valid.
            names (list[str]): Name for each level, top to bottom.
            N_groups (list[int]): Number of units at each level, top to bottom.
            group_ids (list[array]): One obs-level index array per level, top to bottom.
                group_ids[i][obs] = which level-i group observation obs belongs to.
            mu (list[array|None] | None, optional): Top-level global mean override
                (only element 0 is used). If None, global intercept and beta are auto-sampled.
            sigma (list[dist|None] | None, optional): Per-level scale priors, length = n_levels.
                None elements fall back to Exponential(1).
            L_corr (list[dist|None] | None, optional): Per-level Cholesky-factor priors
                (non-centered only), length = n_levels. None → LKJCholesky(N_vars, 2).
            corr (list[dist|None] | None, optional): Per-level correlation-matrix priors
                (centered only), length = n_levels. None → LKJ(N_vars, 2).
            centered (bool): Use centered parameterization. Defaults to False.
            sample (bool): Sample from posterior wrapper. Defaults to False.

        Returns:
            tuple: (varying_intercepts, varying_slopes) arrays, one value per observation.

        Examples:
            # 2 levels, 2 variables (intercept + 1 slope) — most common case
            a, b = m.effects.nested_varying_effects(
                N_vars=2,
                names=["region", "group"],
                N_groups=[5, 20],
                group_ids=[region_id, group_id],
            )

            # 3 levels, 2 variables — N_vars unchanged as levels increase
            a, b = m.effects.nested_varying_effects(
                N_vars=2,
                names=["country", "region", "group"],
                N_groups=[3, 10, 40],
                group_ids=[country_id, region_id, group_id],
            )

            # 2 levels, 4 variables (intercept + 3 slopes) — N_vars > n_levels is valid
            a, b1, b2, b3 = m.effects.nested_varying_effects(
                N_vars=4,
                names=["region", "group"],
                N_groups=[5, 20],
                group_ids=[region_id, group_id],
            )

            # centered parameterization
            a, b = m.effects.nested_varying_effects(
                N_vars=2,
                names=["region", "group"],
                N_groups=[5, 20],
                group_ids=[region_id, group_id],
                centered=True,
            )

            # custom priors per level
            a, b = m.effects.nested_varying_effects(
                N_vars=2,
                names=["region", "group"],
                N_groups=[5, 20],
                group_ids=[region_id, group_id],
                sigma=[
                    m.dist.exponential(2, shape=(2,)),    # region level
                    m.dist.exponential(0.5, shape=(2,)),  # group level
                ],
            )
        """
        n_levels = len(names)

        # Top-level global mean
        if mu is not None:
            current_parent_effects = mu[0] if isinstance(mu, list) else mu
        else:
            alpha_bar = dist.normal(0, 2, name="global_intercept", sample=sample, shape=(1,))
            if N_vars > 1:
                beta_bar = dist.normal(0, 1, name="global_beta", sample=sample, shape=(N_vars - 1,))
                current_parent_effects = jnp.concat([alpha_bar, beta_bar])
            else:
                current_parent_effects = alpha_bar

        for i, name in enumerate(names):
            n_group = N_groups[i]

            sigma_i = (sigma[i] if (sigma is not None and sigma[i] is not None)
                       else dist.exponential(1, shape=(N_vars,), name=f"sigma_{name}", sample=sample))

            if centered:
                if N_vars == 1:
                    if i == 0:
                        current_parent_effects = dist.normal(
                            current_parent_effects[0], sigma_i[0],
                            shape=(n_group, 1), name=f"{name}_effects", sample=sample
                        )
                    else:
                        pid = _parent_ids(group_ids[i], group_ids[i - 1], n_group)
                        current_parent_effects = dist.normal(
                            current_parent_effects[pid, 0], sigma_i[0],
                            name=f"{name}_effects", sample=sample
                        )[:, None]
                else:
                    corr_i = (corr[i] if (corr is not None and corr[i] is not None)
                              else dist.lkj(dimension=N_vars, concentration=2, name=f"corr_{name}", sample=sample))
                    cov = jnp.diag(sigma_i) @ corr_i @ jnp.diag(sigma_i)
                    if i == 0:
                        current_parent_effects = dist.multivariate_normal(
                            current_parent_effects, cov, shape=(n_group,), name=f"{name}_effects", sample=sample
                        )
                    else:
                        pid = _parent_ids(group_ids[i], group_ids[i - 1], n_group)
                        current_parent_effects = dist.multivariate_normal(
                            current_parent_effects[pid], cov, name=f"{name}_effects", sample=sample
                        )
            else:
                if N_vars == 1:
                    z = dist.normal(0, 1, name=f"z_{name}", shape=(n_group,), sample=sample)
                    effects_deviation = (sigma_i[0] * z)[:, None]
                else:
                    L_corr_i = (L_corr[i] if (L_corr is not None and L_corr[i] is not None)
                                else dist.lkj_cholesky(dimension=N_vars, concentration=2, name=f"L_corr_{name}", sample=sample))
                    z = dist.normal(0, 1, name=f"z_{name}", shape=(N_vars, n_group), sample=sample)
                    effects_deviation = ((sigma_i[..., None] * L_corr_i) @ z).T
                if i == 0:
                    current_parent_effects = current_parent_effects + effects_deviation
                else:
                    pid = _parent_ids(group_ids[i], group_ids[i - 1], n_group)
                    current_parent_effects = current_parent_effects[pid] + effects_deviation

        obs_effects        = current_parent_effects[group_ids[-1]]
        varying_intercepts = obs_effects[:, 0]
        if N_vars == 1:
            return varying_intercepts
        varying_slopes = obs_effects[:, 1:]
        if varying_slopes.shape[1] == 1:
            varying_slopes = varying_slopes[:, 0]
        return varying_intercepts, varying_slopes

