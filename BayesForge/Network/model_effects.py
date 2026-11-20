from BayesForge.Network.util import array_manip
import jax 
from jax import jit
import jax.numpy as jnp
import numpy as np
from numpyro import deterministic
from BayesForge.Distributions.np_dists import UnifiedDist as dist
from functools import partial
dist = dist()
class Neteffect(array_manip):
    # Former no-op __init__ only allocated a dead dict
    # that was never read or written anywhere in BayesForge; removed. The
    # inherited array_manip.__init__ is a pass, so construction is unchanged.

    @staticmethod 
    @jit
    def logit(x):
        """
        Computes the logit transformation.

        Parameters
        ----------
        x : float or array-like
            Input value(s) in the range (0, 1).

        Returns
        -------
        float or array-like
            The logit-transformed value(s): log(x) - log(1 - x), computed
            with ``log1p`` so values close to 1 do not overflow the ratio.
        """
        return jnp.log(x) - jnp.log1p(-x)

    # Sender receiver  ----------------------
    
    def nodes_random_effects(self,N_id,  sr_sigma_mu = 0, sr_sigma_sd = 2.5, cholesky_dim = 2, cholesky_density = 2.5, sample = False):
        """Correlated sender and receiver random effects for each node (non-centred).

        Defines ``sr_raw`` ~ Normal(0, 1), ``sr_sigma`` ~ TruncatedNormal(mu, sd, low=0)
        and ``sr_L`` ~ LKJCholesky, and returns ``sr_rf = (diag(sr_sigma) sr_L sr_raw)^T``.

        Args:
            N_id (int): Number of nodes.
            sr_sigma_mu (float): Location of the prior on the two standard deviations. Defaults to 0.
            sr_sigma_sd (float): Scale of that prior. Defaults to 2.5.
            cholesky_dim (int): Dimension of the correlation matrix (2: sender, receiver).
            cholesky_density (float): LKJ concentration; larger values favour weak
                correlations. Defaults to 2.5.
            sample (bool): Draw values directly instead of defining model sites.

        Returns:
            tuple: ``(sr_rf, sr_raw, sr_sigma, sr_L)``; ``sr_rf`` has shape (N_id, 2)
            with sender effects in column 0 and receiver effects in column 1.
        """
        sr_raw =  dist.normal(0, 1, shape=(2, N_id), name = 'sr_raw', sample = sample)
        sr_sigma =  dist.truncated_normal(sr_sigma_mu, sr_sigma_sd, low = 0, shape= (2,), name = 'sr_sigma', sample = sample)
        sr_L = dist.lkj_cholesky(cholesky_dim, cholesky_density, name = "sr_L", sample = sample)
        sr_sigma_temp = jnp.expand_dims(sr_sigma, 1)
        sr_sigma_scaled = sr_sigma_temp * sr_L
        rf = deterministic('sr_rf', jnp.transpose(jnp.matmul(sr_sigma_scaled, sr_raw)))
        return rf, sr_raw, sr_sigma, sr_L
   
    def nodes_terms(self, sender_predictors = None, receiver_predictors = None,
                    s_mu = 0, s_sd = 1, r_mu = 0, r_sd = 1, sample = False):
        """Fixed effects of node predictors on sending and receiving ties.

        Args:
            sender_predictors (2D jax array, optional): Node predictors of sending,
                shape (N, K_s); each column is a node characteristic.
            receiver_predictors (2D jax array, optional): Node predictors of receiving,
                shape (N, K_r). At least one of the two must be given.
            s_mu (float): Prior mean of the sender coefficients. Defaults to 0.
            s_sd (float): Prior sd of the sender coefficients. Defaults to 1.
            r_mu (float): Prior mean of the receiver coefficients. Defaults to 0.
            r_sd (float): Prior sd of the receiver coefficients. Defaults to 1.
            sample (bool): Draw values directly instead of defining model sites.

        Returns:
            tuple: ``(terms, sender_effects, receiver_effects)``; ``terms`` has shape
            (N, 2): the linear predictor of sending (column 0) and receiving (column 1).
        """
        if sender_predictors is None and receiver_predictors is None:
            raise ValueError("At least one of sender_predictors or receiver_predictors must be provided.")
        if sender_predictors is not None:
            N_var_sender = sender_predictors.shape[1]
            sender_effects = dist.normal(s_mu, s_sd, shape=(N_var_sender,), sample = sample, name = 'sender_effects')
            sender_dot = sender_effects @ sender_predictors.T
        else:
            sender_effects = None
            sender_dot = jnp.zeros((receiver_predictors.shape[0], ))

        if receiver_predictors is not None:      
            N_var_receiver = receiver_predictors.shape[1]
            receiver_effects =  dist.normal( r_mu, r_sd, shape= (N_var_receiver,), sample = sample, name = 'receiver_effects')
            receiver_dot = receiver_effects @ receiver_predictors.T
        else:
            receiver_effects = None
            receiver_dot = jnp.zeros((sender_predictors.shape[0], ))


        terms = jnp.stack([sender_dot, receiver_dot], axis = -1)

        return terms, sender_effects, receiver_effects
    
    @staticmethod 
    @jit
    def node_effects_to_dyadic_format(sr_effects):
        """Convert node effects to dyadic (edge list) format.

        Args:
            sr_effects (jax array): Array of node effects with shape [N_nodes, 2].
        Returns:
            jax array: Dyadic effects with shape [N_dyads, 2], where each row represents 
            a dyad (i, j) with sender effect i and receiver effect j.
        """
        ids = jnp.arange(0,sr_effects.shape[0])
        edgl_idx = Neteffect.vec_node_to_edgle(jnp.stack([ids, ids], axis = -1))

        S_i = sr_effects[edgl_idx[:, 0],0]
        S_j = sr_effects[edgl_idx[:, 1],0]
        R_i = sr_effects[edgl_idx[:, 0],1]
        R_j = sr_effects[edgl_idx[:, 1],1]
        return jnp.stack([S_i + R_j, S_j + R_i ], axis = 1)

    def sender_receiver(self,sender_predictors = None, receiver_predictors = None,  
                        #Fixed effect parameters
                        s_mu = 0, s_sd = 2.5, r_mu = 0, r_sd = 2.5,                         
                        #Random effect parameters
                        sr_sigma_mu = 0, sr_sigma_sd = 2.5,
                        cholesky_dim = 2, cholesky_density = 2.5,
                        sample = False):
        """Compute sender-receiver effects combining both fixed and random effects.

        Args:
            sender_predictors (jax array): Predictors for focal individuals.
            receiver_predictors (jax array): Predictors for target individuals.
            s_mu (float, optional): Mean for focal effects. Defaults to 0.
            s_sd (float, optional): SD for focal effects. Defaults to 1.
            r_mu (float, optional): Mean for target effects. Defaults to 0.
            r_sd (float, optional): SD for target effects. Defaults to 1.
            sr_mu (float, optional): Mean for random effects. Defaults to 0.
            sr_sd (float, optional): SD for random effects. Defaults to 1.
            sr_sigma_mu (float, optional): Rate parameter for random effects. Defaults to 0.
            sr_sigma_sd (float, optional): Rate parameter for random effects. Defaults to 1.
            cholesky_dim (int, optional): Dimension for Cholesky decomposition. Defaults to 2.
            cholesky_density (int, optional): Density parameter for Cholesky. Defaults to 2.
            sample (bool, optional): Whether to sample from distributions. Defaults to False.

        Returns:
            jax array: Combined dyadic effects.
        """        
        if sender_predictors is None and receiver_predictors is None:
            raise ValueError("At least one of sender_predictors or receiver_predictors must be provided.")

        if sender_predictors is not None:
            N_var_sender = sender_predictors.shape[1]
            N_id = sender_predictors.shape[0] 
        else:
            N_var_sender = 0

        if receiver_predictors is not None:            
            N_var_receiver = receiver_predictors.shape[1]
            N_id = receiver_predictors.shape[0] 
        else:
            N_var_receiver = 0


        sr_ff, focal_effects, target_effects = self.nodes_terms(
            sender_predictors, receiver_predictors,
            s_mu = s_mu, s_sd = s_sd, r_mu = r_mu, r_sd = r_sd, 
            sample = sample 
        )
        sr_rf, sr_raw, sr_sigma, sr_L = self.nodes_random_effects(
            N_id, sr_sigma_mu = sr_sigma_mu, 
            sr_sigma_sd = sr_sigma_sd, cholesky_dim = cholesky_dim, 
            cholesky_density = cholesky_density,  sample = sample
            ) # shape = N_id
        
        sr_to_dyads = Neteffect.node_effects_to_dyadic_format(sr_ff + sr_rf) # sr_ff and sr_rf are nodal values that need to be converted to dyadic values
        return sr_to_dyads

    # dyadic effects ------------------------------------------
    @staticmethod 
    @jit
    def prepare_dyadic_effect(dyadic_effect_mat):
        """Prepare dyadic effect matrix for processing.

        Args:
            dyadic_effect_mat (jax array): Dyadic effect matrix to process.

        Returns:
            jax array: Processed dyadic effects in edge list format.
        """        
        if dyadic_effect_mat.ndim == 2:
            return Neteffect.mat_to_edgl(dyadic_effect_mat)
        elif dyadic_effect_mat.ndim == 3:
            return jax.vmap(Neteffect.mat_to_edgl, in_axes=2, out_axes=2)(dyadic_effect_mat)

    @staticmethod 
    def dyadic_random_effects(N_dyads,  dr_sigma_mu = 0, dr_sigma_sd = 2.5, cholesky_dim = 2, cholesky_density = 2.5, sample = False, diag = False):
        """Generate random effects for dyadic models.

        Args:
            N_dyads (int): Number of dyads.
            dr_mu (float, optional): Mean for random effects. Defaults to 0.
            dr_sd (float, optional): SD for random effects. Defaults to 1.
            dr_sigma (float, optional): Sigma parameter for random effects. Defaults to 1.
            cholesky_dim (int, optional): Dimension for Cholesky decomposition. Defaults to 2.
            cholesky_density (int, optional): Density parameter for Cholesky. Defaults to 2.
            sample (bool, optional): Whether to sample from distributions. Defaults to False.
            diag (bool, optional): Whether to print diagnostic information. Defaults to False.

        Returns:
            tuple: Contains random effects, raw effects, sigma, and Cholesky decomposition matrix.
        """
        dr_raw =  dist.normal(0, 1, shape=(2,N_dyads), name = 'dr_raw', sample = sample)
        dr_sigma = dist.truncated_normal(dr_sigma_mu, dr_sigma_sd, low = 0, shape=(1,), name = 'dr_sigma', sample = sample )
        dr_L = dist.lkj_cholesky(cholesky_dim, cholesky_density, name = 'dr_L', sample = sample)
        dr_rf =  deterministic('dr_rf',jnp.transpose(jnp.matmul(dr_sigma * dr_L, dr_raw)))
        if diag :
            print("dr_raw--------------------------------------------------------------------------------")
            print(dr_raw)
            print("dr_sigma--------------------------------------------------------------------------------")
            print(dr_sigma)
            print("dr_L--------------------------------------------------------------------------------")
            print(dr_L)
            print("rf--------------------------------------------------------------------------------")
            print(dr_rf)
        return dr_rf, dr_raw, dr_sigma, dr_L # we return everything to get posterior distributions for each parameters

    @staticmethod 
    def dyadic_terms(dyadic_predictors, d_m = 0, d_sd = 2.5, sample = False):
        """Calculate fixed effects for dyadic terms.

        Args:
            dyadic_predictors (jax array): Predictors for dyadic terms.
            d_m (float, optional): Mean for dyad effects. Defaults to 0.
            d_sd (float, optional): SD for dyad effects. Defaults to 1.
            sample (bool, optional): Whether to sample from distributions. Defaults to False.
            diag (bool, optional): Whether to print diagnostic information. Defaults to False.

        Returns:
            tuple: Contains fixed effects and dyadic predictors.
        """        
  
        
        if dyadic_predictors.ndim == 2:
            dyad_effects = dist.normal(d_m, d_sd, name= 'dyad_effects', shape = (1,), sample = sample)
            dr_ff = dyad_effects * dyadic_predictors

        elif dyadic_predictors.ndim == 3:
            dyad_effects = dist.normal(d_m, d_sd, name= 'dyad_effects', shape = (dyadic_predictors.shape[2],), sample = sample)
            dr_ff = jnp.tensordot(dyadic_predictors, dyad_effects, axes=[2, 0])
            
        return dr_ff, dyad_effects

    @staticmethod 
    def dyadic_effect(dyadic_predictors = None, shape = None, d_m = 0, d_sd = 2.5, # Fixed effect arguments
                     dr_sigma_mu = 0, dr_sigma_sd = 2.5, cholesky_dim = 2, cholesky_density = 2.5,
                     sample = False):
        """Compute dyadic effects combining both fixed and random components.
        
        Args:
            dyadic_predictors (jax array, optional): Edge-list predictors of shape
                (N_dyads, 2, K) — column 0 is direction i->j, column 1 j->i, one
                slab per predictor k. A 2-D array is read as an (N_dyads, K)
                edge list with one coefficient shared by the K columns.
                Warning (RELEASE_AUDIT m8): m.net2.dyadic_effect reads a 2-D
                input as an (N, N) predictor matrix instead, so the same 2-D
                array means different things in m.net and m.net2.
            shape (int, optional): Number of dyads, N*(N-1)/2 — used only when
                dyadic_predictors is None (random effects only).
            d_m (float, optional): Mean for fixed effects. Defaults to 0.
            d_sd (float, optional): SD for fixed effects. Defaults to 1.
            dr_sigma (float, optional): Sigma parameter for random effects. Defaults to 1.
            cholesky_dim (int, optional): Dimension for Cholesky decomposition. Defaults to 2.
            cholesky_density (int, optional): Density parameter for Cholesky. Defaults to 2.
            sample (bool, optional): Whether to sample from distributions. Defaults to False.
            
        Returns:
            jax array: Combined dyadic effects.
        """                     
        if dyadic_predictors is None and shape is None:
            # RELEASE_AUDIT m7: this used to print and return the error string,
            # which then propagated as a str into the downstream logit sum.
            raise ValueError(
                'Argument shape must be defined if argument dyadic_predictors is not define')
        if dyadic_predictors is not None :
            dr_ff, dyad_effects = Neteffect.dyadic_terms(dyadic_predictors, d_m = d_m, d_sd = d_sd, sample = sample)
            dr_rf, dr_raw, dr_sigma, dr_L =  Neteffect.dyadic_random_effects(dr_ff.shape[0], dr_sigma_mu = dr_sigma_mu, dr_sigma_sd = dr_sigma_sd,
            cholesky_dim = cholesky_dim, cholesky_density = cholesky_density, sample = sample)
            return dr_ff + dr_rf
        else:
            dr_rf, dr_raw, dr_sigma, dr_L =  Neteffect.dyadic_random_effects(shape,  dr_sigma_mu = dr_sigma_mu, dr_sigma_sd = dr_sigma_sd,
            cholesky_dim = cholesky_dim, cholesky_density = cholesky_density, sample = sample)
        return  dr_rf
  
    @staticmethod 
    @jit
    def block_prior_to_edglelist(v, b):
        """Convert block vector id group belonging to edgelist of i->j group values

        Args:
            v (1D array):  Vector of id group belonging
            b (2D array): Matrix of block model prior matrix (squared)

        Returns:
            2D array: Shape (N_dyads, 2): the block prior for i -> j and for j -> i of each dyad.
        """

        v = Neteffect.vec_node_to_edgle(jnp.stack([v, v], axis= 1)).astype(int)

        return jnp.stack([b[v[:,0],v[:,1]], b[v[:,1],v[:,0]]], axis = 1)

    # RELEASE_AUDIT m6: an orphaned second @staticmethod used to sit here,
    # double-decorating block_model (TypeError at import on Python <= 3.9).

    @staticmethod
    def block_model(group, N_group, N_by_group, b_ij_sd=2.5, b_ij_mu=None,
                    sample=False, name=''):
        """Sample block-model B matrix and convert to dyadic edgelist contributions.

        Args:
            group (1D int array): 0-indexed group membership for each node (length N).
            N_group (int): Number of groups.
            N_by_group (1D int array): Number of nodes per group.
            b_ij_sd (float or (N_group,N_group) array): Prior SD for B[i,j] on logit
                scale. Defaults to 2.5.
            b_ij_mu ((N_group,N_group) array, optional): Prior mean for B[i,j] on the
                **logit scale**. When provided (e.g. STRAND's block_mu_np reshaped as a
                matrix), it replaces the generic block_build_mu_ij formula so that wide-
                format and long-format models share identical block priors.
                When None (default), the prior mean is computed via block_build_mu_ij.
            sample (bool): Whether to draw samples. Defaults to False.
            name (str): Name suffix for the sampled site ('b_<name>').

        Returns:
            (N_dyads, 2) array: Logit-scale block contributions for each directed dyad.
        """
        if b_ij_mu is None:
            # Generic formula: derive probability from group sizes, then logit-transform.
            # Clip to (0, 1) first so a saturated (full or empty) block can never
            # push the prior mean to +-inf.
            mu_prob = jnp.clip(
                Neteffect.block_build_mu_ij(N_by_group, N_group), 1e-6, 1 - 1e-6
            )
            mu_ij = Neteffect.logit(mu_prob)
        else:
            # Caller supplies the prior mean directly on the logit scale
            mu_ij = b_ij_mu
        b = dist.normal(mu_ij, b_ij_sd, sample=sample, name=f'b_{name}')
        return Neteffect.block_prior_to_edglelist(group, b)

    # BISoN edge-weight builtins -----------------------------------------------

    def edge_weights(self, num_edges, prior_mu=0.0, prior_sigma=2.5,
                     partial_pooling=True, sample=False, name='edge_weight'):
        """Sample edge weight parameters for BISoN-style dyadic inference.

        Args:
            num_edges (int): Number of unique dyads/edges.
            prior_mu (float): Prior mean on logit/log scale. Defaults to 0.0.
            prior_sigma (float): Prior SD (direct) or half-normal scale (partial pooling).
                Defaults to 2.5.
            partial_pooling (bool): If True, uses non-centered parameterization with a shared
                sigma hyperprior — decouples edge_sigma from edge geometry for better mixing.
                Defaults to True.
            sample (bool): Draw from prior (DGP mode) vs declare latent variable. Defaults to False.
            name (str): Base name for sampled sites. Defaults to 'edge_weight'.

        Returns:
            jax array: Edge weight values of shape (num_edges,).
        """
        if partial_pooling:
            edge_sigma = dist.half_normal(prior_sigma, name=f'{name}_sigma', sample=sample)
            edge_raw   = dist.normal(0, 1, shape=(num_edges,), name=name, sample=sample)
            return prior_mu + edge_sigma * edge_raw
        else:
            return dist.normal(prior_mu, prior_sigma, shape=(num_edges,), name=name, sample=sample)

    @staticmethod
    def nodal_fixed_effects(design_fixed, num_fixed, prior_mu=0.0, prior_sigma=2.5,
                             sample=False, name='beta_fixed'):
        """Sample fixed nodal effects and return their predictor contribution.

        Args:
            design_fixed (jax array): Design matrix of shape (num_rows, num_fixed).
            num_fixed (int): Number of fixed predictors.
            prior_mu (float): Prior mean for coefficients. Defaults to 0.0.
            prior_sigma (float): Prior SD for coefficients. Defaults to 2.5.
            sample (bool): Draw from prior (DGP mode) vs declare latent variable. Defaults to False.
            name (str): Name for the sampled site. Defaults to 'beta_fixed'.

        Returns:
            jax array: Predictor contribution of shape (num_rows,).
        """
        beta = dist.normal(prior_mu, prior_sigma, shape=(num_fixed,), name=name, sample=sample)
        return jnp.dot(design_fixed, beta)

    @staticmethod
    def nodal_random_effects(design_random, random_group_index, num_random, num_random_groups,
                              prior_mean_mu=0.0, prior_mean_sigma=1.0, prior_std_sigma=1.0,
                              centered=False, sample=False, name='beta_random'):
        """Sample hierarchical random nodal effects and return predictor contribution.

        Args:
            design_random (jax array): Design matrix of shape (num_rows, num_random).
            random_group_index (jax int array): 1-based group membership per random effect,
                shape (num_random,).
            num_random (int): Number of random effect levels (e.g. number of nodes).
            num_random_groups (int): Number of groups (e.g. sex categories).
            prior_mean_mu (float): Prior mean for group-level means. Defaults to 0.0.
            prior_mean_sigma (float): Prior SD for group-level means. Defaults to 1.0.
            prior_std_sigma (float): Half-normal scale for group-level SDs. Defaults to 1.0.
            centered (bool): Use centered parameterization. Preferred for undirected networks
                where u_i + u_j shift non-identifiability causes non-centered mode-switching.
                Non-centered (default) is better for directed networks. Defaults to False.
            sample (bool): Draw from prior (DGP mode) vs declare latent variable. Defaults to False.
            name (str): Base name for sampled sites. Defaults to 'beta_random'.

        Returns:
            jax array: Predictor contribution of shape (num_rows,).
        """
        # RELEASE_AUDIT W7: codes are 1-based (group_idx_0 = code - 1), so
        # 0-based codes from m.index() would silently map group 0 to the last
        # group. Fail loudly when the codes are concrete (not a tracer).
        if not isinstance(random_group_index, jax.core.Tracer):
            _codes = np.asarray(random_group_index)  # numpy: see block_build_mu_ij
            if _codes.size and (int(_codes.min()) < 1
                                or int(_codes.max()) > num_random_groups):
                raise ValueError(
                    f"nodal_random_effects: random_group_index is 1-based "
                    f"(got min={int(_codes.min())}, max={int(_codes.max())}); "
                    f"codes from m.index() are 0-based: pass index + 1")
        group_idx_0 = random_group_index - 1
        group_mu    = dist.normal(prior_mean_mu, prior_mean_sigma, shape=(num_random_groups,),
                                   name=f'{name}_group_mu', sample=sample)
        group_sigma = dist.half_normal(prior_std_sigma, shape=(num_random_groups,),
                                        name=f'{name}_group_sigma', sample=sample)
        if centered:
            # Sample actual effects directly — avoids mode-switching on the u_i+u_j shift ridge
            # in undirected networks. Posterior for `name` is the actual random effect value.
            beta = dist.normal(group_mu[group_idx_0], group_sigma[group_idx_0],
                               shape=(num_random,), name=name, sample=sample)
            return jnp.dot(design_random, beta)
        else:
            beta_raw    = dist.normal(0, 1, shape=(num_random,), name=name, sample=sample)
            beta_actual = group_mu[group_idx_0] + group_sigma[group_idx_0] * beta_raw
            return jnp.dot(design_random, beta_actual)

    @staticmethod
    def block_build_mu_ij(N_by_group, N_group):
        """Build block-model prior means matching STRAND's internal logic.

        This function programmatically replicates the formula in STRAND's
        ``block_set_to_dyadic_block_set.R``.

        Formula:
            Base = 0.1 (diagonal/ingroup) or 0.01 (off-diagonal/outgroup)
            SS   = sqrt(N_i * 0.5 + N_j * 0.5)
            Mu   = Base / SS

        Raises
        ------
        ValueError
            When a concrete (non-traced) ``N_by_group`` contains a 0: an
            empty group divides the base rate by zero, which would silently
            produce ``inf`` probabilities and ``NaN`` prior means downstream.
        """
        if not isinstance(N_by_group, jax.core.Tracer):
            # numpy, not jnp: a jnp op run while the model is traced returns
            # a tracer even on a concrete input.
            sizes = np.asarray(N_by_group)
            empty = np.flatnonzero(sizes == 0)
            if empty.size:
                idx = ", ".join(str(int(i)) for i in empty.tolist())
                raise ValueError(
                    f"block model: group(s) {idx} have no member; drop empty "
                    "levels before building the block prior."
                )
        return Neteffect._block_build_mu_ij_jit(N_by_group, N_group)

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def _block_build_mu_ij_jit(N_by_group, N_group):
        # N_group is a static value known at compile time.
        base_rate = jnp.tile(0.01, (N_group, N_group))
        base_rate = base_rate.at[jnp.diag_indices_from(base_rate)].set(0.1)

        # STRAND additive logic for sample size normalization
        ni_05 = N_by_group[:, None] * 0.5
        nj_05 = N_by_group[None, :] * 0.5
        ss = jnp.sqrt(ni_05 + nj_05)

        mu_ij = base_rate / ss
        return mu_ij