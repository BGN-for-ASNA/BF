import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
import numpyro
import numpyro.distributions as dist
from BayesForge.Network.Net import Neteffect
from BayesForge.Distributions.np_dists import UnifiedDist as _UnifiedDist

# Module-level instance: the class's methods are instance methods.
dist = _UnifiedDist()
from BayesForge.Utils.SampledData import SampledData

def logit(x):
    # RELEASE_AUDIT S4: clip to (eps, 1 - eps) so the endpoints stay finite
    # (the block prior feeds base-rate and padded entries straight in, and a
    # 0/1 there used to poison the whole trace with -inf/+inf), and use
    # log(x) - log1p(-x) for accuracy near 0. Values already inside
    # (eps, 1 - eps) are unchanged up to float rounding.
    eps = jnp.finfo(jnp.result_type(x, jnp.float32)).eps
    x = jnp.clip(x, eps, 1 - eps)
    return jnp.log(x) - jnp.log1p(-x)


def _check_block_labels(block_predictors, index1):
    """RELEASE_AUDIT S5: block labels must be integer, >= 0, 0-based contiguous.

    Both block importers (SRM.import_block_predictors and
    SRM.import_block_predictors2) derive the number of groups of a column from
    ``max + 1``, so a column has to hold every label 0..max: a 1-based column
    silently dropped group 0 and a gapped one silently dropped the missing
    label. Called right after the optional index1 shift.
    """
    labels = np.asarray(block_predictors)
    if labels.ndim != 2 or labels.shape[0] == 0:
        return
    for a in range(labels.shape[1]):
        col = np.asarray(labels[:, a], dtype=float)
        if not np.all(np.isfinite(col)):
            raise ValueError(
                f'Error: Block factor column {a} must hold 0-based contiguous '
                f'integer labels, got non-finite values.')
        if not np.all(col == np.floor(col)):
            raise ValueError(
                f'Error: Block factor column {a} must hold 0-based contiguous '
                f'integer labels, got non-integer values.')
        present = {int(x) for x in np.unique(col)}
        top = int(col.max())
        if min(present) < 0:
            raise ValueError(
                f'Error: Block factor column {a} must hold 0-based contiguous '
                f'labels, got negative labels {sorted(present)}.')
        if present != set(range(top + 1)):
            missing = sorted(set(range(top + 1)) - present)
            msg = (f'Error: Block factor column {a} must hold 0-based contiguous '
                   f'labels 0..{top}; missing {missing}.')
            if min(present) == 1:
                msg += ' Labels start at 1: pass index1=True to shift them to 0-based.'
            raise ValueError(msg)


class SRM:
    def __init__(self, network, print_info = False):
        
        if network.ndim!=2:
             raise ValueError('Error: Network dimension must be 2')

        self.N_id = network.shape[0]        
        self.N_dyad = (self.N_id * (self.N_id - 1) )/ 2
        self.N_dyad = int(self.N_dyad)
        self.network = Neteffect.mat_to_edgl(network)
        self.print_info = print_info
        self.focal_predictors = None
        self.focal_predictors_names = None
        self.receiver_predictors = None
        self.receiver_predictors_names = None
        self.dyadic_predictors = None
        self.dyadic_predictors_names = None
        self.exposure = None
        self.block_predictors = None
        self.block_predictors_names = None
        self.blocks_base_rate = None     
        self.block_intercept = None
        self.dyadic_intercept = None
        
    def import_focal_predictors(self, focal_predictors, names = None):
        if self.print_info:
            print('--------------------------------------------------------------------------------')
            print("Ensure that the focal predictors are in the same order as the network")
            print("Ensure that the focal predictors are have a (V,N) shape, where V is the number of focal predictors and N is the number of nodes")
        # if it is a data frame convert to jax array
        if isinstance(focal_predictors, pd.DataFrame):
            focal_predictors = jnp.array(focal_predictors) # transpose for the dot product

        # if jax array check if it is a 2D array
        if isinstance(focal_predictors, (jnp.ndarray, SampledData)):
            if focal_predictors.ndim != 2:
                raise ValueError('Error: Focal factors must be a 2D array')


        self.focal_predictors = focal_predictors.T
        self.focal_predictors_names = names
        if self.print_info:
            print(f'Focal factors {self.focal_predictors_names}, imported')

    def import_receiver_predictors(self, receiver_predictors, names = None):
        if self.print_info:
            print('--------------------------------------------------------------------------------')
            print("Ensure that the focal predictors are in the same order as the network")
            print("Ensure that the focal predictors are have a (V,N) shape, where V is the number of focal predictors and N is the number of nodes")
        # if it is a data frame convert to jax array
        if isinstance(receiver_predictors, pd.DataFrame):
            receiver_predictors = jnp.array(receiver_predictors)
        # if jax array check if it is a 2D array
        if isinstance(receiver_predictors, (jnp.ndarray, SampledData)):
            if receiver_predictors.ndim != 2:
                raise ValueError('Error: Focal factors must be a 2D array')
 

        self.receiver_predictors = receiver_predictors.T
        self.receiver_predictors_names = names
        if self.print_info:
            print(f'Focal factors {self.receiver_predictors_names}, imported')
    
    def import_dyadic_predictors(self, dyadic_predictors, names = None, intercept_present = False):
        if self.print_info:
            print('--------------------------------------------------------------------------------')
        if isinstance(dyadic_predictors, (jnp.ndarray, SampledData)):
            if dyadic_predictors.ndim == 2:
                #! self.network is already an edge list (N_dyad, 2) (see __init__),
                #! so a 2-D dyadic predictor must be a full (N_id, N_id) matrix.
                if dyadic_predictors.shape != (self.N_id, self.N_id):
                    raise ValueError(f'Dyadic predictor matrix must be (N, N) = ({self.N_id}, {self.N_id})')
                #! Handle it exactly like a 3-D input with one predictor:
                dyadic_predictors = dyadic_predictors[:, :, None]
            if dyadic_predictors.ndim == 3:
                if dyadic_predictors.shape[:2] != (self.N_id, self.N_id):
                    raise ValueError(f'Dyadic predictor matrix must be (N, N) = ({self.N_id}, {self.N_id})')
                tmp2 =  []
                for a in range(dyadic_predictors.shape[2]):
                    tmp2.append(Neteffect.mat_to_edgl(dyadic_predictors[:,:,a]))               

                tmp2 = jnp.transpose(jnp.stack(tmp2), (1,2,0))
                #dyadic_predictors = tmp2[,,0:dim(data$dyad_set)[3]]
                self.dyadic_predictors = tmp2
                self.dyadic_predictors_names = names
                if not intercept_present:
                    dyadic_intercept = jnp.ones((self.N_dyad, 2))
                    self.dyadic_predictors= jnp.concatenate([dyadic_intercept[..., None], self.dyadic_predictors], axis = -1)


                if self.print_info:
                    print(f'Dyadic factors {self.dyadic_predictors_names}, imported')

            else:
                raise ValueError('Error: Dyadic factors must be a 2D or 3D array')
        else:
            raise TypeError('Error: Dyadic factors must be a jax array')
    
    def import_exposure(self, exposure):
        print('--------------------------------------------------------------------------------')
        # if jax array check if it is a 2D array
        if isinstance(exposure, (jnp.ndarray, SampledData)):
            if exposure.ndim != 1:
                raise ValueError('Error: Exposure must be a 1D array')
            else:
                self.exposure = exposure
                if self.print_info:
                    print(f'Exposure {self.exposure}, imported')
        else:
            raise TypeError('Error: Exposure must be a jax array')

    def import_block_predictors2(self, block_predictors, names = None, index1 = False, intercept_present = False):
        """Import block factors for the base-rate / block_prior path.

        Builds ``self.block_predictors`` (an intercept column of ones is
        prepended unless ``intercept_present``), then the per-variable base
        rates and mu_ij matrices (prepare_block_base_rate), the dyad-level
        block indicators (prepare_block_to_edglelist) and the padded mu_ij
        tensor consumed by block_prior (prepare_block_effects).

        See also SRM.import_block_predictors, the other block importer: it
        builds dat_B / Block_edgl for the Stan-like block model instead.
        """
        print('Be sure that the block predictors are integers starting at 0')
        if self.print_info:
            print('--------------------------------------------------------------------------------')
        # if it is a data frame convert to jax array
        if isinstance(block_predictors, pd.DataFrame):
            block_predictors = jnp.array(block_predictors)
        # if jax array check if it is a 2D array
        if isinstance(block_predictors, (jnp.ndarray, SampledData)):
            if block_predictors.ndim != 2:
                raise ValueError('Error: Block(s) factors must be a 2D or 3D array')
                

        if index1:
            block_predictors = block_predictors - 1
        # RELEASE_AUDIT S5: validate the labels before the intercept column is
        # added, so the message points at the user's own columns.
        _check_block_labels(block_predictors, index1)
        self.block_predictors = block_predictors
        if not intercept_present:
            block_intercept = jnp.ones(block_predictors.shape[0])
            self.block_predictors = jnp.concat([block_intercept[..., None], block_predictors], axis = -1)
        self.block_predictors_names = names
        self.prepare_block_base_rate()
        self.prepare_block_to_edglelist()
        self.prepare_block_effects()
        if self.print_info:
            print(f'Focal factors {self.block_predictors_names}, imported')

    def prepare_block_base_rate(self):
        blocks_base_rate = []
        blocks_mu_ij = []

        for a in range(self.block_predictors.shape[1]):
            # RELEASE_AUDIT S6: "sum() == 0" is not a test for "no groups": a
            # column of labels (-1, +1) sums to 0 but holds two groups. A
            # column without groups is a constant column, and for a constant
            # column the result (1 group, N_by_group = [N_id]) is unchanged.
            if jnp.unique(self.block_predictors[:,a]).shape[0] == 1:
                N_groups = 1
                N_by_group = jnp.array([self.N_id])
            else:   
                N_by_group = jnp.unique(self.block_predictors[:,a], return_counts = True)[1]
                N_groups = N_by_group.shape[0]
            base_rate = jnp.tile(0.01, (N_groups,N_groups))
            blocks_base_rate.append(base_rate.at[jnp.diag_indices_from(base_rate)].set(0.1))
            blocks_mu_ij.append(base_rate/jnp.sqrt(jnp.outer(N_by_group, N_by_group)))
        
        self.blocks_base_rate = blocks_base_rate
        self.blocks_mu_ij = blocks_mu_ij
    
    def prepare_block_to_edglelist(self):
        blocks = self.block_predictors
        blocks_edgl = []
        for a in range(blocks.shape[1]):
            blocks_edgl.append(Neteffect.vec_node_to_edgle(jnp.stack([blocks[:,a], blocks[:,a]], axis= 1).astype(int)))
        blocks_edgl = jnp.stack(blocks_edgl, axis = 1).transpose(0,2,1)
        self.blocks_edgl = blocks_edgl
    
    def prepare_block_effects(self):
        blocks_mu_ij = self.blocks_mu_ij        
        # --- Preparation for JIT ---
        max_dim = max(b.shape[0] for b in blocks_mu_ij)
        #    We find the largest dimension and pad the smaller arrays.
        #! Pad with 0.5, not NaN: update_block_effect applies logit() to the whole
        #! padded matrix, and logit(0.5) = 0 is a finite prior mean. The padded
        #! entries are never indexed by Neteffect.block_prior_to_edglelist.
        blocks_mu_ij_padded =  [jnp.pad(b, [(0, max_dim - b.shape[0]), (0, max_dim - b.shape[1])], constant_values=0.5) for b in blocks_mu_ij]
        blocks_mu_ij_padded = jnp.stack(blocks_mu_ij_padded)
        num_blocks = len(blocks_mu_ij)
        self.num_blocks = num_blocks
        self.blocks_mu_ij_padded = blocks_mu_ij_padded

    @staticmethod
    def update_block_effect(i, tmp, blocks_padded, block_predictors, b_ij_sd):
        """
        The body of the fori_loop, executed for each block.
        This function should not be JIT-compiled directly, as it's part of the loop.
        """
        # Select the current block and predictor vector
        current_block_mu = blocks_padded[i]
        current_predictor = block_predictors[:, i]

        # Calculate 'b' using the helper functions
        b = dist.normal(logit(current_block_mu), b_ij_sd, name = 'b' + str(i))

        update = Neteffect.block_prior_to_edglelist(current_predictor, b)
        tmp = tmp + jnp.sum(update, axis=1, keepdims=True) # Assumption: sum updates

        return tmp

    @staticmethod
    def block_prior_fn(initial_tmp, blocks_padded, block_predictors, num_blocks, b_ij_sd):
        """
        JIT-compatible function to run the update loop over all blocks.
        """
        # Create a lambda that captures the static arguments for fori_loop
        loop_body_fn = lambda i, tmp: SRM.update_block_effect(i, tmp, blocks_padded, block_predictors, b_ij_sd)

        # Run the loop
        final_tmp = jax.lax.fori_loop(0, num_blocks, loop_body_fn, initial_tmp)

        return final_tmp
      
    def block_prior(self):
        # RELEASE_AUDIT S2: use the default float dtype instead of asking for
        # float64, which silently downcast (with a warning) when x64 is off.
        tmp = jnp.zeros_like(self.blocks_edgl[:,:,0], dtype=jnp.result_type(float))
        Block_effect = SRM.block_prior_fn(tmp, self.blocks_mu_ij_padded, self.block_predictors, self.num_blocks, b_ij_sd = 0.1)
        return Block_effect
    
    def import_block_predictors(self, block_predictors, names = None, index1 = False):
        """Import block factors for the Stan-like block model (dat_B / Block_edgl).

        Builds ``self.dat_B`` — one row per block variable v and group pair
        (i, j), holding the indices, the base rate, the group sizes, the prior
        mean Mu and the prior scale Sigma — and ``self.Block_edgl``, the
        dyad-level design of every block parameter, consumed by SRM.model and
        block_effect_srm_class.

        See also SRM.import_block_predictors2, the other block importer: it
        builds the base rates and the padded block effects used by block_prior.
        """
        #! index1 for R users: labels 1..g are shifted to 0..g-1 first, so the
        #! block tensor below and self.block_predictors use the same labels.
        if index1:
            block_predictors = block_predictors - 1
        # RELEASE_AUDIT S5: labels must be 0-based contiguous per column.
        _check_block_labels(block_predictors, index1)
        N_id = block_predictors.shape[0]
        N_vars = block_predictors.shape[1]
        N_groups_per_var = (block_predictors.max(axis=0) + 1).astype(int)
        max_N_groups_per_var = max(N_groups_per_var)
        # RELEASE_AUDIT S1: the (v, i, j) grid used to be filled by a triple
        # Python loop of .at[].set (O(V*G^2) dispatches). This is setup code,
        # not jitted, so it is built with numpy instead: dat_B is bit-identical
        # (same row order, same values) and the padding entries keep 0, so they
        # are still dropped by the B_In == 1 filter below.
        float_dtype = jnp.result_type(float)
        bp_np = np.asarray(block_predictors)
        N_groups_np = np.asarray(N_groups_per_var, dtype = int)
        N_per_group = np.ones((N_vars, max_N_groups_per_var), dtype = float_dtype)
        for a in range(N_vars):
            N_per_group[a,:N_groups_np[a]] = np.bincount(bp_np[:,a].astype(int), minlength = N_groups_np[a])
        N_pars = (N_groups_per_var * N_groups_per_var).sum()
    
        B_V = np.zeros((N_vars, max_N_groups_per_var, max_N_groups_per_var), dtype = float_dtype)
        B_I = np.zeros_like(B_V)
        B_J = np.zeros_like(B_V)
        B_In = np.zeros_like(B_V)
        B_Base = np.zeros_like(B_V)
        B_SS = np.zeros_like(B_V)
        for v in range(N_vars):
            G = N_groups_np[v]
            ii, jj = np.meshgrid(np.arange(G), np.arange(G), indexing = "ij")
            B_V[v,ii,jj] = v
            B_I[v,ii,jj] = ii
            B_J[v,ii,jj] = jj
            B_In[v,ii,jj] = 1
            B_Base[v,ii,jj] = np.where(ii == jj, 0.1, 0.01)
            B_SS[v,ii,jj] = np.sqrt(N_per_group[v,ii]*0.5 + N_per_group[v,jj]*0.5)
    
        B_SS = B_SS.flatten()
        B_V = B_V.flatten()
        B_I = B_I.flatten()
        B_J = B_J.flatten()
        B_In = B_In.flatten()
        B_Base = B_Base.flatten()
        dat_B = jnp.stack([B_V, B_I, B_J, B_In, B_Base, B_SS], axis = -1)
        dat_B = dat_B[jnp.where(B_In==1)]
        Sigma = jnp.full((N_pars,), 2.5)
        Mu = logit(dat_B[:,-2]/dat_B[:,-1])
    
        dat_B = jnp.concatenate([dat_B, Mu[:,None], Sigma[:,None]], axis = 1)


        def build_block_tensor(dat_B, block_predictors):
            dat_B2 = dat_B[:, :3].astype(int)
            dat_B2_col0 = dat_B2[:, 0]
            dat_B2_col1 = dat_B2[:, 1]
            dat_B2_col2 = dat_B2[:, 2]
            block_preds_indexed = block_predictors[:, dat_B2_col0]
            scrap_i_matrix = (block_preds_indexed == dat_B2_col1)
            scrap_j_matrix = (block_preds_indexed == dat_B2_col2)
            # 5. Transpose the matrices to get a shape of (N_pars, N_id)
            scrap_i_T = scrap_i_matrix.T
            scrap_j_T = scrap_j_matrix.T
            Y_optimized = (scrap_i_T[:, :, np.newaxis] * scrap_j_T[:, np.newaxis, :]).astype(int)
            return  jnp.array(Y_optimized.transpose(1,2,0))
        
        block_tensor = build_block_tensor(dat_B, block_predictors)

        Block_edgl =  []
        for a in range(block_tensor.shape[2]):
            Block_edgl.append(Neteffect.mat_to_edgl(block_tensor[:,:,a]))  
        Block_edgl = jnp.transpose(jnp.stack(Block_edgl), (1,2,0))

        self.Block_edgl = Block_edgl
        self.dat_B = dat_B
        self.block_names = names
        self.N_vars_Block = N_vars
        self.max_N_groups_per_var = max_N_groups_per_var
        self.N_pars = N_pars
        self.block_predictors = block_predictors

    @staticmethod
    def block_effect_srm_class(Block_edgl, dat_B, sample = False):
        coefs = dist.normal(dat_B[:,-2], dat_B[:,-1], name = 'block_effects',  sample = sample)
        return jnp.sum(coefs * Block_edgl, axis=2)
    

    def model(self, Block_edgl, focal_predictors, receiver_predictors, dyadic_predictors, network, dat_B):
        # Block ---------------------------------------
        B = self.block_effect_srm_class(Block_edgl, dat_B)

        ## SR shape =  N individuals---------------------------------------
        sr =  Neteffect.sender_receiver(focal_predictors,receiver_predictors)

        # Dyadic shape = N dyads--------------------------------------
        dr = Neteffect.dyadic_effect(dyadic_predictors)

        dist.bernoulli(logits = B + sr + dr , obs=network)

    def numpyro_bsrm_model(self, network, focal_predictors, receiver_predictors, dyadic_predictors,
                          block_predictors, exposure, outcome_mode=1, export_network=0):
        """
        NumPyro-compatible BSRM model that matches the reference implementation exactly.
        """
        # Get dimensions
        N_id = focal_predictors.shape[0] if focal_predictors is not None else receiver_predictors.shape[0]
        N_dyads = network.shape[0]

        # RELEASE_AUDIT W4: this model consumes ordered dyads (i -> j, i != j),
        # N*(N-1) rows. SRM.__init__ stores the N*(N-1)/2 upper-triangle dyads
        # built by mat_to_edgl; passing them would silently misalign every
        # dyad, so fail loudly instead.
        if N_dyads != N_id * (N_id - 1):
            raise ValueError(
                f"numpyro_bsrm_model expects ordered dyads: network must have "
                f"N*(N-1) = {N_id * (N_id - 1)} rows for N={N_id}, got "
                f"{N_dyads}. SRM(A).network holds the N*(N-1)/2 upper-triangle "
                f"dyads from mat_to_edgl and cannot be passed directly.")

        # Create edgelist indices (use the edgelist passed as network parameter)
        # The network parameter is the edgelist outcomes, so we need to create indices
        ids = jnp.arange(N_id)
        S, R = jnp.meshgrid(ids, ids)
        mask = S != R
        long_ids_int = jnp.stack([S[mask], R[mask]], axis=1)

        # Prepare data matrices - expand individual predictors to dyad level
        if focal_predictors is not None:
            long_focal_set_np = focal_predictors[long_ids_int[:, 0]]  # (N_dyads, N_vars)
        else:
            long_focal_set_np = jnp.zeros((N_dyads, 0))
            
        if receiver_predictors is not None:
            long_target_set_np = receiver_predictors[long_ids_int[:, 1]]  # (N_dyads, N_vars)
        else:
            long_target_set_np = jnp.zeros((N_dyads, 0))
            
        long_dyad_set_np = dyadic_predictors if dyadic_predictors is not None else jnp.zeros((N_dyads, 0))

        # Get block data if provided
        if block_predictors is not None:
            # Block design: one effect per (block variable, group); groups are
            # 0-based labels 0..g_v-1 of each column of block_predictors.
            bp = np.asarray(block_predictors).astype(int)
            n_groups = bp.max(axis=0) + 1
            offsets = np.concatenate([[0], np.cumsum(n_groups)[:-1]])
            N_blocks = int(n_groups.sum())
            ids_np = np.asarray(long_ids_int)
            design = np.zeros((N_dyads, N_blocks))
            rows = np.arange(N_dyads)
            for v in range(bp.shape[1]):
                # For dyad (i,j), effects for i's group and j's group
                design[rows, offsets[v] + bp[ids_np[:, 0], v]] = 1
                design[rows, offsets[v] + bp[ids_np[:, 1], v]] = 1
            long_block_set_np = jnp.asarray(design)
            block_mu_np = jnp.zeros(N_blocks)
            block_sigma_np = jnp.full(N_blocks, 2.5)
            
            N_var_block = N_blocks
        else:
            long_block_set_np = jnp.zeros((N_dyads, 0))
            block_mu_np = jnp.array([])
            block_sigma_np = jnp.array([])
            N_var_block = 0

        # Get dimensions
        N_var_focal = long_focal_set_np.shape[1]
        N_var_target = long_target_set_np.shape[1]
        N_var_dyad = long_dyad_set_np.shape[1]

        # Priors (matching NumPyro reference)
        prior_12_1, prior_12_2 = 0.0, 2.5  # focal effects
        prior_13_1, prior_13_2 = 0.0, 2.5  # target effects
        prior_14_1, prior_14_2 = 0.0, 2.5  # dyad effects
        prior_15_1, prior_15_2 = 0.0, 2.5  # sr sigma
        prior_16_1, prior_16_2 = 0.0, 2.5  # dr sigma
        prior_17_1 = 2.5  # sr LKJ
        prior_18_1 = 2.5  # dr LKJ
        prior_23_1, prior_23_2 = 0.0, 2.5  # error sigma

        # Scaling factors (A_F, A_T, A_D from NumPyro)
        A_F = jnp.ones(N_var_focal) if N_var_focal > 0 else jnp.array([])
        A_T = jnp.ones(N_var_target) if N_var_target > 0 else jnp.array([])
        A_D = jnp.ones(N_var_dyad) if N_var_dyad > 0 else jnp.array([])

        # Outcome and exposure data
        long_outcome_set_np = network
        long_exposure_set_np = exposure if exposure is not None else jnp.ones_like(network)

        # Call the NumPyro model function (imported or defined elsewhere)
        # This would be the actual NumPyro model code
        return self._numpyro_bsrm_model_core(
            long_outcome_set_np, long_exposure_set_np, long_ids_int,
            block_mu_np, block_sigma_np,
            prior_12_1, prior_12_2, prior_13_1, prior_13_2, prior_14_1, prior_14_2,
            prior_15_1, prior_15_2, prior_16_1, prior_16_2, prior_17_1, prior_18_1,
            prior_23_1, prior_23_2,
            long_focal_set_np, long_target_set_np, long_dyad_set_np, long_block_set_np,
            N_dyads, N_id, N_var_focal, N_var_target, N_var_dyad, N_var_block,
            A_F, A_T, A_D, outcome_mode, export_network
        )

    def _numpyro_bsrm_model_core(self, long_outcome_set_np, long_exposure_set_np, long_ids_int,
                                block_mu_np, block_sigma_np,
                                prior_12_1, prior_12_2, prior_13_1, prior_13_2, prior_14_1, prior_14_2,
                                prior_15_1, prior_15_2, prior_16_1, prior_16_2, prior_17_1, prior_18_1,
                                prior_23_1, prior_23_2,
                                long_focal_set_np, long_target_set_np, long_dyad_set_np, long_block_set_np,
                                N_dyads, N_id, N_var_focal, N_var_target, N_var_dyad, N_var_block,
                                A_F, A_T, A_D, outcome_mode, export_network):
        """
        Core NumPyro BSRM model implementation that matches the reference exactly.
        """
        if outcome_mode not in (1, 2, 3, 4):
            raise ValueError(f"outcome_mode must be 1 (bernoulli), 2 (binomial), 3 (poisson) "
                             f"or 4 (gaussian), got {outcome_mode!r}")
        # Observation noise (for Gaussian outcome)
        error_sigma = dist.truncated_normal(prior_23_1, prior_23_2, low=0.0, name="error_sigma")

        # Block, focal, target, dyad effects
        if N_var_block > 0:
            block_effects = dist.normal(block_mu_np, block_sigma_np, name="block_effects")
        else:
            block_effects = jnp.array([])

        if N_var_focal > 0:
            focal_effects = dist.normal(jnp.full((N_var_focal,), prior_12_1), jnp.full((N_var_focal,), prior_12_2), name="focal_effects")
        else:
            focal_effects = jnp.array([])

        if N_var_target > 0:
            target_effects = dist.normal(jnp.full((N_var_target,), prior_13_1), jnp.full((N_var_target,), prior_13_2), name="target_effects")
        else:
            target_effects = jnp.array([])

        if N_var_dyad > 0:
            dyad_effects = dist.normal(jnp.full((N_var_dyad,), prior_14_1), jnp.full((N_var_dyad,), prior_14_2), name="dyad_effects")
        else:
            dyad_effects = jnp.array([])

        # Linear predictor for each observation (2 dimensions for BSRM)
        mu = jnp.zeros((N_dyads, 2))
        if N_var_block > 0:
            block_contrib = jnp.tensordot(long_block_set_np, block_effects, axes=1)
            mu += block_contrib[:, None]  # broadcast to (N_dyads, 2)
        if N_var_focal > 0:
            focal_contrib = jnp.tensordot(long_focal_set_np, focal_effects * A_F, axes=1)
            mu += focal_contrib[:, None]  # broadcast to (N_dyads, 2)
        if N_var_target > 0:
            target_contrib = jnp.tensordot(long_target_set_np, target_effects * A_T, axes=1)
            mu += target_contrib[:, None]  # broadcast to (N_dyads, 2)
        if N_var_dyad > 0:
            dyad_contrib = jnp.tensordot(long_dyad_set_np, dyad_effects * A_D, axes=1)
            mu += dyad_contrib[:, None]  # broadcast to (N_dyads, 2)

        # Dyad random effects
        dr_L = numpyro.sample("dr_L", numpyro.distributions.LKJCholesky(2, prior_18_1))
        dr_sigma = dist.truncated_normal(prior_16_1, prior_16_2, low=0.0, name="dr_sigma")
        dr_sigma_scaled = jnp.sqrt(dr_sigma)
        dr_L_scaled = dr_L * dr_sigma_scaled

        # RELEASE_AUDIT S3: both branches of the old `if export_network == 1:`
        # held the same statement, so the branch was dead code.
        dr_raw = numpyro.sample("dr_raw", numpyro.distributions.Normal(jnp.zeros((2, N_dyads)), 1))

        dr = jnp.transpose(jnp.matmul(dr_L_scaled, dr_raw))

        # Sender-receiver effects
        sr_L = numpyro.sample("sr_L", numpyro.distributions.LKJCholesky(2, prior_17_1))
        sr_sigma = numpyro.sample("sr_sigma", numpyro.distributions.TruncatedNormal(jnp.full((2,), prior_15_1), scale=jnp.full((2,), prior_15_2), low=jnp.full((2,), 0.0)))
        sr_sigma_temp = jnp.expand_dims(sr_sigma, 1)
        sr_sigma_scaled = sr_sigma_temp * sr_L
        sr_raw = numpyro.sample("sr_raw", numpyro.distributions.Normal(jnp.zeros((2, N_id)), 1))

        gr = jnp.transpose(jnp.matmul(sr_sigma_scaled, sr_raw))

        # Expand sender/receiver effects into observation-level data
        i_idx = long_ids_int[:, 0]
        j_idx = long_ids_int[:, 1]

        S_i = gr[i_idx, 0]
        S_j = gr[j_idx, 0]
        R_i = gr[i_idx, 1]
        R_j = gr[j_idx, 1]
        gr_long = jnp.stack([S_i + R_j, S_j + R_i], axis=1)

        # Outcome models
        linear_model = mu + dr + gr_long
        if outcome_mode == 1:
            # Bernoulli (binary)
            dist.bernoulli(logits=linear_model, obs=long_outcome_set_np)
            if export_network == 1:
                numpyro.deterministic("p", jax.nn.sigmoid(linear_model))

        elif outcome_mode == 2:
            # Binomial
            dist.binomial(total_count=long_exposure_set_np, logits=linear_model, obs=long_outcome_set_np)
            if export_network == 1:
                numpyro.deterministic("p", jax.nn.sigmoid(linear_model))

        elif outcome_mode == 3:
            # Poisson
            dist.poisson(rate=jnp.exp(linear_model), obs=long_outcome_set_np)
            if export_network == 1:
                numpyro.deterministic("p", jnp.exp(linear_model))

        elif outcome_mode == 4:
            # Gaussian
            dist.normal(loc=linear_model, scale=error_sigma, obs=long_outcome_set_np)
            if export_network == 1:
                numpyro.deterministic("p", linear_model)

        # Store correlation matrices for diagnostics
        numpyro.deterministic("G_corr", jnp.matmul(sr_L, sr_L.T))
        numpyro.deterministic("D_corr", jnp.matmul(dr_L, dr_L.T))

