"""
Matrix-form SRM network effects — shard-compatible alternative to model_effects.py.

All methods return (N, N) arrays instead of (N_dyads, 2) edgelists, enabling
clean axis-0 sharding (sender/row dimension) without XLA allgather deadlocks.

Root cause of the deadlock in the edgelist version
---------------------------------------------------
`node_effects_to_dyadic_format` in Neteffect does a scatter-gather:
    S_i = sr_effects[edgl_idx[:, 0], 0]   # (N_nodes) → (N_dyads)
When dyadic_predictors is sharded along axis 0, the dyadic fixed-effect output
(dr_ff) is (N_dyads/n_dev, 2) per device. Adding the unsharded sr (N_dyads, 2)
forces XLA to allgather sr across all devices simultaneously. With 4 vmapped
chains, one device is always staggered behind the NUTS tree rendezvous → deadlock.

Fix: compute SR as an outer sum, shard along the sender (row) axis
------------------------------------------------------------------
    s (N,) ⊗ r (N,)  →  SR_mat[i,j] = s[i] + r[j]
    s[:, None] + r[None, :]            ← no scatter-gather, no allgather

The (N, N) logit matrix shards cleanly along axis 0 when dyadic_predictors_mat
(N, N, K) and Y_mat (N, N) are sharded along axis 0.

Sharding notes for callers
--------------------------
Shard along axis 0:   Y_mat, dyadic_predictors_mat, sender_predictors
Keep replicated:      receiver_predictors, Any/Merica group vectors,
                      N_by_grp_* arrays
                      (group vectors need all N column labels for the outer
                       indexing in block_model; receiver_predictors needs all
                       N rows for the outer product columns in sender_receiver)

N divisibility:       choose n_devices that divides N (e.g. n_devices=20 for N=400)

Usage
-----
    from BayesForge.Network.model_effects2 import NeteffectMatrix

    def model(Y_mat, dyadic_predictors_mat, sender_predictors, receiver_predictors,
              Any, Merica, N_by_grp_Any, N_by_grp_Merica, **_):
        N = Y_mat.shape[0]
        B_any    = NeteffectMatrix.block_model(Any,    1, N_by_grp_Any,    name='intercept')
        B_merica = NeteffectMatrix.block_model(Merica, 3, N_by_grp_Merica, name='Merica')
        SR_mat   = m.net2.sender_receiver(sender_predictors, receiver_predictors)
        D_mat    = NeteffectMatrix.dyadic_effect(dyadic_predictors_mat)
        logits   = B_any + B_merica + SR_mat + D_mat
        # Mask the diagonal with -inf, never by multiplying the logits:
        # logits * (1 - eye) sets the diagonal to logit 0 -> p = 0.5, a
        # constant -N*log(2) contribution to the log-likelihood that also
        # contradicts a 1 in Y_mat's diagonal without warning. The where
        # idiom (same as metrics.py off-diagonal masking) makes the diagonal
        # a true exclusion: zero contribution while obs is 0.
        logits   = jnp.where(jnp.eye(N, dtype=bool), -jnp.inf, logits)
        m.dist.bernoulli(logits=logits, obs=Y_mat, name='Y_mat')  # Y_mat diagonal must be 0
"""

import jax.numpy as jnp
from numpyro import deterministic

from BayesForge.Network.model_effects import Neteffect
from BayesForge.Distributions.np_dists import UnifiedDist as _UnifiedDist

dist = _UnifiedDist()


class NeteffectMatrix(Neteffect):
    """Matrix-form network effects. All primary outputs are (N, N) arrays.

    Inherits utilities from Neteffect (logit, block_build_mu_ij, edgl_to_mat,
    mat_to_edgl, vec_node_to_edgle, …) but overrides the three core methods
    so that every output lives in (N, N) space rather than (N_dyads, 2) space.
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Block model — (N, N)
    # ------------------------------------------------------------------
    @staticmethod
    def block_model(group, N_group, N_by_group, b_ij_sd=2.5, b_ij_mu=None,
                    sample=False, name='', group_row=None):
        """Sample block B and return B[group[i], group[j]] as (N, N) or (N_shard, N).

        Identical prior to Neteffect.block_model. The scatter-gather over
        N_dyads is replaced by a direct (G, G) matrix lookup: since B is
        at most (G, G) with G typically 1–5, this is trivially replicated
        on all devices and requires no allgather.

        Args:
            group (1D int array): 0-indexed group membership, shape (N,).
                Keep replicated across devices (needed for column dimension).
            N_group (int): Number of groups G.
            N_by_group (1D int array): Count of nodes per group.
            b_ij_sd (float or (G,G) array): Prior SD on logit scale.
            b_ij_mu ((G,G) array, optional): Prior mean on logit scale.
            sample (bool): DGP mode if True.
            name (str): Suffix for the numpyro site name 'b_<name>'.
            group_row (1D int array, optional): Row-shard group indices, shape (N_shard,).
                When provided, returns (N_shard, N) using group_row for rows and
                group for columns. Use this when data is sharded along axis 0 so
                only N_shard rows are processed per device.

        Returns:
            (N, N) or (N_shard, N) array: logit-scale block contribution.
        """
        if b_ij_mu is None:
            mu_ij = Neteffect.logit(Neteffect.block_build_mu_ij(N_by_group, N_group))
        else:
            mu_ij = b_ij_mu
        b = dist.normal(mu_ij, b_ij_sd, sample=sample, name=f'b_{name}')  # (G, G)
        row_idx = group if group_row is None else group_row
        return b[row_idx[:, None], group[None, :]]  # (N_shard, N) or (N, N)

    # ------------------------------------------------------------------
    # Sender-receiver — (N, N)
    # ------------------------------------------------------------------
    def sender_receiver(self, sender_predictors=None, receiver_predictors=None,
                        s_mu=0, s_sd=2.5, r_mu=0, r_sd=2.5,
                        sr_sigma_mu=0, sr_sigma_sd=2.5,
                        cholesky_dim=2, cholesky_density=2.5,
                        sample=False):
        """Sender-receiver effects as (N, N) outer sum — no scatter-gather.

        Computes identical fixed + random effects as Neteffect.sender_receiver
        but returns the full (N, N) matrix via outer sum instead of a
        (N_dyads, 2) scatter-gather:

            SR_mat[i,j] = s[i] + r[j]   (direction i → j)
            SR_mat[j,i] = s[j] + r[i]   (direction j → i, automatic from matrix)

        This is mathematically equivalent to Neteffect.node_effects_to_dyadic_format
        but avoids the indexed gather that triggers XLA allgather when dyads are
        sharded across devices.

        Sharding: shard sender_predictors along axis 0; keep receiver_predictors
        replicated. s becomes (N/n_dev,) per device via sharding propagation;
        r stays (N,) replicated. The outer sum s[:,None]+r[None,:] then produces
        (N/n_dev, N) locally on each device without cross-device communication.

        Args:
            sender_predictors (jax array): (N, P_s) sender design matrix.
                Shard along axis 0 for data parallelism.
            receiver_predictors (jax array): (N, P_r) receiver design matrix.
                Keep replicated — all N columns needed for outer product.
            s_mu, s_sd (float): Prior mean/SD for sender fixed effects.
            r_mu, r_sd (float): Prior mean/SD for receiver fixed effects.
            sr_sigma_mu, sr_sigma_sd (float): Half-normal prior for random SD.
            cholesky_dim (int): LKJ dimension (must be 2 for sender/receiver).
            cholesky_density (float): LKJ concentration parameter.
            sample (bool): DGP mode if True.

        Returns:
            (N, N) array: logit-scale sender-receiver contribution.
        """
        if sender_predictors is None and receiver_predictors is None:
            raise ValueError(
                "At least one of sender_predictors or receiver_predictors must be provided."
            )

        N_id = (
            sender_predictors if sender_predictors is not None else receiver_predictors
        ).shape[0]

        # ── fixed effects (node-level vectors) ─────────────────────────────
        if sender_predictors is not None:
            s_eff = dist.normal(
                s_mu, s_sd, shape=(sender_predictors.shape[1],),
                name='sender_effects', sample=sample,
            )
            # When sender_predictors is sharded (N/n_dev, P_s), output is
            # (N/n_dev,) per device — sharding propagates naturally.
            s_fixed = sender_predictors @ s_eff  # (N,) or (N/n_dev,) if sharded
        else:
            s_fixed = jnp.zeros(N_id)

        if receiver_predictors is not None:
            r_eff = dist.normal(
                r_mu, r_sd, shape=(receiver_predictors.shape[1],),
                name='receiver_effects', sample=sample,
            )
            # receiver_predictors must remain replicated so r spans all N nodes.
            r_fixed = receiver_predictors @ r_eff  # (N,) replicated
        else:
            r_fixed = jnp.zeros(N_id)

        # ── random effects (Cholesky-correlated, node-level) ───────────────
        # sr_raw is a global (2, N) latent variable; XLA replicates it on all
        # devices as part of the NUTS state. sr_rf (N, 2) is also replicated.
        sr_raw = dist.normal(0, 1, shape=(2, N_id), name='sr_raw', sample=sample)
        sr_sigma = dist.truncated_normal(
            sr_sigma_mu, sr_sigma_sd, low=0, shape=(2,),
            name='sr_sigma', sample=sample,
        )
        sr_L = dist.lkj_cholesky(
            cholesky_dim, cholesky_density, name='sr_L', sample=sample,
        )
        sr_rf = deterministic(
            'sr_rf',
            jnp.transpose(jnp.matmul(sr_sigma[:, None] * sr_L, sr_raw)),
        )  # (N, 2)

        # ── total sender / receiver per node ───────────────────────────────
        # sr_rf[:, 0] is (N,) replicated. When added to sharded s_fixed
        # (N/n_dev,), XLA broadcasts the replicated array to match the shard:
        # each device takes its own rows from sr_rf[:, 0] and adds to s_fixed.
        # No allgather — same pattern as alpha + beta*X_shard in simple models.
        s = s_fixed + sr_rf[:, 0]  # (N,) or (N/n_dev,) if sender sharded
        r = r_fixed + sr_rf[:, 1]  # (N,) replicated

        # ── outer sum — the core change from Neteffect ─────────────────────
        # s[:,None] is (N/n_dev, 1) sharded; r[None,:] is (1, N) replicated.
        # Output: (N/n_dev, N) per device — pure local broadcast, no allgather.
        return s[:, None] + r[None, :]  # (N, N)

    # ------------------------------------------------------------------
    # Dyadic effect — (N, N)
    # ------------------------------------------------------------------
    @staticmethod
    def dyad_index_maps(N):
        """Static row-shardable index maps for the dyadic random-effect gather.

        Returns two (N, N) int arrays (``eid``, ``dir_idx``) such that, for the
        edgelist random effects ``dr_rf_edgl`` (N_dyads, 2),

            dr_rf_edgl[eid, dir_idx]  ==  edgl_to_mat(dr_rf_edgl, N)   (off-diagonal)

        i.e. a per-row GATHER replaces the full-matrix SCATTER. When ``eid`` and
        ``dir_idx`` are sharded along axis 0, each device builds only its
        ``(N/k, N)`` row block from the replicated ``dr_rf_edgl`` — no redundant
        full-matrix construction, no array collective in the forward pass.

        The diagonal maps to (edge 0, dir 0); callers MUST mask the diagonal
        downstream (the SRM always does), exactly as for ``edgl_to_mat`` whose
        diagonal is zero.

        Args:
            N (int): Number of nodes.

        Returns:
            (eid, dir_idx): two (N, N) int32 jax arrays.
        """
        import numpy as _np
        ur, uc = _np.triu_indices(N, 1)
        P = ur.shape[0]
        eid = _np.zeros((N, N), dtype=_np.int32)
        dir_idx = _np.zeros((N, N), dtype=_np.int32)
        pos = _np.arange(P, dtype=_np.int32)
        eid[ur, uc] = pos          # i<j → upper triangle → edgl column 0 (i→j)
        eid[uc, ur] = pos          # i>j → lower triangle → edgl column 1 (j→i)
        dir_idx[uc, ur] = 1
        return jnp.asarray(eid), jnp.asarray(dir_idx)

    @staticmethod
    def dyadic_effect(dyadic_predictors=None, shape=None,
                      d_m=0, d_sd=2.5,
                      dr_sigma_mu=0, dr_sigma_sd=2.5,
                      cholesky_dim=2, cholesky_density=2.5,
                      sample=False, eid=None, dir_idx=None):
        """Dyadic fixed + random effects in (N, N) matrix form.

        Fixed effects:
            dyadic_predictors (N, N, K) @ dyad_effects (K,) → (N, N)
            When dyadic_predictors is sharded along axis 0, tensordot output
            is (N/n_dev, N) per device — genuine data parallelism on the
            dominant N²-scale computation.

        Random effects:
            Sampled as dr_raw (2, N_dyads) in edgelist space to preserve
            the Cholesky reciprocity correlation (i→j correlated with j→i).
            Converted to (N, N) via edgl_to_mat (local scatter-put, no allgather).
            When added to sharded dr_ff_mat, XLA takes the matching row shard
            from the replicated (N, N) random-effect matrix automatically.

        Args:
            dyadic_predictors (jax array): (N, N, K) directed predictor matrices.
                dyadic_predictors[:, :, k] is the k-th dyadic predictor (N×N).
                Shard along axis 0 for data parallelism.
                Alternatively (N, N) for a single predictor.
            shape (int): N_dyads — use when dyadic_predictors is None
                (random effects only, no fixed predictors). Must be exactly
                N*(N-1)/2 for an integer N, else ValueError.
            d_m, d_sd (float): Prior mean/SD for dyadic fixed effects.
            dr_sigma_mu, dr_sigma_sd (float): Half-normal prior for random SD.
            cholesky_dim (int): LKJ dimension (must be 2 for dyadic model).
            cholesky_density (float): LKJ concentration parameter.
            sample (bool): DGP mode if True.

        Returns:
            (N, N) array: logit-scale dyadic contribution.
        """
        if dyadic_predictors is None and shape is None:
            raise ValueError("Either dyadic_predictors or shape must be provided.")

        # ── fixed effects ───────────────────────────────────────────────────
        if dyadic_predictors is not None:
            N = dyadic_predictors.shape[0]
            N_dyads = N * (N - 1) // 2

            if dyadic_predictors.ndim == 3:
                K = dyadic_predictors.shape[2]
                dyad_eff = dist.normal(
                    d_m, d_sd, shape=(K,), name='dyad_effects', sample=sample,
                )
                # tensordot contracts the K axis: (N, N, K) × (K,) → (N, N)
                # When dyadic_predictors is sharded along axis 0:
                #   local shape (N/n_dev, N, K) × (K,) → (N/n_dev, N) per device
                dr_ff_mat = jnp.tensordot(dyadic_predictors, dyad_eff, axes=[[2], [0]])
            else:
                # (N, N) single predictor
                dyad_eff = dist.normal(
                    d_m, d_sd, shape=(1,), name='dyad_effects', sample=sample,
                )
                dr_ff_mat = dyad_eff[0] * dyadic_predictors  # (N, N)
        else:
            N_dyads = shape
            N = int(round((1 + (1 + 8 * shape) ** 0.5) / 2))
            if N * (N - 1) // 2 != shape:
                # RELEASE_AUDIT m8: a non-triangular N_dyads used to round N
                # and die later in an opaque JAX broadcast error.
                raise ValueError(
                    f"dyadic_effect: shape={shape} is not N*(N-1)/2 dyads "
                    "for any integer N; pass the number of upper-triangle "
                    "dyads (1, 3, 6, 10, 15, ...).")
            dr_ff_mat = jnp.zeros((N, N))

        # ── random effects in edgelist space → scatter to matrix ───────────
        # Sampled globally as (2, N_dyads) to preserve the Cholesky reciprocity
        # structure. dr_rf_mat is (N, N) replicated; when added to sharded
        # dr_ff_mat, XLA selects the local row shard automatically.
        dr_raw = dist.normal(0, 1, shape=(2, N_dyads), name='dr_raw', sample=sample)
        dr_sigma = dist.truncated_normal(
            dr_sigma_mu, dr_sigma_sd, low=0, shape=(1,),
            name='dr_sigma', sample=sample,
        )
        dr_L = dist.lkj_cholesky(
            cholesky_dim, cholesky_density, name='dr_L', sample=sample,
        )
        dr_rf_edgl = deterministic(
            'dr_rf',
            jnp.transpose(jnp.matmul(dr_sigma * dr_L, dr_raw)),
        )  # (N_dyads, 2)

        if eid is not None and dir_idx is not None:
            # Row-shardable GATHER: each device builds only its (N/k, N) block by
            # gathering from the replicated dr_rf_edgl. Forward pass is a per-row
            # map (no array collective); the gradient back to dr_rf_edgl is a
            # scatter-add reduced across devices (one collective on the small
            # (N_dyads, 2) array). Replaces the full replicated edgl_to_mat
            # scatter. Diagonal is masked downstream by the caller.
            dr_rf_mat = dr_rf_edgl[eid, dir_idx]  # (N, N) or (N/k, N) if eid sharded
        else:
            # scatter-put: edgelist (N_dyads, 2) → (N, N), full (N, N) per device.
            # edgl_to_mat is JIT-compiled with N as a static arg.
            dr_rf_mat = NeteffectMatrix.edgl_to_mat(dr_rf_edgl, N)  # (N, N) replicated

        return dr_ff_mat + dr_rf_mat  # (N, N), axis-0 sharding from dr_ff_mat

    # ------------------------------------------------------------------
    # Data conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def edgelist_to_matrix_predictors(dyadic_predictors_edgl, N):
        """Convert (N_dyads, 2, K) edgelist predictors → (N, N, K) matrix.

        Applies edgl_to_mat independently for each predictor k, stacking
        the K resulting (N, N) matrices along axis 2.

        Args:
            dyadic_predictors_edgl (jax array): (N_dyads, 2, K) edgelist predictors.
            N (int): Number of nodes.

        Returns:
            (N, N, K) jax array.
        """
        K = dyadic_predictors_edgl.shape[2]
        slices = [
            NeteffectMatrix.edgl_to_mat(dyadic_predictors_edgl[:, :, k], N)
            for k in range(K)
        ]
        return jnp.stack(slices, axis=2)  # (N, N, K)

    @staticmethod
    def edgelist_to_matrix_outcome(network_edgl, N):
        """Convert (N_dyads, 2) edgelist outcome → (N, N) directed adjacency matrix.

        Delegates to edgl_to_mat inherited from array_manip.

        Args:
            network_edgl (jax array): (N_dyads, 2) binary outcome edgelist.
            N (int): Number of nodes.

        Returns:
            (N, N) jax array with 0 on the diagonal.
        """
        return NeteffectMatrix.edgl_to_mat(network_edgl, N)
