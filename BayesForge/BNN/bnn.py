from BayesForge.Distributions.np_dists import UnifiedDist as dist
from BayesForge.BNN.activations import activation
from BayesForge.Utils.dev_warning import warn_dev

from numpyro import deterministic
import jax.numpy as jnp
import jax


def _prior_scale(fan_in, prior_scale=None):
    """
    Std for a weight prior (MAJ-5, opt-in).

    prior_scale=None keeps the historical N(0, 1) prior (scale 1.0) so
    existing results do not move. prior_scale="fan_in" gives
    N(0, 1/sqrt(fan_in)) (He/Glorot-style), with fan_in the first dimension
    of the weight matrix. A float is used as the scale directly.
    """
    if prior_scale is None:
        return 1.0
    if prior_scale == "fan_in":
        return float(fan_in) ** -0.5
    return float(prior_scale)


class bnn(activation):
    """
    The bnn class is designed to build Bayesian Neural Networks (BNNs). It provides methods for creating network layers with specified prior distributions and activation functions. Additionally, it includes a specific two-layer BNN model for covariance estimation and a utility function to compute a correlation matrix from posterior samples.

    State
    -----
    layer_attention, layer_toeplitz, layer_compound_symmetry and
    layer_diagonal store the weights they build as instance attributes
    (emb_q, emb_kv, q_k_v_proj, out, log_sigma, raw_alpha, log_diag,
    log_variances, raw_rho, b). These keep only the weights of the LAST layer
    built, for inspection: each call overwrites them and reads them back only
    within that same call, and no other method reads them later. Two models
    sharing m.bnn therefore do not share weights through these attributes
    (RELEASE_AUDIT W13).
    """
    def __init__(self, rand_seed = True):
        super().__init__()

        self.dist = dist(rand_seed)
        # Create the mapping in the constructor
        self._activation_map = {
            # Standard functions
            "relu": self.relu,
            "tanh": self.tanh,
            "sigmoid": self.sigmoid,
            "softmax": self.softmax,
            "linear": self.linear,

            # Advanced/Modern functions
            "leaky_relu": self.leaky_relu,
            "elu": self.elu,
            "gelu": self.gelu,
            "silu": self.silu,
            "swish": self.silu,  # Common alias for SiLU

            # Specialty functions
            "softplus": self.softplus,
            "exp": self.exp,
        }

    def available_activations(self):
        """
        Returns a list of available activation functions.
        
        This method retrieves the names of all activation functions defined in the class.
        """
        return list(self._activation_map.keys())

    def layer_linear(self, X, dist, activation=None, bias=False, verbose=True):
        """
        Adds a layer to the BNN: matmul(X, W) + bias, then an activation.

        Parameters:
        - X (jnp.ndarray): Input array; its last dimension is the layer's input size.
        - dist (jnp.ndarray): The weight matrix W of shape (in, out) -- a
          tensor of weights, not a distribution object (MIN-11: the parameter
          name is kept for backward compatibility). Its shape defines the
          layer's input/output dimensions.
        - activation (str): The name of the activation function to use after this layer ('relu', 'tanh', 'sigmoid', 'softmax').
          'softmax' and 'exp' are meant for the OUTPUT layer only (MIN-7): in a
          hidden layer softmax normalises each row and exp is unbounded, which
          silently loses information or overflows. Nothing enforces it.
        - bias (jnp.ndarray or False): Optional additive bias (default False).
        - verbose (bool): Print the in-development warning (default True).
        """
        if verbose:
            warn_dev("bnn")
        if bias is None:
            raise ValueError("layer_linear: bias=None is not supported; use "
                             "bias=False (the default) for no bias, or pass "
                             "a bias array (MIN-10).")
        # MIN-4: the weight matrix must be (in, out); catch a PyTorch-style
        # (out, in) matrix before the matmul silently transposes it.
        w_shape = getattr(dist, "shape", None)
        if (w_shape is not None and len(w_shape) >= 2 and X.ndim >= 1
                and w_shape[-2] != X.shape[-1]):
            raise ValueError(f"layer_linear: incompatible shapes -- the last "
                             f"dimension of X ({X.shape[-1]}) must match the "
                             f"first dimension of the weight matrix "
                             f"{tuple(w_shape)}.")
        if bias is not False:
            prod = jnp.matmul(X, dist) + bias
        else:
            prod = jnp.matmul(X, dist)

        # 2. Get and store the activation function object (MAJ-8: look it up
        # in _activation_map, not by attribute name, so mapped names such as
        # "swish" resolve to their implementation).
        if activation is None:
            return prod
        else:
            activation_func = self._activation_map.get(activation)
            if activation_func is None:
                raise ValueError(f"Unknown activation function: {activation!r}. "
                                 f"Available: {self.available_activations()}")
            return activation_func(prod)

    def layer_cholesky(self, X, dim, min_diag=1e-3, diag_offset=0.0):
        """
        Maps the last axis of X to lower-triangular Cholesky factors.

        The D(D+1)/2 values of the last axis fill the lower triangle (row-major,
        as jnp.tril_indices); the diagonal is made positive with
        softplus(x + diag_offset) + min_diag, so L @ L.T is always a valid
        covariance matrix. Works on any batch shape, e.g. (n, K, D(D+1)/2) ->
        (n, K, D, D): one factor per row and per mixture component.

        Parameters:
        - X (jnp.ndarray): Input of shape (..., dim * (dim + 1) // 2).
        - dim (int): Dimension D of the factors.
        - min_diag (float): Lower bound added to the diagonal.
        - diag_offset (float): Shift inside the softplus (0.5413 makes the
          diagonal start at 1 when X is 0).

        Returns:
        - jnp.ndarray of shape (..., dim, dim).
        """
        n_tril = dim * (dim + 1) // 2
        if X.shape[-1] != n_tril:
            raise ValueError(f"layer_cholesky: last axis has {X.shape[-1]} values, "
                             f"expected dim*(dim+1)/2 = {n_tril}.")
        rows, cols = jnp.tril_indices(dim)
        L = jnp.zeros(X.shape[:-1] + (dim, dim)).at[..., rows, cols].set(X)
        diag = jnp.arange(dim)
        return L.at[..., diag, diag].set(jax.nn.softplus(L[..., diag, diag] + diag_offset) + min_diag)

    @staticmethod
    def mixture_normal_size(n_components, dim):
        """Number of network outputs needed by layer_mixture_normal: K * (1 + D + D(D+1)/2)."""
        return n_components * (1 + dim + dim * (dim + 1) // 2)

    def layer_mixture_normal(self, X, n_components, dim, min_diag=1e-3, diag_offset=0.0,
                             sample=False, shape=(), seed=None, name=None, obs=None):
        """
        Output layer of a mixture density network.

        Splits the last axis of X into the parameters of a mixture of
        n_components multivariate Normal distributions in dim dimensions:
        K mixture logits, K means of size D, and K Cholesky factors
        (through layer_cholesky). Works on any batch shape of X.

        Parameters:
        - X (jnp.ndarray): Network output of shape (..., mixture_normal_size(K, D)).
        - n_components (int): Number of mixture components K.
        - dim (int): Dimension D of each component.
        - min_diag, diag_offset: Passed to layer_cholesky.
        - sample (bool): If False (default), return the BF distribution object
          (with log_prob / sample). If True, draw `shape` samples from it.
        - shape (tuple): Number of draws when sample=True.
        - seed: Seed for sample=True draws (default: the model's RNG stream).
        - name, obs: Inside a model, declare a sample site `name` (observed at
          `obs` if given), e.g. the likelihood of a mixture density network
          trained with m.svi() or m.fit(). `name` and `obs` require
          sample=False, and `obs` requires `name`: any other combination
          raises a ValueError instead of silently declaring no site (MIN-12).

        Returns:
        - MixtureSameFamily distribution, samples of shape shape + batch + (D,),
          or the value of the sample site when `name` is given.
        """
        if sample and (name is not None or obs is not None):
            raise ValueError("layer_mixture_normal: name/obs require sample=False; "
                             "with sample=True no site is declared and they would "
                             "be ignored (MIN-12).")
        if obs is not None and name is None:
            raise ValueError("layer_mixture_normal: obs requires name; without a "
                             "site name no likelihood is declared (MIN-12).")
        K, D = n_components, dim
        size = self.mixture_normal_size(K, D)
        if X.shape[-1] != size:
            raise ValueError(f"layer_mixture_normal: last axis has {X.shape[-1]} values, "
                             f"expected K*(1+D+D(D+1)/2) = {size}.")
        logits = X[..., :K]
        loc = X[..., K:K + K * D].reshape(X.shape[:-1] + (K, D))
        L = self.layer_cholesky(X[..., K + K * D:].reshape(X.shape[:-1] + (K, D * (D + 1) // 2)),
                                D, min_diag=min_diag, diag_offset=diag_offset)
        mixing = self.dist.categorical(logits=logits, create_obj=True)
        components = self.dist.multivariate_normal(loc, scale_tril=L, create_obj=True)
        if name is not None and not sample:
            return self.dist.mixture_same_family(mixing, components, name=name, obs=obs)
        return self.dist.mixture_same_family(mixing, components, create_obj=not sample,
                                             sample=sample, shape=shape, seed=seed)

    def layer_attention(self, b_kv, b_q, d_model=32,  sample=True, name = '', seed = None,
                        prior_scale=None, weights=None):
        """
        Builds the Bayesian attention layer weights (embeddings and Q/K/V and
        output projections).

        prior_scale (MAJ-5, opt-in): None (default) keeps the historical
        N(0, 1) prior on the output projection weight; "fan_in" uses
        N(0, 1/sqrt(fan_in)) with fan_in = d_model * b_q (first dimension of
        the output weight); a float is used as the scale directly.

        Note (MIN-9): the default sample=True draws the weights from the
        prior and returns fixed values -- inside a numpyro model use
        sample=False so the weights become learnable sample sites.

        weights (MIN-6): optional 4-tuple (emb_q, emb_kv, q_k_v_proj, out)
        of pre-computed weight arrays; when given, no draw happens (so the
        call is safe inside vmap/scan, where the seed would be a tracer)
        and the arrays may carry a leading batch dimension -- the block
        then returns shape batch + (b_q, b_kv).
        """
        warn_dev("bnn")
        # Layers
        ### Dimensions 
        self.b_q, self.b_kv, self.d_model = b_q, b_kv, d_model
        if weights is None:
            ### Create learnable vector embeddings for each feature in the Query  block a
            self.emb_q = self.dist.normal(0,1,shape=(self.b_q,self.d_model), sample=sample, name=f'attention_q_{name}', seed = seed)


            ### Create learnable vector embeddings for each feature in the Key/ Value block 
            self.emb_kv= self.dist.normal(0,1,shape=(self.b_kv,d_model),sample=sample, name=f'attention_kv_{name}', seed = seed)    

            ### Define linear layers to project embeddings into Query, Key, and Value spaces
            self.q_k_v_proj = self.dist.normal(0,1,shape=(self.d_model,self.d_model,3),sample=sample, name=f'attention_q_k_v_{name}', seed = seed)  

            ### Define a final output layer to map the attention context to the desired matrix shape.
            self.out = self.dist.normal(0, _prior_scale(self.d_model * self.b_q, prior_scale), shape=(self.d_model * self.b_q, self.b_q * self.b_kv),  sample=sample, name=f'attention_out_{name}', seed = seed) 
        else:
            # MIN-6: explicit weights skip the draws, so the block can run
            # inside vmap/scan without a tracer seed reaching self.dist.
            self.emb_q, self.emb_kv, self.q_k_v_proj, self.out = weights

        # Performs the attention mechanism
        ## three dense (linear) layers : Q = X W_Q,  K = X W_K, V = X W_V
        ## the three projection layers in attention do not have activation functions. They are purely linear transformations.
        Q = self.layer_linear(X = self.emb_q, dist = self.q_k_v_proj[..., 0])
        K = self.layer_linear(X = self.emb_kv, dist = self.q_k_v_proj[..., 1])
        V = self.layer_linear(X = self.emb_kv, dist = self.q_k_v_proj[..., 2])    

        ## Attention mechanism: 
        ### Calculate dot-product similarity scores between all Queries and Keys. Scale for stability.
        scores =jnp.matmul(Q, jnp.swapaxes(K, -1, -2)) / jnp.sqrt(self.d_model)

        #### Convert raw scores into attention weights (probabilities) using softmax.
        attn = jax.nn.softmax(scores, axis=-1)

        ####Compute the context vector as a weighted average of the Value vectors. Now we normalize the similarity scores across all possible `j` for each `i`
        context = jnp.matmul(attn, V)  

        # Reshape and pass through the final output layer to get the L_ij
        # block. MIN-6: keep the leading batch dimensions instead of forcing
        # a batch of 1, so batched or vmapped calls reshape correctly.
        batch = context.shape[:-2]
        out = self.layer_linear(context.reshape(batch + (-1,)), self.out)
        return out.reshape(batch + (self.b_q, self.b_kv))

    def layer_toeplitz(self, block_size = 32, sample = False, name = '', seed = None):
        """
        Models a diagonal covariance block C_ii with a diagonal structure.

        This is the simplest structure, assuming all variables in the block are
        uncorrelated with each other. It only learns their individual variances.

        Args:
            block_size (int): The dimension of this square block.
        """
        warn_dev("bnn")
        self.b = block_size
        # Learnable parameter for the base variance (raw draw; the softplus
        # below maps it to a positive variance, MIN-3).
        self.log_sigma = self.dist.log_normal(1,sample=sample, name = f'toeplitz_log_sigma_{name}', seed = seed)
        # Learnable parameter controlling correlation decay.
        self.raw_alpha = self.dist.normal(0.5,1,sample=sample, name = f'toeplitz_raw_alpha_{name}', seed = seed)
        # Learnable parameter for the diagonal adjustment.
        self.log_diag = self.dist.normal(0,1,sample=sample, name = f'toeplitz_log_diag_{name}', seed = seed)

        ## Constructs the Toeplitz matrix from the learned parameters.
        sigma, alpha = jax.nn.softplus(self.log_sigma) + 1e-6, jax.nn.sigmoid(self.raw_alpha)
        idx = jnp.arange(self.b)

        # Create the first row of the Toeplitz matrix: [sigma, sigma*alpha, sigma*alpha^2, ...].
        toeplitz_row = sigma * (alpha ** idx)
        indices = jnp.abs(idx[:, None] - idx[None, :])
        # Build the full Toeplitz matrix by indexing the first row with the lag matrix.
        toeplitz = toeplitz_row[indices]
        diag = jax.nn.softplus(self.log_diag) + 1e-5
        result = toeplitz + diag * jnp.eye(self.b)
        return result

    def layer_compound_symmetry(self, block_size, sample= True, name='', seed = None):
        """
        Build a Compound Symmetry covariance block.

        All variables share a common variance (sigma^2) and correlation (rho).
        C = sigma^2 * [(1 - rho) I + rho 11^T]

        Returns a (size, size) SPD matrix.

        Note (MIN-9): the default sample=True draws the parameters from the
        prior and returns fixed values -- inside a numpyro model use
        sample=False so they become learnable sample sites.
        """
        warn_dev("bnn")
        # Store the block size.
        self.b = block_size

        # Learnable parameter for the common variance (raw draw; the softplus
        # below maps it to a positive sigma, MIN-3).
        self.log_sigma = self.dist.log_normal(1, sample = sample, seed = seed, name = f'compound_symmetry_log_sigma_{name}')

        # Learnable parameter for the common correlation (raw, to be mapped to valid range).
        self.raw_rho = self.dist.normal(0, 1, sample = sample, seed = seed, name = f'compound_symmetry_raw_rho_{name}')

        # Learnable diagonal offsets.
        self.log_diag = self.dist.normal(0, 1, shape = (self.b,), sample = sample, seed = seed, name = f'compound_symmetry_log_diag_{name}')

        # Constructs the compound symmetry matrix from the learned parameters.
        # Convert log_sigma to positive sigma.
        sigma = jax.nn.softplus(self.log_sigma) + 1e-6
        # Define the mathematically valid range for rho to ensure the matrix is PD.
        low = -1.0 / (self.b - 1.0) + 1e-6 if self.b > 1 else 0.0
        high = 0.999

        # Map the output of sigmoid (0, 1) to the valid range (low, high).
        rho = low + jax.nn.sigmoid(self.raw_rho) * (high - low)
        # Create identity and all-ones matrices as building blocks.
        I = jnp.eye(self.b)
        ones = jnp.ones((self.b, self.b))

        # Construct the matrix using its mathematical formula (MAJ-4: the
        # common variance is sigma**2, matching the docstring C = sigma^2 *
        # [(1 - rho) I + rho 11^T]).
        comp_sym = sigma**2 * ((1.0 - rho) * I + rho * ones)
        # Convert log_diag to positive offsets.
        diag = jax.nn.softplus(self.log_diag) + 1e-5

        # Add the diagonal offsets.
        return comp_sym +  diag * jnp.eye(self.b)
 
    def layer_diagonal(self, block_size, sample= True, name='', seed = None):
        """
        Build a Diagonal covariance block.

        Variables are assumed independent; only individual variances are learned.

        Returns a (size, size) SPD diagonal matrix.

        Note (MIN-9): the default sample=True draws the variances from the
        prior and returns a fixed matrix -- inside a numpyro model use
        sample=False so they become learnable sample sites.
        """
        warn_dev("bnn")
        # MIN-16: block_size is used directly; the old block-size attribute
        # store was a copy-paste leftover from
        # layer_toeplitz/layer_compound_symmetry (which still need it).
        # Learnable vector of per-variable variances (raw draws; the softplus
        # below maps them to positive variances, MIN-3).
        self.log_variances = self.dist.normal(0, 1, sample = sample, shape = (block_size,), name = f"log_variances_{name}",seed=seed)

        #Constructs the diagonal matrix from the learned variances
        # Convert log-variances to positive variances.
        variances = jax.nn.softplus(self.log_variances) + 1e-6
        # Create a diagonal matrix from the variances vector.
        return jnp.diag(variances)

    def scaled_dot_product_attention(self, Q, K, V):
        """Compute scaled dot-product attention."""
        d_k = Q.shape[-1]
        scores = jnp.matmul(Q, K.T) / jnp.sqrt(d_k)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        return jnp.matmul(attn_weights, V), attn_weights

    def __call__(self, X):
        """Forward pass through the Bayesian attention mechanism.

        Uses the Q/K/V projection weights drawn by the last ``layer_attention``
        call (``self.q_k_v_proj``, shape (d_model, d_model, 3)); X has shape
        (..., d_model).
        """
        if getattr(self, "q_k_v_proj", None) is None:
            raise RuntimeError("bnn.__call__ needs projection weights: call layer_attention(...) first.")
        Q = self.layer_linear(X=X, dist=self.q_k_v_proj[:, :, 0])
        K = self.layer_linear(X=X, dist=self.q_k_v_proj[:, :, 1])
        V = self.layer_linear(X=X, dist=self.q_k_v_proj[:, :, 2])

        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)
        return attn_output, attn_weights
        
    def cov(self,hidden_dim,N,a, b, sample = False, prior_scale=None, prefix=''):
        """
        Creates a Bayesian Neural Network (BNN) with two layers for covariance estimation.
        The first layer maps the input to a hidden dimension using a normal distribution,
        and the second layer outputs two values per N (offsets for a and b).
        Parameters:
        - hidden_dim (int): The number of hidden units in the first layer.
        - N (int): The number of data points, which determines the size of the input and output.
        - a (jnp.ndarray): The first set of offsets for the covariance matrix.
        - b (jnp.ndarray): The second set of offsets for the covariance matrix.
        - prior_scale (MAJ-5, opt-in): None (default) keeps the historical
          N(0, 1) prior on W1 and W2; "fan_in" uses N(0, 1/sqrt(fan_in)) with
          fan_in the first dimension of each weight (N for W1, hidden_dim for
          W2); a float is used as the scale directly.
        - prefix (str): Namespace for the numpyro site names (MIN-8). With
          the default '' the historical names 'W1', 'W2' and 'rf' are kept;
          a non-empty prefix declares '<prefix>/W1', '<prefix>/W2' and
          '<prefix>/rf', so cov() can be called twice in one model.
        """
        warn_dev("bnn")
        # First layer weights/biases: note these are treated as latent parameters
        W1 = self.dist.normal(0, _prior_scale(N, prior_scale), shape=(N, hidden_dim), name=f'{prefix}/W1' if prefix else 'W1', sample=sample)

        # Second layer weights/biases
        W2 = self.dist.normal(0, _prior_scale(hidden_dim, prior_scale), shape=(hidden_dim, 2), name=f'{prefix}/W2' if prefix else 'W2', sample=sample)

        # MIN-5: eye(N) @ W1 == W1, so the dense N x N one-hot input matrix
        # is skipped entirely (it cost O(N^2) memory for an identity product).
        hidden = jnp.tanh(W1)  # shape: (N, hidden_dim)

        # Second layer: output two values per cafe (offsets for a and b)
        delta = jnp.dot(hidden, W2)        # shape: (N, 2)    

        return deterministic(f'{prefix}/rf' if prefix else 'rf', jnp.stack([jnp.broadcast_to(a, (N,)), jnp.broadcast_to(b, (N,))], axis=-1) + delta)

    def get_rho(self, posterior):
        """
        Correlation matrix rho of the (a, b) offsets from posterior samples.

        rho is the 2x2 Pearson correlation between the a and b columns of
        the posterior means: for a 3-D input of shape (draws, N, 2) the
        draws are averaged first (axis 0), so rho describes the correlation
        of the per-observation mean offsets, not the mean of the per-draw
        correlations (MIN-14: this distinction was undocumented). A 2-D
        input (N, 2) is used as is. Returns a (2, 2) array.

        Guards (MAJ-6): raises ValueError when N < 2 or the last dimension
        is not 2 (previously this divided by N - 1 = 0 or indexed the wrong
        axis and returned NaN silently), and when a column has zero or
        non-finite variance -- rho is undefined there, so no NaN/Inf is
        ever returned.
        """
        warn_dev("bnn")
        posterior = jnp.asarray(posterior)
        if posterior.ndim == 3:
            a_b = jnp.mean(posterior, axis=0)
        elif posterior.ndim == 2:
            a_b = posterior
        else:
            raise ValueError(f"get_rho: expected posterior of shape (draws, N, 2) "
                             f"or (N, 2), got {posterior.shape}.")
        if a_b.ndim != 2 or a_b.shape[-1] != 2:
            raise ValueError(f"get_rho: expected posterior of shape (draws, N, 2) "
                             f"or (N, 2), got {posterior.shape}.")
        N = a_b.shape[0]
        if N < 2:
            raise ValueError(f"get_rho: need N >= 2 observations to estimate a "
                             f"correlation, got N = {N} (shape {posterior.shape}).")

        # 1. Compute sample covariance matrix
        mean_a_b = jnp.mean(a_b, axis=0)  # Mean of [a_cafe, b_cafe]

        centered_data = a_b - mean_a_b  # Center data by subtracting the mean

        cov_sample = jnp.dot(centered_data.T, centered_data) / (N - 1)  # Covariance matrix

        # 2. Extract sigma (standard deviations) from the diagonal of the covariance matrix
        sigma = jnp.sqrt(jnp.diagonal(cov_sample))  # Extract standard deviations (sqrt of     variance)
        if jnp.any(sigma <= 0) or not jnp.all(jnp.isfinite(sigma)):
            raise ValueError("get_rho: a column has zero variance; correlation is undefined")

        # 3. Compute Rho (correlation matrix)
        rho = cov_sample / (sigma[:, None] * sigma[None, :])  # Normalize covariance to obtain correlation matrix
        return rho

    def make_pd_and_cholesky_jax(self, A):
        """Cholesky factor of a symmetrised matrix with escalating jitter.

        MIN-2: a fixed jitter of 1e-6 is not enough for a badly conditioned
        matrix (e.g. a correlated posterior correlation matrix). The jitter
        is multiplied by 10 (1e-6, 1e-5, ..., up to 1e-2) until the
        factorisation is finite; if the matrix is still not positive definite
        at jitter 1e-2, a ValueError is raised instead of silently returning
        a non-finite factor.
        """
        A = 0.5 * (A + A.T)
        n = A.shape[-1]
        jitter = 1e-6
        if isinstance(A, jax.core.Tracer):
            # Under jit/vmap/a model trace the finiteness test cannot be
            # evaluated in Python: escalate with jnp.where instead of raising.
            L = jnp.linalg.cholesky(A + jitter * jnp.eye(n))
            for _ in range(4):
                jitter *= 10
                L = jnp.where(jnp.all(jnp.isfinite(L)), L,
                              jnp.linalg.cholesky(A + jitter * jnp.eye(n)))
            return L
        for _ in range(5):  # tries 1e-6, 1e-5, 1e-4, 1e-3, 1e-2
            try:
                L = jnp.linalg.cholesky(A + jitter * jnp.eye(n))
                if jnp.all(jnp.isfinite(L)):
                    return L
            except Exception:
                pass
            jitter *= 10
        raise ValueError("make_pd_and_cholesky_jax: matrix is not positive "
                         "definite even after a jitter of 1e-2; check the "
                         "input matrix.")