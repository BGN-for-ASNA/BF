from BayesForge.Distributions.np_dists import UnifiedDist as dist
from BayesForge.BNN.activations import activation

from numpyro import deterministic
import jax.numpy as jnp
import jax

class bnn(activation):
    """
    The bnn class is designed to build Bayesian Neural Networks (BNNs). It provides methods for creating network layers with specified prior distributions and activation functions. Additionally, it includes a specific two-layer BNN model for covariance estimation and a utility function to compute a correlation matrix from posterior samples.
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
        Adds a layer to the BNN with the specified prior distribution and activation function.

        Parameters:
        - prior_dist (bf.dist): The prior distribution for the weights of the layer. The shape of the distribution defines the layer's input/output dimensions.
        - activation (str): The name of the activation function to use after this layer ('relu', 'tanh', 'sigmoid', 'softmax').
        - verbose (bool): Print the in-development warning (default True).
        """
        if verbose:
            print("⚠️This function is still in development. Use it with caution. ⚠️")
        if bias is not False:
            prod = jnp.matmul(X, dist) + bias
        else:
            prod = jnp.matmul(X, dist) 

        # 2. Get and store the activation function object.
        if activation is None:
            return prod
        else:
            try:
                activation_func = getattr(self, activation)
            except AttributeError:
                raise ValueError(f"Unknown activation function: '{activation}'")    
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
          trained with m.svi() or m.fit().

        Returns:
        - MixtureSameFamily distribution, samples of shape shape + batch + (D,),
          or the value of the sample site when `name` is given.
        """
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

    def layer_attention(self, b_kv, b_q, d_model=32,  sample=True, name = '', seed = None):
        print("⚠️This function is still in development. Use it with caution. ⚠️")
        # Layers
        ### Dimensions 
        self.b_q, self.b_kv, self.d_model = b_q, b_kv, d_model
        ### Create learnable vector embeddings for each feature in the Query  block a
        self.emb_q = self.dist.normal(0,1,shape=(self.b_q,self.d_model), sample=sample, name=f'attention_q_{name}', seed = seed)


        ### Create learnable vector embeddings for each feature in the Key/ Value block 
        self.emb_kv= self.dist.normal(0,1,shape=(self.b_kv,d_model),sample=sample, name=f'attention_kv_{name}', seed = seed)    

        ### Define linear layers to project embeddings into Query, Key, and Value spaces
        self.q_k_v_proj = self.dist.normal(0,1,shape=(self.d_model,self.d_model,3),sample=sample, name=f'attention_q_k_v_{name}', seed = seed)  

        ### Define a final output layer to map the attention context to the desired matrix shape.
        self.out = self.dist.normal(0,1,shape=(self.d_model * self.b_q, self.b_q * self.b_kv),  sample=sample, name=f'attention_out_{name}', seed = seed) 

        # Performs the attention mechanism
        ## three dense (linear) layers : Q = X W_Q,  K = X W_K, V = X W_V
        ## the three projection layers in attention do not have activation functions. They are purely linear transformations.
        Q = self.layer_linear(X = self.emb_q, dist = self.q_k_v_proj[:,:,0])
        K = self.layer_linear(X = self.emb_kv, dist = self.q_k_v_proj[:,:,1])
        V = self.layer_linear(X = self.emb_kv, dist = self.q_k_v_proj[:,:,2])    

        ## Attention mechanism: 
        ### Calculate dot-product similarity scores between all Queries and Keys. Scale for stability.
        scores =jnp.matmul(Q, K.T) / jnp.sqrt(self.d_model ** 0.5)

        #### Convert raw scores into attention weights (probabilities) using softmax.
        attn = jax.nn.softmax(scores, axis=-1)

        ####Compute the context vector as a weighted average of the Value vectors. Now we normalize the similarity scores across all possible `j` for each `i`
        context = jnp.matmul(attn, V)  

        # Reshape and pass through the final output layer to get the L_ij block.
        return self.layer_linear(context.reshape(1, -1), self.out).reshape(self.b_q, self.b_kv) 

    def layer_toeplitz(self, block_size = 32, sample = False, name = '', seed = None):
        print("⚠️This function is still in development. Use it with caution. ⚠️")
        """
        Models a diagonal covariance block C_ii with a diagonal structure.

        This is the simplest structure, assuming all variables in the block are
        uncorrelated with each other. It only learns their individual variances.

        Args:
            block_size (int): The dimension of this square block.
        """
        self.b = block_size
        # Learnable parameter for the base variance (stored in log space for unconstrained optimization).
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
        """
        print("⚠️This function is still in development. Use it with caution. ⚠️")
        # Store the block size.
        self.b = block_size

        # Learnable parameter for the common variance (log space).
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

        # Construct the matrix using its mathematical formula.
        comp_sym = sigma * ((1.0 - rho) * I + rho * ones)
        # Convert log_diag to positive offsets.
        diag = jax.nn.softplus(self.log_diag) + 1e-5

        # Add the diagonal offsets.
        return comp_sym +  diag * jnp.eye(self.b)
 
    def layer_diagonal(self, block_size, sample= True, name='', seed = None):
        """
        Build a Diagonal covariance block.

        Variables are assumed independent; only individual variances are learned.

        Returns a (size, size) SPD diagonal matrix.
        """
        print("⚠️This function is still in development. Use it with caution. ⚠️")
        # Store the block size.
        self.b = block_size
        # Learnable vector of per-variable variances (stored in log space).
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
        """Forward pass through the Bayesian attention mechanism."""
        Q = self.layer(X, self.q_proj)
        K = self.layer(X, self.k_proj)
        V = self.layer(X, self.v_proj)

        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V)
        return attn_output, attn_weights
        
    def cov(self,hidden_dim,N,a, b, sample = False):
        """
        Creates a Bayesian Neural Network (BNN) with two layers for covariance estimation.
        The first layer maps the input to a hidden dimension using a normal distribution,
        and the second layer outputs two values per N (offsets for a and b).
        Parameters:
        - hidden_dim (int): The number of hidden units in the first layer.
        - N (int): The number of data points, which determines the size of the input and output.
        - a (jnp.ndarray): The first set of offsets for the covariance matrix.
        - b (jnp.ndarray): The second set of offsets for the covariance matrix.
        """
        print("⚠️This function is still in development. Use it with caution. ⚠️")
        # First layer weights/biases: note these are treated as latent parameters
        W1 = self.dist.normal(0, 1, shape=(N, hidden_dim), name='W1', sample=sample)

        # Second layer weights/biases
        W2 = self.dist.normal(0, 1, shape=(hidden_dim, 2), name='W2', sample=sample)

        # Create one-hot encoding for each N (each row is a one–hot vector)
        X = jnp.eye(N)

        hidden = jnp.tanh(jnp.dot(X, W1))  # shape: (N, hidden_dim)

        # Second layer: output two values per cafe (offsets for a and b)
        delta = jnp.dot(hidden, W2)        # shape: (N, 2)    

        return deterministic('rf', jnp.stack([a, b]) + delta) 

    def get_rho(self, posterior):
        """

        """
        print("⚠️This function is still in development. Use it with caution. ⚠️")
        a_b = jnp.mean(posterior, axis=0) 
        N= a_b.shape[0]

        # 1. Compute sample covariance matrix
        mean_a_b = jnp.mean(a_b, axis=0)  # Mean of [a_cafe, b_cafe]

        centered_data = a_b - mean_a_b  # Center data by subtracting the mean

        cov_sample = jnp.dot(centered_data.T, centered_data) / (N - 1)  # Covariance matrix

        # 2. Extract sigma (standard deviations) from the diagonal of the covariance matrix
        sigma = jnp.sqrt(jnp.diagonal(cov_sample))  # Extract standard deviations (sqrt of     variance)

        # 3. Compute Rho (correlation matrix)
        rho = cov_sample / (sigma[:, None] * sigma[None, :])  # Normalize covariance to obtain correlation matrix
        return rho

    def make_pd_and_cholesky_jax(self, A):
        A = 0.5 * (A + A.T)
        jitter = 1e-6
        A += jitter * jnp.eye(A.shape[-1])
        return jnp.linalg.cholesky(A)