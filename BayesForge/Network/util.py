import warnings
from BayesForge.Utils.Gaussian import gaussian
from BayesForge.Utils.Effects import effects
import jax
from jax import jit
import jax.numpy as jnp
import numpy as np

gaussian = gaussian()
effects = effects()

from functools import partial


# vector related functions -----------------------------------
@partial(
    jit,
    static_argnums=(
        1,
        2,
    ),
)
def vec_to_mat_jax(arr, N, K):
    return jnp.tile(arr, (N, K))


# Matrices related functions ------------------------------------------------------------------
def upper_tri(array, diag=1):
    """Extracts the upper triangle elements of a 2D JAX array.

    Args:
        array (2D array): A JAX 2D array.
        diag (int): Integer indicating if diagonal must be kept or not.
                    diag=1 excludes the diagonal, diag=0 includes it.
    """
    upper_triangle_indices = jnp.triu_indices(array.shape[0], k=diag)
    upper_triangle_elements = array[upper_triangle_indices]
    return upper_triangle_elements


# JIT compile the function with static_argnums
get_upper_tri = jit(upper_tri, static_argnums=(1,))


def lower_tri(array, diag=-1):
    """Extracts the lower triangle elements of a 2D JAX array.

    Args:
        array (2D array): A JAX 2D array.
        diag (int): Integer indicating if diagonal must be kept or not.
                    diag=0 includes the diagonal, diag=-1 excludes it.
    """
    lower_triangle_indices = jnp.tril_indices(array.shape[0], k=diag)
    lower_triangle_elements = array[lower_triangle_indices]
    return lower_triangle_elements


# JIT compile the function with static_argnums
get_lower_tri = jit(lower_tri, static_argnums=(1,))


class array_manip:
    """A class for efficient array manipulations using JAX.
    This class provides methods for checking symmetry, converting vectors to matrices, extracting triangle elements, and converting matrices to edge lists.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    @jax.jit
    def is_symmetric(arr, rtol=1e-5, atol=1e-8):
        """
        Efficiently check if a 2D array is symmetric using JIT  compilation.

        Parameters:
        -----------
        arr : jax.numpy.ndarray
            Input 2D array to check for symmetry
        rtol : float, optional
            Relative tolerance for comparison (default: 1e-5)
        atol : float, optional
            Absolute tolerance for comparison (default: 1e-8)

        Returns:
        --------
        bool
            True if the array is symmetric, False otherwise
        """
        # Check if array is 2D
        if arr.ndim != 2:
            return False

        # Check if square matrix
        if arr.shape[0] != arr.shape[1]:
            return False

        # Compare array with its transpose
        return jnp.allclose(arr, arr.T, rtol=rtol, atol=atol)
        # Matrix manipulations -------------------------------------

    @staticmethod
    @partial(jit, static_argnums=(1,))
    def vec_to_mat(vec, shape=()):
        """Convert a vector to a matrix.

        Args:
            vec (1D array): A JAX 1D array.
            shape (tuple): A tuple indicating the shape of the matrix.

        Returns:
            (2D array): return a matrix of shape shape.
        """
        return jnp.tile(vec, shape)

    def get_tri(self, array, type="upper", diag=0):
        """Extracts the upper, lower, or both triangle elements of a 2D JAX array.

        Args:
            array (2D array): A JAX 2D array.
            type (str): A string indicating which part of the triangle to extract.
                        It can be 'upper', 'lower', or 'both'.
            diag (int): Integer indicating if diagonal must be kept or not.
                        diag=1 excludes the diagonal, diag=0 includes it.

        Returns:
            If argument type is 'upper', 'lower', it return a 1D JAX array containing the requested triangle elements.
            If argument type is 'both', it returns a 2D JAX array whose first column holds the upper triangle and whose second column holds the lower triangle (RELEASE_AUDIT m16: the docstring used to state the reverse of the code).
        """
        if diag != 0 and diag != 1:
            raise ValueError("diag must be 0 or 1")
        if type == "upper":
            upper_triangle_indices = jnp.triu_indices(array.shape[0], k=diag)
            triangle_elements = array[upper_triangle_indices]
        elif type == "lower":
            lower_triangle_indices = jnp.tril_indices(array.shape[0], k=-diag)
            triangle_elements = array[lower_triangle_indices]
        elif type == "both":
            upper_triangle_indices = jnp.triu_indices(array.shape[0], k=diag)
            lower_triangle_indices = jnp.tril_indices(array.shape[0], k=-diag)
            upper_triangle_elements = array[upper_triangle_indices]
            lower_triangle_elements = array[lower_triangle_indices]
            triangle_elements = jnp.stack(
                (upper_triangle_elements, lower_triangle_elements), axis=1
            )
        else:
            raise ValueError("type must be 'upper', 'lower', or 'both'")

        return triangle_elements

    @staticmethod
    def mat_to_edgl(mat):
        """Convert a matrix to an edge list.

        Args:
            mat (2D array): A JAX 2D array; must be square (N, N).

        Returns:
            (2D array): return and edgelist of all dyads combination (excluding diagonal).
            First column represent the value fo individual i  in the first column of argument sr, the second column the value of j in the second column of argument sr

        Raises:
            ValueError: If ``mat`` is not a square (N, N) matrix. A non-square
                input would otherwise be silently truncated to ``shape[0]`` rows.
        """
        if not isinstance(mat, jax.core.Tracer):
            mat = jnp.asarray(mat)
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
                raise ValueError(
                    f"mat_to_edgl expects a square (N, N) matrix, got shape {mat.shape}."
                )
            # numpy, not jnp: a jnp op run while a model is traced returns a
            # tracer even on a concrete input, and bool() of it raises.
            if bool(np.any(np.diagonal(np.asarray(mat)) != 0)):
                warnings.warn(
                    "mat_to_edgl: the diagonal (self-ties) is not part of the "
                    "dyad list and is ignored.",
                    UserWarning,
                    stacklevel=2,
                )
        return array_manip._mat_to_edgl_jit(mat)

    @staticmethod
    @jit
    def _mat_to_edgl_jit(mat):
        N = mat.shape[0]
        # From to
        urows, ucols = jnp.triu_indices(N, k=1)
        ft = mat[(urows, ucols)]

        m2 = jnp.transpose(mat)
        tf = m2[(urows, ucols)]
        return jnp.stack([ft, tf], axis=-1)

    @staticmethod
    def edgl_to_mat(edgl, N_id=None):
        """Dyad edge list (n_dyads, 2) -> (N_id, N_id) adjacency.

        ``N_id`` is optional: a full dyad list has n_dyads = N_id*(N_id-1)/2
        rows, so the node count is recovered from ``edgl.shape[0]`` when not
        given.
        """
        if N_id is None:
            import math
            n = int(edgl.shape[0])                       # N*(N-1)/2
            N_id = (1 + math.isqrt(1 + 8 * n)) // 2
            if N_id * (N_id - 1) // 2 != n:
                raise ValueError(
                    f"edgl has {n} rows, not a full dyad list for any N; "
                    "pass N_id explicitly."
                )
        return array_manip._edgl_to_mat(edgl, int(N_id))

    @staticmethod
    @partial(jit, static_argnums=(1,))
    def _edgl_to_mat(edgl, N_id):
        m = jnp.zeros((N_id, N_id))
        urows, ucols = jnp.triu_indices(N_id, 1)
        m = m.at[(ucols, urows)].set(edgl[:, 1])
        m = m.at[(urows, ucols)].set(edgl[:, 0])
        return m

    # RELEASE_AUDIT m16: a commented-out diagonal-removal helper used to sit
    # here; dead code, dropped (jnp.diagonal masking covers that use case).

    @staticmethod
    @jit
    def vec_node_to_edgle(sr):
        """Pair node-level values into dyads (upper triangle, i < j).

        Args:
            sr (2D array): Node values, shape (N, 2): column 0 is the value used
                for the first node of a dyad, column 1 for the second.

        Returns:
            (2D array): Shape (N*(N-1)/2, 2); row (i, j) holds ``sr[i, 0]`` and ``sr[j, 1]``.
        """
        N = sr.shape[0]
        urows, ucols = jnp.triu_indices(N, k=1)
        ft = sr[urows, 0]
        tf = sr[ucols, 1]
        return jnp.stack([ft, tf], axis=-1)

    def vec_to_edgl(self, vec):
        """Convert a vector to an edge list.

        Args:
            vec (1D array): A JAX 1D array.

        Returns:
            (2D array): return and edgelist of all dyads combination (excluding diagonal).
            First column represent the value fo individual i  in the first column of argument sr, the second column the value of j in the second column of argument sr
        """
        N = vec.shape[0]

        urows, ucols = jnp.triu_indices(N, k=1)
        ft = vec[urows]
        tf = vec[ucols]
        return jnp.stack([ft, tf], axis=-1)

    @staticmethod
    @jit
    def to_binary_matrix(m):
        """Binarise a weight matrix: 1 where the weight is positive, 0 elsewhere.

        Args:
            m (jax.Array): Weight matrix.

        Returns:
            jax.Array: Integer matrix of 0s and 1s with the same shape.
        """
        return jnp.where(m > 0, 1, 0)

    # RELEASE_AUDIT m10: dyad_vec_to_mat used to be duplicated here as a
    # staticmethod identical to the module-level function below; the method
    # was removed (no caller in BayesForge or Test used it through the class)
    # and the single remaining definition gained exact length validation.


# RELEASE_AUDIT m10: single definition of dyad_vec_to_mat (module level, the
# one every caller resolves today). Length validation raises ValueError with
# the expected convention instead of a JAX broadcast error.
@partial(jit, static_argnums=(1, 2))
def dyad_vec_to_mat(vec, include_diagonal=False, symmetric=False):
    """Convert dyad vector to N×N matrix. N deduced from vec length.

    Vector length conventions:
        diag=False, sym=True  : N*(N-1)/2  — unique undirected dyads
        diag=False, sym=False : N*(N-1)    — both directions; first half→upper, second half→lower
        diag=True,  sym=True  : N*(N+1)/2  — unique undirected dyads + diagonal
        diag=True,  sym=False : N²         — full matrix, row-major

    Args:
        vec: 1D JAX array
        include_diagonal: vec includes diagonal elements (default False)
        symmetric: produce symmetric matrix (default False)

    Returns:
        (N, N) JAX array

    Raises:
        ValueError: If the vector length does not match the convention chosen
            by the flags for any integer N (previously an opaque JAX
            broadcast error, RELEASE_AUDIT m10).
    """
    n = vec.shape[0]
    if not include_diagonal and symmetric:
        N = int(round((1 + (1 + 8 * n) ** 0.5) / 2))
        if N * (N - 1) // 2 != n:
            raise ValueError(
                f"dyad_vec_to_mat: vector length {n} is not N*(N-1)/2 for any "
                "integer N (include_diagonal=False, symmetric=True); expected "
                "1, 3, 6, 10, 15, ... dyads.")
        rows, cols = jnp.triu_indices(N, k=1)
        mat = jnp.zeros((N, N)).at[rows, cols].set(vec)
        return mat + mat.T
    elif not include_diagonal and not symmetric:
        N = int(round((1 + (1 + 4 * n) ** 0.5) / 2))
        if N * (N - 1) != n:
            raise ValueError(
                f"dyad_vec_to_mat: vector length {n} is not N*(N-1) for any "
                "integer N (include_diagonal=False, symmetric=False); expected "
                "2, 6, 12, 20, ... entries (both directions of every dyad).")
        n_dyads = n // 2
        rows, cols = jnp.triu_indices(N, k=1)
        mat = jnp.zeros((N, N)).at[rows, cols].set(vec[:n_dyads])
        return mat.at[cols, rows].set(vec[n_dyads:])
    elif include_diagonal and symmetric:
        N = int(round((-1 + (1 + 8 * n) ** 0.5) / 2))
        if N * (N + 1) // 2 != n:
            raise ValueError(
                f"dyad_vec_to_mat: vector length {n} is not N*(N+1)/2 for any "
                "integer N (include_diagonal=True, symmetric=True); expected "
                "1, 3, 6, 10, 15, ... entries (dyads + diagonal).")
        rows, cols = jnp.triu_indices(N, k=0)
        mat = jnp.zeros((N, N)).at[rows, cols].set(vec)
        return mat + mat.T - jnp.diag(jnp.diag(mat))
    else:  # include_diagonal=True, symmetric=False — full matrix row-major
        N = int(round(n ** 0.5))
        if N * N != n:
            raise ValueError(
                f"dyad_vec_to_mat: vector length {n} is not a perfect square "
                "N*N (include_diagonal=True, symmetric=False).")
        return vec.reshape(N, N)


# One implementation (RELEASE_AUDIT m10), still reachable as a method:
# m.net.dyad_vec_to_mat(...) / m.net2.dyad_vec_to_mat(...) inherit it.
array_manip.dyad_vec_to_mat = staticmethod(dyad_vec_to_mat)
