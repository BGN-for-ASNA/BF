import warnings

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax import vmap
from functools import partial
from typing import Optional, Tuple


def _is_symmetric(m):
    """True when the concrete square matrix ``m`` equals its transpose.

    Evaluates eagerly (outside any jit trace); returns False for non-square
    or 1x1 matrices, where the in/out double-counting warning is moot.
    """
    a = np.asarray(m)  # numpy: jnp ops are traced inside a model trace
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] <= 1:
        return False
    return bool(np.all(a == a.T))


def _is_concrete(x):
    """True when ``x`` is a real array rather than a JIT trace.

    Python-level validation (``raise ValueError``) is only possible outside a
    trace: inside one the input is a ``jax.core.Tracer`` and the condition
    cannot be evaluated, so the guards built on this helper are skipped and
    the traced computation is returned unchanged.
    """
    return isinstance(x, np.ndarray) or not isinstance(x, jax.core.Tracer)


def _path_count_dtype():
    """Accumulator dtype for shortest-path counts (RELEASE_AUDIT Network m1).

    Brandes' counting step sums one term per shortest path, so ``sigma`` and
    ``delta`` grow with the number of paths: on dense graphs that count
    exceeds float32's range (3.4e38) around N ~ 30-40 in the worst case and
    silently turns the result into inf/NaN. The counts are therefore float64
    whenever ``jax_enable_x64`` is on. With x64 off (the default) they stay
    float32 and the overflow risk remains; see ``met.betweenness``.
    """
    return jnp.float64 if jax.config.jax_enable_x64 else jnp.float32


class met:
    """Network metrics class for computing various graph metrics using JAX.
    This class provides methods to compute clustering coefficients, eigenvector centrality, Dijkstra's algorithm for shortest paths, and other network metrics. 
    It leverages JAX's capabilities for efficient computation on large graphs.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    @partial(jit, static_argnames=("total",))
    def normalize(x, m, total=False):
        """Divide a node-level measure by the maximum possible degree.

        Args:
            x (jax.Array): Node-level values, shape (N,).
            m (jax.Array): The network's adjacency matrix, shape (N, N).
            total (bool): ``x`` is a directed total (in + out) degree, whose
                maximum is ``2 (N - 1)``; otherwise the maximum is ``N - 1``.

        Returns:
            jax.Array: ``x / (N - 1)``, or ``x / (2 (N - 1))`` when
            ``total=True``; zeros when ``N <= 1`` (never NaN).
        """
        n = m.shape[0]
        if n <= 1:
            return jnp.zeros_like(x)
        max_degree = 2 * (n - 1) if total else (n - 1)
        return x / max_degree

        
    # Network utils
    # Nodal measures----------------------------------------------------------------------------------
    ## Clustering_coefficient----------------------------------------------------------------------------------
    @staticmethod 
    @jax.jit
    def triangles_and_degree(adj_matrix):
        """
        Computes the number of triangles and the degree for each node in    a graph.
        Self-loops (the diagonal) are ignored and an asymmetric matrix is
        symmetrised as ``(A + A.T) > 0``, matching NetworkX's undirected
        clustering semantics.
        This function is optimized for JAX and is JIT-compatible.
        """
        # Ensure the adjacency matrix is boolean for logical operations,
        # without self-loops, and symmetrised for the undirected formula.
        no_self_loop = ~jnp.eye(adj_matrix.shape[0], dtype=bool)
        adj_matrix_bool = (adj_matrix > 0) & no_self_loop
        adj_matrix_bool = adj_matrix_bool | adj_matrix_bool.T

        # Calculate the degree of each node
        degrees = jnp.sum(adj_matrix_bool, axis=1)

        # Compute the number of triangles for each node
        # This is equivalent to the diagonal of the cube of the     adjacency matrix
        num_triangles = jnp.diag(jnp.linalg.matrix_power(adj_matrix_bool.   astype(jnp.int32), 3)) / 2

        return degrees, num_triangles

    @staticmethod
    def clustering_coefficient(adj_matrix):
        """
        Computes the clustering coefficient for each node in the graph.

        Self-loops are ignored and the matrix is binarized (``A > 0``). A
        symmetric matrix gets the undirected coefficient; a non-symmetric one
        gets the directed coefficient of Fagiolo (2007), as
        ``nx.clustering`` on a ``DiGraph``. Both agree on a symmetric matrix.
        The result is in [0, 1]. The core computations are JIT-compatible
        (``met._clustering_coefficient_jit``,
        ``met._directed_clustering_coefficient_jit``).
        """
        a = jnp.asarray(adj_matrix)
        if a.ndim == 2 and a.shape[0] == a.shape[1] and not bool(jnp.all(a == a.T)):
            return met._directed_clustering_coefficient_jit(a)
        return met._clustering_coefficient_jit(adj_matrix)

    @staticmethod
    @jax.jit
    def _directed_clustering_coefficient_jit(adj_matrix):
        """
        Directed (binary) clustering coefficient, Fagiolo (2007).

        c_i = (S^3)_ii / (2 * (d_tot_i * (d_tot_i - 1) - 2 * d_bi_i)), with
        B = (A > 0) without self-loops, S = B + B.T, d_tot = in + out degree
        and d_bi = (B @ B)_ii the number of reciprocated ties. It equals
        ``nx.clustering(G)`` for a ``nx.DiGraph`` G, and the undirected
        coefficient when B is symmetric.
        """
        a = jnp.asarray(adj_matrix)
        b = (a > 0).astype(jnp.result_type(a.dtype, float))
        b = b * (1 - jnp.eye(b.shape[0], dtype=b.dtype))
        s = b + b.T
        triangles = jnp.diagonal(s @ s @ s)
        d_tot = b.sum(axis=0) + b.sum(axis=1)
        d_bi = jnp.diagonal(b @ b)
        denominator = 2 * (d_tot * (d_tot - 1) - 2 * d_bi)
        return jnp.where(denominator > 0, triangles / jnp.where(denominator > 0, denominator, 1), 0)

    @staticmethod
    @jax.jit
    def _clustering_coefficient_jit(adj_matrix):
        """
        Computes the clustering coefficient for each node in the graph.
        This function is optimized for JAX and is JIT-compatible.
        """
        degrees, num_triangles = met.triangles_and_degree(adj_matrix)

        # To avoid division by zero for nodes with degree less than 2,
        # we calculate the denominator and use jnp.where to handle these    cases.
        denominator = degrees * (degrees - 1)

        # The clustering coefficient is set to 0 where the denominator  is 0.
        clusterc = jnp.where(denominator > 0, 2 * num_triangles /   denominator, 0)

        return clusterc

    @staticmethod
    def onnela_clustering_coefficient(mat):
        """
        Weighted clustering coefficient (Onnela et al., 2005).

        This is the definition NetworkX computes via
        ``nx.clustering(G, weight=...)``. Unlike ``clustering_coefficient``
        (which binarizes the input), it uses the geometric mean of each
        triangle's edge weights, so weak edges contribute weakly:

            w_hat = w / max(w)                            # scale by max weight
            tri_i = sum_{j,k} (w_hat_ij w_hat_jk w_hat_ki) ** (1/3)
                  = diag( cbrt(w_hat) @ cbrt(w_hat) @ cbrt(w_hat) )_i
            cc_i  = tri_i / (deg_i * (deg_i - 1))         # deg = unweighted degree

        ``w`` is ``mat`` with its diagonal zeroed: a self-loop is not part of
        any triangle, so it is dropped before the maximum weight, the
        geometric mean and the degree are taken (it used to inflate ``max_w``
        and the einsum, which pulled a hub of a complete triangle down to
        0.46 instead of 1.0). The result is bounded in [0, 1] and reduces to
        the unweighted clustering coefficient when all edge weights are
        equal, matching NetworkX.

        The degree is a row sum, so an asymmetric matrix is read as directed
        (out-edges) and the undirected formula is applied to it as given.

        Args:
            mat (jax.Array): Weight matrix, shape (N, N); entry [i, j] is the
                weight of the tie from i to j. The diagonal is ignored.

        Returns:
            jax.Array: Weighted clustering coefficient per node, in [0, 1].

        Raises:
            ValueError: If a concrete (non-traced) matrix contains negative
                weights: the cube root of a negative weight is not a real
                triangle intensity, and the coefficient would silently come
                out as 0.
        """
        if _is_concrete(mat):
            a = np.asarray(mat)  # numpy: jnp ops are traced inside a model trace
            if a.ndim == 2 and a.shape[0] == a.shape[1] and bool(np.any(a < 0)):
                raise ValueError(
                    "onnela_clustering_coefficient: negative weights are not "
                    "supported (the geometric mean of a negative weight is not "
                    "a real triangle intensity, the coefficient would be 0); "
                    "shift the weights to be non-negative or drop those ties."
                )
        return met._onnela_clustering_coefficient_jit(mat)

    @staticmethod
    @jax.jit
    def _onnela_clustering_coefficient_jit(mat):
        """
        Computes the weighted (Onnela) clustering coefficient for each node.

        JIT core of ``met.onnela_clustering_coefficient``, without the input
        validation. This function is optimized for JAX and is JIT-compatible.
        """
        w = jnp.asarray(mat)
        n = w.shape[0]
        # Self-loops carry no triangle: zero the diagonal first so it stays
        # out of max_w, of the einsum below and of the degree.
        w = w * (1.0 - jnp.eye(n))

        max_w = jnp.max(w)
        max_w = jnp.where(max_w > 0, max_w, 1.0)
        w_cbrt = jnp.cbrt(w / max_w)

        # Triangle intensity = diagonal of (cbrt(w_hat))^3; the zero diagonal
        # means only genuine i-j-k triangles contribute.
        tri = jnp.einsum("ij,jk,ki->i", w_cbrt, w_cbrt, w_cbrt)

        deg = jnp.sum(w > 0, axis=1)
        denom = deg * (deg - 1)
        return jnp.where(denom > 0, tri / denom, 0.0)

    @staticmethod
    def cc(m, nodes=None, weighted=False):
        """Local clustering coefficient of each node.

        Args:
            m (jax.Array): Adjacency (or weight) matrix, shape (N, N); entry [i, j] is the tie from i to j.
            nodes (array of int, optional): Return only these nodes (default: all).
            weighted (bool): Use the weighted (Onnela et al. 2005) coefficient
                instead of the binary one. Defaults to False.

        Returns:
            jax.Array: Clustering coefficient per node, in [0, 1].
        """
        out = met.onnela_clustering_coefficient(m) if weighted else met.clustering_coefficient(m)
        return out if nodes is None else out[jnp.asarray(nodes)]

    @staticmethod
    def weighted_cc(m, nodes=None):
        """Weighted local clustering coefficient (Onnela et al. 2005) of each node.

        Same as ``cc(m, nodes, weighted=True)``.

        Args:
            m (jax.Array): Weight matrix, shape (N, N).
            nodes (array of int, optional): Return only these nodes (default: all).

        Returns:
            jax.Array: Weighted clustering coefficient per node, in [0, 1].
        """
        return met.cc(m, nodes=nodes, weighted=True)

    ## eigenvector----------------------------------------------------------------------------------
    @staticmethod
    @partial(jit, static_argnames=['max_iter', 'tol', 'component_aware'])
    def eigenvector(A, max_iter=100, tol=1.0e-6, component_aware=False):
        """
        Eigenvector centrality by power iteration, matching NetworkX behavior.
        
        Computes eigenvector centrality using power iteration on (A.T + I).
        The transpose matches NetworkX's eigenvector_centrality for directed
        graphs, where a node's centrality comes from its in-edges; for
        symmetric (undirected) matrices A.T and A are identical.
        For weighted graphs, edge weights are used in the computation.
        
        Args:
            A: Adjacency matrix (n x n), can be weighted
            max_iter: Maximum number of iterations
            tol: Convergence tolerance; the iteration stops when the L1
                change of the vector falls below n * tol (or max_iter is
                reached)
            component_aware: If True, run the power iteration per connected
                component (weakly connected components for directed
                matrices) and normalise each component's vector to unit L2
                norm within the component, so every component keeps a
                non-zero ranking. Default False matches NetworkX standard
                behavior (a single iteration on the full graph; on
                disconnected graphs the dominant component wins).
            
        Returns:
            Eigenvector centrality values (L2 normalized over the whole
            graph, or per component when component_aware=True)
        """
        A = jnp.asarray(A)
        dtype = jnp.result_type(A.dtype, float)
        A = A.astype(dtype)
        n = A.shape[0]
        
        if n == 0:
            return jnp.array([], dtype=dtype)
        
        # NetworkX uses (I + A) @ x for the iteration, and the transpose
        # (I + A.T) @ x for directed graphs (in-edges); A.T == A when
        # symmetric, so the transpose is always safe.
        A_modified = A.T + jnp.eye(n, dtype=dtype)
        
        if component_aware:
            # Label the (weakly) connected components by max-label
            # propagation over the boolean adjacency (BFS-style, at most n
            # rounds), then normalise the vector per component.
            adjacency_bool = jnp.logical_or(A != 0, A.T != 0) | jnp.eye(n, dtype=bool)
            
            def label_body(state):
                labels, i, changed = state
                neighbor_max = jnp.max(jnp.where(adjacency_bool, labels[None, :], -1), axis=1)
                new_labels = jnp.maximum(labels, neighbor_max)
                changed = jnp.any(new_labels != labels)
                return new_labels, i + 1, changed
            
            labels, _, _ = lax.while_loop(
                lambda state: (state[1] < n) & state[2],
                label_body,
                (jnp.arange(n), jnp.int32(0), True),
            )
            component_sizes = jnp.bincount(labels, length=n)[labels]
            
            def normalize_fn(x):
                sumsq = jnp.zeros(n, dtype=x.dtype).at[labels].add(x * x)
                norm = jnp.sqrt(sumsq[labels])
                return jnp.where(norm > 0, x / jnp.where(norm > 0, norm, 1.0), x)
            
            # Uniform within each component (unit L2 norm per component).
            x = jnp.ones(n, dtype=dtype) / jnp.sqrt(component_sizes.astype(dtype))
        else:
            def normalize_fn(x):
                # Normalize by L2 norm
                norm = jnp.sqrt(jnp.sum(x * x))
                norm = jnp.where(norm > 0, norm, 1.0)
                return x / norm
            
            # Initialize with uniform vector (sum = 1)
            x = jnp.ones(n, dtype=dtype) / n
        
        def cond_fn(state):
            _, i, l1_change = state
            return (i < max_iter) & (l1_change >= n * tol)
        
        def body_fn(state):
            x, i, _ = state
            # Power iteration: x_new = (I + A.T) @ x
            x_new = normalize_fn(A_modified @ x)
            l1_change = jnp.sum(jnp.abs(x_new - x))
            return x_new, i + 1, l1_change
        
        # Iterate until the L1 change drops below n * tol, or max_iter.
        x_final, _, _ = lax.while_loop(
            cond_fn, body_fn, (x, jnp.int32(0), jnp.asarray(jnp.inf, dtype=dtype))
        )
        
        # Final normalization
        x_final = normalize_fn(x_final)
        
        return x_final

    ## Dijkstra----------------------------------------------------------------------------------
    @staticmethod 
    @jit
    def dijkstra_jax(adjacency_matrix, source):
        """
        Compute the shortest path from a source node to all other nodes using Dijkstra's algorithm.

        Semantics (RELEASE_AUDIT Network m5): entries are edge *lengths*
        (distances, larger = farther), not strengths; a neighbour is any entry
        ``> 0``, so zero and negative entries both mean "no edge"; rows are
        read as out-edges, hence an asymmetric matrix is a directed graph.
        Siblings with different semantics: ``met.single_source_dijkstra``
        (same length reading, masks ``> 0`` to inf up front and is the driver
        behind ``met.geodesic_distance``/``met.diameter``) and
        ``met._dijkstra_betweenness`` (presence from the adjacency matrix,
        lengths from a separate weight matrix, plus the path counts).
        See ``met.dijkstra`` for the public wrapper.

        Dijkstra's algorithm finds the shortest paths between nodes in a graph, particularly useful
        for graphs with non-negative edge weights. This function uses JAX for efficient computation.

        Parameters:
        -----------
        adjacency_matrix : jax.numpy.ndarray
            A square (n x n) adjacency matrix representing the graph. The element at (i, j)
            represents the weight of the edge from node i to node j. Non-zero values indicate
            a connection, and higher values indicate longer paths.

        source : int
            The index of the source node from which the shortest paths are computed.

        Returns:
        --------
        jax.numpy.ndarray
            A 1D array of length n where each element represents the shortest distance from the
            source node to the corresponding node. The source node will have a distance of 0.

        """
        n = adjacency_matrix.shape[0]
        visited = jnp.zeros(n, dtype=bool)
        dist = jnp.inf * jnp.ones(n)
        dist = dist.at[source].set(0)

        def body_fn(carry):
            visited, dist = carry

            # Find the next node to process
            u = jnp.argmin(jnp.where(visited, jnp.inf, dist))
            visited = visited.at[u].set(True)

            # Update distances to all neighbors
            def update_dist(v, dist):
                return jax.lax.cond(
                    jnp.logical_and(jnp.logical_not(visited[v]), adjacency_matrix[u, v] > 0),
                    lambda _: jnp.minimum(dist[v], dist[u] + adjacency_matrix[u, v]),
                    lambda _: dist[v],
                    None
                )

            dist = lax.fori_loop(0, n, lambda v, dist: dist.at[v].set(update_dist(v, dist)), dist)

            return visited, dist

        def cond_fn(carry):
            visited, _ = carry
            return jnp.any(jnp.logical_not(visited))

        # Loop until all nodes are visited
        visited, dist_final = lax.while_loop(cond_fn, body_fn, (visited, dist))

        return dist_final

    @staticmethod 
    def dijkstra(m,  source):
        return met.dijkstra_jax(m, source)
    

    ## Strength----------------------------------------------------------------------------------
    @staticmethod 
    @jit    
    def outstrength_jit(x):
        """JIT-compiled out-strength: row sums of the weight matrix ``x``."""
        return jnp.sum(x, axis=1)

    @staticmethod 
    @jit
    def instrength_jit(x):
        """JIT-compiled in-strength: column sums of the weight matrix ``x``."""
        return jnp.sum(x, axis=0)

    @staticmethod 
    @jit
    def strength_jit(x):
        """JIT-compiled total strength: out-strength + in-strength of ``x``."""
        return met.outstrength_jit(x) +  met.instrength_jit(x)

    @staticmethod 
    def strength(m, sym = False):
        """Strength (sum of tie weights) of each node.

        Args:
            m (jax.Array): Weight matrix, shape (N, N); entry [i, j] is the tie from i to j.
            sym (bool): The network is symmetric (undirected): count each tie
                once, i.e. return the out-strength. Defaults to False
                (out-strength + in-strength). A UserWarning is raised when
                ``sym=False`` but the matrix is symmetric, since the in+out
                total then double-counts every tie.

        Returns:
            jax.Array: Strength per node, shape (N,).
        """
        if not sym and _is_symmetric(m):
            warnings.warn(
                "strength: the matrix is symmetric (undirected) but sym=False "
                "sums in- and out-strength, i.e. twice the undirected strength; "
                "pass sym=True.",
                UserWarning,
                stacklevel=2,
            )
        if sym :
            return met.outstrength_jit(m)
        else:
            return met.strength_jit(m)

    
    @staticmethod 
    def outstrength(m):
        """Out-strength of each node: sum of the weights of its outgoing ties (row sums).

        Args:
            m (jax.Array): Weight matrix, shape (N, N); entry [i, j] is the tie from i to j.

        Returns:
            jax.Array: Out-strength per node, shape (N,).
        """
        return met.outstrength_jit(m)
    
    @staticmethod 
    def instrength(m):
        """In-strength of each node: sum of the weights of its incoming ties (column sums).

        Args:
            m (jax.Array): Weight matrix, shape (N, N); entry [i, j] is the tie from i to j.

        Returns:
            jax.Array: In-strength per node, shape (N,).
        """
        return met.instrength_jit(m)

    ## Degree----------------------------------------------------------------------------------
    @staticmethod 
    @jit
    def outdegree_jit(x):
        """JIT-compiled out-degree: number of non-zero entries in each row of ``x``."""
        mask = x != 0
        return jnp.sum(mask, axis=1)

    @staticmethod 
    @jit
    def indegree_jit(x):
        """JIT-compiled in-degree: number of non-zero entries in each column of ``x``."""
        mask = x != 0
        return jnp.sum(mask, axis=0)

    @staticmethod 
    @jit
    def degree_jit(x):
        """JIT-compiled total degree: in-degree + out-degree of ``x``."""
        return met.indegree_jit(x) + met.outdegree_jit(x)

    @staticmethod 
    def degree(m, sym = False, normalize=False):
        """Degree (number of ties) of each node.

        Args:
            m (jax.Array): Adjacency (or weight) matrix, shape (N, N); entry [i, j] is the tie from i to j.
            sym (bool): The network is symmetric (undirected): count each tie
                once. Defaults to False (in-degree + out-degree). A UserWarning
                is raised when ``sym=False`` but the matrix is symmetric, since
                the in+out total then double-counts every tie.
            normalize (bool): Divide by the maximum possible degree: N - 1, or
                2 (N - 1) for the directed total (in + out) degree. Zeros for
                N <= 1. Defaults to False.

        Returns:
            jax.Array: Degree per node, shape (N,).
        """
        if not sym and _is_symmetric(m):
            warnings.warn(
                "degree: the matrix is symmetric (undirected) but sym=False "
                "sums in- and out-degree, i.e. twice the undirected degree; "
                "pass sym=True.",
                UserWarning,
                stacklevel=2,
            )
        if sym :
            degree =  met.indegree_jit(m)
        else:
            degree = met.degree_jit(m)

        if normalize:
            return met.normalize(degree, m, total=not sym)
        else:
            return degree
    
    @staticmethod 
    def indegree(m, normalize=False):
        """In-degree of each node: number of incoming ties (non-zero column entries).

        Args:
            m (jax.Array): Adjacency (or weight) matrix, shape (N, N); entry [i, j] is the tie from i to j.
            normalize (bool): Divide by N - 1. Defaults to False.

        Returns:
            jax.Array: In-degree per node, shape (N,).
        """
        degree =  met.indegree_jit(m)
        if normalize:
            return met.normalize(degree,m)
        else:
            return degree
    
    @staticmethod 
    def outdegree(m, normalize=False):
        """Out-degree of each node: number of outgoing ties (non-zero row entries).

        Args:
            m (jax.Array): Adjacency (or weight) matrix, shape (N, N); entry [i, j] is the tie from i to j.
            normalize (bool): Divide by N - 1. Defaults to False.

        Returns:
            jax.Array: Out-degree per node, shape (N,).
        """
        degree = met.outdegree_jit(m)
        if normalize:
            return met.normalize(degree,m)
        else:
            return degree
    
    # Global measures----------------------------------------------------------------------------------
    @staticmethod
    @jit
    def density(m):
        """
        Compute the binary network density: the fraction of possible edges
        that are present (a non-zero entry counts as an edge; weights are
        ignored). Self-loops (the diagonal) are ignored, as in NetworkX.

        For a directed matrix the density is ``nnz / (n (n - 1))``; for a
        symmetric (undirected) matrix the same formula equals
        ``nnz_upper / (n (n - 1) / 2)``. Returns 0.0 for ``n <= 1``
        (NetworkX returns 0).

        Args:
            m: JAX array representing the adjacency matrix of a graph.

        Returns:
            Network density as a float.
        """
        n_nodes = m.shape[0]
        if n_nodes <= 1:
            return jnp.asarray(0.0)
        # Ignore the diagonal: self-loops are not edges for density.
        off_diagonal = jnp.where(jnp.eye(n_nodes, dtype=bool), 0, m)
        n_actual_edges = jnp.count_nonzero(off_diagonal) / 2  # both halves of a symmetric matrix

        # Density formula
        density = n_actual_edges / (n_nodes * (n_nodes - 1) / 2)
        return density

    @staticmethod
    @jit
    def single_source_dijkstra(m, src):
        """
        Computes the shortest path from a source node to all other nodes
        in a weighted graph using Dijkstra's algorithm.

        ``m`` is an adjacency/weight matrix: a 0 entry means "no edge" (as in
        ``geodesic_distance``), positive entries are edge lengths.

        Semantics (RELEASE_AUDIT Network m5): entries are edge *lengths*
        (distances, larger = farther), not strengths; the ``> 0`` mask is
        applied here (0 becomes inf), so no pre-masking is needed; rows are
        out-edges, so an asymmetric matrix is a directed graph. Negative
        entries are treated as "no edge" too, never as lengths. Siblings with
        different semantics: ``met.dijkstra_jax`` (same length reading, tests
        the mask per relaxation) and ``met._dijkstra_betweenness`` (presence
        from the adjacency matrix, lengths from a separate weight matrix).
        """
        n_nodes = m.shape[0]
        m = jnp.where(m > 0, m, jnp.inf)
        
        # Initialize distances and visited status
        dist = jnp.full((n_nodes,), jnp.inf).at[src].set(0)
        visited = jnp.zeros((n_nodes,), dtype=bool)

        def relax_step(carry, _):
            dist, visited = carry
            
            # Find the closest unvisited node
            unvisited_dist = jnp.where(visited, jnp.inf, dist)
            u = jnp.argmin(unvisited_dist)
            visited = visited.at[u].set(True)
            
            # Relax distances for neighbors of the selected node
            new_dist = jnp.minimum(dist, dist[u] + m[u])
            return (new_dist, visited), None

        # The loop runs n_nodes times to ensure all nodes are visited
        (dist, _), _ = jax.lax.scan(relax_step, (dist, visited), None, length=n_nodes)

        return dist

    @staticmethod
    @jit
    def geodesic_distance(m):
        """
        Compute the geodesic distance in a weighted graph using Dijkstra's algorithm in JAX.

        Here only zero entries are turned into "no edge" up front, but the
        inner ``met.single_source_dijkstra`` re-masks with ``> 0``, so a
        negative entry also ends up as "no edge", never as a length: the
        invalid-for-Dijkstra case (negative lengths) cannot reach the
        relaxation. See the semantics notes of the three Dijkstra variants in
        ``met.dijkstra_jax``.
        Args:
            m: 2D JAX array representing the weighted adjacency matrix of a graph.

        Returns:
            A 2D JAX array containing the shortest path distances between all pairs of nodes.
        """
        # Replace 0s with infinity for non-existent edges, but keep diagonal as 0
        m = jnp.where(m == 0, jnp.inf, m)
        m = m.at[jnp.diag_indices_from(m)].set(0)
        
        n_nodes = m.shape[0]

        # Use vmap to run Dijkstra from each node as a source.
        # in_axes=(None, 0) means the first argument (m) is broadcasted (the same for all calls)
        # and the second argument (the source nodes) is mapped over.
        distances = jax.vmap(met.single_source_dijkstra, in_axes=(None, 0))(m, jnp.arange(n_nodes))
        return distances
    
    @staticmethod
    def diameter(m):
        """
        Compute the diameter of a graph using the geodesic distance.
        Args:
            m: 2D JAX array representing the weighted adjacency matrix of a graph. 
            
        Returns:
            The diameter of the graph.

        Raises:
            ValueError: If the graph is disconnected (some infinite
                distances), matching NetworkX's ``diameter``.
        """
        diameter = jnp.max(met.geodesic_distance(m))
        if not bool(jnp.isfinite(diameter)):
            raise ValueError(
                "diameter: the graph is disconnected (infinite distances); "
                "compute it per connected component."
            )
        return diameter


    # Betweenness centrality  ----------------------------------------------------------------------------------

    # --- Paste your helper functions here ---
    @staticmethod
    @partial(jit, static_argnames=['n_nodes'])
    def _dijkstra_betweenness(
        adjacency_matrix: jnp.ndarray,
        weight_matrix: jnp.ndarray,
        source: int,
        n_nodes: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Dijkstra from ``source`` for Brandes' betweenness: order, predecessors, counts.

        Semantics (RELEASE_AUDIT Network m5): edge *presence* comes from
        ``adjacency_matrix > 0`` (a tie exists) while the edge *lengths* come
        from the separate ``weight_matrix`` and are read as distances; with
        ``weight_matrix`` equal to the adjacency matrix, strengths are used as
        distances. Paths of equal length are merged with an absolute tolerance
        of 1e-9, and rows are out-edges, so an asymmetric matrix is a directed
        graph. Unlike ``met.dijkstra_jax`` and ``met.single_source_dijkstra``
        (single matrix, lengths only, no counts) this also returns the
        predecessor matrix and ``sigma``, the number of shortest paths.
        """
        # RELEASE_AUDIT Network m1: the count accumulators follow
        # jax_enable_x64 (see _path_count_dtype), dist shares the dtype so the
        # carry stays homogeneous.
        pc_dtype = _path_count_dtype()
        initial_dist = jnp.full(n_nodes, jnp.inf, dtype=pc_dtype).at[source].set(0.0)
        initial_sigma = jnp.zeros(n_nodes, dtype=pc_dtype).at[source].set(1.0)
        initial_P = jnp.zeros((n_nodes, n_nodes), dtype=pc_dtype)
        initial_visited = jnp.zeros(n_nodes, dtype=bool)
        initial_S = jnp.full(n_nodes, -1, dtype=jnp.int32)
        initial_s_idx = 0
        initial_state = (initial_dist, initial_sigma, initial_P, initial_visited, initial_S, initial_s_idx)
        def body_fun(_, state):
            dist, sigma, P_matrix, visited, S, s_idx = state
            unvisited_dist = jnp.where(visited, jnp.inf, dist)
            u = jnp.argmin(unvisited_dist)
            min_dist_u = unvisited_dist[u]
            do_update = min_dist_u < jnp.inf
            S = S.at[s_idx].set(jnp.where(do_update, u, S[s_idx]))
            s_idx = jnp.where(do_update, s_idx + 1, s_idx)
            visited = visited.at[u].set(jnp.where(do_update, True, visited[u]))
            new_dist_v = dist[u] + weight_matrix[u, :]
            is_neighbor = adjacency_matrix[u, :] > 0
            is_unvisited = ~visited
            shorter_path_found = do_update & is_neighbor & is_unvisited & (new_dist_v < dist)
            equal_path_found = do_update & is_neighbor & is_unvisited & (jnp.abs(new_dist_v - dist) < 1e-9)
            dist = jnp.where(shorter_path_found, new_dist_v, dist)
            sigma_after_reset = jnp.where(shorter_path_found, 0.0, sigma)
            sigma_to_add = jnp.where(shorter_path_found | equal_path_found, sigma[u], 0.0)
            sigma = sigma_after_reset + sigma_to_add
            P_matrix = jnp.where(shorter_path_found[:, jnp.newaxis], 0.0, P_matrix)
            predecessor_update_mask = shorter_path_found | equal_path_found
            new_col_u = jnp.where(predecessor_update_mask, 1.0, P_matrix[:, u])
            P_matrix = P_matrix.at[:, u].set(new_col_u)
            return dist, sigma, P_matrix, visited, S, s_idx
        _, final_sigma, final_P, _, final_S, final_s_idx = lax.fori_loop(0, n_nodes, body_fun, initial_state)
        output_mask = jnp.arange(n_nodes) < final_s_idx
        gather_indices = jnp.maximum(0, final_s_idx - 1 - jnp.arange(n_nodes))
        reversed_S_padded = final_S[gather_indices]
        S_reversed = jnp.where(output_mask, reversed_S_padded, -1)
        return S_reversed, final_P, final_sigma

    @staticmethod
    @partial(jit, static_argnames=['n_nodes'])
    def bfs(adjacency_matrix, source, n_nodes):
        """ Breadth-First Search (BFS)"""
        # RELEASE_AUDIT Network m1: sigma counts shortest paths, see
        # _path_count_dtype.
        pc_dtype = _path_count_dtype()
        dist = jnp.full(n_nodes, -1, dtype=jnp.int32).at[source].set(0)
        sigma = jnp.zeros(n_nodes, dtype=pc_dtype).at[source].set(1.0)
        P_matrix = jnp.zeros((n_nodes, n_nodes), dtype=pc_dtype)
        layer_mask = (dist == 0)
        def body_fun(i, state):
            dist, sigma, P_matrix, layer_mask = state
            neighbors_matrix = jnp.where(layer_mask[:, None], adjacency_matrix, 0)
            potential_next_layer = (neighbors_matrix.sum(axis=0) > 0)
            newly_discovered_mask = potential_next_layer & (dist == -1)
            dist = jnp.where(newly_discovered_mask, i + 1, dist)
            is_in_next_layer_mask = (dist == i + 1)
            predecessor_mask = (adjacency_matrix.T > 0) & layer_mask[None, :]
            predecessor_sigmas = jnp.where(predecessor_mask, sigma[None, :], 0)
            sigma_contribution = predecessor_sigmas.sum(axis=1)
            sigma = jnp.where(is_in_next_layer_mask, sigma + sigma_contribution, sigma)
            P_update = jnp.where(is_in_next_layer_mask[None, :] & layer_mask[:, None] & (adjacency_matrix > 0), 1.0,    0)
            P_matrix = P_matrix + P_update.T
            return dist, sigma, P_matrix, is_in_next_layer_mask
        dist, sigma, P_matrix, _ = lax.fori_loop(0, n_nodes, body_fun, (dist, sigma, P_matrix, layer_mask))
        S = jnp.flip(jnp.argsort(dist, stable=True))
        return S, P_matrix, sigma, dist

    @staticmethod
    @partial(jit, static_argnames=['n_nodes'])
    def _optimized_accumulate_basic(betweenness, S, P, sigma, source, n_nodes):
        # RELEASE_AUDIT Network m1: delta accumulates path counts, see
        # _path_count_dtype.
        initial_delta = jnp.zeros(n_nodes, dtype=_path_count_dtype())
        initial_state = (betweenness, initial_delta)
        def body_fun(i, state):
            betweenness, delta = state
            # S is padded with -1 for unreachable nodes: those entries must be
            # skipped, not wrapped around to the last node by negative indexing.
            valid = S[i] >= 0
            w = jnp.maximum(S[i], 0)
            predecessors_mask = (P[w, :] > 0) & valid
            coeff = (sigma / sigma[w]) * (1.0 + delta[w])
            delta_update = jnp.where(predecessors_mask, coeff, 0.0)
            delta_new = delta + delta_update
            betweenness_update = jnp.where((w == source) | ~valid, 0.0, delta[w])
            betweenness_new = betweenness.at[w].add(betweenness_update)
            return (betweenness_new, delta_new)
        final_betweenness, _ = lax.fori_loop(0, n_nodes, body_fun, initial_state)
        return final_betweenness

    @staticmethod
    @partial(jit, static_argnames=['n_nodes'])
    def _optimized_accumulate_endpoints(betweenness, S, P, sigma, source, n_nodes, s_len):
        # s_len is a traced value under vmap, so it cannot bound the loop: loop
        # over n_nodes and skip the -1 padding of S.
        betweenness = betweenness.at[source].add(s_len - 1)
        # RELEASE_AUDIT Network m1: delta accumulates path counts, see
        # _path_count_dtype.
        initial_delta = jnp.zeros(n_nodes, dtype=_path_count_dtype())
        initial_state = (betweenness, initial_delta)
        def body_fun(i, state):
            betweenness, delta = state
            valid = S[i] >= 0
            w = jnp.maximum(S[i], 0)
            betweenness_update_val = delta[w] + 1
            betweenness = betweenness.at[w].add(jnp.where((w != source) & valid, betweenness_update_val, 0.0))
            predecessors_mask = (P[w, :] > 0) & valid
            safe_sigma_w = jnp.where(sigma[w] == 0, 1.0, sigma[w])
            coeff = (sigma / safe_sigma_w) * (1.0 + delta[w])
            is_valid_update = predecessors_mask & (sigma[w] > 0)
            delta_update = jnp.where(is_valid_update, coeff, 0.0)
            delta = delta + delta_update
            return (betweenness, delta)
        final_betweenness, _ = lax.fori_loop(0, n_nodes, body_fun, initial_state)
        return final_betweenness

    @staticmethod
    @jit
    def _rescale_jax(betweenness, n_nodes, normalized, k, endpoints, n_sampled, directed=False):
        def scale_fn(_):
            def small_graph(_):
                return betweenness
            def normal_graph(_):
                def endpoints_scale(_):
                    # networkx: 1 / (n (n - 1)) over ordered pairs; an undirected
                    # graph here counts each pair once, hence the factor 2.
                    return lax.cond(directed, lambda _: 1.0 / (n_nodes * (n_nodes - 1)),
                                    lambda _: 2.0 / (n_nodes * (n_nodes - 1)), operand=None)
                def directed_or_undirected(_):
                    return lax.cond(directed, lambda _: 1.0 / ((n_nodes - 1) * (n_nodes - 2)), lambda _: 2.0 /  ((n_nodes - 1) * (n_nodes - 2)), operand=None)
                scale = lax.cond(endpoints, endpoints_scale, directed_or_undirected, operand=None)
                scale = lax.cond((k is not None) & (n_sampled < n_nodes), lambda s: s * n_nodes / n_sampled, lambda     s: s, operand=scale)
                return betweenness * scale
            return lax.cond(n_nodes <= 2, small_graph, normal_graph, operand=None)
        return lax.cond(normalized, scale_fn, lambda _: betweenness, operand=None)


    @staticmethod
    @partial(jit, static_argnames=['n_nodes', 'endpoints', 'use_weights'])
    def _vmapped_betweenness_computation(
        adjacency_matrix: jnp.ndarray,
        weight_matrix: Optional[jnp.ndarray],
        source_nodes: jnp.ndarray,
        n_nodes: int,
        endpoints: bool,
        use_weights: bool
    ) -> jnp.ndarray:
        def single_source_logic(s):
            # RELEASE_AUDIT Network m1: per-source betweenness sums deltas of
            # path counts, see _path_count_dtype.
            b_s = jnp.zeros(n_nodes, dtype=_path_count_dtype())
            if use_weights:
                S, P, sigma = met._dijkstra_betweenness(adjacency_matrix, weight_matrix, s, n_nodes=n_nodes)
            else:
                S, P, sigma, bfs_dist = met.bfs(adjacency_matrix, s, n_nodes=n_nodes)
                # bfs orders every node, reachable or not: mark the unreachable
                # ones -1, as dijkstra does, so the accumulators skip them.
                S = jnp.where(bfs_dist[S] >= 0, S, -1)
            s_len = (S != -1).sum()
            if endpoints:
                b_s = met._optimized_accumulate_endpoints(b_s, S, P, sigma, s, n_nodes, s_len)
            else:
                b_s = met._optimized_accumulate_basic(b_s, S, P, sigma, s, n_nodes)
            return b_s
        all_b_s = vmap(single_source_logic)(source_nodes)
        return all_b_s.sum(axis=0)

    # --- CORRECTED MAIN FUNCTION ---
    @staticmethod
    def betweenness(
        adjacency_matrix: jnp.ndarray,
        n_nodes: int,
        k: int = None,
        weight_matrix: jnp.ndarray = None,
        normalized: bool = True,
        endpoints: bool = False,
        directed: bool = False,
    ) -> jnp.ndarray:
        """Betweenness centrality of each node (Brandes' algorithm, JAX).

        The fraction of shortest paths between other pairs of nodes that pass
        through each node. Matches ``networkx.betweenness_centrality``.

        Args:
            adjacency_matrix (jax.Array): Binary adjacency matrix, shape (N, N).
            n_nodes (int): Number of nodes, N.
            k (int, optional): Sample k source nodes. Not implemented; must be None.
            weight_matrix (jax.Array, optional): Edge lengths (distances) for
                weighted shortest paths. Defaults to unweighted paths.
            normalized (bool): Normalise by the number of node pairs. Defaults to True.
            endpoints (bool): Count path endpoints. Defaults to False.
            directed (bool): Treat the network as directed. Defaults to False.

        Returns:
            jax.Array: Betweenness per node, shape (N,).

        Raises:
            NotImplementedError: If ``k`` is given.
            ValueError: If ``adjacency_matrix`` is not a square 2-D matrix, if
                ``n_nodes`` disagrees with its shape, or if it is asymmetric
                while ``directed=False`` (RELEASE_AUDIT Network m1). The
                checks run on concrete input only; inside a JIT trace they
                are skipped.

        Note:
            Path counting (RELEASE_AUDIT Network m1): ``sigma`` and ``delta``
            count shortest paths and grow combinatorially, so on dense graphs
            they exceed float32's range (3.4e38, around N ~ 30-40 in the
            worst case) and the result silently becomes inf/NaN. The
            accumulators therefore follow ``jax_enable_x64``: enable it
            (``jax.config.update("jax_enable_x64", True)``) for dense or
            large networks. With x64 off, the default, the counts stay
            float32 and the overflow risk remains.
        """
        if k is not None:
            raise NotImplementedError("Sampling (k < n_nodes) is not implemented.")

        # RELEASE_AUDIT Network m1: validate the matrix/n_nodes pair before
        # tracing. An asymmetric matrix with directed=False used to compute
        # every path in one direction only and then halve the sum, which is
        # wrong in silence; a mismatched n_nodes used to fail later, deep in
        # a broadcast error.
        if _is_concrete(adjacency_matrix):
            a = np.asarray(adjacency_matrix)  # numpy: jnp ops are traced inside a model trace
            if a.ndim != 2 or a.shape[0] != a.shape[1]:
                raise ValueError(
                    f"betweenness: adjacency_matrix must be a square 2-D "
                    f"matrix, got shape {a.shape}."
                )
            if n_nodes != a.shape[0]:
                raise ValueError(
                    f"betweenness: n_nodes={n_nodes} does not match "
                    f"adjacency_matrix.shape[0]={a.shape[0]}."
                )
            if not directed and a.shape[0] > 1 and not _is_symmetric(a):
                raise ValueError(
                    "betweenness: adjacency_matrix is not symmetric but "
                    "directed=False: shortest paths would be counted in one "
                    "direction only and then halved. Pass directed=True for a "
                    "directed network, or symmetrise the matrix first."
                )

        source_nodes = jnp.arange(n_nodes)
        n_sampled = n_nodes

        betweenness = met._vmapped_betweenness_computation(
            adjacency_matrix,
            weight_matrix,
            source_nodes,
            n_nodes=n_nodes,
            endpoints=endpoints,
            use_weights=(weight_matrix is not None)
        )

        # Correct for double-counting in undirected graphs.

        if not directed:
            betweenness /= 2.0

        betweenness = met._rescale_jax(
            betweenness, n_nodes, normalized, k, endpoints, n_sampled, directed
        )
        return betweenness
    