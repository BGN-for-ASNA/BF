from BayesForge import bf
import numpy as np
import pandas as pd
import jax.numpy as jnp
import jax
import networkx as nx
import sys
import os

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Initialize logging to both console and log.txt
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log.txt")
log_file = open(log_file_path, "w")
sys.stdout = Tee(sys.stdout, log_file)

m = bf(platform='cpu')

print('Loading data...')
data_path = os.path.dirname(os.path.abspath("")) + "/BF/resources/data/"
G_undirected = nx.karate_club_graph()
G_directed = nx.DiGraph(G_undirected)  # Convert to directed graph

# Get the adjacency matrix for JAX
adj_matrix_np = nx.to_numpy_array(G_directed)
adj_matrix_jax = jnp.array(adj_matrix_np)


def beteeweness_run_all_tests():
    """Iterates through all test configurations and compares JAX vs NetworkX."""
    n_nodes = 15
    seed = 42
    rng = np.random.default_rng(seed)
    
    test_configs = [
        (directed, weighted, normalized)
        for directed in [False, True]
        for weighted in [False, True]
        for normalized in [False, True]
    ]
    
    total_passed = 0
    for directed, weighted, normalized in test_configs:
        print("-" * 70)
        print(f"Testing: Directed={directed}, Weighted={weighted}, Normalized={normalized}")
        
        # 1. Generate graph data
        adj_matrix_np = (rng.random((n_nodes, n_nodes)) < 0.3).astype(np.float32)
        np.fill_diagonal(adj_matrix_np, 0)
        
        weight_matrix_np = None
        if weighted:
            # Generate random weights between 1 and 10
            # Ensure the weight matrix is float32 to match the JAX functions' internal state.
            weight_matrix_np = (rng.uniform(1, 10, size=(n_nodes, n_nodes)) * adj_matrix_np).astype(np.float32)
        
        if not directed:
            adj_matrix_np = np.tril(adj_matrix_np) + np.tril(adj_matrix_np, -1).T
            if weighted:
                weight_matrix_np = np.tril(weight_matrix_np) + np.tril(weight_matrix_np, -1).T

        # 2. Create NetworkX graph
        graph_type = nx.DiGraph if directed else nx.Graph
        G = nx.from_numpy_array(adj_matrix_np, create_using=graph_type)
        
        nx_weight_arg = None
        if weighted:
            nx_weight_arg = 'weight'
            for i, j, data in G.edges(data=True):
                data[nx_weight_arg] = weight_matrix_np[i, j]

        # 3. Run JAX implementation
        adj_matrix_jax = jnp.array(adj_matrix_np)
        weight_matrix_jax = jnp.array(weight_matrix_np) if weighted else None
        
        jax_bc = m.net.betweenness(
            adj_matrix_jax,
            n_nodes=n_nodes,
            weight_matrix=weight_matrix_jax,
            normalized=normalized,
            directed=directed
        ).block_until_ready()

        # 4. Run NetworkX implementation
        nx_bc_dict = nx.betweenness_centrality(
            G,
            normalized=normalized,
            weight=nx_weight_arg
        )
        nx_bc_array = jnp.array([nx_bc_dict[i] for i in range(n_nodes)])

        # 5. Compare results
        try:
            # Use a slightly higher tolerance for float32 vs float64 comparisons
            is_close = jnp.allclose(jax_bc, nx_bc_array, atol=1e-5, rtol=1e-5)
            assert is_close
            print("✅ PASS: JAX and NetworkX results are consistent.")
            total_passed += 1
        except AssertionError:
            print("❌ FAIL: JAX and NetworkX results differ.")
            print(f"   JAX result: {jax_bc}")
            print(f"   NX result:  {nx_bc_array}")

    print("-" * 70)
    print(f"SUMMARY: Passed {total_passed} out of {len(test_configs)} test cases.")


def nx_geodesic(G_undirected, weighted  = True):
    if weighted:
        lengths = dict(nx.all_pairs_dijkstra_path_length(G_undirected))
    else:
        lengths = dict(nx.all_pairs_shortest_path_length(G_undirected))
    nodes = list(G_undirected.nodes)

    # Initialize a distance matrix with 'inf' for unreachable pairs
    n = len(nodes)
    dist_matrix = np.full((n, n), np.inf)

    # Fill the matrix with the shortest path lengths
    for i, source in enumerate(nodes):
        for target, length in lengths[source].items():
            j = nodes.index(target)
            dist_matrix[i, j] = length

    # Set the diagonal to 0 (distance from a node to itself)
    np.fill_diagonal(dist_matrix, 0)

    return dist_matrix


def nx_geodesic_any(G, weighted=True):
    """Shortest-path length matrix for any (di)graph, inf for unreachable pairs."""
    if weighted:
        lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    else:
        lengths = dict(nx.all_pairs_shortest_path_length(G))
    nodes = list(G.nodes)
    n = len(nodes)
    dist_matrix = np.full((n, n), np.inf)
    for i, source in enumerate(nodes):
        for target, length in lengths[source].items():
            j = nodes.index(target)
            dist_matrix[i, j] = length
    np.fill_diagonal(dist_matrix, 0)
    return dist_matrix


_test_stats = {"passed": 0, "total": 0}

def run_test(name, test_fn):
    """Utility to run a test function and log the result."""
    print(f"Testing {name}------------------------")
    _test_stats["total"] += 1
    try:
        test_fn()
        _test_stats["passed"] += 1
        print(f"✅ PASS: {name}")
    except Exception as e:
        print(f"❌ FAIL: {name}")
        print(f"   Error: {e}")

run_test("degree (symmetric)", lambda: np.testing.assert_array_almost_equal(
    m.net.degree(adj_matrix_jax,  sym = True), 
    np.array(list(dict(nx.degree(G_undirected)).values()))
))

run_test("degree (asymmetric)", lambda: np.testing.assert_array_almost_equal(
    m.net.degree(adj_matrix_jax,  sym = False), 
    np.array(list(dict(nx.degree(G_undirected)).values()))*2
))

run_test("indegree", lambda: np.testing.assert_array_almost_equal(
    m.net.indegree(adj_matrix_jax,  normalize = True), 
    np.array(list(dict(nx.in_degree_centrality(G_directed)).values()))
))

run_test("outdegree", lambda: np.testing.assert_array_almost_equal(
    m.net.outdegree(adj_matrix_jax,  normalize = True), 
    np.array(list(dict(nx.out_degree_centrality(G_directed)).values()))
))

run_test("strength (symmetric)", lambda: np.testing.assert_array_almost_equal(
    m.net.strength(adj_matrix_jax,  sym = True), 
    np.array(list(dict(G_undirected.degree(weight="weight")).values()))
))

run_test("strength (asymmetric)", lambda: np.testing.assert_array_almost_equal(
    m.net.strength(adj_matrix_jax,  sym = False), 
    np.array(list(dict(G_undirected.degree(weight="weight")).values()))*2
))

run_test("instrength", lambda: np.testing.assert_array_almost_equal(
    m.net.instrength(adj_matrix_jax), 
    np.array(list(dict(G_directed.in_degree(weight="weight")).values()))
))

run_test("outstrength", lambda: np.testing.assert_array_almost_equal(
    m.net.outstrength(adj_matrix_jax), 
    np.array(list(dict(G_directed.out_degree(weight="weight")).values()))
))

run_test("clustering coefficient", lambda: np.testing.assert_array_almost_equal(
    m.net.cc(adj_matrix_jax),
    np.array(list(dict(nx.clustering(G_directed)).values()))
))

# Weighted clustering coefficient (Onnela et al.) must match NetworkX's
# nx.clustering(G, weight='weight'). NetworkX defines it only for undirected
# graphs, so build undirected weighted graphs from the same numpy matrices the
# JAX side consumes (from_numpy_array copies the weights exactly).
def nx_weighted_cc(W):
    G = nx.from_numpy_array(W)  # undirected, weighted
    return np.array([nx.clustering(G, weight='weight')[i] for i in range(W.shape[0])])

# 1. The karate-club adjacency (this NetworkX build ships it weighted, 0-7).
adj_undirected_np = nx.to_numpy_array(G_undirected)
run_test("weighted cc / Onnela (karate weighted)", lambda: np.testing.assert_array_almost_equal(
    m.net.weighted_cc(jnp.array(adj_undirected_np)),
    nx_weighted_cc(adj_undirected_np)
))

# 2. A random continuous, symmetric, zero-diagonal weight matrix.
_cc_rng = np.random.default_rng(42)
_n = adj_undirected_np.shape[0]
_cc_cont_np = _cc_rng.random((_n, _n))
_cc_cont_np = np.tril(_cc_cont_np, -1)
_cc_cont_np = _cc_cont_np + _cc_cont_np.T           # symmetric, zero diagonal
_cc_cont_jax = jnp.array(_cc_cont_np)
run_test("weighted cc / Onnela (continuous matrix)", lambda: np.testing.assert_array_almost_equal(
    m.net.weighted_cc(_cc_cont_jax),
    nx_weighted_cc(_cc_cont_np)
))

# 3. The opt-in flag on cc() must route to the NetworkX weighted (Onnela) result.
run_test("cc(weighted=True) / Onnela vs NetworkX", lambda: np.testing.assert_array_almost_equal(
    m.net.cc(_cc_cont_jax, weighted=True),
    nx_weighted_cc(_cc_cont_np)
))

run_test("eigenvector centrality weighted", lambda: np.testing.assert_array_almost_equal(
    m.net.eigenvector(adj_matrix_jax), 
    np.array(list(dict(nx.eigenvector_centrality(G_undirected, max_iter=1000, weight='weight')).values())),
    decimal=5
))

run_test("eigenvector centrality unweighted", lambda: (
    adj_matrix_jax_unweighted := m.net.to_binary_matrix(adj_matrix_jax),
    np.testing.assert_array_almost_equal(
        m.net.eigenvector(adj_matrix_jax_unweighted), 
        np.array(list(dict(nx.eigenvector_centrality(G_undirected, max_iter=1000)).values())),
        decimal=5
    )
)[1])

run_test("density", lambda: np.testing.assert_array_almost_equal(
    m.net.density(adj_matrix_jax), 
    np.array(nx.density(G_undirected))
))

run_test("diameter", lambda: np.testing.assert_array_almost_equal(
    m.net.diameter(m.net.to_binary_matrix(adj_matrix_jax)), 
    np.array(nx.diameter(G_undirected))
))

run_test("geodesic_distance binary", lambda: np.testing.assert_array_almost_equal(
    m.net.geodesic_distance(m.net.to_binary_matrix(adj_matrix_jax)),
    np.array(nx_geodesic(G_undirected, weighted=False))
))

run_test("geodesic_distance weighted", lambda: np.testing.assert_array_almost_equal(
    m.net.geodesic_distance(adj_matrix_jax),
    np.array(nx_geodesic(G_undirected, weighted=True))
))

# ---------------------------------------------------------------------------
# Directed, asymmetric network
# ---------------------------------------------------------------------------
print("\n=== Directed, asymmetric network ===")
rng = np.random.default_rng(7)
_n_dir = 30
W = rng.random((_n_dir, _n_dir)) * (rng.random((_n_dir, _n_dir)) < 0.15)
np.fill_diagonal(W, 0)
G_dir = nx.from_numpy_array(W, create_using=nx.DiGraph)
W_jax = jnp.array(W)
W_bin_jax = m.net.to_binary_matrix(W_jax)

run_test("directed: indegree", lambda: np.testing.assert_array_almost_equal(
    m.net.indegree(W_jax, normalize=True),
    np.array([nx.in_degree_centrality(G_dir)[i] for i in range(_n_dir)])
))

run_test("directed: outdegree", lambda: np.testing.assert_array_almost_equal(
    m.net.outdegree(W_jax, normalize=True),
    np.array([nx.out_degree_centrality(G_dir)[i] for i in range(_n_dir)])
))

run_test("directed: instrength", lambda: np.testing.assert_array_almost_equal(
    m.net.instrength(W_jax),
    np.array([G_dir.in_degree(i, weight="weight") for i in range(_n_dir)])
))

run_test("directed: outstrength", lambda: np.testing.assert_array_almost_equal(
    m.net.outstrength(W_jax),
    np.array([G_dir.out_degree(i, weight="weight") for i in range(_n_dir)])
))

run_test("directed: degree (asymmetric)", lambda: np.testing.assert_array_almost_equal(
    m.net.degree(W_jax, sym=False),
    np.array([dict(G_dir.degree())[i] for i in range(_n_dir)])
))

run_test("directed: cc", lambda: np.testing.assert_array_almost_equal(
    m.net.cc(W_bin_jax),
    np.array([nx.clustering(G_dir)[i] for i in range(_n_dir)])
))

run_test("directed: density", lambda: np.testing.assert_array_almost_equal(
    m.net.density(W_jax),
    np.array(nx.density(G_dir))
))

run_test("directed: geodesic_distance weighted", lambda: np.testing.assert_array_almost_equal(
    np.asarray(m.net.geodesic_distance(W_jax)),
    nx_geodesic_any(G_dir, weighted=True)
))

run_test("directed: geodesic_distance binary", lambda: np.testing.assert_array_almost_equal(
    np.asarray(m.net.geodesic_distance(W_bin_jax)),
    nx_geodesic_any(G_dir, weighted=False)
))

# ---------------------------------------------------------------------------
# Disconnected network
# ---------------------------------------------------------------------------
print("\n=== Disconnected network ===")
G_disc = nx.disjoint_union(nx.complete_graph(5), nx.path_graph(6))
G_disc.add_nodes_from([11, 12])
A_disc_np = nx.to_numpy_array(G_disc, nodelist=range(13))
A_disc_jax = jnp.array(A_disc_np)

run_test("disconnected: degree", lambda: np.testing.assert_array_almost_equal(
    m.net.degree(A_disc_jax, sym=True),
    np.array([dict(nx.degree(G_disc))[i] for i in range(13)])
))

run_test("disconnected: density", lambda: np.testing.assert_array_almost_equal(
    m.net.density(A_disc_jax),
    np.array(nx.density(G_disc))
))

run_test("disconnected: cc", lambda: np.testing.assert_array_almost_equal(
    m.net.cc(A_disc_jax),
    np.array([nx.clustering(G_disc)[i] for i in range(13)])
))

def check_geodesic_disconnected():
    got = np.asarray(m.net.geodesic_distance(A_disc_jax))
    exp = nx_geodesic_any(G_disc, weighted=False)
    np.testing.assert_array_equal(np.isinf(got), np.isinf(exp))
    np.testing.assert_array_almost_equal(got[np.isfinite(got)], exp[np.isfinite(exp)])

run_test("disconnected: geodesic_distance", check_geodesic_disconnected)

run_test("disconnected: eigenvector", lambda: np.testing.assert_array_almost_equal(
    m.net.eigenvector(A_disc_jax),
    np.array([nx.eigenvector_centrality(G_disc, max_iter=1000)[i] for i in range(13)]),
    decimal=5
))


# component_aware=True: NetworkX run on each connected component separately
# (each component normalised on its own; an isolated node gets 1.0).
def nx_eigenvector_per_component(G, n):
    ref = np.zeros(n)
    for comp in nx.connected_components(G):
        H = G.subgraph(comp)
        ec = {next(iter(comp)): 1.0} if len(H) == 1 else nx.eigenvector_centrality(H, max_iter=1000)
        for k, v in ec.items():
            ref[k] = v
    return ref


run_test("disconnected: eigenvector (component_aware)", lambda: np.testing.assert_array_almost_equal(
    m.net.eigenvector(A_disc_jax, component_aware=True),
    nx_eigenvector_per_component(G_disc, 13),
    decimal=5
))

try:
    nx_diam_disc = nx.diameter(G_disc)
except Exception as e:
    nx_diam_disc = f"raised {type(e).__name__}: {e}"
try:
    bf_diam_disc = np.asarray(m.net.diameter(m.net.to_binary_matrix(A_disc_jax)))
except Exception as e:
    bf_diam_disc = f"raised {type(e).__name__}: {e}"
print(f"disconnected: diameter — NetworkX: {nx_diam_disc}, BayesForge: {bf_diam_disc}")

beteeweness_run_all_tests()

print(f"\nSUMMARY metrics: {_test_stats['passed']}/{_test_stats['total']} passed")
print("\nAll tests completed.")
sys.stdout = sys.stdout.files[0]   # restore the terminal before closing the log
log_file.close()
