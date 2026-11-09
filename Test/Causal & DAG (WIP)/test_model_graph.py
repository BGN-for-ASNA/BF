"""Test suite for the probabilistic model-graph extraction (m.graph).

Runs under pytest or standalone (see test_causal_dag.py for the rationale).

Covers: latent / observed / deterministic / data node classification, edge
reconstruction from distribution parameters, the ``name=`` site-name override,
indexed dependencies (e.g. a_grp[group]), and the matplotlib renderer.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from BayesForge import bf
from BayesForge.PostModel.model_graph import (
    LATENT, OBSERVED, DETERMINISTIC, DATA,
)

m = bf("cpu", print_devices_found=False)


# Module-level model functions: inspect.getsource needs real, importable source
# (lambdas / REPL-defined functions are not inspectable).
def simple_linear(x, y):
    alpha = m.dist.normal(0, 1, name="alpha")
    beta = m.dist.normal(0, 1, name="beta")
    sigma = m.dist.exponential(1, name="sigma")
    mu = alpha + beta * x
    m.dist.normal(mu, sigma, obs=y)


def hierarchical(x, y, group):
    alpha = m.dist.normal(0, 1, name="alpha")
    beta = m.dist.normal(0, 1, name="beta")
    tau = m.dist.exponential(1, name="tau")
    sigma = m.dist.exponential(1, name="sigma")
    a_grp = m.dist.normal(0, tau, name="a_grp", shape=(5,))
    mu = alpha + a_grp[group] + beta * x
    m.dist.normal(mu, sigma, obs=y)


def named_override(x, y):
    # the sample-site name differs from the python LHS variable
    b = m.dist.normal(0, 1, name="slope")
    mu = b * x
    m.dist.normal(mu, 1.0, obs=y, name="likelihood")


# ---------------------------------------------------------------------------
# 1. Node classification, edges and labels (static AST extraction)
# ---------------------------------------------------------------------------
def test_node_classification():
    """Every node gets a role: priors LATENT, intermediate maths DETERMINISTIC,
    the ``obs=`` site OBSERVED, and function arguments DATA."""
    g = m.graph(simple_linear, draw=False)
    assert g.kinds["alpha"] == LATENT
    assert g.kinds["beta"] == LATENT
    assert g.kinds["sigma"] == LATENT
    assert g.kinds["mu"] == DETERMINISTIC
    assert g.kinds["y"] == OBSERVED
    assert g.kinds["x"] == DATA


def test_simple_linear_edges():
    """Edges are rebuilt from the distribution arguments: alpha, beta and x
    feed mu, then mu and sigma feed the observed y."""
    g = m.graph(simple_linear, draw=False)
    e = set(g.edges)
    assert ("alpha", "mu") in e
    assert ("beta", "mu") in e
    assert ("x", "mu") in e
    assert ("mu", "y") in e
    assert ("sigma", "y") in e


def test_distribution_labels():
    """The distribution of each sampled node is recorded; deterministic nodes
    have no distribution (None)."""
    g = m.graph(simple_linear, draw=False)
    assert g.dist_of["alpha"] == "normal"
    assert g.dist_of["sigma"] == "exponential"
    assert g.dist_of["y"] == "normal"
    assert g.dist_of["mu"] is None  # deterministic has no distribution


# ---------------------------------------------------------------------------
# 2. Hierarchical sites, name override and pruning
# ---------------------------------------------------------------------------
def test_hierarchical_scale_edge():
    """The scale tau reaches the group effect a_grp, and the indexed
    dependency a_grp[group] is resolved (plus the index variable itself)."""
    g = m.graph(hierarchical, draw=False)
    e = set(g.edges)
    # non-centered-style scale parameter feeds the group effects
    assert ("tau", "a_grp") in e
    # indexed dependency a_grp[group] is detected, plus the index variable
    assert ("a_grp", "mu") in e
    assert ("group", "mu") in e
    assert g.kinds["a_grp"] == LATENT


def test_name_kwarg_overrides_node_name():
    """``name=`` replaces the Python left-hand side as the node identity, for
    latent and observed sites alike."""
    g = m.graph(named_override, draw=False)
    # the site name 'slope' (not the LHS 'b') becomes the node identity
    assert "slope" in g.nodes
    assert "b" not in g.nodes
    assert "likelihood" in g.nodes
    assert g.kinds["likelihood"] == OBSERVED
    assert ("slope", "mu") in set(g.edges)
    assert ("mu", "likelihood") in set(g.edges)


def test_unused_inputs_are_pruned():
    """A function argument never referenced in the body is dropped from the
    graph, while a referenced one is kept."""
    def with_extra(x, y, unused):
        a = m.dist.normal(0, 1, name="a")
        mu = a + x
        m.dist.normal(mu, 1.0, obs=y)
    g = m.graph(with_extra, draw=False)
    assert "unused" not in g.nodes   # never referenced -> pruned
    assert "x" in g.nodes            # referenced -> kept


# ---------------------------------------------------------------------------
# 3. Renderer
# ---------------------------------------------------------------------------
def test_draw_runs_headless():
    """The matplotlib renderer builds an Axes without a display (Agg backend)."""
    g = m.graph(simple_linear, draw=False)
    ax = g.draw(title="test")
    assert ax is not None
    plt.close("all")


def test_graph_method_returns_object_when_not_drawing():
    """``draw=False`` returns the graph object (nodes/edges) instead of drawing."""
    g = m.graph(hierarchical, draw=False)
    assert hasattr(g, "nodes") and hasattr(g, "edges")
    assert len(g.nodes) > 0


# ---------------------------------------------------------------------------
# 4. Loops, JAX ops, varying effects and the trace fallback
# ---------------------------------------------------------------------------
def test_for_loop_and_jax_ops_graph_and_latex():
    """A model written with a Python ``for`` loop and ``jnp.stack`` is readable
    both from the AST and from an execution trace, and renders in LaTeX.

    Steps:
      1. static graph: the loop variable ``z`` is one node fed by ``z0``;
      2. traced graph: running the model with data recovers the same edges and
         the observed shape;
      3. LaTeX: the loop, the recursion and the likelihood are written out.
    """
    import jax.numpy as jnp
    from BayesForge.PostModel import to_latex as TL
    TL.display = lambda *a, **k: None

    def ar1_loop(y, T):
        s = m.dist.exponential(1, name="s")
        z0 = m.dist.normal(0, 1, name="z0")
        z = [z0]
        for t in range(1, T):
            z.append(z[t - 1] * 0.9)
        m.dist.normal(jnp.stack(z), s, obs=y)

    # Static AST graph: the list built in the loop collapses to a single node z.
    g_static = m.graph(ar1_loop, draw=False)
    assert ("z0", "z") in set(g_static.edges)
    assert ("z", "y") in set(g_static.edges)
    assert ("s", "y") in set(g_static.edges)

    # Dynamic trace-backed graph: execute with data, same edges plus shapes.
    g_traced = m.graph(ar1_loop, draw=False, data=dict(y=jnp.zeros(5), T=5))
    assert ("z0", "z") in set(g_traced.edges)
    assert ("z", "y") in set(g_traced.edges)
    assert g_traced.shapes["y"] == (5,)

    # LaTeX rendering of loop + jnp.stack.
    tex = m.latex(ar1_loop)
    assert r"\text{for } t \in 1, \dots, T - 1\text{:}" in tex
    assert r"z[t] = z[t - 1] \cdot 0.9" in tex
    assert r"y \sim \text{Normal}(z, s)" in tex


def test_varying_effects_and_link_functions_graph_and_latex():
    """A GLMM built with ``varying_effects`` and a link function exposes the
    latent correlation and group effects, and renders the link in LaTeX.

    Steps:
      1. graph: the correlation matrix ``cafe_L_corr`` and the standardised
         effects ``cafe_z`` appear, feeding alpha and beta, which feed p;
      2. LaTeX: the inverse-logit, the sqrt/abs transform and the Cholesky
         factor of the group are typeset.
    """
    import jax.numpy as jnp
    from BayesForge.PostModel import to_latex as TL
    TL.display = lambda *a, **k: None

    def glmm_model(gid, x, y):
        alpha, beta = m.effects.varying_effects(
            N_vars=1, N_group=3, group_id=gid, group_name="cafe", centered=False
        )
        p = m.link.inv_logit(alpha + beta * jnp.sqrt(jnp.abs(x)))
        m.dist.binomial(1, p, obs=y, name="y")

    gid = jnp.array([0, 0, 1, 1, 2, 2])
    x = jnp.linspace(-1.0, 1.0, 6)
    y = jnp.array([0, 1, 0, 1, 1, 1])

    # 1. Graph: correlation -> standardised effects -> alpha/beta -> p -> y.
    g = m.graph(glmm_model, draw=False, data=dict(gid=gid, x=x, y=y))
    e = set(g.edges)
    assert "cafe_L_corr" in g.nodes
    assert "cafe_z" in g.nodes
    assert ("cafe_z", "alpha") in e
    assert ("cafe_z", "beta") in e
    assert ("alpha", "p") in e
    assert ("beta", "p") in e
    assert ("p", "y") in e

    # 2. LaTeX: link function, jnp.sqrt/jnp.abs and the Cholesky factor.
    tex = m.latex(glmm_model)
    assert r"\text{logit}^{-1}" in tex
    assert r"\sqrt{\left|x\right|}" in tex
    assert r"\mathbf{L}_{\text{cafe}}" in tex


def test_sourceless_model_trace_fallback():
    """When Python source is unavailable (R/Julia-defined models), graph and
    LaTeX fall back to an execution trace.

    Steps:
      1. build a callable with ``exec`` so ``inspect.getsource`` fails;
      2. graph from a trace with data: priors a, b and the input x feed y;
      3. LaTeX from the same trace still prints the priors and the likelihood.
    """
    import functools
    import jax.numpy as jnp
    from BayesForge.PostModel import to_latex as TL
    TL.display = lambda *a, **k: None

    # 1. Create a callable whose Python source cannot be inspected
    #    (simulating R/Julia).
    ns = {"m": m, "jnp": jnp}
    exec(
        "def r_model(x, y):\n"
        "    a = m.dist.normal(0, 1, name='a')\n"
        "    b = m.dist.normal(0, 1, name='b')\n"
        "    m.dist.normal(a + b * x, 1, obs=y, name='y')\n",
        ns,
    )
    r_model = ns["r_model"]
    data = dict(x=jnp.ones(4), y=jnp.zeros(4))

    # 2. Graph rebuilt from the trace: a, b and x all feed y.
    g = m.graph(r_model, draw=False, data=data)
    assert set(g.edges) == {("a", "y"), ("b", "y"), ("x", "y")}

    # 3. LaTeX from the same trace still prints priors and likelihood.
    tex = m.latex(r_model, data=data)
    assert r"y \sim \text{Normal}" in tex
    assert r"a \sim \text{Normal}(0, 1)" in tex


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception:
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed (model-graph suite)")
    raise SystemExit(1 if failed else 0)
