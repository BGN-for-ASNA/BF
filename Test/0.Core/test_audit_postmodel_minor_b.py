"""Regression tests for RELEASE_AUDIT.md, module PostModel (model_graph, model_trace) minors."""
import functools
import inspect
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.distributions.transforms import AffineTransform

from BayesForge import bf
from BayesForge.PostModel import model_graph as mg
from BayesForge.PostModel.model_graph import (
    LATENT, OBSERVED, DETERMINISTIC, ModelGraph, model_to_graph,
)
from BayesForge.PostModel.model_trace import _param_leaves, trace_model
from BayesForge.PostModel.to_latex import _latex_from_trace, to_latex

m = bf(platform="cpu", print_devices_found=False)


# ---------------------------------------------------------------------------
# m6 — the nested_varying_effects static table scanned only call.keywords,
# but `names` is the second POSITIONAL parameter (the docstring example and
# Test/0.Core/test_audit_postmodel_minor_a.py both pass it positionally), so
# every per-level site (sigma_*, L_corr_*, z_*) silently vanished.
# ---------------------------------------------------------------------------
def model_m6(r_id, g_id, y):
    a = m.effects.nested_varying_effects(2, ["region", "group"], [5, 20],
                                         [r_id, g_id])
    m.dist.normal(a, 1.0, obs=y)


def test_m6_nested_varying_effects_positional_names():
    g = model_to_graph(model_m6)
    keys = set(g.nodes)
    assert {"sigma_region", "L_corr_region", "z_region",
            "sigma_group", "L_corr_group", "z_group"} <= keys
    assert {"global_intercept", "global_beta"} <= keys


# ---------------------------------------------------------------------------
# m7 — the static effect tables added corr_{g}, {g}_sd_intercept, L_Rho_{g},
# ... unconditionally, even when the caller supplies the hyperprior via
# corr= / sd_intercept= / ... (Effects.py then never creates the site), so
# the AST-only graph was over-populated with ghost sites.
# ---------------------------------------------------------------------------
def model_m7_default(gid, y):
    a = m.effects.varying_effects(1, 10, gid, group_name="age")
    m.dist.normal(a, 1.0, obs=y)


def model_m7_overridden(gid, my_corr, sd_int, y):
    a = m.effects.varying_effects(1, 10, gid, group_name="age",
                                  corr=my_corr, sd_intercept=sd_int)
    m.dist.normal(a, 1.0, obs=y)


def test_m7_supplied_hyperpriors_drop_ghost_sites():
    g_def = model_to_graph(model_m7_default)
    assert {"age_L_corr", "age_sd_intercept", "age_sd_beta",
            "global_intercept", "global_beta", "age_z"} <= set(g_def.nodes)

    g_ovr = model_to_graph(model_m7_overridden)
    keys = set(g_ovr.nodes)
    assert "age_L_corr" not in keys        # corr=my_corr supplied
    assert "age_sd_intercept" not in keys  # sd_intercept=sd_int supplied
    # sites whose hyperprior was left None must stay
    assert "age_z" in keys
    assert "age_sd_beta" in keys
    assert "global_intercept" in keys


# ---------------------------------------------------------------------------
# m19 — _param_leaves read only the BASE distribution's arg_constraints:
# parameters held by a TransformedDistribution's transforms were neither
# named nor added to the leaves (dependencies lost silently), and the
# pytree fallback named its anonymous leaves "arg0", "arg1", ...
# ---------------------------------------------------------------------------
def test_m19_transform_params_traced_fallback_leaves_unnamed():
    td = dist.TransformedDistribution(
        dist.Normal(0.0, 1.0), [AffineTransform(2.0, 3.0)])
    names, leaves = _param_leaves(td)
    # base names unchanged, transform parameters traced as extra leaves
    assert names == ["loc", "scale"]
    assert len(leaves) == 4  # base loc/scale + transform loc/scale

    class _NoArgDist:
        arg_constraints = {}

        def __init__(self, v):
            self.v = jnp.asarray(v)

    names0, leaves0 = _param_leaves(_NoArgDist(jnp.ones(3)))
    # fallback leaves are traced but never named "arg0" (m15 placeholders)
    assert names0 == []
    assert len(leaves0) == 1


# ---------------------------------------------------------------------------
# m11 — line_offset = start_line - func.lineno: getsourcelines starts at the
# first decorator while func.lineno points at the `def`, so for a decorated
# model every traced site was attached to the LHS of the WRONG line.
# ---------------------------------------------------------------------------
def _identity_deco(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def _m11_helper(N):
    z = numpyro.sample("z", numpyro.distributions.Normal(0.0, 1.0))
    return z * 2.0


@_identity_deco
def model_m11(N=None, y=None):
    mu = _m11_helper(N)
    sigma = numpyro.sample("sigma", numpyro.distributions.HalfNormal(1.0))
    numpyro.sample("y", numpyro.distributions.Normal(mu, sigma), obs=y)


def test_m11_decorated_model_line_mapping():
    g = model_to_graph(model_m11, data={"N": 2, "y": jnp.zeros(2)})
    assert g.exact is True
    # site z is created inside the helper called on the `mu = ...` line
    assert ("z", "mu") in g.edges
    # the off-by-one line mapping attached it to the NEXT assignment instead
    assert ("z", "sigma") not in g.edges


# ---------------------------------------------------------------------------
# m12 — draw() leaked one open matplotlib figure per call and never rendered
# the plates captured from the trace (the "plate nesting" promise).
# ---------------------------------------------------------------------------
def test_m12_draw_closes_figure_and_renders_plates():
    plt.close("all")
    g = ModelGraph(["x", "y"], [("x", "y")],
                   {"x": LATENT, "y": OBSERVED},
                   {"x": "normal", "y": "normal"},
                   shapes={"x": (5,), "y": (5,)},
                   plates={"x": ("i",), "y": ("i",)})
    before = set(plt.get_fignums())
    try:
        ax = g.draw(show=False)
        assert ax is not None
        # the figure is closed when draw returns (headless Agg, not interactive)
        assert set(plt.get_fignums()) == before
        # the captured plate is actually drawn (dashed rectangle)
        assert any(isinstance(p, Rectangle) for p in ax.patches)
    finally:
        plt.close("all")


# ---------------------------------------------------------------------------
# m27 — the "experimental" notice now goes through BF's warn_dev helper
# (RuntimeWarning, emitted once per key; reset_dev_warnings() re-arms).
# ---------------------------------------------------------------------------
def model_m27(x, y):
    m.dist.normal(0.0, 1.0, obs=y)


def test_m27_experimental_notice_is_a_warning_not_print():
    from BayesForge.Utils.dev_warning import reset_dev_warnings
    reset_dev_warnings()
    with pytest.warns(RuntimeWarning, match="still in development"):
        model_to_graph(model_m27)
    # the notice goes through BF's shared warn_dev helper, no local state
    assert not hasattr(mg, "_WARNED")
    assert not hasattr(mg, "_warn_experimental")


def test_to_latex_emits_the_dev_warning():
    from BayesForge.PostModel.to_latex import to_latex
    from BayesForge.Utils.dev_warning import reset_dev_warnings
    reset_dev_warnings()
    with pytest.warns(RuntimeWarning, match="still in development"):
        to_latex(model_m27, display=False)


# ---------------------------------------------------------------------------
# m20 — malformed `data` surfaced as raw internal errors: a non-mapping blew
# up inside dict()/get_trace, an unknown key raised an unreadable TypeError,
# and numpy scalars were traced as 0-d arrays, breaking `range(N)` in the
# second pass and leaving the graph with no edges.
# ---------------------------------------------------------------------------
def test_m20_malformed_data_gets_guided_errors():
    def model_m20(N=None, y=None):
        numpyro.sample("y", numpyro.distributions.Normal(0.0, 1.0), obs=y)

    with pytest.raises(TypeError, match="mapping"):
        trace_model(model_m20, [1, 2, 3])
    with pytest.raises(ValueError, match="nope"):
        trace_model(model_m20, {"nope": jnp.zeros(3)})

    def model_m20_scalar(N=None, y=None):
        mu = numpyro.sample("mu", numpyro.distributions.Normal(0.0, 1.0))
        s = 0.0
        for i in range(N):
            s = s + mu
        numpyro.deterministic("s", s)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        tr = trace_model(model_m20_scalar, {"N": np.int64(3), "y": jnp.zeros(3)})
    assert not any("dependency tracing failed" in str(w.message) for w in rec)
    assert tr.parents["s"] == {"mu"}


# ---------------------------------------------------------------------------
# m21 — numpyro.param (SVI) sites: invisible in the trace and unlisted by
# to_latex, but the AST table treated them as latent distribution nodes
# labelled "param" — three files, three behaviours. The AST path now ignores
# them like the other two views.
# ---------------------------------------------------------------------------
def model_m21(init, y):
    q = numpyro.param("q_mu", init)
    m.dist.normal(q, 1.0, obs=y)


def test_m21_svi_param_sites_ignored_in_ast():
    g = model_to_graph(model_m21)
    assert "q_mu" not in g.nodes
    assert "param" not in set(g.dist_of.values())
    # the assignment stays visible as a plain deterministic node
    assert g.kinds["q"] == DETERMINISTIC


# ---------------------------------------------------------------------------
# m8 — a bare latent call (no name=, no obs=) became a node named
# `obs_<dist>`: a lying prefix, and two bare calls of the same distribution
# merged into one node.
# ---------------------------------------------------------------------------
def model_m8(mu, y):
    m.dist.normal(mu, 1.0)
    m.dist.normal(mu, 2.0)
    m.dist.normal(mu, 1.0, obs=y)


def test_m8_bare_latent_sites_not_labelled_observed():
    g = model_to_graph(model_m8)
    assert "obs_normal" not in g.nodes
    latents = [n for n in g.nodes if g.kinds[n] == LATENT]
    assert len(latents) == 2  # two distinct nodes, not one merged
    assert all("obs" not in n for n in latents)


# ---------------------------------------------------------------------------
# m9/m10 — multi-target assignments (`a = b = expr`) produced no nodes, and
# an input used only as a plate size (`with numpyro.plate("i", N)`) was never
# scanned as a dependency, so N was pruned and the graph hid a model input.
# ---------------------------------------------------------------------------
def model_m9m10(N, x, y):
    a = b = x * 2.0
    with numpyro.plate("i", N):
        z = numpyro.sample("z", numpyro.distributions.Normal(a, 1.0))
    m.dist.normal(z, 1.0, obs=y)


def test_m9_m10_multi_target_assignments_and_plate_sizes():
    g = model_to_graph(model_m9m10)
    assert {"a", "b"} <= set(g.nodes)
    assert g.kinds["a"] == DETERMINISTIC
    assert g.kinds["b"] == DETERMINISTIC
    assert "N" in g.nodes
    assert ("N", "z") in g.edges
    assert ("a", "z") in g.edges


# ---------------------------------------------------------------------------
# m15/m16 — "hollow" renderings instead of honest output: a dependency held
# by a TransformedDistribution's transform vanished (x looked independent of
# sigma), anonymous leaves could print as `arg0` placeholders, and observed
# sites must carry the obs marker (M5 regression). The unknown-distribution
# and empty-trace bullets live in to_latex code outside this fix's scope.
# ---------------------------------------------------------------------------
def model_m15(y=None):
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    numpyro.sample("x", dist.TransformedDistribution(
        dist.Normal(0.0, 1.0), [AffineTransform(sigma, 1.0)]))
    numpyro.sample("y", dist.Normal(0.0, 1.0), obs=y)


def test_m15_m16_transform_dependency_and_observed_marker():
    tr = trace_model(model_m15, {"y": jnp.zeros(3)})
    # the transform parameter carries the sigma dependency (m19/m15)
    assert "sigma" in tr.parents["x"]
    tex = "\n".join(_latex_from_trace(model_m15, {"y": jnp.zeros(3)}))
    # observed sites are marked, not rendered as latents
    assert r"y^{\text{obs}}" in tex
    assert "arg0" not in tex


# ---------------------------------------------------------------------------
# m13/m14/m24 — API coherence: to_latex announced align* but emitted
# gather*, and display(Latex(...)) printed raw LaTeX outside IPython;
# `mu = numpyro.deterministic("mean", e)` rendered under the Python name
# while model_graph uses the site name; numpyro.sample("x", loi) got the
# distribution label "sample"; plate headers denied subsampling, ignored
# dim= and claimed 1-based indices.
# ---------------------------------------------------------------------------
def model_api_latex(x, y):
    beta = m.dist.normal(0.0, 1.0, name="beta")
    m.dist.normal(beta * x, 1.0, obs=y)


def model_api_det(x, y):
    e = m.dist.normal(0.0, 1.0, name="e")
    mu = numpyro.deterministic("mean", e * 2.0)
    m.dist.normal(mu, 1.0, obs=y)


def model_api_dist_in_var(x, y):
    loi = numpyro.distributions.Normal(0.0, 1.0)
    z = numpyro.sample("z", loi)
    m.dist.normal(z, 1.0, obs=y)


def model_api_dist_param(d, y):
    w = numpyro.sample("w", d)
    m.dist.normal(w, 1.0, obs=y)


def model_api_plate(N, n, y):
    with numpyro.plate("i", N, subsample_size=n, dim=-2):
        m.dist.normal(0.0, 1.0, obs=y)


def test_m13_m14_m24_api_coherence(capsys):
    # m13: docstring matches the emitted environment; display is an opt-in
    # keyword and nothing is ever printed outside IPython.
    sig = inspect.signature(to_latex)
    assert "display" in sig.parameters
    assert sig.parameters["display"].default is True
    assert "align*" not in (to_latex.__doc__ or "")
    assert "gather*" in (to_latex.__doc__ or "")
    capsys.readouterr()
    tex = to_latex(model_api_latex)
    assert tex.startswith("\\begin{gather*}")
    assert capsys.readouterr().out == ""

    # m14: deterministic sites render under the SITE name, like model_graph.
    tex = to_latex(model_api_det)
    assert r"mean = e \cdot 2.0" in tex
    assert r"\mu = e" not in tex

    # m13/m14/m24: a distribution passed through a variable resolves to its
    # real name, and never to the primitive label "sample".
    g = model_to_graph(model_api_dist_in_var)
    assert g.dist_of["z"] == "normal"
    g2 = model_to_graph(model_api_dist_param)
    assert g2.dist_of["w"] is None

    # m24: subsample_size, dim and 0-based indexing are shown, not denied.
    tex = to_latex(model_api_plate)
    assert r"\in 0, \dots, n" in tex
    assert "subsample" in tex
    assert "dim" in tex


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
