"""Regression tests for RELEASE_AUDIT.md, module PostModel, majors M1-M5."""

import jax.numpy as jnp
import numpyro
import pytest

from BayesForge import bf
from BayesForge.PostModel.model_graph import model_to_graph
from BayesForge.PostModel.to_latex import _latex_from_trace, to_latex

m = bf(platform="cpu", print_devices_found=False)


# ---------------------------------------------------------------------------
# M1 — the audit claimed the LaTeX table referenced a varying_slope helper
# that did not exist in Effects.py; the premise is stale (the helper was
# added in commit 94e80673 and uses valid dist.* calls), so the table
# branches are live and must expand the LKJ-Cholesky block instead of
# emitting a generic placeholder line.
#
# The helper is called from a module-level model (a real `def`, so
# inspect.getsource works); the AST tables must name the expanded nodes
# after the helper's real numpyro sites: L_Rho_{group_name},
# sigma_{group_name}, z_{group_name} (group_name defaults to "group").
# ---------------------------------------------------------------------------

def model_m1(category_idx=None, grouping_idx=None, num_categories=None,
             num_groups=None, y=None):
    beta = m.effects.varying_slope_non_centered(
        category_idx, grouping_idx, num_categories, num_groups)
    sigma = numpyro.sample("sigma", numpyro.distributions.HalfNormal(1.0))
    numpyro.sample("y", numpyro.distributions.Normal(beta, sigma), obs=y)


def test_to_latex_expands_varying_slope_non_centered_block():
    # to_latex(model) takes a model FUNCTION (it reads the source), not a str
    tex = to_latex(model_m1)
    # the LKJ/Cholesky hierarchy, not a generic "beta ~ ..." placeholder
    assert (r"\text{diag}(\sigma_{\text{group}}) \cdot \mathbf{L}_{\text{group}}"
            r" \cdot \mathbf{Z}_{\text{group}}") in tex
    assert r"\mathbf{Z}_{\text{group}} \sim \text{Normal}(0, 1)" in tex
    assert r"\mathbf{L}_{\text{group}} \sim \text{LKJCholesky}(2)" in tex
    assert r"\sigma_{\text{group}} \sim \text{Exponential}(1)" in tex
    # the generic fallback would render \text{Varying\_slope\_non\_centered}(...)
    assert "Varying" not in tex


def test_graph_varying_slope_non_centered_has_direct_sample_nodes():
    # no data: the graph comes from the AST table, which must expand the
    # helper into its real numpyro site names (group_name defaults to "group")
    g = model_to_graph(model_m1)
    keys = set(g.nodes)
    # direct numpyro.sample sites keep their raw names as graph nodes
    assert {"L_Rho_group", "sigma_group", "z_group"} <= keys
    assert "sigma" in keys


# ---------------------------------------------------------------------------
# M2 — model_to_graph() swallowed the tracing exception silently and still
# reported the static graph as exact; it must warn and set exact=False.
# ---------------------------------------------------------------------------

def model_m2(y=None):
    sigma = numpyro.sample("sigma", numpyro.distributions.HalfNormal(1.0))
    numpyro.sample("y", numpyro.distributions.Normal(0.0, sigma), obs=y)


def model_m2_boom(x=None):
    # accepts the data keys trace_model passes as kwargs, then raises inside
    # its body so the warning must name the exception type that occurred
    raise KeyError("boom")


def test_m2_tracing_failure_warns_and_marks_graph_inexact():
    # the model raises while it is traced: warn and mark the graph inexact
    with pytest.warns(UserWarning, match="tracing the model failed"):
        g = model_to_graph(model_m2_boom, data={"x": 1})
    assert g.exact is False


def test_m2_warning_names_the_failure():
    with pytest.warns(UserWarning) as rec:
        model_to_graph(model_m2_boom, data={"x": 1})
    text = " ".join(str(w.message) for w in rec)
    assert "KeyError" in text
    assert "may miss or invent edges" in text


def test_m2_successful_tracing_keeps_exact_flag_true():
    # data keys must match the model parameters exactly (trace_model calls
    # the model with the data dict as keyword arguments)
    g = model_to_graph(model_m2, data={"y": jnp.zeros(3)})
    assert g.exact is True


# ---------------------------------------------------------------------------
# M4 — a `with numpyro.plate(...)` body was reversed line-by-line, so the
# plate header was emitted after its body.
# ---------------------------------------------------------------------------

def model_m4(data=None):
    with numpyro.plate("obs", 5):
        sigma = numpyro.sample("sigma", numpyro.distributions.HalfNormal(1.0))


def test_m4_with_plate_header_precedes_its_body():
    tex = to_latex(model_m4)
    lines = [l for l in tex.split("\n") if l.strip() and not l.strip().startswith("\\begin")
             and not l.strip().startswith("\\end")]
    assert any("for" in l and "obs" in l for l in lines), lines
    assert any("\\sigma" in l and "\\sim" in l for l in lines), lines
    # the header must come BEFORE the body it introduces (the M4 bug)
    i_header = next(i for i, l in enumerate(lines) if "for" in l and "obs" in l)
    i_body = next(i for i, l in enumerate(lines) if "\\sigma" in l and "\\sim" in l)
    assert i_header < i_body, lines


# ---------------------------------------------------------------------------
# M5 — the traced LaTeX path emitted no plate headers and marked observed
# sites with a trailing "% observed" comment.
# ---------------------------------------------------------------------------

Y_M5 = jnp.zeros(5)


def model_m5(y=None):
    with m.dist.plate("obs", 5):
        sigma = m.dist.half_normal(1.0, name="sigma")
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, sigma, name="y", obs=y)


def test_m5_trace_latex_marks_observed_sites_and_plates():
    # trace_model calls the model with the data dict as kwargs: the single
    # data key "y" must be a parameter of the model
    lines = _latex_from_trace(model_m5, {"y": Y_M5})
    tex = "\n".join(lines)
    # exactly the one observed site carries the marker, on its LHS
    assert tex.count(r"^{\text{obs}}") == 1
    assert r"y^{\text{obs}}" in tex
    assert r"\sigma \sim" in tex
    # plate header with the real size, in the AST-renderer style
    assert r"\text{for } obs \in 1, \dots, 5\text{:}" in tex
    # nested sites are indented one \quad per plate level
    assert any(l.startswith("\\quad") for l in lines), lines
    assert not any("observed" in l for l in lines), lines


# ---------------------------------------------------------------------------
# M3 — the bypass filter dropped traced edges whose intermediate was a
# deterministic node outside the trace (the audit's sigma -> y example).
# ---------------------------------------------------------------------------

def model_m3(y=None):
    sigma = numpyro.sample("sigma", numpyro.distributions.HalfNormal(1.0))
    mu = sigma * 2.0
    numpyro.deterministic("mu_site", mu)
    numpyro.sample("y", numpyro.distributions.Normal(mu, sigma), obs=y)


def test_m3_edge_through_bypassed_deterministic_is_kept():
    g = model_to_graph(model_m3, data={"y": jnp.zeros(3)})
    # y depends on sigma directly (as the Normal scale) AND through the
    # deterministic mu_site; the filter keeps edges whose source is a direct
    # jaxpr parent, so the traced sigma -> y dependency must survive it.
    # (mu_site itself stays a node: the filter prunes indirect edges, not
    # recorded deterministic sites.)
    assert g.exact is True
    assert ("sigma", "y") in g.edges


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
