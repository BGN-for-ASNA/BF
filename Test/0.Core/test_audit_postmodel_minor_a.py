"""Regression tests for RELEASE_AUDIT.md, module PostModel (to_latex) minors."""
import warnings

import jax.numpy as jnp
import numpyro
import pytest
import scipy.stats

import BayesForge.PostModel.to_latex as tl
from BayesForge import bf
from BayesForge.PostModel.to_latex import (
    to_latex, _body_lines, _escape_text, convert_to_greek)

m = bf(platform="cpu", print_devices_found=False)


# ---------------------------------------------------------------------------
# m1 — `"".join(part.capitalize() ...)` rendered lkj / lkj_cholesky as
# "Lkj" / "LkjCholesky" instead of the LKJ / LKJCholesky used by the block
# tables. The AST path never executes the model, so the helper calls below
# only need to parse.
# ---------------------------------------------------------------------------
def model_m1_lkj(y=None):
    R = m.dist.lkj(2, name="R")
    L = m.dist.lkj_cholesky(2, 2.0, name="L")
    m.dist.normal(0.0, 1.0, obs=y)


def test_m1_lkj_class_names():
    tex = to_latex(model_m1_lkj)
    assert r"\text{LKJ}(2)" in tex
    assert r"\text{LKJCholesky}(2, 2.0)" in tex
    assert "Lkj" not in tex


# ---------------------------------------------------------------------------
# m2 — `sigma ~ TruncatedNormal(0, 2.5)` omitted the real support low = 0
# (Network/model_effects.py: dist.truncated_normal(mu, sd, low=0, ...)).
# ---------------------------------------------------------------------------
def model_m2_truncated(y=None):
    sr = m.net.sender_receiver(X, name="sr")
    m.dist.normal(sr, 1.0, obs=y)


def test_m2_truncated_normal_support():
    tex = to_latex(model_m2_truncated)
    assert r"\sigma_{sr} \sim \text{TruncatedNormal}_{[0, \infty)}(0, 2.5)" in tex
    assert r"\text{TruncatedNormal}(0, 2.5)" not in tex


# ---------------------------------------------------------------------------
# m3 — K_SE(D, eta^2, rho^2, sigma^2) presented sigma^2 as a free kernel
# hyperparameter; gaussian_process (Utils/Gaussian.py) keeps sigmaq = 0.01
# FIXED and samples only etasq ~ Exponential(2), rhosq ~ Exponential(0.5).
# ---------------------------------------------------------------------------
def model_m3_gp(y=None):
    f = m.gaussian.gaussian_process(D)
    m.dist.normal(f, 1.0, obs=y)


def test_m3_gp_kernel_fixed_nugget():
    tex = to_latex(model_m3_gp)
    assert r"\sigma_{\text{fix}}^{2}" in tex
    assert r"\rho^{2}, \sigma^{2}" not in tex
    # the two genuinely sampled priors must stay
    assert r"\eta^{2} \sim \text{Exponential}(2)" in tex
    assert r"\rho^{2} \sim \text{Exponential}(0.5)" in tex


# ---------------------------------------------------------------------------
# m4/m5 — the renderer forced the non-centered parameterization even when
# the code asks for the centered one: edge_weights(partial_pooling=False)
# is a single Normal(prior_mu, prior_sigma) site, and
# nested_varying_effects(centered=True) samples corr_{name} ~ LKJ and
# {name}_effects ~ MVN (Effects.py / model_effects.py). The defaults
# (partial_pooling=True, centered=False) must keep the old decomposition.
# ---------------------------------------------------------------------------
def model_m4_edge_direct(y=None):
    w = m.net.edge_weights(10, partial_pooling=False)
    m.dist.normal(w, 1.0, obs=y)


def model_m4_edge_pooled(y=None):
    w = m.net.edge_weights(10)
    m.dist.normal(w, 1.0, obs=y)


def model_m5_nested_centered(y=None):
    a = m.effects.nested_varying_effects(
        2, ["r", "g"], [5, 20], [g1, g2], centered=True)
    m.dist.normal(a, 1.0, obs=y)


def model_m5_nested_noncentered(y=None):
    a = m.effects.nested_varying_effects(2, ["r", "g"], [5, 20], [g1, g2])
    m.dist.normal(a, 1.0, obs=y)


def test_m4_m5_centered_parameterizations():
    tex = to_latex(model_m4_edge_direct)
    assert r"w \sim \text{Normal}(0, 2.5)" in tex
    assert r"\mu_{w}" not in tex
    # default partial_pooling=True keeps the non-centered decomposition
    tex = to_latex(model_m4_edge_pooled)
    assert r"w = \mu_{w} + \sigma_{w} \cdot z_{w}" in tex

    tex = to_latex(model_m5_nested_centered)
    assert r"\text{MultivariateNormal}" in tex
    assert r"\mathbf{R}_{\ell} \sim \text{LKJ}(2)" in tex
    assert r"\mathbf{Z}_{\ell}" not in tex
    # default centered=False keeps the Z/L recursion
    tex = to_latex(model_m5_nested_noncentered)
    assert r"\mathbf{Z}_{\ell} \sim \text{Normal}(0, 1)" in tex


# ---------------------------------------------------------------------------
# m17 — mathematically wrong expression translations: slogdet rendered as
# det, generic ppf/cdf claimed the standard normal Phi/Phi^-1, `is` rendered
# as "=", `//` produced "\lfloor / \rfloor", keyword-only calls leaked
# "x=a" into \sqrt{...}, repeat(x, k) dropped k, _escape_text missed % & #,
# and convert_to_greek silently lowercased uppercase names without a
# dedicated command (Epsilon -> \epsilon).
# ---------------------------------------------------------------------------
def model_m17_exprs(y=None):
    A = jnp.eye(3)
    d = jnp.det(A)
    ldet = jnp.slogdet(A)
    qt = scipy.stats.t.ppf(0.95, 3.0)
    qn = scipy.stats.norm.ppf(0.95)
    p = scipy.stats.gamma.cdf(qt, 2.0)
    k = n_obs // 2
    rep = jnp.repeat(qt, 3)
    s = jnp.sqrt(x=k)
    w = 1.0 if A is None else 2.0
    m.dist.normal(w, 1.0, obs=y)


def test_m17_mathematically_wrong_translations():
    tex = to_latex(model_m17_exprs)
    # slogdet is log|det|, plain det keeps \det
    assert r"ldet = \log\left|\det\left(A\right)\right|" in tex
    assert r"d = \det\left(A\right)" in tex
    # Phi^-1 stays reserved for the normal; t.ppf / gamma.cdf say ppf / cdf
    assert r"\Phi^{-1}\left(0.95\right)" in tex
    assert r"\text{ppf}\left(0.95\right)" in tex
    assert r"\text{cdf}\left(qt\right)" in tex
    assert r"\Phi\left(qt\right)" not in tex
    # a // b is the floor of the fraction, not "a \lfloor / \rfloor b"
    assert r"\left\lfloor \frac{n_{obs}}{2} \right\rfloor" in tex
    assert r"\lfloor / \rfloor" not in tex
    # keyword-only call feeds the VALUE, not "x=k"
    assert r"\sqrt{k}" in tex
    assert r"\sqrt{x=k}" not in tex
    # repeat keeps its repetition count
    assert r"\text{repeat}\left(qt, 3\right)" in tex
    # `is` is not `=`
    assert r"\text{is}" in tex
    assert r"A = None" not in tex
    # LaTeX specials escaped, uppercase Greek without a command kept as text
    esc = _escape_text("50% & #1_x")
    assert r"\%" in esc and r"\&" in esc and r"\#" in esc and r"\_" in esc
    assert convert_to_greek("Epsilon") == r"\text{Epsilon}"
    assert convert_to_greek("Sigma") == r"\Sigma"
    assert convert_to_greek("sigma") == r"\sigma"


# ---------------------------------------------------------------------------
# m23 — degenerate line-by-line fallbacks: _body_lines dropped a single
# line even when the def signature spans several lines, and the to_latex
# fallback reversed lines instead of groups, separating `for` headers from
# their bodies.
# ---------------------------------------------------------------------------
def model_m23_loop(y=None):
    z = 0.0
    for t in range(3):
        z = z + 1.0
    sigma = numpyro.sample("sigma", numpyro.distributions.HalfNormal(1.0))
    numpyro.sample("y", numpyro.distributions.Normal(z, sigma), obs=y)


def test_m23_line_by_line_fallbacks_keep_headers_and_bodies(monkeypatch):
    # (a) _body_lines degenerate path: the py2 `print` statement makes
    # ast.parse fail, so the fallback branch is taken; a multi-line def
    # header must be dropped whole, not one line at a time.
    src = 'def model(\n    x,\n    y):\n    print "boom"\n    a = 1.0\n'
    out = _body_lines(src)
    assert "a = 1.0" in out
    assert "x," not in out
    assert "y):" not in out
    # correct case unchanged: single-line def through the AST path
    assert _body_lines("def model(x):\n    a = 1.0\n") == ["a = 1.0"]

    # (b) to_latex line-by-line fallback: forcing the AST statement path
    # to fail must still keep the `for` header ABOVE its body line when
    # the statement groups are reversed.
    def boom(*args, **kwargs):
        raise RuntimeError("forced AST statement failure")
    monkeypatch.setattr(tl, "_stmt_to_latex_lines", boom)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # m26 announces the fallback
        tex = tl.to_latex(model_m23_loop)
    lines = [l for l in tex.split("\n") if l.strip()]
    idx_header = next(i for i, l in enumerate(lines) if r"\text{for } t \in" in l)
    idx_body = next(i for i, l in enumerate(lines) if "z + 1.0" in l)
    assert idx_header < idx_body


# ---------------------------------------------------------------------------
# m26 — the AST -> line-by-line -> trace fallback chain swallowed every
# exception silently; each bascule must now emit a UserWarning naming what
# was skipped (the fallbacks themselves are kept).
# ---------------------------------------------------------------------------
def test_m26_fallback_chain_warns(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("forced AST failure")
    monkeypatch.setattr(tl, "_stmt_to_latex_lines", boom)
    with pytest.warns(UserWarning, match="full AST rendering failed"):
        tex = tl.to_latex(model_m23_loop)
    assert "gather*" in tex

    # second bascule: line-by-line also fails -> the trace fallback is
    # announced too (the trace itself is stubbed, no sampling needed).
    monkeypatch.setattr(tl, "_body_lines", boom)
    monkeypatch.setattr(tl, "_latex_from_trace",
                        lambda model, data: [r"x \sim \text{Normal}(0, 1)"])
    with pytest.warns(UserWarning) as record:
        tex = tl.to_latex(model_m23_loop, data={})
    messages = [str(w.message) for w in record]
    assert any("full AST rendering failed" in msg for msg in messages)
    assert any("execution-trace rendering" in msg for msg in messages)
    assert r"x \sim \text{Normal}(0, 1)" in tex
