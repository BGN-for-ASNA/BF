"""Regression test for RELEASE_AUDIT.md, module 08 (PostModel/to_latex.py), critical C1."""
from BayesForge import bf
from BayesForge.PostModel.to_latex import to_latex

m = bf(platform="cpu", print_devices_found=False)


def model_vis(group, gi, gs, y):
    a, b = m.effects.varying_intercept_slope(10, group, gi, gs)
    m.dist.normal(a + b, 1.0, obs=y)


def model_ve(group, x, y):
    a, b = m.effects.varying_effects(1, 10, group, x)
    m.dist.normal(a + b, 1.0, obs=y)


def _text(out):
    return out if isinstance(out, str) else str(out)


def test_c1_varying_intercept_slope_has_no_invented_global_priors():
    tex = _text(to_latex(model_vis))
    assert "\\bar{\\alpha} \\sim" not in tex
    assert "\\bar{\\beta} \\sim" not in tex


def test_c1_varying_effects_keeps_its_sampled_global_priors():
    tex = _text(to_latex(model_ve))
    assert "\\bar{\\alpha} \\sim \\text{Normal}(0, 2)" in tex
