"""Regression tests for RELEASE_AUDIT.md, module Utils (Effects, Gaussian, link) minors."""
import ast
import inspect

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.handlers
import pytest

import BayesForge.Utils.Effects as effects_mod
import BayesForge.Utils.Gaussian as gaussian_mod
import BayesForge.Utils.link as link_mod
from BayesForge.Utils.Effects import effects
from BayesForge.Utils.Gaussian import gaussian
from BayesForge.Utils.link import link


def _module_func(module, name):
    """Return the ast.FunctionDef of a top-level (possibly nested) function."""
    src = inspect.getsource(module)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node, src
    raise AssertionError(f"{name} not found in {module.__name__}")


# --- [Mineure] Docstrings « centered »/« non-centered » inversées --------------
def test_centered_noncentered_docstrings_match_code():
    ve = effects.varying_effects.__doc__
    # Default is centered=False and the default branch is the non-centered one.
    assert "non-centered" in ve
    assert '"centered" parameterization to sample' not in ve
    # corr is used as a Cholesky factor in the non-centered branch.
    corr_doc = ve[ve.index("corr ("):]
    assert "Cholesky" in corr_doc[:500]
    # varying_intercept samples alpha ~ N(a_bar, sigma): centered.
    vi = effects.varying_intercept.__doc__
    assert "centered parameterization" in vi
    assert "non-centered" not in vi


# --- [Mineure] Code mort et pièges divers (Effects) -----------------------------
def test_effects_dead_code_and_traps():
    # `import jax`, not the cosmetic `import jax as jax`.
    tree = ast.parse(inspect.getsource(effects_mod))
    jax_asnames = [
        alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "jax"
    ]
    assert jax_asnames == [None]

    # diag_pre_multiply: no materialised jnp.diag in the code, exact for 2D m.
    node, src = _module_func(effects_mod, "diag_pre_multiply")
    code = "\n".join(ast.get_source_segment(src, s) or "" for s in node.body[1:])
    assert "jnp.diag" not in code
    v = jnp.array([2.0, -1.0, 0.5])
    m = jnp.arange(6.0).reshape(3, 2)
    np.testing.assert_allclose(effects.diag_pre_multiply(v, m), np.diag(v) @ m)

    # varying_intercept_slope: the dead pre-assignment was removed, so
    # varying_intercept is assigned exactly once.
    node, src = _module_func(effects_mod, "varying_intercept_slope")
    assigns = [
        stmt
        for stmt in ast.walk(node)
        if isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "varying_intercept" for t in stmt.targets)
    ]
    assert len(assigns) == 1

    # nested_varying_effects: a length-1 mu with N_vars=2 warns instead of
    # silently broadcasting the same offset onto intercept and slopes.
    with pytest.warns(UserWarning, match="mu should be"), numpyro.handlers.seed(rng_seed=0):
        effects.nested_varying_effects(
            N_vars=2,
            names=["a", "b"],
            N_groups=[2, 3],
            group_ids=[jnp.array([0, 0, 1, 1, 1]), jnp.array([0, 1, 2, 2, 2])],
            mu=[jnp.zeros(1)],
        )

    # The return arity depending on N_vars is documented (kept behaviour).
    doc = effects.nested_varying_effects.__doc__
    assert "array or tuple" in doc and "N_vars == 1" in doc
    with numpyro.handlers.seed(rng_seed=0):
        out1 = effects.nested_varying_effects(
            N_vars=1, names=["g"], N_groups=[3],
            group_ids=[jnp.array([0, 1, 2])], sample=True)
        out2 = effects.nested_varying_effects(
            N_vars=2, names=["g"], N_groups=[3],
            group_ids=[jnp.array([0, 1, 2])], sample=True)
    assert not isinstance(out1, tuple) and out1.shape == (3,)
    assert isinstance(out2, tuple) and len(out2) == 2


# --- [Mineure] Corps de classe dupliqué (Gaussian) ------------------------------
def test_gaussian_class_defined_once_and_kernel_periodic_local():
    tree = ast.parse(inspect.getsource(gaussian_mod))
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "gaussian"][0]
    names = [n.name for n in cls.body if isinstance(n, ast.FunctionDef)]
    assert len(names) == len(set(names)), f"duplicate method definitions: {names}"
    for expected in ("__init__", "distance_matrix", "kernel_sq_exp", "kernel_exp",
                     "kernel_periodic", "kernel_periodic_local",
                     "gaussian_process", "gaussian_process2"):
        assert expected in names

    # The kernel in effect is exp(-m^2 / (2 ls^2)), not the dead exp(-(m^2/2) ls^2).
    m = jnp.array([0.0, 0.5, 1.0, 2.0])
    ls, sig, per = 1.3, 2.0, 1.7
    K = gaussian.kernel_periodic_local(m, sigma=sig, length_scale=ls, period=per)
    periodic = sig**2 * jnp.exp(-2 * jnp.sin(jnp.pi * m / per) ** 2 / ls**2)
    np.testing.assert_allclose(K, periodic * jnp.exp(-m**2 / (2 * ls**2)), rtol=1e-5)
    assert not np.allclose(K, periodic * jnp.exp(-(m**2 / 2) * ls**2))


# --- [Mineure] gaussian_process — contrat et sites nommés ------------------------
def test_gaussian_process_contract():
    doc = gaussian.gaussian_process.__doc__
    assert "draw" in doc.lower()
    assert "NOT a length-scale" in doc  # etasq is the amplitude
    assert "Inverse squared length-scale" in doc  # rhosq is the decay rate
    for site in ("'etasq'", "'rhosq'", "'kernel'"):
        assert site in doc

    D = gaussian.distance_matrix(jnp.arange(4.0))
    tr = numpyro.handlers.trace(
        numpyro.handlers.seed(lambda: gaussian.gaussian_process(D), rng_seed=0)
    ).get_trace()
    sites = {s["name"] for s in tr.values() if s["type"] == "sample"}
    assert {"etasq", "rhosq", "kernel"} <= sites
    assert tr["kernel"]["value"].shape == (4,)


# --- [Mineure] inv_logit_scale — docstring sur le range --------------------------
def test_inv_logit_scale_range_documented():
    doc = link.inv_logit_scale.__doc__
    assert "between -1 and 1" in doc
    assert "between 0 and 1" not in doc
    assert float(link.inv_logit_scale(0.0)) == 0.0
    assert float(link.inv_logit_scale(8.0)) < 1.0
    assert float(link.inv_logit_scale(-8.0)) > -1.0
    assert float(link.inv_logit_scale(2.0)) == pytest.approx(2 / (1 + np.exp(-2)) - 1, rel=1e-5)


# --- [Mineure] link — code mort et cohérence de classe ----------------------------
def test_link_dead_code_and_class_consistency():
    # No unreachable statement after the return of logit.
    node, src = _module_func(link_mod, "logit")
    body = node.body
    for i, stmt in enumerate(body):
        if isinstance(stmt, ast.Return):
            assert i == len(body) - 1, "statement after return in logit"
    assert "inv_logit(x) - 0.5" not in inspect.getsource(link_mod)

    # Uniform recompilation behaviour: every link transform is jitted.
    for name in ("logit", "inv_logit", "inv_logit_scale", "probit", "inv_probit",
                 "log", "exp", "cloglog", "inv_cloglog", "reciprocal", "sqrt", "square"):
        assert "Pjit" in type(getattr(link, name)).__name__, name

    # reciprocal(0) -> inf, documented rather than silent.
    assert bool(jnp.isinf(link.reciprocal(jnp.float32(0.0))))
    assert "inf" in link.reciprocal.__doc__

    # Class docstring documents the missing Jacobians and the LKJcor placement.
    assert "Jacob" in link.__doc__
    assert "density" in link.__doc__ and "not a link" in link.__doc__
