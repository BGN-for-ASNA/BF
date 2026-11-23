"""Regression tests for RELEASE_AUDIT.md, np_dists.py minors m1-m10."""

import ast
import os

import pytest

_SRC = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "BayesForge", "Distributions", "np_dists.py"))


def _source():
    with open(_SRC, encoding="utf-8") as fh:
        return fh.read()


def _func(name):
    """Return the ast.FunctionDef of the top-level method ``name``."""
    tree = ast.parse(_source())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name} not found in {_SRC}")


def _docstring(name):
    return ast.get_docstring(_func(name)) or ""


# ---------------------------------------------------------------- m1

def test_m1_lkj_authors_are_lewandowski_kurowicka_joe():
    src = _source()
    assert "Leonard" not in src
    assert "Lewandowski-Kurowicka-Joe" in src


# ---------------------------------------------------------------- m2

def test_m2_power_law_alpha_constraint_matches_formula():
    doc = _docstring("lower_truncated_power_law")
    assert "Must be less than -1" not in doc
    assert "Must be greater than 1" in doc
    # the formula line and the Args line must now agree
    assert r"\alpha > 1" in doc


# ---------------------------------------------------------------- m3

def test_m3_scale_documented_as_scale_not_sample():
    src = _source()
    assert "*sample* (float)" not in src
    for name in ("asymmetric_laplace_quantile", "truncated_normal"):
        assert "*scale* (float): The scale parameter" in _docstring(name), name


# ---------------------------------------------------------------- m4

def test_m4_improper_uniform_docstring_clean_and_complete():
    doc = _docstring("improper_uniform")
    # TruncatedDistribution contamination removed (it belongs to
    # truncated_distribution, which keeps its own references).
    assert "Truncated" not in doc
    assert "Greene" not in doc
    assert "LaplacesDemon" not in doc
    # Args list the method's real parameters (signature: support,
    # batch_shape, event_shape, validate_args, name, obs, mask, infer,
    # sample, seed, shape, event, create_obj, to_jax; the house style
    # documents all but validate_args/infer/to_jax).
    for param in ("support", "batch_shape", "event_shape", "shape", "event",
                  "mask", "create_obj", "sample", "seed", "obs", "name"):
        assert f"*{param}*" in doc, param


# ---------------------------------------------------------------- m5

def test_m5_bernoullilogits_returns_only_in_own_method():
    tree = ast.parse(_source())
    seen = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        doc = ast.get_docstring(node) or ""
        if node.name == "bernoulli_logits":
            seen += doc.count("BernoulliLogits distribution")
            continue
        # RelaxedBernoulliLogits legitimately contains the substring.
        cleaned = doc.replace("RelaxedBernoulliLogits", "")
        assert "BernoulliLogits distribution" not in cleaned, (
            f"stale BernoulliLogits Returns in {node.name}")
    assert seen > 0, "bernoulli_logits itself lost its Returns wording"


# ---------------------------------------------------------------- m6

def test_m6_gaussian_copula_beta_validate_args_defaults_to_none():
    fn = _func("gaussian_copula_beta")
    argnames = [a.arg for a in fn.args.args]
    idx = argnames.index("validate_args")
    defaults = fn.args.defaults
    default = defaults[idx - (len(argnames) - len(defaults))]
    assert isinstance(default, ast.Constant) and default.value is None


# ---------------------------------------------------------------- m7

def test_m7_no_unused_matplotlib_or_dist_import():
    src = _source()
    assert "import matplotlib" not in src
    assert "import numpyro.distributions as dist" not in src
    # nothing ever referenced plt./dist. in code, so nothing broke:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Name):
            assert node.id not in ("plt", "dist")


# ---------------------------------------------------------------- m8

def test_m8_infer_caller_name_has_no_dead_return():
    fn = _func("_infer_caller_name")
    body = [s for s in fn.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.Try), (
        "unreachable statement after the try/except in _infer_caller_name")


# ---------------------------------------------------------------- m9

def test_m9_seeds_beyond_32_bits_are_not_truncated():
    import jax
    from BayesForge.Distributions.np_dists import UnifiedDist

    d = UnifiedDist()
    a = d.normal(0, 1, sample=True, seed=2**40)
    b = d.normal(0, 1, sample=True, seed=2**40 + 2**32)
    assert float(a) != float(b), (
        "seeds 2**40 and 2**40 + 2**32 collapsed to the same PRNGKey")

    # seeds below 2**32 keep exactly the old code path: PRNGKey(seed)
    c = d.normal(0, 1, sample=True, seed=5)
    assert float(c) == float(jax.random.normal(jax.random.PRNGKey(5)))
    e = d.normal(0, 1, sample=True, seed=5)
    assert float(c) == float(e)

    # negative seeds draw (no uint32 overflow) and differ from their absolute value
    n = d.normal(0, 1, sample=True, seed=-7)
    p = d.normal(0, 1, sample=True, seed=7)
    assert float(n) != float(p)


# ---------------------------------------------------------------- m10

def test_m10_empty_infer_dict_is_kept_on_the_site():
    src = _source()
    assert "dict(infer) if infer else None" not in src
    assert "dict(infer) if infer is not None else None" in src

    numpyro = pytest.importorskip("numpyro")
    from BayesForge.Distributions.np_dists import UnifiedDist

    d = UnifiedDist()

    def model():
        d.normal(0, 1, name="x", infer={})

    trace = numpyro.handlers.trace(
        numpyro.handlers.seed(model, 0)).get_trace()
    assert trace["x"]["infer"] == {}
