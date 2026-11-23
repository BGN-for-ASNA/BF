"""Drift detector for RELEASE_AUDIT.md Integration X3: the Build generators must reproduce the production distribution wrappers (except the header)."""

import ast
import pathlib

import pytest

HERE = pathlib.Path(__file__).parent
REPO_ROOT = HERE.parents[1]
BUILD_DIR = REPO_ROOT / "BayesForge" / "Build"
DIST_DIR = REPO_ROOT / "BayesForge" / "Distributions"

# Production methods that no generator run can ever produce, so their absence
# from the generated file is not drift. Each entry is a name defined on the
# production ``UnifiedDist`` but never emitted by the matching generator.
HAND_MAINTAINED_NP = {
    # Session RNG plumbing and the REPL helper, copied into np_dists.py by hand.
    "_draw_key",
    "_resolve_infer",
    "normalize_seed",
    "track",
    # Hand-written wrappers with no numpyro class of the same name.
    "car",
    "factor",
    "sine_bivariate_vonmises",  # numpyro.SineBivariateVonMises generates sine_bivariate_von_mises
    # The generator's naive CamelCase split (re.sub(r"(?<!^)(?=[A-Z])", "_", ...))
    # turns LKJ/LKJCholesky into l_k_j/l_k_j_cholesky, never lkj/lkj_cholesky.
    "lkj",
    "lkj_cholesky",
}

HAND_MAINTAINED_TFP = {
    # Hand-written convenience wrappers: tfd has no class with this name in its
    # namespace (Bernoulli exposes .probs/.logits classmethods, not BernoulliProbs).
    "asymmetric_laplace",
    "asymmetric_laplace_quantile",
    "bernoulli_logits",
    "bernoulli_probs",
    "beta_proportion",
    "binomial_logits",
    "binomial_probs",
    "categorical_logits",
    "categorical_probs",
    "delta",
    "discrete_uniform",
    "gamma_poisson",
    "gaussian_random_walk",
    "multinomial_logits",
    "multinomial_probs",
    "negative_binomial_logits",
    "negative_binomial_probs",
    "truncated_distribution",
    "unit",
    "wishart",
    "zero_inflated_poisson",
    # Naive CamelCase split: tfd.LKJ generates l_k_j, tfd.MultivariateNormalDiag
    # generates multivariate_normal_diag.
    "lkj",
    "multivariate_normal",
    # generate() skips tfd.Distribution itself (obj is not tfd.Distribution guard).
    "distribution",
}


def _load_generator(filename):
    """Execute a Build generator file and return its module namespace.

    Done with ``compile``/``exec`` instead of an import statement so this test
    module keeps its top-level imports to the standard library plus pytest; the
    generators import jax/numpyro/TFP at module level, so callers must run
    ``pytest.importorskip`` for the right backend first.
    """
    path = BUILD_DIR / filename
    namespace = {"__name__": path.stem, "__file__": str(path)}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


def _unified_method_names(path):
    """Return the method names defined on the ``UnifiedDist`` class in ``path``."""
    tree = ast.parse(pathlib.Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "UnifiedDist":
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"no UnifiedDist class found in {path}")


def test_np_generator_reproduces_np_dists_methods(tmp_path):
    """Compare UnifiedDist method *names* only: the generator header states the
    production np_dists.py is maintained by hand, so bodies and signatures
    legitimately differ (infer/to_jax/enum kwargs, _draw_key/_mask_draws
    plumbing). Drift here means a numpyro distribution lost its wrapper.
    """
    pytest.importorskip("numpyro")
    generator = _load_generator("Build_NP_light.py")

    out_path = tmp_path / "dists_generated.py"
    generator["generate"](str(out_path), docs=generator["read_doc_cache"]())

    generated = _unified_method_names(out_path)
    production = _unified_method_names(DIST_DIR / "np_dists.py")
    assert generated, "Build_NP_light.py generated no UnifiedDist methods"

    missing = sorted(set(production) - generated - HAND_MAINTAINED_NP)
    assert not missing, (
        "Build_NP_light.py no longer generates methods that np_dists.py defines: "
        f"{missing}. Either the generator lost a distribution or np_dists.py grew a "
        "hand-written method that must be added to HAND_MAINTAINED_NP."
    )


def test_tfp_generator_reproduces_tfp_dists_methods(tmp_path):
    """Build_TFP_light.py's docstring says tfp_dists.py is hand-maintained and
    the generated file is only a diffing starting point, so no byte comparison
    is expected. Signatures are not compared either: the production file
    deliberately reorders arguments, adds defaults and wrapper arguments
    (infer, to_jax, enum, name, ...). What must not drift is the set of
    UnifiedDist method names.
    """
    pytest.importorskip("tensorflow_probability")
    generator = _load_generator("Build_TFP_light.py")

    out_path = tmp_path / "tfp_dists_generated.py"
    generator["generate"](str(out_path))

    generated = _unified_method_names(out_path)
    production = _unified_method_names(DIST_DIR / "tfp_dists.py")
    assert generated, "Build_TFP_light.py generated no UnifiedDist methods"

    missing = sorted(set(production) - generated - HAND_MAINTAINED_TFP)
    assert not missing, (
        "Build_TFP_light.py no longer generates methods that tfp_dists.py defines: "
        f"{missing}. Either the generator lost a distribution or tfp_dists.py grew a "
        "hand-written method that must be added to HAND_MAINTAINED_TFP."
    )


def test_generators_refuse_to_overwrite_production_files(tmp_path):
    """Both generators must raise ValueError, before writing anything, when the
    output path is named like a production wrapper. The paths are built under
    tmp_path so a broken guard cannot touch the real files.
    """
    pytest.importorskip("numpyro")
    pytest.importorskip("tensorflow_probability")
    generators = [_load_generator("Build_NP_light.py"), _load_generator("Build_TFP_light.py")]

    for generator in generators:
        for production_name in ("np_dists.py", "tfp_dists.py"):
            out_path = tmp_path / production_name
            with pytest.raises(ValueError, match="Refusing to overwrite production file"):
                generator["generate"](str(out_path))
            assert not out_path.exists(), f"{out_path} was written despite the guard"
