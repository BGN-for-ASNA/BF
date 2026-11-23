"""Regression tests for RELEASE_AUDIT.md, tfp_dists.py minors 1-17."""
import inspect
import os
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("tensorflow_probability")

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd

from BayesForge.Distributions.tfp_dists import UnifiedDist

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "BayesForge" / "Distributions" / "tfp_dists.py"
SRC = SRC_PATH.read_text()
HEADER = "\n".join(SRC.splitlines()[:20])


def _doc(name):
    return getattr(UnifiedDist, name).__doc__


# --- minor 1: single (effective) delta definition ---------------------------

def test_minor1_delta_defined_once_with_v_signature():
    assert SRC.count("def delta(") == 1
    assert list(inspect.signature(UnifiedDist.delta).parameters)[0] == "v"
    dist = UnifiedDist.delta(v=2.0, wrap=False)
    assert float(dist.loc) == 2.0


# --- minor 2: log_normal builds tfd.LogNormal once --------------------------

def test_minor2_log_normal_built_once_and_works():
    assert SRC.count("tfd.LogNormal(") == 1
    dist = UnifiedDist.log_normal(create_obj=True)
    assert dist.name == "LogNormal"
    assert jnp.isfinite(dist.log_prob(1.0))
    draws = UnifiedDist.log_normal(sample=True, shape=(4,), seed=7)
    again = UnifiedDist.log_normal(sample=True, shape=(4,), seed=7)
    assert draws.shape == (4,)
    assert bool(jnp.all(draws == again))


# --- minor 3: non-samplable wrappers raise clearly on sample=True -----------

NON_SAMPLABLE = {
    "distribution": dict(dtype=jnp.float32),
    "auto_composite_tensor_distribution": dict(dtype=jnp.float32),
}


@pytest.mark.parametrize("name", sorted(NON_SAMPLABLE))
def test_minor3_non_samplable_wrappers(name):
    fn = getattr(UnifiedDist, name)
    assert "sample=True is unsupported" in fn.__doc__
    with pytest.raises(NotImplementedError, match="sample=True"):
        fn(reparameterization_type=tfd.NOT_REPARAMETERIZED, validate_args=False,
           allow_nan_stats=True, sample=True, seed=0, **NON_SAMPLABLE[name])
    # construction without sampling keeps working (unchanged behaviour)
    dist = fn(reparameterization_type=tfd.NOT_REPARAMETERIZED, validate_args=False,
              allow_nan_stats=True, create_obj=True, **NON_SAMPLABLE[name])
    assert isinstance(dist, tfd.Distribution)


# --- minors 4 and 5: header comments ----------------------------------------

def test_minor4_header_documents_partial_lkj_rename():
    assert "cholesky_l_k_j" in HEADER
    assert "p_e_r_t" in HEADER
    # the generated names are kept as methods (no rename, doc-only fix)
    assert hasattr(UnifiedDist, "cholesky_l_k_j")
    assert hasattr(UnifiedDist, "p_e_r_t")


def test_minor5_header_documents_wrap_not_warp():
    assert 'wrap=True, *args, **kwargs"' in HEADER
    assert "not `warp`" in HEADER
    assert "wrap" in inspect.signature(UnifiedDist.normal).parameters
    assert "warp" not in inspect.signature(UnifiedDist.normal).parameters


# --- minor 6: multivariate_normal_diag requires loc or scale_diag -----------

def test_minor6_multivariate_normal_diag_requires_args():
    with pytest.raises(ValueError, match="loc or scale_diag"):
        UnifiedDist.multivariate_normal_diag(create_obj=True)
    by_loc = UnifiedDist.multivariate_normal_diag(loc=jnp.zeros(3), create_obj=True)
    assert tuple(by_loc.event_shape) == (3,)
    by_scale = UnifiedDist.multivariate_normal_diag(scale_diag=jnp.ones(2), create_obj=True)
    assert tuple(by_scale.event_shape) == (2,)


# --- minor 7: dtype follows jax_enable_x64 ----------------------------------

_DTYPE_PROBE = """
import jax.numpy as jnp
from BayesForge.Distributions.tfp_dists import UnifiedDist
print(jnp.dtype(UnifiedDist.continuous_bernoulli(probs=0.5, create_obj=True).dtype).name)
print(jnp.dtype(UnifiedDist.flat_dirichlet(concentration_shape=(3,), create_obj=True).dtype).name)
print(jnp.dtype(UnifiedDist.spherical_uniform(dimension=3, create_obj=True).dtype).name)
"""


@pytest.mark.parametrize("x64,expected", [("0", "float32"), ("1", "float64")])
def test_minor7_dtype_follows_x64(x64, expected):
    env = dict(os.environ, JAX_PLATFORMS="cpu", JAX_ENABLE_X64=x64)
    proc = subprocess.run([sys.executable, "-c", _DTYPE_PROBE], cwd=str(REPO_ROOT),
                          env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    # the BayesForge import banner may print a line first: keep the last 3 tokens
    assert proc.stdout.split()[-3:] == [expected] * 3


# --- minor 8: truncated_distribution keyword note ---------------------------

def test_minor8_truncated_distribution_note():
    doc = _doc("truncated_distribution")
    assert "differs from the numpyro backend" in doc
    assert "base_dist" in doc


# --- minor 9: half_cauchy loc note ------------------------------------------

def test_minor9_half_cauchy_note():
    doc = _doc("half_cauchy")
    assert "differs from the numpyro backend" in doc
    assert "loc" in doc


# --- minor 10: delta log_density/event_dim note -----------------------------

def test_minor10_delta_note():
    doc = _doc("delta")
    assert "differs from the numpyro backend" in doc
    assert "log_density" in doc and "event_dim" in doc


# --- minor 11: beta_binomial total_count dtype documented -------------------

def test_minor11_beta_binomial_total_count_documented():
    doc = _doc("beta_binomial")
    assert "total_count" in doc and "result_type" in doc
    # behaviour unchanged: integer total_count follows the concentrations' dtype
    dist = UnifiedDist.beta_binomial(total_count=10, concentration1=1.0,
                                     concentration0=2.0, create_obj=True)
    assert dist.total_count.dtype == jnp.float32


# --- minor 12: beta_proportion validates mean in (0, 1) ---------------------

@pytest.mark.parametrize("mean", [0.0, 1.0, 1.2, -0.5])
def test_minor12_beta_proportion_rejects_bad_mean(mean):
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        UnifiedDist.beta_proportion(mean=mean, concentration=2.0)


def test_minor12_beta_proportion_accepts_valid_mean():
    dist = UnifiedDist.beta_proportion(mean=0.3, concentration=2.0, wrap=False)
    assert float(dist.concentration1) == pytest.approx(0.6)
    assert float(dist.concentration0) == pytest.approx(1.4)
    draws = UnifiedDist.beta_proportion(mean=0.5, concentration=2.0,
                                        sample=True, shape=(4,), seed=1)
    assert draws.shape == (4,) and bool(jnp.all((draws > 0) & (draws < 1)))


def test_minor12_beta_proportion_skips_validation_for_tracers():
    def f(m):
        UnifiedDist.beta_proportion(mean=m, concentration=2.0, wrap=False)
        return jnp.asarray(0.0)

    assert float(jax.jit(f)(1.2)) == 0.0


# --- minor 13: zero_inflated_poisson rate default matches np ----------------

def test_minor13_zero_inflated_poisson_rate_default():
    assert inspect.signature(UnifiedDist.zero_inflated_poisson).parameters["rate"].default == 1.0


# --- minor 14: tfb alias is used, so it stays -------------------------------

def test_minor14_tfb_alias_used():
    assert "tfb = tfp.bijectors" in SRC
    assert "tfb.Chain" in SRC


# --- minor 15: package docstring lists both backends ------------------------

def test_minor15_package_docstring():
    import BayesForge.Distributions as dist_pkg

    doc = dist_pkg.__doc__ or ""
    assert "np_dists" in doc and "tfp_dists" in doc
    init_src = (REPO_ROOT / "BayesForge" / "Distributions" / "__init__.py").read_text()
    assert not any(line.startswith(("import ", "from ")) for line in init_src.splitlines())


# --- minors 16 and 17: naming/default divergences documented ----------------

@pytest.mark.parametrize("name", ["doublesided_maxwell", "binomial",
                                  "generalized_pareto", "johnson_s_u"])
def test_minor16_name_conventions_documented(name):
    assert "differs from the numpyro backend" in _doc(name)


def test_minor17_von_mises_note_and_aligned_defaults():
    assert "differs from the numpyro backend" in _doc("von_mises")
    sigs = {name: inspect.signature(getattr(UnifiedDist, name)).parameters
            for name in ("normal", "student_t", "cauchy", "laplace",
                         "log_normal", "exponential")}
    for name in ("normal", "cauchy", "laplace", "log_normal"):
        assert sigs[name]["loc"].default == 0.0
        assert sigs[name]["scale"].default == 1.0
    assert sigs["student_t"]["loc"].default == 0.0
    assert sigs["student_t"]["scale"].default == 1.0
    assert sigs["exponential"]["rate"].default == 1.0
