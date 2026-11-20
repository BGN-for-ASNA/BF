"""varying_intercept_slope: Cholesky scale passed directly, and a correlation
matrix passed by mistake as L_corr is rejected."""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

import BayesForge.Utils.Effects as effects_mod
from BayesForge.Utils.Effects import effects

R = jnp.array([[1.0, 0.6], [0.6, 1.0]])
L = jnp.linalg.cholesky(R)
SD = jnp.array([2.0, 0.5])
GROUP = jnp.array([0, 1, 2])

# Run varying intercept slope model
def _run(L_corr, group_std=SD):
    return numpyro.handlers.seed(
        lambda: effects.varying_intercept_slope(
            N_group=3, group=GROUP, global_intercept=0.0, global_slope=0.0,
            group_std=group_std, L_corr=L_corr),
        rng_seed=0)()

# Ensure Cholesky factor is correctly used by MVN
def test_mvn_receives_the_scaled_cholesky_factor(monkeypatch):
    seen = {}
    original = effects_mod.dist.multivariate_normal

    def spy(*args, **kwargs):
        seen.update(kwargs)
        seen["n_args"] = len(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(effects_mod.dist, "multivariate_normal", spy)
    _run(L)
    assert seen.get("covariance_matrix") is None and seen["n_args"] == 1
    np.testing.assert_allclose(np.asarray(seen["scale_tril"]),
                               np.diag(np.asarray(SD)) @ np.asarray(L), rtol=1e-6)

# Ensure correlation matrix instead of cholesky factor raises error
def test_correlation_matrix_instead_of_cholesky_factor_raises():
    with pytest.raises(ValueError, match="cholesky"):
        _run(R)

# Ensure valid cholesky factor and traced factor are accepted
def test_valid_cholesky_factor_and_traced_factor_are_accepted():
    a, b = _run(L)
    assert a.shape == (3,) and b.shape == (3,)
    # Under jit L_corr is a tracer: no value check, no error.
    out = jax.jit(lambda Lc: _run(Lc)[0])(L)
    assert out.shape == (3,)


# --- review follow-up ---------------------------------------------------------
# Ensure error message distinguish covariance from correlation
def test_message_tells_covariance_from_correlation():
    with pytest.raises(ValueError, match="correlation matrix R"):
        _run(R)
    with pytest.raises(ValueError, match="full .*covariance"):
        _run(jnp.array([[4.0, 0.6], [0.6, 1.0]]))
    with pytest.raises(ValueError, match="rows do not have unit norm"):
        _run(jnp.array([[2.0, 0.0], [0.6, 0.8]]))

# Ensure default lkj cholesky factor is not checked
def test_default_lkj_cholesky_factor_is_not_checked(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("default L_corr should not be checked")
    monkeypatch.setattr(effects_mod, "_check_cholesky_factor", boom)
    numpyro.handlers.seed(
        lambda: effects.varying_intercept_slope(
            N_group=3, group=GROUP, global_intercept=0.0, global_slope=0.0,
            group_std=SD, sample=True),
        rng_seed=0)()

# Ensure group_std must have two entries
def test_group_std_must_have_two_entries():
    with pytest.raises(ValueError, match="group_std must have shape"):
        _run(L, group_std=jnp.array([1.0, 2.0, 3.0]))

# Ensure bad shapes for diag_pre_multiply raise error even under jit
def test_diag_pre_multiply_rejects_wrong_shapes_even_under_jit():
    v, m2 = jnp.array([1.0, 2.0]), jnp.eye(2)
    np.testing.assert_allclose(np.asarray(effects.diag_pre_multiply(v, m2)), np.diag([1.0, 2.0]))
    for bad_v, bad_m in ((v, jnp.ones(2)), (jnp.eye(2), m2), (jnp.ones(3), m2)):
        with pytest.raises(ValueError, match="diag_pre_multiply"):
            effects.diag_pre_multiply(bad_v, bad_m)
    with pytest.raises(ValueError, match="diag_pre_multiply"):
        jax.jit(lambda a: effects.diag_pre_multiply(a, jnp.ones(2)))(v)

# Ensure warnign about mu shape is raised also for traced mu
def test_nested_mu_shape_warning_also_for_traced_mu():
    import warnings

    def run(mu0):
        with numpyro.handlers.seed(rng_seed=0):
            effects.nested_varying_effects(
                N_vars=2, names=["a", "b"], N_groups=[2, 3],
                group_ids=[jnp.array([0, 0, 1, 1, 1]), jnp.array([0, 1, 2, 2, 2])],
                mu=[mu0])
        return 0.0

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        jax.jit(run)(jnp.zeros(1))
    assert any("mu should be" in str(w.message) for w in rec)
