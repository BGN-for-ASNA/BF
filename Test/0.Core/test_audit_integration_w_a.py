"""Regression tests for RELEASE_AUDIT.md, Integration W2, W4-W8."""

import jax.numpy as jnp
import numpyro
import pytest

from BayesForge import bf
from BayesForge.Models.SRM import SRM
from BayesForge.NBDA.NBDA import NBDA
from BayesForge.Network.model_effects import Neteffect
from BayesForge.Utils.link import link

# Module-level instance for the model-tracing tests (W6), like the other
# audit regression files. W2 builds its own bf right before asserting, so a
# bf constructed by another test file in the same run cannot clobber the
# binding identity it checks.
m = bf(platform="cpu", print_devices_found=False, rand_seed=7)


# ---------------------------------------------------------------------------
# W2 — model_effects2, PCA and survival_old singletons were missing from
# bf._bind_library_rng: their sample=True draws went through time_ns() per
# call (non-reproducible DGP even with a single bf + set_seed), and each
# pca/survival_old instance built a fresh UnifiedDist with _rng=None.
# ---------------------------------------------------------------------------
# Ensure RNG of library is bounded to the modules of bf instance
def test_w2_bind_library_rng_covers_model_effects2_pca_surv():
    m7 = bf(platform="cpu", print_devices_found=False, rand_seed=7)

    import BayesForge.Network.model_effects2 as ME2
    import BayesForge.Models.PCA as PCA_mod
    import BayesForge.Models.surv as surv_mod

    # module-level singletons rebound to THIS instance's RNG
    assert ME2.dist._rng is m7._rng
    assert PCA_mod._dist._rng is m7._rng
    assert surv_mod.dist._rng is m7._rng

    # survival_old used to build a fresh dist() per instance (surv.py L73)
    assert surv_mod.survival_old(m7).dist is surv_mod.dist

# Ensure PCA module share the same dist with bound RNG across instances
def test_w2_pca_instances_share_bound_singleton():
    m7 = bf(platform="cpu", print_devices_found=False, rand_seed=7)
    import BayesForge.Models.PCA as PCA_mod

    p = PCA_mod.pca()
    assert p.dist is PCA_mod._dist
    assert p.dist._rng is m7._rng


# ---------------------------------------------------------------------------
# W4 — SRM.numpyro_bsrm_model consumes ordered dyads (N*(N-1) rows) while
# SRM(A).network holds the N*(N-1)/2 upper-triangle dyads from mat_to_edgl;
# passing the latter silently misaligned every dyad. Must raise ValueError.
# ---------------------------------------------------------------------------
# Ensure SRM rejects upper-triangle edgelist
def test_w4_bsrm_rejects_upper_triangle_edgelist():
    adj = jnp.array([[0.0, 1.0, 0.0, 1.0],
                     [1.0, 0.0, 1.0, 0.0],
                     [0.0, 1.0, 0.0, 1.0],
                     [1.0, 0.0, 1.0, 0.0]])
    srm = SRM(adj)
    assert srm.network.shape[0] == 6  # N(N-1)/2 upper-triangle dyads, N=4

    model = numpyro.handlers.trace(
        numpyro.handlers.seed(srm.numpyro_bsrm_model, rng_seed=0))
    with pytest.raises(ValueError, match="ordered dyads"):
        model.get_trace(srm.network, jnp.zeros((4, 1)), None, None, None, None)


# ---------------------------------------------------------------------------
# W5 — NBDA used the UnifiedDist CLASS (unbound methods swallowed the first
# positional argument as self): Normal(0, 2) became Normal(loc=2, scale=1)
# and Uniform(0.0001, 0.9999) a degenerate Uniform(0.9999, 1.0). Fixed by
# the instance `dist = _UnifiedDist()` (NBDA.py L6-7).
# ---------------------------------------------------------------------------
# Ensure NBDA priors use instance_dist
def test_w5_nbda_priors_use_instance_dist():
    nbda = NBDA()
    # Minimal shapes so priors() takes the no-covariate branches.
    nbda.T_social = jnp.zeros((1, 1, 1, 1))
    nbda.T_asocial = jnp.zeros((1, 1, 1))

    traced = numpyro.handlers.trace(numpyro.handlers.seed(nbda.priors, rng_seed=0))
    tr = traced.get_trace()

    fn = tr["social_learning"]["fn"]
    fn = getattr(fn, "base_dist", fn)
    assert float(jnp.ravel(fn.loc)[0]) == 0.0
    assert float(jnp.ravel(fn.scale)[0]) == 2.0

    ces = tr["ces_alpha"]["fn"]
    ces = getattr(ces, "base_dist", ces)
    assert float(ces.low) < 0.5 < float(ces.high)


# ---------------------------------------------------------------------------
# W6 — Effects.varying_slope_non_centered referenced m.dist with no `m` in
# scope -> NameError on every call (fixed in 94e80673 to dist.*). Args
# mirror model_m1 in test_audit_postmodel_major.py (2 categories, 2 groups).
# ---------------------------------------------------------------------------

def model_w6(category_idx=None, grouping_idx=None, num_categories=None,
             num_groups=None, y=None):
    beta = m.effects.varying_slope_non_centered(
        category_idx, grouping_idx, num_categories, num_groups)
    sigma = numpyro.sample("sigma", numpyro.distributions.HalfNormal(1.0))
    numpyro.sample("y", numpyro.distributions.Normal(beta, sigma), obs=y)

# Ensure varying_slope_non_centered traces without NameError
def test_w6_varying_slope_non_centered_traces_without_name_error():
    traced = numpyro.handlers.trace(numpyro.handlers.seed(model_w6, rng_seed=0))
    tr = traced.get_trace(
        category_idx=jnp.array([0, 1, 0, 1]),
        grouping_idx=jnp.array([0, 0, 1, 1]),
        num_categories=2,
        num_groups=2,
        y=jnp.zeros(4),
    )
    assert {"L_Rho_group", "sigma_group", "z_group", "sigma", "y"} <= set(tr)


# ---------------------------------------------------------------------------
# W7 — nodal_random_effects documents 1-based random_group_index and does
# group_idx_0 = code - 1, so 0-based codes from m.index() silently mapped
# group 0 to the last group. Concrete codes must now be range-checked.
# ---------------------------------------------------------------------------

def test_w7_nodal_random_effects_rejects_zero_based_codes():
    design = jnp.ones((2, 5))
    with pytest.raises(ValueError, match="1-based"):
        Neteffect.nodal_random_effects(design, jnp.array([0, 1, 2, 0, 1]),
                                       5, 3, sample=True)
    out = Neteffect.nodal_random_effects(design, jnp.array([1, 2, 3, 1, 2]),
                                         5, 3, sample=True)
    assert out.shape == (2,)


# ---------------------------------------------------------------------------
# W8 — link.inv_logit_scale is (sigmoid - 0.5) * 2, image (-1, 1), but the
# docstring claimed "between 0 and 1". Docstring now says (-1, 1) and NBDA
# clips the social influence weight to [0, 1].
# ---------------------------------------------------------------------------

def test_w8_inv_logit_scale_documented_on_minus_one_one():
    assert "-1" in link.inv_logit_scale.__doc__
    assert float(link.inv_logit_scale(0.0)) == 0.0
