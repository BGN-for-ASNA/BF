"""Regression tests for RELEASE_AUDIT.md, Integration V16-V18 / X10-X11 / W16.

V16      m.diag diagnose() follows m's CURRENT fit (no stale snapshot)
V17/X10  a failed ArviZ InferenceData build warns and stashes the cause
V18      sampler_stats() names the composite sampler that cannot record NUTS stats
X11      summary() docstring documents that the two backends read different sources
X16      __setstate__ warns when this machine's device count differs from save time
W16      fit() documents model identification and warns when the model is a lambda
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge import bf

# Helper function to import BF
def _m(seed=0):
    return bf(platform="cpu", print_devices_found=False, rand_seed=seed)

# Helper function to create a model
def _model(m):
    def model(y):
        mu = m.dist.normal(0.0, 1.0, name="mu")
        m.dist.normal(mu, 1.0, obs=y)
    return model

# Helper function to fit the model
def _fit(m, y, num_samples=50):
    m.fit(_model(m), obs={"y": jnp.asarray(y)}, num_samples=num_samples,
          num_warmup=num_samples, num_chains=1, progress_bar=False)
    return m

# Fixture to get a fitted model
@pytest.fixture(scope="module")
def fitted():
    return _fit(_m(), jnp.zeros(5))


# V16 ------------------------------------------------------------------------
# Ensure diagnose() use last fit
def test_v16_diagnose_follows_current_fit():
    # A `d = m.diag` kept across a second fit must diagnose the NEW fit, not
    # the snapshot taken when d was built: prior N(0,1) + 5 obs at 0 puts the
    # posterior mean of mu at 0; the same model with obs at 3 puts it at 2.5.
    m = _fit(_m(), jnp.zeros(5))
    d = m.diag
    assert float(np.asarray(d.posterior_samples["mu"]).mean()) < 1.0
    _fit(m, jnp.full(5, 3.0))
    report = d.diagnose()
    assert isinstance(report, str)
    assert "Processing complete" in report
    mu = np.asarray(d.posterior_samples["mu"], dtype=float)
    assert mu.mean() > 1.0


# V17 / X10 ------------------------------------------------------------------
# Ensure to_az failure warns and stashes cause
def test_v17_to_az_failure_warns_and_stashes_cause(fitted, monkeypatch):
    from BayesForge.Diagnostic import Diag, patch_diag

    def boom(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(Diag.diag, "to_az", boom)
    if getattr(fitted, "_az_diag", None) is not None:
        del fitted._az_diag
    with pytest.warns(RuntimeWarning, match="boom"):
        cached = patch_diag._az_diag_for(fitted)
    assert isinstance(getattr(cached, "_to_az_error", None), RuntimeError)


# V18 ------------------------------------------------------------------------
# Ensure sampler_stats() names the composite sampler that cannot record NUTS stats
def test_v18_sampler_stats_names_composite_sampler(fitted, monkeypatch):
    # Composite kernels record no NUTS state, so the error must say which
    # sampler was used instead of blaming an extra_fields override.
    monkeypatch.setattr(fitted, "sampler_name", "mixed_hmc", raising=False)
    monkeypatch.setattr(fitted, "extra_fields_by_chain", None, raising=False)
    with pytest.raises(RuntimeError, match="mixed_hmc"):
        fitted.sampler_stats()


# X11 ------------------------------------------------------------------------
def test_x11_summary_docstring_documents_different_sources():
    assert "different sources" in bf.summary.__doc__


# W16 ------------------------------------------------------------------------
# Ensure fit() docstring documents model identification
def test_w16_fit_docstring_documents_model_identification():
    assert "How the model is identified" in bf.fit.__doc__

# Ensure lambda model warns
def test_w16_lambda_model_warns():
    m = _m()
    model = _model(m)
    with pytest.warns(UserWarning, match="lambda"):
        m.fit(lambda y: model(y), obs={"y": jnp.zeros(5)}, num_samples=20,num_warmup=20, num_chains=1, progress_bar=False)
    assert m.run_model_name == "<lambda>"


# X16 ------------------------------------------------------------------------
def test_x16_setstate_warns_on_device_count_mismatch(fitted):
    state = fitted.__getstate__()
    # A saved device count that cannot match this machine's: under
    # JAX_PLATFORMS=cpu the mesh rebuild yields one device, but another test
    # may have forced more, so pick a value that mismatches either way.
    state["n_devices"] = 4 if len(jax.devices("cpu")) != 4 else 9
    state["_platform"] = "cpu"
    m2 = object.__new__(type(fitted))
    with pytest.warns(RuntimeWarning):
        m2.__setstate__(state)
