"""Regression tests for RELEASE_AUDIT.md, Integration V5-V11 (empirical trace)."""

import inspect

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from BayesForge import bf

# Fixture for a default model instance
def _m(seed=0, **kwargs):
    return bf(platform="cpu", print_devices_found=False, rand_seed=seed, **kwargs)

# Helper function to define a two-parameter model
def _two_param_model(m):
    """Two latent parameters (a, b): a single-parameter model would make the
    V11 include/exclude filter a no-op and the V5 mutation loop trivial."""
    def model(y):
        a = m.dist.normal(0.0, 1.0, name="a")
        b = m.dist.normal(0.0, 1.0, name="b")
        m.dist.normal(a + b, 1.0, obs=y)
    return model

# Helper function to fit the model
def _fit(seed=0, **fit_kwargs):
    m = _m(seed)
    m.fit(_two_param_model(m), obs={"y": jnp.array([0.1, -0.2, 0.3])},
          num_samples=16, num_warmup=16, num_chains=2, progress_bar=False,
          **fit_kwargs)
    return m


# V5 -------------------------------------------------------------------------------
# Ensure posterior is a copy of the sampler cache
def test_v5_posteriors_is_a_copy_of_the_sampler_cache():
    m = _fit()
    assert m.posteriors is not m.sampler.get_samples()
    assert m.posteriors_by_chain is not m.sampler.get_samples(group_by_chain=True)
    before = {k: np.asarray(v).copy() for k, v in m.sampler.get_samples().items()}
    for k in m.posteriors:
        m.posteriors[k] = m.posteriors[k] * 0
    for k, v in m.sampler.get_samples().items():
        assert np.array_equal(np.asarray(v), before[k]), k


# V6 -------------------------------------------------------------------------------
# Ensure stats per chain without num_steps works
def test_v6_sampler_stats_per_chain_without_num_steps():
    # extra_fields without 'num_steps' left steps=None -> range(len(None)).
    m = _fit(extra_fields=("diverging",))
    rows = m.sampler_stats(per_chain=True)
    assert [row["chain"] for row in rows] == [0, 1]
    assert all("divergences" in row for row in rows)


# V7 (Data C1) -----------------------------------------------------------------------
# Ensure OHE does not duplicate columns
def test_v7_ohe_does_not_duplicate_columns():
    m = _m()
    df = pd.DataFrame({"g": pd.Categorical(["a", "b", "a", "b"]),
                       "x": [1.0, 2.0, 3.0, 4.0]})
    m.data(df)
    m.OHE(["g"])
    assert not m.df.columns.duplicated().any()
    assert "x" in m.df.columns


# V8 (M-01) --------------------------------------------------------------------------
# Ensure log prob is conditioned on the data
def test_v8_log_prob_conditions_on_the_data():
    m = _m()
    m.data_on_model = {"y": jnp.array([1.0, 2.0, 3.0])}
    init_params, potential_fn, _, _ = m.log_prob(
        _two_param_model(m), **m.data_on_model)
    assert np.isfinite(-float(potential_fn(init_params.z)))


# V9 (M-04) --------------------------------------------------------------------------
# Ensure TFP posteriors honour the chain contract
def test_v9_tfp_posteriors_honour_the_chain_contract():
    pytest.importorskip("tensorflow_probability")
    m = _m(backend="tfp")

    def model(y):
        a = yield m.dist.normal(0.0, 1.0, name="a")
        yield m.dist.normal(a, 1.0, obs=y)

    m.fit(model, obs={"y": jnp.array([0.1, -0.2, 0.3])},
          num_samples=8, num_warmup=8, num_chains=2, progress_bar=False)
    assert m.posteriors["a"].shape[0] == 2 * 8
    assert m.posteriors_by_chain["a"].shape[0] == 2


# V10 --------------------------------------------------------------------------------
# Ensure to_az_tfp rejects misaligned var names
def test_v10_to_az_tfp_rejects_misaligned_var_names():
    from BayesForge.Diagnostic.Diag2 import diagWIP

    class _FakeTFPSampler:
        num_chains = 2

        def __init__(self):
            self.model_info = {"a": None, "b": None}   # 2 names
            self.posterior = [np.zeros((2, 4))]        # but 1 sampled site
            self.sample_stats = [np.zeros((2, 4)) for _ in range(5)]

        def get_samples(self, group_by_chain=False):
            return {"a": np.zeros((2, 4))}

    d = diagWIP(_FakeTFPSampler())
    with pytest.raises(ValueError, match=r"2 var_names .* 1 TFP posterior"):
        d.to_az(backend="tfp")
    assert "zip(var_names, self.sampler.posterior, strict=True)" in \
        inspect.getsource(diagWIP.to_az)


# V11 --------------------------------------------------------------------------------
# Ensure summary include does not filter persistently
def test_v11_summary_include_does_not_filter_persistently():
    m = _fit()
    assert {"a", "b"} <= set(m.posteriors)
    m.summary(include=["a"])
    assert set(m.posteriors) == set(m.posteriors_full)
    assert set(m.posteriors_by_chain) == set(m.posteriors_by_chain_full)
