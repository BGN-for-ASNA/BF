"""Regression tests for RELEASE_AUDIT.md, module 06 (Diagnostic), critical bugs 1 and 2."""
import numpy as np
import jax.numpy as jnp
import pytest

from BayesForge.Diagnostic.Diag2 import diagWIP
from BayesForge.Samplers.SBI.simulate import SBIPosterior


class _FakeTfpSampler:
    """Mimics mcmc_tfp after NUTS: posterior list + 5 trace_fn statistics."""
    num_chains = 1

    def __init__(self):
        self.model_info = {"mu": {}, "sigma": {}}
        self.posterior = [jnp.zeros((1, 6)), jnp.ones((1, 6))]
        # trace_fn order: target_log_prob, leapfrogs_taken, has_divergence, energy, log_accept_ratio
        self.sample_stats = (
            jnp.full((1, 6), -10.0),
            jnp.full((1, 6), 31.0),
            jnp.zeros((1, 6), dtype=bool),
            jnp.full((1, 6), 12.0),
            jnp.full((1, 6), -0.25),
        )

    def get_samples(self, group_by_chain=True):
        return dict(zip(self.model_info, self.posterior))


def test_bug1_tfp_sample_stats_are_named_in_trace_fn_order():
    d = diagWIP(_FakeTfpSampler())
    tr = d.to_az(backend="tfp")
    ss = tr["sample_stats"]
    assert np.allclose(np.asarray(ss["log_accept_ratio"]), -0.25)
    assert np.allclose(np.asarray(ss["leapfrogs_taken"]), 31.0)


def test_bug1_mismatched_stat_names_raise():
    d = diagWIP(_FakeTfpSampler())
    with pytest.raises(ValueError):
        d.to_az(backend="tfp", sample_stats_name=["a", "b", "c", "d"])


def test_bug2_sbi_pseudo_chains_warn():
    post = SBIPosterior()
    post.posterior = {"theta": jnp.asarray(np.random.default_rng(0).normal(size=32))}
    with pytest.warns(UserWarning, match="pseudo-chains"):
        diagWIP(post)
