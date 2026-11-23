"""Regression tests for RELEASE_AUDIT.md, module 03 (Samplers/mcmc_tfp.py), critical C1-C3."""
import numpy as np
import pytest

pytest.importorskip("tensorflow_probability")

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

from BayesForge import bf
from BayesForge.Samplers import mcmc_tfp as mcmc_tfp_mod


def _model_factory(m):
    def model(y):
        mu = yield m.dist.normal(0.0, 5.0, name='mu')
        sigma = yield m.dist.half_normal(2.0, name='sigma')
        yield m.dist.normal(mu + jnp.zeros_like(y), sigma, obs=y)
    return model


def _bf_tfp():
    m = bf(platform='cpu', backend='tfp', print_devices_found=False)
    y = np.random.default_rng(0).normal(1.0, 0.7, size=40).astype(np.float32)
    m.data_on_model = dict(y=jnp.asarray(y))
    return m


def test_c1_dual_averaging_adapts_over_num_burnin_steps(monkeypatch):
    seen = {}
    real = tfp.mcmc.DualAveragingStepSizeAdaptation

    def spy(*args, **kwargs):
        seen['num_adaptation_steps'] = kwargs.get('num_adaptation_steps')
        return real(*args, **kwargs)

    monkeypatch.setattr(mcmc_tfp_mod.tfp.mcmc, 'DualAveragingStepSizeAdaptation', spy)
    m = _bf_tfp()
    m.fit(_model_factory(m), num_samples=20, num_warmup=100, num_chains=1)
    assert seen['num_adaptation_steps'] == int(0.8 * 100)


def test_c1_dual_averaging_actually_tunes_step_size():
    # Behavioural check of C1: from a deliberately tiny initial step size,
    # dual averaging must move the step so that post-warmup acceptance sits
    # near target_accept_prob and NUTS trees stay short. Without adaptation,
    # step_size=1e-3 gives acceptance ~1.0 and trees at max depth.
    m = _bf_tfp()
    model = _model_factory(m)
    s = m.sampler
    s.data_on_model = m.data_on_model
    s.model = model
    s.run(seed=0, model=model, n_chains=1, num_results=200,
          num_burnin_steps=400, target_accept_prob=0.8, step_size=1e-3)
    log_acc = np.asarray(s.sample_stats[4]).ravel()
    acc = np.minimum(1.0, np.exp(log_acc)).mean()
    leapfrogs = np.median(np.asarray(s.sample_stats[1]).ravel())
    assert abs(acc - 0.8) < 0.15, acc
    assert leapfrogs < 64, leapfrogs


def test_c2_custom_target_log_prob_fn_is_used():
    m = _bf_tfp()
    model = _model_factory(m)
    s = m.sampler
    s.data_on_model = m.data_on_model
    s.model = model
    calls = {'n': 0}

    def custom(values):
        calls['n'] += 1
        return s.tensor.log_prob(values)

    s.run(seed=0, model=model, n_chains=1, num_results=10, num_burnin_steps=10,
          target_log_prob_fn=custom)
    assert calls['n'] > 0
    assert len(s.posterior) == 2


def test_c3_chain_starts_from_the_constrained_init_state():
    """TransformedTransitionKernel takes current_state in the constrained space."""
    m = _bf_tfp()
    model = _model_factory(m)
    s = m.sampler
    s.data_on_model = m.data_on_model
    s.model = model
    s.run(seed=0, model=model, n_chains=1, num_results=5, num_burnin_steps=0)
    init_sigma = float(s.init_params[1])
    hmc = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.NoUTurnSampler(lambda *p: s.tensor.log_prob(p + (m.data_on_model['y'],)), step_size=1e-3),
        bijector=s.bijectors)
    pkr = hmc.bootstrap_results(s.init_params)
    # the kernel's unconstrained state must map back to the init draw (no double transform)
    back = float(s.bijectors[1].forward(pkr.transformed_state[1]))
    assert np.isclose(back, init_sigma, rtol=1e-5)
    assert init_sigma > 0


# Majors M1-M6 ---------------------------------------------------------------

def test_m1_obs_not_last_raises():
    m = _bf_tfp()
    def bad(y):
        mu = yield m.dist.normal(0.0, 5.0, name='mu')
        yield m.dist.normal(mu + jnp.zeros_like(y), 1.0, obs=y)
        s = yield m.dist.half_normal(2.0, name='s')
    with pytest.raises(ValueError, match="last yields"):
        m.fit(bad, num_samples=20, num_warmup=20, num_chains=1)


def test_m1_several_obs_sites_after_latents_are_supported():
    # M1 rule is about order, not count: several observed sites are fine
    # as long as they all come after the latents (documented in bf.__init__
    # `backend` docstring and Documentation/start/Import_class.qmd).
    m = bf(platform='cpu', backend='tfp', print_devices_found=False)
    rng = np.random.default_rng(0)
    y1 = rng.normal(1.0, 0.7, size=40).astype(np.float32)
    y2 = rng.normal(1.0, 0.7, size=25).astype(np.float32)
    m.data_on_model = dict(y1=jnp.asarray(y1), y2=jnp.asarray(y2))

    def model(y1, y2):
        mu = yield m.dist.normal(0.0, 5.0, name='mu')
        sigma = yield m.dist.half_normal(2.0, name='sigma')
        # Each observed site needs its own name=: unnamed sites all default to "Normal" and TFP rejects the duplicate.
        yield m.dist.normal(mu + jnp.zeros_like(y1), sigma, obs=y1, name='y1')
        yield m.dist.normal(mu + jnp.zeros_like(y2), sigma, obs=y2, name='y2')

    m.fit(model, num_samples=200, num_warmup=300, num_chains=1)
    assert set(m.posteriors) == {'mu', 'sigma'}
    assert abs(float(np.mean(m.posteriors['mu'])) - 1.0) < 0.3
    assert abs(float(np.mean(m.posteriors['sigma'])) - 0.7) < 0.3


def test_m1_latent_between_two_obs_sites_raises():
    m = bf(platform='cpu', backend='tfp', print_devices_found=False)
    m.data_on_model = dict(y1=jnp.zeros(5), y2=jnp.zeros(5))

    def bad(y1, y2):
        mu = yield m.dist.normal(0.0, 5.0, name='mu')
        yield m.dist.normal(mu + jnp.zeros_like(y1), 1.0, obs=y1)
        s = yield m.dist.half_normal(2.0, name='s')
        yield m.dist.normal(mu + jnp.zeros_like(y2), s, obs=y2)

    with pytest.raises(ValueError, match="last yields"):
        m.fit(bad, num_samples=20, num_warmup=20, num_chains=1)


def test_m2_seed_changes_init():
    inits = []
    for s in (1, 2):
        m = _bf_tfp()
        m.fit(_model_factory(m), num_samples=20, num_warmup=20, num_chains=1, seed=s)
        inits.append(np.asarray(m.sampler.init_params[0]))
    assert not np.allclose(inits[0], inits[1])


def test_m3_posteriors_flat_like_numpyro():
    m = _bf_tfp()
    m.fit(_model_factory(m), num_samples=20, num_warmup=20, num_chains=2)
    assert np.asarray(m.posteriors['mu']).shape == (40,)
    assert np.asarray(m.posteriors_by_chain['mu']).shape == (2, 20)


def test_m4_thinning_is_applied():
    m = _bf_tfp()
    # Like numpyro: thinning=3 with num_samples=21 keeps 7 draws. The TFP
    # sampler runs num_results * (k + 1) steps with
    # num_steps_between_results = k = thinning - 1, so 7 kept draws come
    # from max(21 // 3, 1) = 7 num_results.
    m.fit(_model_factory(m), num_samples=21, num_warmup=20, num_chains=1, thinning=3)
    assert np.asarray(m.posteriors['mu']).shape == (7,)
    with pytest.warns(UserWarning, match="ignores"):
        m.fit(_model_factory(m), num_samples=20, num_warmup=20, num_chains=1, dense_mass=True)


def test_m5_beta_parameter_gets_sigmoid_bijector():
    m = _bf_tfp()
    def model(y):
        p = yield m.dist.beta(2.0, 2.0, name='p')
        yield m.dist.normal(p + jnp.zeros_like(y), 1.0, obs=y)
    m.fit(model, num_samples=20, num_warmup=20, num_chains=1)
    p = np.asarray(m.posteriors['p'])
    assert ((p > 0) & (p < 1)).all()


def test_m6_param_named_obs_something_is_kept():
    m = _bf_tfp()
    def model(y):
        obs_rate = yield m.dist.normal(0.0, 5.0, name='obs_rate')
        yield m.dist.normal(obs_rate + jnp.zeros_like(y), 1.0, obs=y)
    m.fit(model, num_samples=20, num_warmup=20, num_chains=1)
    assert 'obs_rate' in m.posteriors
