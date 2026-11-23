"""Every m.diag method is documented and runs on a fitted model."""
import inspect

import jax.numpy as jnp
import matplotlib
import pytest

matplotlib.use("Agg")

from BayesForge import bf


@pytest.fixture(scope="module")
def m():
    m = bf(platform="cpu", rand_seed=5, print_devices_found=False)
    x = jnp.linspace(-1, 1, 40)
    y = 0.5 + 2.0 * x + 0.2 * jnp.sin(9 * x)

    def model(x, y):
        a = m.dist.normal(0, 2, name="a")
        b = m.dist.normal(0, 2, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y)

    m.fit(model, obs=dict(x=x, y=y), num_samples=100, num_warmup=100, num_chains=2,
          chain_method="sequential", progress_bar=False)
    return m


def public_methods(diag):
    return sorted(n for n in dir(diag) if not n.startswith("_") and callable(getattr(diag, n)))


def test_every_diag_method_is_documented(m):
    undocumented = [n for n in public_methods(m.diag) if not inspect.getdoc(getattr(m.diag, n))]
    assert undocumented == []


# Called with defaults only: y, yrep and the posterior come from the fitted model.
NO_ARG_METHODS = [
    "summary", "rhat", "ess", "mcse", "loo", "waic",
    "plot_trace", "posterior", "autocor", "forest", "density", "pair", "rank",
    "ppc_density", "ppc_hist", "ppc_boxplot", "ppc_stat", "ppc_stat_2d", "ppc_intervals",
    "ppc_ribbon", "ppc_error_hist", "ppc_scatter", "ppc_loo_intervals",
    "calibration", "influence", "multimodality",
]


@pytest.mark.parametrize("name", NO_ARG_METHODS)
def test_diag_method_runs(m, name):
    assert getattr(m.diag, name)() is not None


# In development: kept in Diagnostic.ppc / regression_plot, not exposed on m.diag.
IN_DEVELOPMENT = ["ppc_loo_pit", "ppc_km_overlay", "ppc_error_scatter",
                  "ppc_error_binned", "ppc_intervals_sorted", "plot_regression"]


@pytest.mark.parametrize("name", IN_DEVELOPMENT)
def test_in_development_plots_not_on_diag(m, name):
    assert not hasattr(m.diag, name)
    from BayesForge.Diagnostic import ppc, regression_plot
    assert callable(getattr(ppc, name, None) or getattr(regression_plot, name))


def test_ppc_signature_marks_y_yrep_optional(m):
    params = inspect.signature(m.diag.ppc_density).parameters
    assert params["y"].default is None and params["yrep"].default is None


def test_yrep_is_simulated_not_the_observed_data(m):
    # Regression: obs_args was empty before m.sample() ran, so the "posterior
    # predictive" was the observed y repeated and every PPC trivially passed.
    import numpy as np
    yrep = m.diag.get_yrep()
    y = np.asarray(m.data_on_model["y"])
    assert yrep.shape == (200, y.size)
    assert yrep.std(axis=0).mean() > 0.01
    assert not np.allclose(yrep, y[None, :])
    # the replicated data scatter around y with roughly the residual sd
    assert abs(yrep.mean() - y.mean()) < 0.2


def test_network_cc_nodes_subset():
    import jax.numpy as jnp
    from BayesForge.Network.metrics import met
    A = jnp.array([[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
    full = met.cc(A)
    assert jnp.allclose(met.cc(A, nodes=[0, 2]), full[jnp.array([0, 2])])
    assert jnp.allclose(met.weighted_cc(A), met.cc(A, weighted=True))


def test_sample_with_observed_argument_first():
    # Regression: m.sample() built an invalid signature when the observed
    # argument came before another one ("non-default argument follows default").
    import jax.numpy as jnp
    from BayesForge import bf
    m = bf(platform="cpu", rand_seed=2, print_devices_found=False)
    x = jnp.linspace(-1, 1, 15)

    def model(y, x):
        a = m.dist.normal(0, 1, name="a")
        m.dist.normal(a * x, 1.0, obs=y)

    m.fit(model, obs=dict(y=2 * x, x=x), num_samples=30, num_warmup=30, num_chains=1, progress_bar=False)
    pred = m.sample(posterior=True)
    # the observed site is simulated from each of the 30 posterior draws
    assert any(v.shape == (30, 15) for v in pred.values())
