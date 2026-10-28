"""m.diag ppc bindings resolve x= / group= given as a data name.

``m.diag.ppc_*(x="name")`` (and ``group="name"``) may pass the *name* of a
model data entry (``m.data_on_model``) or a column of ``m.df`` instead of the
array: the binding replaces it by ``np.asarray(...)`` of that entry and uses
the name as the axis/group label (``x_label`` / ``group_name``), and raises
ValueError when the name is in neither place.

Fixture: a tiny fitted linear model, like the ``fitted`` fixture of
Test/0.Core/test_sample_api.py."""
import jax.numpy as jnp
import numpy as np
import pytest

from BayesForge import bf

X = jnp.linspace(-1, 1, 30)
Y = 1.0 + 2.0 * X + 0.1 * np.asarray(
    np.random.default_rng(0).normal(size=30), dtype=X.dtype)


@pytest.fixture(scope="module")
def fitted():
    """One bf instance fitted once on (X, Y); yrep drawn once with a seed."""
    m = bf(platform="cpu", rand_seed=1, print_devices_found=False)

    def model(x, y):
        a = m.dist.normal(0, 5, name="a")
        b = m.dist.normal(0, 5, name="b")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a + b * x, s, obs=y, name="y")

    m.fit(model, obs={"x": X, "y": Y},
          num_warmup=100, num_samples=100, num_chains=1, progress_bar=False)
    return m


@pytest.fixture(scope="module")
def yrep(fitted):
    """Posterior predictive draws of the observed site, fixed across calls."""
    return np.asarray(fitted.sample(seed=0)["y"], dtype=float)


def _trace_xy(fig):
    """(x, y) of every trace, as float arrays (segment traces keep Nones)."""
    def arr(v):
        if v is None:
            return None
        seq = list(v)
        if any(t is None for t in seq):
            return np.asarray([np.nan if t is None else t for t in seq])
        return np.asarray(seq, dtype=float)
    return [(arr(t.x), arr(t.y)) for t in fig.data]


def test_ppc_intervals_x_by_name_matches_array(fitted, yrep):
    fig_name = fitted.diag.ppc_intervals(yrep=yrep, x="x")
    fig_arr = fitted.diag.ppc_intervals(yrep=yrep, x=np.asarray(X))

    # the name becomes the x axis title
    assert fig_name.layout.xaxis.title.text == "x"

    # and the same data as passing the array
    assert len(fig_name.data) == len(fig_arr.data)
    for i, ((nx, ny), (ax, ay)) in enumerate(zip(_trace_xy(fig_name),
                                                 _trace_xy(fig_arr))):
        for attr, a, b in (("x", nx, ax), ("y", ny, ay)):
            if a is None:
                assert b is None
                continue
            np.testing.assert_allclose(a, b, rtol=0, atol=0,
                                       err_msg=f"trace {i} {attr!r} differs")


def test_ppc_intervals_x_by_name_plots_the_data(fitted, yrep):
    # the plotted x values are the x data itself (ppc_intervals sorts by x)
    fig = fitted.diag.ppc_intervals(yrep=yrep, x="x")
    plotted = np.asarray([t for t in fig.data[2].x if t is not None], dtype=float)
    np.testing.assert_allclose(plotted, np.sort(np.asarray(X, dtype=float)),
                               rtol=0, atol=0)


def test_ppc_intervals_unknown_x_name_raises(fitted, yrep):
    with pytest.raises(ValueError,
                       match=r"x= 'nope': no such entry in "
                             r"m\.data_on_model or column in m\.df"):
        fitted.diag.ppc_intervals(yrep=yrep, x="nope")


def test_ppc_intervals_unknown_group_name_raises(fitted, yrep):
    with pytest.raises(ValueError,
                       match=r"group= 'nope': no such entry in "
                             r"m\.data_on_model or column in m\.df"):
        fitted.diag.ppc_intervals(yrep=yrep, group="nope")


def test_diag_ppc_error_binned_builds_figure(fitted):
    fig = fitted.diag.ppc_error_binned()
    assert len(fig.data) > 0


def test_diag_ppc_km_overlay_builds_figure(fitted, yrep):
    y = np.abs(np.asarray(Y, dtype=float)) + 0.1
    fig = fitted.diag.ppc_km_overlay(y=y, yrep=np.abs(yrep) + 0.1,
                                     status_y=np.ones(len(y)))
    assert len(fig.data) == 51                      # n=50 predicted + observed
    assert fig.data[-1].name == "y (observed)"
