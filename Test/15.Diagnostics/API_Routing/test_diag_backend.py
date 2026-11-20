"""ArviZ is BF's default diagnostics backend; 'jax' is opt-in and warns.

Covers:
  * m.summary() and m.diag.{summary,rhat,ess,mcse,loo,WAIC} route to ArviZ
  * bf(diag_backend="jax") / m.summary(backend="jax") select jax_diagnostics
    and emit an experimental-feature warning
  * the two backends agree numerically (they are verified equal to 1e-6 by
    test_jax_diagnostics_parity.py; here we check them through the live API)
  * backend-independent surfaces (plots, PPC, sensitivity) work either way

The ArviZ references are read the 1.x way (DataTree results, `prob=` on
``ess(method="tail")``); the statistics and tolerances are unchanged.
"""
import warnings

import numpy as np
import pytest

from BayesForge import bf
from BayesForge.Diagnostic.patch_diag import (
    ARVIZ, JAX, _resolve_backend, bind_diag_to_model,
)
# BayesForge is imported above first: its __init__ patches
# jax.interpreters.pxla, which arviz's own jax import can break.
import arviz as az


def _fit(diag_backend="arviz"):
    m = bf(platform="cpu", diag_backend=diag_backend)
    data_path = m.load.howell1(only_path=True)
    m.data(data_path, sep=";")
    m.df = m.df[m.df.age > 18]
    m.scale(["weight"])

    def model(weight, height):
        a = m.dist.normal(178, 20, name="a")
        b = m.dist.log_normal(0, 1, name="b")
        s = m.dist.uniform(0, 50, name="s")
        m.dist.normal(a + b * weight, s, obs=height, shape=(weight.shape[0],))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(model, num_samples=800, num_chains=4, progress_bar=False)
    return m


def _trace(diag_obj):
    """The ArviZ container of the fitted draws (BayesForge builds it lazily)."""
    trace = getattr(diag_obj, "trace", None)
    if trace is None:
        trace = diag_obj._ensure_trace()
    return trace


def _ds_val(res, param):
    """Scalar metric for `param` from an ArviZ rhat/ess result.

    ArviZ 1.x returns an ``xarray.DataTree`` rooted at the group it computed
    (0.x returned an ``xarray.Dataset``); the variable is at ``res[param]``
    either way, with ``res["posterior"][param]`` as the fallback for builds
    that keep the group as a child node.
    """
    try:
        obj = res[param]
    except (KeyError, TypeError):
        obj = res["posterior"][param]
    return float(np.asarray(obj.values))


@pytest.fixture(scope="module")
def m_az():
    return _fit("arviz")


# ============================== defaults ==============================

def test_arviz_is_the_default(m_az):
    assert m_az.diag_backend == ARVIZ
    assert m_az.diag.backend == ARVIZ


def test_constructing_without_diag_backend_gives_arviz():
    m = bf(platform="cpu")
    assert m.diag_backend == ARVIZ


@pytest.mark.parametrize("given,expected", [
    (None, ARVIZ), ("arviz", ARVIZ), ("ArviZ", ARVIZ),
    ("jax", JAX), ("JAX", JAX),
])
def test_resolve_backend(given, expected):
    assert _resolve_backend(given) == expected


def test_resolve_backend_rejects_nonsense():
    with pytest.raises(ValueError, match="backend must be one of"):
        _resolve_backend("numpy")


# ============================== summary routing ==============================

def test_summary_returns_an_arviz_table(m_az):
    df = m_az.summary()
    assert df is not None and len(df) == 3
    # az.summary(kind="all") columns
    for col in ("mean", "sd", "ess_bulk", "ess_tail", "r_hat"):
        assert col in df.columns, f"missing {col}: {list(df.columns)}"
    assert set(df.index) == {"a", "b", "s"}


def test_summary_jax_backend_warns_and_works(m_az):
    with pytest.warns(UserWarning, match="experimental"):
        df = m_az.summary(backend="jax")
    for col in ("mean", "sd", "mcse_mean", "mcse_sd",
                "ess_bulk", "ess_tail", "r_hat"):
        assert col in df.columns
    assert set(df.index) == {"a", "b", "s"}


def test_summary_jax_helper_warns(m_az):
    with pytest.warns(UserWarning, match="experimental"):
        m_az.summary_jax()


def test_backends_agree_on_summary(m_az):
    az_df = m_az.summary()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        jx_df = m_az.summary(backend="jax", round_to=8)
    az_df8 = m_az.summary(round_to=8)
    for p in ("a", "b", "s"):
        for col in ("mean", "sd", "ess_bulk", "r_hat"):
            assert float(az_df8.loc[p, col]) == pytest.approx(
                float(jx_df.loc[p, col]), rel=1e-6), f"{p}.{col}"
        # ess_tail cannot be compared table-to-table any more: az.summary
        # reports the tail ESS at ArviZ's own probability (1 - ci_prob = 0.11
        # in 1.x), while the jax backend uses the 5%/95% tails of Vehtari et
        # al. (2021) that ArviZ 0.x used. Ask ArviZ for that same estimator and
        # compare the jax column against the reference.
        ref_tail = _ds_val(az.ess(_trace(m_az.diag), method="tail", prob=0.05), p)
        assert float(jx_df.loc[p, "ess_tail"]) == pytest.approx(
            ref_tail, rel=1e-6), f"{p}.ess_tail"
    m_az.summary()   # restore the default table


# ============================== m.diag.* routing ==============================

def test_diag_metrics_use_arviz(m_az):
    """ArviZ returns xarray containers; jax_diagnostics returns plain dicts.

    ArviZ 1.x returns an ``xarray.DataTree`` where 0.x returned an
    ``xarray.Dataset``, so both containers are accepted.
    """
    import xarray as xr
    containers = (xr.Dataset,)
    data_tree = getattr(xr, "DataTree", None)
    if data_tree is not None:
        containers += (data_tree,)
    for name in ("rhat", "ess", "mcse"):
        res = getattr(m_az.diag, name)()
        assert isinstance(res, containers), \
            f"{name}() returned {type(res).__name__}, not one of {containers}"


def test_diag_metric_values_are_sane(m_az):
    rhat = m_az.diag.rhat()
    ess = m_az.diag.ess()
    for p in ("a", "b", "s"):
        assert 0.99 <= _ds_val(rhat, p) <= 1.05
        assert _ds_val(ess, p) > 100


def test_diag_loo_and_waic_run(m_az):
    loo = m_az.diag.loo()
    waic = m_az.diag.WAIC()
    for res in (loo, waic):
        val = getattr(res, "elpd_loo", None)
        if val is None:
            val = getattr(res, "elpd_waic", None)
        if val is None:
            val = res.elpd
        assert np.isfinite(float(val))


def test_jax_backend_binding_warns_and_switches(m_az):
    with pytest.warns(UserWarning, match="experimental"):
        bind_diag_to_model(m_az.diag, m_az, backend="jax")
    try:
        assert m_az.diag.backend == JAX
        assert isinstance(m_az.diag.rhat(), dict)      # jax path returns dicts
        assert isinstance(m_az.diag.ess(), dict)
    finally:
        bind_diag_to_model(m_az.diag, m_az, backend="arviz")
    assert m_az.diag.backend == ARVIZ


def test_backends_agree_through_the_live_api(m_az):
    az_rhat = {p: _ds_val(m_az.diag.rhat(), p) for p in ("a", "b", "s")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bind_diag_to_model(m_az.diag, m_az, backend="jax")
        jx_rhat = {k: float(v) for k, v in m_az.diag.rhat().items()}
        bind_diag_to_model(m_az.diag, m_az, backend="arviz")
    for p in ("a", "b", "s"):
        assert az_rhat[p] == pytest.approx(jx_rhat[p], rel=1e-6)


# ============================== backend-independent ==============================

def test_plots_and_ppc_are_backend_independent(m_az):
    """These are not metric estimators; they must work under either backend."""
    for backend in (ARVIZ, JAX):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bind_diag_to_model(m_az.diag, m_az, backend=backend)
            for name in ("plot_trace", "posterior", "forest", "pair",
                         "ppc_density", "ppc_stat", "calibration",
                         "influence", "multimodality"):
                fig = getattr(m_az.diag, name)()
                assert len(fig.data) > 0, f"{name} produced nothing ({backend})"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bind_diag_to_model(m_az.diag, m_az, backend="arviz")


def test_diagnose_still_available(m_az):
    """diagnose() lives on diagWIP and is not overwritten by the binding."""
    report = m_az.diag.diagnose()
    assert "R-hat" in report and "effective sample size" in report


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
