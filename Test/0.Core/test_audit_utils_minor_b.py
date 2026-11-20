"""Regression tests for RELEASE_AUDIT.md, module Utils (SampledData, rng, infer_discrete, dev_warning, ImportManager) minors."""
import inspect
import warnings

import matplotlib
matplotlib.use("Agg")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import BayesForge.Utils.rng as rng_mod
import BayesForge.Utils.SampledData as sd_mod
from BayesForge.Utils.SampledData import SampledData, _autocorr_1d


@pytest.fixture
def capture_plotly(monkeypatch):
    """Capture plotly figures instead of opening a browser window."""
    go = pytest.importorskip("plotly.graph_objects")
    captured = {}

    def fake_show(self, *args, **kwargs):
        captured["fig"] = self

    monkeypatch.setattr(go.Figure, "show", fake_show)
    return captured


@pytest.fixture
def quiet_mpl(monkeypatch):
    """No matplotlib windows under Agg; close figures afterwards."""
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    yield
    plt.close("all")


# --- [Mineure] hist 3D — axes inversés et bins figés [-4, 4] --------------------
def test_hist_3d_axes_and_bins(capture_plotly):
    rng = np.random.default_rng(0)
    n_samples, n_groups, n_times = 60, 3, 7  # n_times > n_groups: old code IndexError
    data = np.zeros((n_samples, n_groups, n_times))
    for g in range(n_groups):
        for t in range(n_times):
            data[:, g, t] = rng.normal(loc=10.0 * g + t, scale=0.05)

    SampledData(jnp.asarray(data)).hist(nbinsx=30)
    fig = capture_plotly["fig"]
    assert len(fig.data) == n_groups * n_times
    # Trace (g, t) peaks at the mean of column (g, t): axis 1 = groups, axis 2 = times.
    for g in range(n_groups):
        for t in range(n_times):
            tr = fig.data[g * n_times + t]
            assert tr.name == f"Group {g}, Time {t}"
            x = np.asarray(tr.x)
            z = np.asarray(tr.z) - g  # z = counts + g
            assert abs(float(x[int(np.argmax(z))]) - (10.0 * g + t)) < 1.5

    # Out-of-range data is binned, not silently dropped by frozen [-4, 4] edges.
    far = rng.normal(100.0, 1.0, size=(40, 2, 3))
    SampledData(jnp.asarray(far)).hist(nbinsx=20)
    fig2 = capture_plotly["fig"]
    zmax = max(
        (np.asarray(tr.z) - int(tr.name.split()[1].rstrip(","))).max()
        for tr in fig2.data
    )
    assert zmax > 0.0


# --- [Mineure] ridgeline interactif — draws vs catégories ; 3D cassé ------------
def test_ridgeline_interactive_iterates_categories(capture_plotly, quiet_mpl):
    rng = np.random.default_rng(1)
    data = np.vstack([rng.normal(m, 0.5, 200) for m in (0.0, 10.0, 20.0, 30.0)]).T
    sd = SampledData(jnp.asarray(data))

    sd.ridgeline()
    fig = capture_plotly["fig"]
    assert len(fig.data) == 4
    for i in range(4):
        # One violin per category (column), not per draw (row).
        np.testing.assert_allclose(np.asarray(fig.data[i].x), data[:, i])

    # 3D used to be accepted and crash inside go.Violin; now a clear error.
    with pytest.raises(ValueError, match="2D"):
        SampledData(jnp.zeros((10, 2, 3))).ridgeline()

    # The (already correct) non-interactive branch still works under Agg.
    sd.ridgeline(interactive=False)


# --- [Mineure] ppc_plot — RNG global numpy et krach sur données constantes ------
def test_ppc_plot_local_rng_and_constant_data(capture_plotly):
    from BayesForge.Utils.dev_warning import reset_dev_warnings

    src = inspect.getsource(SampledData.ppc_plot)
    assert "np.random.choice" not in src and "default_rng" in src

    rng = np.random.default_rng(2)
    draws = rng.normal(0.0, 1.0, (80, 50))
    obs = rng.normal(0.2, 1.0, 50)
    sd = SampledData(jnp.asarray(draws))

    def run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reset_dev_warnings()
            sd.ppc_plot(obs, n_samples_to_plot=10, seed=3)
        return capture_plotly["fig"]

    # The subset follows the local `seed`, not the global numpy RNG.
    np.random.seed(0)
    f1 = run()
    np.random.seed(999)
    f2 = run()
    assert len(f1.data) == 11  # 10 predictive lines + observed density
    for a, b in zip(f1.data, f2.data):
        np.testing.assert_allclose(np.asarray(a.y), np.asarray(b.y))

    # Constant observed data: clear ValueError instead of LinAlgError.
    with pytest.raises(ValueError, match="constant"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reset_dev_warnings()
            sd.ppc_plot(np.ones(10))

    # Constant predictive draws are skipped instead of crashing the KDE.
    mixed = draws.copy()
    mixed[0] = 1.0
    mixed[1] = -2.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reset_dev_warnings()
        SampledData(jnp.asarray(mixed)).ppc_plot(obs, n_samples_to_plot=80, seed=0)
    assert len(capture_plotly["fig"].data) == 78 + 1


# --- [Mineure] autocorr — NaN sur séries courtes/constantes, np/jnp, syncs ------
def test_autocorr_nan_free_and_single_semantics(capture_plotly, quiet_mpl):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        acf_const = _autocorr_1d(np.ones(5), 10)
        acf_short = _autocorr_1d(np.arange(3.0), 6)
        series = np.random.default_rng(3).normal(size=40)
        acf_ok = _autocorr_1d(series, 10)
    assert not [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert acf_const == [1.0] + [0.0] * 9  # constant: no NaN
    assert len(acf_short) == 6 and all(np.isfinite(acf_short))  # k >= len: no NaN
    for k in range(1, 10):  # valid lags keep the exact previous values
        assert acf_ok[k] == pytest.approx(np.corrcoef(series[:-k], series[k:])[0, 1])

    # Plot paths on the degenerate cases, both backends, no NaN, no warning.
    SampledData(jnp.arange(10.0)).autocorr(lags=20)
    y = np.asarray(capture_plotly["fig"].data[0].y, dtype=float)
    assert len(y) == 20 and np.all(np.isfinite(y))
    SampledData(jnp.ones((12, 2))).autocorr(lags=15, interactive=False)

    # One shared numpy helper: no per-lag jnp device ops, no mixed semantics.
    src = inspect.getsource(SampledData.autocorr)
    assert "jnp.corrcoef" not in src and "np.corrcoef" not in src
    assert "_autocorr_1d" in src


# --- [Mineure] __getattr__ — récursion infinie si _data absent ------------------
def test_getattr_no_recursion_when_data_missing():
    broken = object.__new__(SampledData)  # partially constructed instance
    with pytest.raises(AttributeError):
        broken.anything
    with pytest.raises(AttributeError):
        broken._data
    assert "hist" in dir(broken)  # introspectable without RecursionError

    sd = SampledData(jnp.arange(5.0))
    with pytest.raises(AttributeError, match="SampledData"):
        sd.meann()  # typo names SampledData, not the wrapped jax type


# --- [Mineure] Contrats de types incohérents et code dupliqué -------------------
def test_type_contracts_ufunc_hash_and_dead_code():
    sd = SampledData(jnp.asarray(np.random.default_rng(4).normal(size=(20, 3))))

    # ufunc results keep the wrapper, like the operator sugar does.
    assert isinstance(sd + 1.0, SampledData)
    assert isinstance(np.add(sd, 1.0), SampledData)
    red = np.add.reduce(sd)  # used to leak a bare ndarray
    assert isinstance(red, SampledData)
    np.testing.assert_allclose(np.asarray(red), np.asarray(sd._data).sum(axis=0))

    # out=/where= fail loudly instead of silently breaking semantics.
    with pytest.raises(NotImplementedError):
        np.add(sd, 1.0, out=np.zeros((20, 3)))
    with pytest.raises(NotImplementedError):
        np.add(sd, 1.0, where=np.ones((20, 3), dtype=bool))

    # Element-wise __eq__ => unhashable, now documented.
    with pytest.raises(TypeError):
        hash(sd)
    assert "unhashable" in SampledData.__doc__

    # The duplicated pytree pair is gone; the surviving one still round-trips.
    assert inspect.getsource(sd_mod).count("def _tree_flatten") == 1
    out = jax.tree_util.tree_map(lambda x: x * 2, sd)
    assert isinstance(out, SampledData)
    np.testing.assert_allclose(np.asarray(out._data), 2 * np.asarray(sd._data))

    # __contains__/__bool__ keep numpy semantics and say so.
    v = SampledData(jnp.array([1.0, 2.0, 3.0]))
    assert 2.0 in v and 5.0 not in v
    with pytest.raises(ValueError):
        jnp.array([1.0, 2.0]) in v
    assert "ValueError" in SampledData.__contains__.__doc__


# --- [Mineure] Contrat public en deçà de la fiche module -------------------------
def test_public_contract_documented():
    # Documentation-only heading: the class docstring must state the real
    # surface (hdi/to_jax, delegated jnp stats with ddof=0, no export,
    # NaN in plots, unhashable) instead of the old "statistiques, export" claim.
    doc = SampledData.__doc__
    for token in ("hdi", "to_jax", "ddof", "export", "NaN", "unhashable"):
        assert token in doc


# --- [Mineure] rng stream() — ordre ; garde Tracer sur API privée ----------------
def test_rng_stream_order_dependence_documented_and_guard_kept():
    from BayesForge.Utils.rng import RNG

    # Same base seed -> identical stream; consecutive keys are independent.
    r1, r2 = RNG(5), RNG(5)
    k1 = [np.asarray(r1.stream()) for _ in range(3)]
    k2 = [np.asarray(r2.stream()) for _ in range(3)]
    for a, b in zip(k1, k2):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(k1[0], k1[1])

    # Order dependence and the mutating split are documented (module docstring).
    assert "statistically independent" in rng_mod.__doc__
    assert "order-dependent" in RNG.stream.__doc__
    # The Tracer guard is kept but flagged as a private-API dependency.
    assert "jax.core.Tracer" in inspect.getsource(RNG.stream)
    doc = RNG.stream.__doc__.lower()
    assert "private" in doc and "upgrade" in doc


# --- [Mineure] set_seed incohérent avec le constructeur ---------------------------
def test_set_seed_consistent_with_constructor_globals():
    from BayesForge.Utils.rng import RNG

    # Opt-in path (bf(seed_global_rngs=True)): after seed_globals(), set_seed
    # must re-sync the numpy globals with the new base, like the constructor.
    r = RNG(1)
    r.seed_globals()
    r.set_seed(42)
    a = np.random.rand()
    np.random.seed(42)
    assert a == np.random.rand()

    # Default path: set_seed never touches the user's globals.
    np.random.seed(7)
    x0 = np.random.rand()
    np.random.seed(7)
    RNG(1).set_seed(42)
    assert np.random.rand() == x0

    # The JAX stream rewinds to the new base either way.
    r = RNG(1)
    r.set_seed(9)
    np.testing.assert_array_equal(
        np.asarray(r.stream()), np.asarray(RNG(9).stream())
    )


# --- [Mineure] infer_discrete — hypothèses d'entrée, API privée numpyro ----------
def test_infer_discrete_validates_inputs_and_smoke():
    import numpyro
    import numpyro.distributions as dist

    import BayesForge.Utils.infer_discrete as id_mod
    from BayesForge.Utils.infer_discrete import sample_discrete_posterior

    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="empty"):
        sample_discrete_posterior(lambda: None, {}, key)
    with pytest.raises(ValueError, match="at least one draw"):
        sample_discrete_posterior(lambda: None, {"a": jnp.zeros((0, 2))}, key)
    with pytest.raises(ValueError, match="batch"):
        sample_discrete_posterior(
            lambda: None, {"a": jnp.zeros((5, 2)), "b": jnp.zeros((4, 2))}, key
        )

    # The private numpyro dependency is documented as such.
    assert "_sample_posterior" in id_mod.__doc__

    # Smoke test for that private API (the audit noted none existed):
    # y|z ~ Bern(0.95 if z else 0.05) with all-ones y pins every z to 1.
    def model(y=None):
        p = numpyro.sample("p", dist.Beta(1.0, 1.0))
        with numpyro.plate("obs", 4):
            z = numpyro.sample("z", dist.Bernoulli(0.5))
            numpyro.sample("y", dist.Bernoulli(0.95 * z + 0.05 * (1 - z)), obs=y)

    out = sample_discrete_posterior(
        model, {"p": jnp.full((6,), 0.5)}, key, {"y": jnp.ones(4)}
    )
    assert "z" in out and out["z"].shape[0] == 6
    assert float(np.mean(np.asarray(out["z"]))) > 0.5


# --- [Mineure] dev_warning — print au lieu de warnings.warn -----------------------
def test_dev_warning_uses_warnings_warn():
    from BayesForge.Utils import dev_warning as dw

    src = inspect.getsource(dw.warn_dev)
    assert "print(" not in src and "warnings.warn" in src

    dw.reset_dev_warnings()
    with pytest.warns(RuntimeWarning, match="development"):
        dw.warn_dev("audit_b")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dw.warn_dev("audit_b")  # same key: still deduplicated
    assert len(rec) == 0
    dw.reset_dev_warnings()
    with pytest.warns(RuntimeWarning):
        dw.warn_dev("audit_b")  # reset re-arms the anti-duplication set
    dw.reset_dev_warnings()


# --- [Mineure] ImportManager — get_module sans délai, erreurs, executor -----------
def test_import_manager_no_executor_clear_errors():
    import BayesForge.Utils.ImportManager as im_mod
    from BayesForge.Utils.ImportManager import LazyImporter

    # No executor at all any more: nothing to time out on, nothing to close,
    # no atexit hook needed (the audit's "close the executor" is moot).
    src = inspect.getsource(im_mod)
    assert "ThreadPoolExecutor" not in src and "concurrent.futures" not in src

    li = LazyImporter()
    li.schedule_import("json", "j")
    m1 = li.get_module("j")
    assert m1.__name__ == "json"
    assert li.get_module("j") is m1  # cached, imported once
    li.schedule_import("csv", "j")  # duplicate alias keeps the first mapping
    assert li.get_module("j").__name__ == "json"

    with pytest.raises(ImportError, match="not scheduled"):
        li.get_module("nope")

    # The import error surfaces at the get_module call, naming the package,
    # with the original ImportError chained (no deferred background surprise).
    li.schedule_import("not_a_module_xyz", "z")
    with pytest.raises(ImportError, match="pip install") as ei:
        li.get_module("z")
    assert "not_a_module_xyz" in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)

    li.shutdown()  # documented no-op
