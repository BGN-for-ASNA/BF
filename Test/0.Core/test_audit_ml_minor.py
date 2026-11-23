"""
Regression tests for the RELEASE_AUDIT.md "Module ML — minors" block (m1–m8).

m1  KMeans ignored ``tol``: it always ran the full ``n_iterations`` budget
m2  ``self.predictions`` was initialised to ``{}`` although it holds an array
m3  ``plot()`` required a jnp array, assumed 2-D input, hard-coded an
    unvalidated style, and built its contour grid with a fixed step
m4  float64 input was silently downcast to float32 when ``jax_enable_x64`` is off
m5  ``predict`` documented a ``centroids`` argument it does not accept
m6  ``ml.fit`` stored ``self.y`` / ``self.model_params`` and never read them
m7  unused imports (``matplotlib`` in KMeans, ``os``/``numpy``/``matplotlib`` in ml)
m8  the handler class is named ``ml`` while the module is also ``ml``

Only m1 and m4 need jax; the m3 / m5 / m6 / m7 / m8 checks read the source text
so they stay cheap and never build a plot.
"""

import importlib
import inspect
import warnings
from pathlib import Path

import numpy as onp
import pytest

from BayesForge.ML.KMeans import JAXKMeans as KMeans
from BayesForge.ML.ml import ml

_REPO = Path(__file__).resolve().parents[2]
_KMEANS_SRC = _REPO / "BayesForge" / "ML" / "KMeans.py"
_ML_SRC = _REPO / "BayesForge" / "ML" / "ml.py"

# Import the ML module
def _ml_module():
    """The BayesForge.ML.ml *module*: the package attribute of that name is
    the class, which is exactly the m8 collision."""
    return importlib.import_module("BayesForge.ML.ml")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _blobs(seed: int = 0):
    """Three well separated 2-D Gaussian blobs as float32."""
    rng = onp.random.default_rng(seed)
    parts = [
        rng.normal(loc=loc, scale=0.2, size=(60, 2))
        for loc in ([0.0, 0.0], [6.0, 6.0], [0.0, 6.0])
    ]
    X = onp.vstack(parts).astype(onp.float32)
    y = onp.repeat(onp.arange(3), 60)
    return X, y


def _fit(**kwargs):
    X, _ = _blobs()
    return KMeans(n_clusters=3, random_state=7, **kwargs).fit(X)


@pytest.fixture(autouse=True)
def _close_figures():
    """plot() opens a figure; do not leak it into the rest of the suite."""
    yield
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a hard dependency
        return
    plt.close("all")


# ---------------------------------------------------------------------------
# m1 — tol is honoured, tol=0 reproduces the old fixed-budget behaviour
# ---------------------------------------------------------------------------

def test_tol_zero_runs_the_full_iteration_budget():
    assert _fit(n_iterations=500, tol=0.0).n_iter_ == 500


def test_positive_tol_stops_early():
    km = _fit(n_iterations=500, tol=1e-4)
    assert isinstance(km.n_iter_, int)
    assert 1 <= km.n_iter_ < 500


def test_default_tol_is_positive():
    assert KMeans().tol == pytest.approx(1e-4)
    assert _fit(n_iterations=500).n_iter_ < 500


def test_tol_does_not_change_the_result_on_converged_data():
    """The point of the fix: early stopping must not move the optimum."""
    X, _ = _blobs()
    fixed = KMeans(n_clusters=3, random_state=7, tol=0.0).fit(X, n_iterations=500)
    early = KMeans(n_clusters=3, random_state=7, tol=1e-4).fit(X, n_iterations=500)
    onp.testing.assert_allclose(
        onp.asarray(early.centroids_), onp.asarray(fixed.centroids_), rtol=1e-4, atol=1e-5
    )
    assert onp.array_equal(onp.asarray(early.labels_), onp.asarray(fixed.labels_))


def test_fit_tol_overrides_the_constructor():
    X, _ = _blobs()
    km = KMeans(n_clusters=3, random_state=7, n_iterations=500, tol=0.0)
    km.fit(X, tol=1e-4)
    assert km.n_iter_ < 500


@pytest.mark.parametrize("bad_tol", [-1.0, onp.nan, onp.inf])
def test_invalid_tol_raises_value_error(bad_tol):
    with pytest.raises(ValueError, match="tol"):
        KMeans(tol=bad_tol)


def test_negative_tol_in_fit_raises_value_error():
    X, _ = _blobs()
    with pytest.raises(ValueError, match="tol"):
        KMeans(n_clusters=3).fit(X, tol=-0.5)


def test_exhausted_budget_warns():
    """Silent budget exhaustion was the audit complaint (m1).

    One iteration only: on well-separated blobs the k-means++ centres settle
    after a single Lloyd update, so with n_iterations=2 the last centroid
    move is exactly 0.0 <= tol — the run converged inside the budget and
    warning would be wrong. With one iteration the budget is cut while the
    centroids are still moving (first update from the k-means++ draws),
    which is the case the warning exists for.
    """
    X, _ = _blobs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        KMeans(n_clusters=3, random_state=7, n_iterations=1, tol=1e-12).fit(X)
    assert any("without converging" in str(w.message) for w in caught)


def test_tol_zero_never_warns_about_the_budget():
    X, _ = _blobs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        KMeans(n_clusters=3, random_state=7, n_iterations=5).fit(X, tol=0.0)
    assert not [w for w in caught if "without converging" in str(w.message)]


def test_clustering_matches_sklearn():
    pytest.importorskip("sklearn")
    from sklearn.metrics import adjusted_rand_score

    X, y = _blobs()
    labels = onp.asarray(_fit(n_iterations=300).labels_)
    assert adjusted_rand_score(y, labels) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# m2 — predictions is an array slot, not a dict
# ---------------------------------------------------------------------------

def test_predictions_is_none_before_predict():
    assert _fit(n_iterations=20).predictions is None


def test_predict_stores_an_array():
    X, _ = _blobs()
    km = _fit(n_iterations=20)
    out = km.predict(X)
    assert not isinstance(km.predictions, dict)
    assert onp.asarray(km.predictions).shape == (X.shape[0],)
    assert onp.array_equal(onp.asarray(out), onp.asarray(km.predictions))


# ---------------------------------------------------------------------------
# m3 — plot() and input handling
# ---------------------------------------------------------------------------

def test_fit_accepts_a_plain_list():
    X, _ = _blobs()
    km = KMeans(n_clusters=3, random_state=7, n_iterations=20, tol=0.0).fit(X.tolist())
    assert km.labels_ is not None


def test_fit_rejects_non_2d_input():
    X, _ = _blobs()
    with pytest.raises(ValueError, match="2-D"):
        KMeans(n_clusters=3).fit(X[:, 0])
    with pytest.raises(ValueError, match="2-D"):
        KMeans(n_clusters=3).fit([[[1.0, 2.0]]])


def test_predict_rejects_non_2d_input():
    km = _fit(n_iterations=20)
    with pytest.raises(ValueError, match="2-D"):
        km.predict([1.0, 2.0, 3.0])


def test_plot_grid_size_bounds_the_mesh(monkeypatch):
    """The contour grid is grid_size x grid_size, whatever the data extent."""
    pytest.importorskip("matplotlib")
    import matplotlib.axes
    import matplotlib.pyplot as plt

    captured = {}

    def fake_contour(self, *args, **kwargs):
        captured["xx"] = onp.asarray(args[0])
        return None

    monkeypatch.setattr(matplotlib.axes.Axes, "contour", fake_contour)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)

    X, _ = _blobs()
    _fit(n_iterations=20).plot(X, grid_size=7)
    assert captured["xx"].shape == (7, 7)


def test_plot_rejects_invalid_grid_size():
    X, _ = _blobs()
    with pytest.raises(ValueError, match="grid_size"):
        _fit(n_iterations=20).plot(X, grid_size=1)


def test_plot_warns_when_data_is_not_the_fitted_one(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    X, _ = _blobs()
    km = _fit(n_iterations=20)
    other, _ = _blobs(seed=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        km.plot(other, grid_size=5)
    assert any("not the data this model was fitted on" in str(w.message) for w in caught)


def test_plot_style_is_validated_against_matplotlib():
    src = _KMEANS_SRC.read_text()
    assert "plt.style.available" in src
    assert "plt.style.context" in src


def test_contour_is_no_longer_labelled_decision_boundaries():
    assert "Decision Boundaries" not in _KMEANS_SRC.read_text()


def test_contour_grid_uses_a_point_count_not_an_open_ended_step():
    src = _KMEANS_SRC.read_text()
    assert "jnp.arange(min_val" not in src
    assert "step = 0.02" not in src


# ---------------------------------------------------------------------------
# m4 — float64 downcast is announced
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_downcast_warning_follows_the_x64_flag(dtype):
    import jax

    X, _ = _blobs()
    X = X.astype(dtype)
    x64 = bool(jax.config.jax_enable_x64)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        KMeans(n_clusters=3, random_state=7, n_iterations=5).fit(X, tol=0.0)
    downcast = [w for w in caught if "float32" in str(w.message)]

    if dtype == "float64" and not x64:
        assert downcast, "float64 input must warn when jax_enable_x64 is off"
    else:
        assert not downcast


def test_no_downcast_warning_when_x64_is_enabled():
    import jax

    previous = bool(jax.config.jax_enable_x64)
    X, _ = _blobs()
    try:
        jax.config.update("jax_enable_x64", True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            KMeans(n_clusters=3, random_state=7, n_iterations=5).fit(
                X.astype(onp.float64), tol=0.0
            )
        assert not [w for w in caught if "float32" in str(w.message)]
    finally:
        jax.config.update("jax_enable_x64", previous)


# ---------------------------------------------------------------------------
# m5 — predict signature and docstring
# ---------------------------------------------------------------------------

def test_predict_takes_only_newdata():
    # m5 removed the documented-but-unaccepted ``centroids`` argument; the
    # data parameter keeps its existing name X.
    assert list(inspect.signature(KMeans.predict).parameters) == ["self", "X"]


def test_predict_docstring_no_longer_documents_parameters():
    assert "Parameters" not in (inspect.getdoc(KMeans.predict) or "")


# ---------------------------------------------------------------------------
# m6 — the handler stores what it actually uses
# ---------------------------------------------------------------------------

def test_handler_exposes_centroids_and_labels():
    # The handler builds and eagerly fits a JAXKMeans, so the fitted state it
    # used lives on handler.model (RELEASE_AUDIT m6).
    X, _ = _blobs()
    handler = ml().KMEANS(X, n_clusters=3, random_state=7, n_iterations=20, tol=0.0)
    assert handler.fit(X) is handler
    assert handler.model.centroids_ is not None
    assert handler.model.labels_ is not None
    assert handler.model.n_iter_ == 20


def test_handler_drops_the_unused_y_and_model_params():
    src = _ML_SRC.read_text()
    assert "self.y" not in src
    assert "self.model_params" not in src


def test_handler_records_predictions():
    X, _ = _blobs()
    handler = ml().KMEANS(X, n_clusters=3, random_state=7, n_iterations=20, tol=0.0)
    assert handler.predictions is None
    handler.predict(X)
    assert handler.predictions is not None


# ---------------------------------------------------------------------------
# m7 — no unused imports
# ---------------------------------------------------------------------------

def test_kmeans_has_no_unused_imports():
    src = _KMEANS_SRC.read_text()
    head, plot_body = src.split("def plot", 1)
    assert "fori_loop" not in src
    assert "import matplotlib.pyplot as plt" not in head
    assert "import matplotlib.pyplot as plt" in plot_body


def test_ml_has_no_unused_imports():
    src = _ML_SRC.read_text()
    for dead in ("import os", "import numpy as onp", "import matplotlib.pyplot as plt",
                 "Union"):
        assert dead not in src, f"{dead} is still imported or referenced"


# ---------------------------------------------------------------------------
# m8 — the ml / ml naming collision is documented and importable both ways
# ---------------------------------------------------------------------------

def test_ml_alias_exists():
    assert _ml_module().ML is ml


def test_naming_collision_is_documented():
    assert "ML = ml" in (_ml_module().__doc__ or "")


def test_handler_forwards_tol():
    X, _ = _blobs()
    handler = ml().KMEANS(X, n_clusters=2, tol=1e-2)
    assert handler.model.tol == pytest.approx(1e-2)
