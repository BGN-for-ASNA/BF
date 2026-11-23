"""Regression tests for RELEASE_AUDIT.md, module Utils, majors corr_heatmap, lazy imports, global RNG seeding."""
import numpy as np
import pytest

from BayesForge import bf


def test_bf_does_not_reseed_numpy():
    # Default (seed_global_rngs=False): constructing bf() must leave the
    # user's numpy.random stream untouched.
    np.random.seed(123)
    a = np.random.rand()
    np.random.seed(123)
    bf(platform="cpu", rand_seed=7, print_devices_found=False)
    b = np.random.rand()
    assert a == b

    # Opt-in (seed_global_rngs=True): two constructions with the same base
    # seed must seed numpy.random identically.
    bf(platform="cpu", rand_seed=7, print_devices_found=False, seed_global_rngs=True)
    x = np.random.rand()
    bf(platform="cpu", rand_seed=7, print_devices_found=False, seed_global_rngs=True)
    y = np.random.rand()
    assert x == y


def test_corr_heatmap_is_correlation(monkeypatch):
    go = pytest.importorskip("plotly.graph_objects")
    from BayesForge.Utils.SampledData import SampledData

    digits = 5
    draws = np.random.default_rng(0).normal(size=(500, 3))
    captured = {}

    def fake_show(self, *args, **kwargs):
        captured["fig"] = self

    monkeypatch.setattr(go.Figure, "show", fake_show)
    SampledData(draws).corr_heatmap(digits=digits)

    fig = captured["fig"]
    z = np.asarray(fig.data[0].z)
    assert z.shape == (3, 3)
    expected = np.corrcoef(draws, rowvar=False)
    assert np.allclose(z, expected, atol=1e-8)
    # Annotations are the correlation values rounded to `digits`.
    text = np.asarray(fig.data[0].text, dtype=object)
    assert text.shape == (3, 3)
    assert text[0, 1] == f"{expected[0, 1]:.{digits}f}"


def test_lazy_importer_no_threads():
    from BayesForge.Utils.ImportManager import LazyImporter

    li = LazyImporter()
    li.schedule_import("json", "j")
    assert li.get_module("j").__name__ == "json"

    li.schedule_import("not_a_module_xyz", "z")
    with pytest.raises(ImportError, match="pip install"):
        li.get_module("z")

    assert not hasattr(li, "_executor") or li._executor is None
