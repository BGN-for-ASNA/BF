"""Regression tests for RELEASE_AUDIT.md, module Network, minors m11-m15, m17."""
import numpy as np
import pandas as pd
import pytest

from BayesForge.Network import layouts as lay
from BayesForge.Network import vizfeatures as vf


# --------------------------------------------------------------------------- #
# m11 — min-max normalisations: constant column must not produce NaN
# --------------------------------------------------------------------------- #
def test_m11_minmax_norm_constant_column_no_nan():
    from BayesForge.Network.explorer import _minmax_norm

    v = _minmax_norm(np.full(5, 3.0))
    assert np.isfinite(v).all()
    assert np.allclose(v, 0.5)  # documented neutral value

    # non-constant column keeps the historical behaviour (min lifted to floor)
    v = _minmax_norm(np.array([1.0, 2.0, 3.0]), floor=1e-3)
    assert np.allclose(v, [1e-3, 0.5, 1.0])

    # all-NaN column: no range either -> neutral, no NaN out
    v = _minmax_norm(np.array([np.nan, np.nan]))
    assert np.isfinite(v).all() and np.allclose(v, 0.5)


def test_m11_format_att_constant_opacity_column_no_nan():
    from BayesForge.Network.explorer import NetExplorer

    ex = NetExplorer()
    df = pd.DataFrame({"id": ["a", "b", "c"], "op": [2.0, 2.0, 2.0]})
    d, _ori = ex.format_att(df, col_id="id", node_opacity="op")
    assert np.isfinite(np.asarray(d["opacity"], dtype=float)).all()
    assert np.allclose(d["opacity"], 0.5)


# --------------------------------------------------------------------------- #
# m12 — vis_net must validate len(df) == N at the boundary
# --------------------------------------------------------------------------- #
def test_m12_check_node_table_helper():
    from BayesForge.Network.explorer import _check_node_table

    _check_node_table(2, 2)  # aligned: no raise
    with pytest.raises(ValueError, match="one row per node"):
        _check_node_table(3, 2)
    with pytest.raises(ValueError, match="one row per node"):
        _check_node_table(1, 2)


def test_m12_vis_net_length_mismatch_raises(tmp_path):
    from BayesForge.Network.explorer import NetExplorer

    ex = NetExplorer()
    df = pd.DataFrame({"id": ["a", "b", "c"]})  # 3 rows
    adj = np.array([[0.0, 1.0], [0.0, 0.0]])    # 2x2 matrix
    with pytest.raises(ValueError, match="one row per node"):
        ex.vis_net(df=df, m=adj, col_id="id", browser=False, out_dir=tmp_path)


# --------------------------------------------------------------------------- #
# m13 — layouts: N=0 / N=1 guards, disconnected spectral warns + fallback
# --------------------------------------------------------------------------- #
def test_m13_layouts_empty_and_single_node():
    for fn in (lay.spectral, lay.mds, lay.clustered, lay.layered):
        lx, ly = fn(np.zeros((0, 0)))
        assert lx.shape == (0,) and ly.shape == (0,)
        lx, ly = fn(np.zeros((1, 1)))
        assert np.allclose(np.column_stack([lx, ly]), [[0.0, 0.0]])

    # metric-based layouts take vectors, not matrices
    for lx, ly in (lay.radial(np.zeros(0)), lay.arc(np.array([])),
                   lay.geo(np.array([]), np.array([]))):
        assert lx.shape == (0,) and ly.shape == (0,)
    for lx, ly in (lay.radial(np.zeros(1)), lay.arc(np.array([7.0])),
                   lay.geo(np.array([3.0]), np.array([-1.0]))):
        assert np.allclose(np.column_stack([lx, ly]), [[0.0, 0.0]])

    # dispatch keeps the same contract
    lx, ly = lay.compute("spectral", np.zeros((0, 0)))
    assert lx.shape == (0,) and ly.shape == (0,)


def test_m13_spectral_disconnected_warns_and_falls_back():
    # two disjoint edges -> 2 components -> 2 zero eigenvalues
    adj = np.array([[0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0]])
    with pytest.warns(UserWarning, match="disconnected"):
        lx, ly = lay.spectral(adj)
    assert lx.shape == (4,) and ly.shape == (4,)
    assert np.isfinite(lx).all() and np.isfinite(ly).all()
    assert lx.min() >= 0.0 and lx.max() <= 1.0
    assert ly.min() >= 0.0 and ly.max() <= 1.0


def test_m13_spectral_connected_does_not_warn():
    A = np.array([[0.0, 1.0, 1.0, 0.0],
                  [1.0, 0.0, 1.0, 1.0],
                  [1.0, 1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
    lx, ly = lay.spectral(A)  # no warning expected
    assert lx.shape == (4,) and np.isfinite(lx).all()


# --------------------------------------------------------------------------- #
# m14 — chord_matrix: vectorised output identical to the reference loop
# --------------------------------------------------------------------------- #
def test_m14_chord_matrix_matches_reference_double_loop():
    rng = np.random.default_rng(0)
    a = (rng.random((6, 6)) > 0.4) * rng.integers(1, 5, (6, 6))
    a = a.astype(float)
    groups = np.array([0, 0, 1, 1, 2, 2])

    M, labels = lay.chord_matrix(groups, a)

    ref = np.zeros((3, 3))
    for i in range(6):
        for j in range(6):
            if a[i, j]:
                ref[groups[i], groups[j]] += a[i, j]
    assert labels == [0, 1, 2]
    assert np.allclose(M, ref)


def test_m14_chord_matrix_string_groups_and_length_guard():
    a = np.array([[0.0, 1.0], [2.0, 0.0]])
    M, labels = lay.chord_matrix(np.array(["x", "y"]), a)
    assert labels == ["x", "y"]
    assert np.allclose(M, [[0.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match="one group per node"):
        lay.chord_matrix(np.array(["x"]), a)


# --------------------------------------------------------------------------- #
# m15 — density fallback consistent directed/undirected, self-loops excluded
# --------------------------------------------------------------------------- #
def test_m15_density_fallback_directed_vs_undirected():
    # n=3, E=2: directed E/(n(n-1)) = 2/6, undirected 2E/(n(n-1)) = 2/3
    assert vf._density(3, 2, undirected=False) == pytest.approx(2 / 6)
    assert vf._density(3, 2, undirected=True) == pytest.approx(2 * 2 / 6)
    assert vf._density(1, 0, undirected=True) == 0.0
    assert vf._density(0, 0, undirected=False) == 0.0


def test_m15_network_stats_ignores_self_loops():
    # audit case: [[1,1],[1,0]] used to report density 1.5 / mean degree 1.5
    st = vf.network_stats(np.array([[1.0, 1.0], [1.0, 0.0]]))
    assert st["edges"] == 1
    assert st["mean_degree"] == pytest.approx(1.0)
    assert st["density"] == pytest.approx(1.0)  # 1 edge / (2*1/2)
    assert st["directed"] is False


def test_m15_network_stats_directed_density():
    # one directed edge 0->1: E/(n(n-1)) = 1/2
    st = vf.network_stats(np.array([[0.0, 1.0], [0.0, 0.0]]))
    assert st["directed"] is True
    assert st["edges"] == 1
    assert st["density"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# m17 — HTML/JS injection: user labels must be escaped in the generated page
# --------------------------------------------------------------------------- #
def test_m17_js_json_escapes_script_break():
    from BayesForge.Network.explorer import _js_json

    out = _js_json("</script><b>")
    assert "</script>" not in out
    assert "\\u003c" in out
    # still valid JSON
    import json

    assert json.loads(out) == "</script><b>"


def test_m17_js_val_and_tooltip_escape():
    from BayesForge.Network.explorer import NetExplorer

    assert "<script>" not in NetExplorer._js_val("<script>alert(1)</script>")
    # same single-quoted literal format as before m17 (NetExplorer tests parse it)
    assert NetExplorer._js_val("ind1") == "'ind1'"
    assert NetExplorer._js_val("a'b") == "'a\\'b'"
    ex = NetExplorer()
    ori = ["id", "size'x</script>", None, None, None, None, None]
    js = ex._tooltip_js(ori)
    assert "</script>" not in js
    assert "\\u003c" in js


def test_m17_label_escaped_in_generated_html(tmp_path):
    from BayesForge.Network.explorer import NetExplorer

    ex = NetExplorer()
    df = pd.DataFrame({"id": ["<script>x</script>", "b"]})
    adj = np.array([[0.0, 1.0], [0.0, 0.0]])
    ex.vis_net(df=df, m=adj, col_id="id", browser=False,
               out_dir=tmp_path, metrics=False)
    html = (tmp_path / "NetExplorer.html").read_text(encoding="utf-8")
    # the raw payload must not survive as an executable script tag...
    assert "<script>x</script>" not in html
    # ...it appears only with "<" escaped as \\u003c
    assert "\\u003cscript>x\\u003c/script>" in html
