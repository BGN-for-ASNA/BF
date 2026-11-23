"""Regression tests for RELEASE_AUDIT.md, module Causal, minors."""

import inspect
import os
import re
import subprocess
import sys
import warnings

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: never pop a window during tests

from BayesForge.Causal import dag as dag_mod
from BayesForge.Causal.dag import DAG

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _g(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DAG(spec)


# -- m1: d_separated(x, x) ------------------------------------------------------

def test_m1_d_separated_self_raises():
    g = _g("z -> x, z -> y, x -> y")
    with pytest.raises(ValueError):
        g.d_separated("x", "x")


# -- m2: back-door deviation (y forbidden in z): kept, but documented ----------

def test_m2_backdoor_y_forbidden_is_documented():
    g = _g("z -> x, z -> y, x -> y")
    # behaviour kept on purpose: y is rejected in the adjustment set...
    assert g.is_valid_adjustment_set("x", "y", {"y"}) is False
    # ...and the deviation from Pearl's criterion is now documented.
    doc = inspect.getdoc(DAG.is_valid_adjustment_set) or ""
    assert "Pearl" in doc and "outcome" in doc


# -- m3: empty node names -------------------------------------------------------

def test_m3_empty_node_name_raises():
    with pytest.raises(ValueError):
        DAG([("a", "")])          # tuple path bypassed _parse_spec before
    with pytest.raises(ValueError):
        DAG([("  ", "b")])        # whitespace-only is empty too
    g = _g("a -> b")              # valid names still work
    g.add_edge("b", "c")
    assert g.nodes == {"a", "b", "c"}


# -- m4: quadratic costs / deque in _topo_order ---------------------------------

def test_m4_complexity_documented_and_deque_used():
    for fn in (DAG.add_edge, DAG._is_open, DAG.check, DAG._all_paths):
        doc = inspect.getdoc(fn) or ""
        assert "Complexity" in doc, f"{fn.__name__} lacks a complexity note"
    src = inspect.getsource(DAG._topo_order)
    code = src.split('"""', 2)[-1]  # the docstring may mention the old pop(0)
    assert "popleft()" in code and "pop(0)" not in code


# -- m5: list copies in path enumeration (documented) ---------------------------

def test_m5_path_copy_cost_documented():
    doc = inspect.getdoc(DAG._all_paths) or ""
    assert "prefix" in doc and "memory" in doc


# -- m6: experimental notice is a warning, not print + global -------------------

def test_m6_experimental_notice_is_a_warning_not_print():
    with pytest.warns(UserWarning, match="experimental"):
        DAG()
    # the one-shot global state is gone; Python's warning filter dedupes
    assert not hasattr(dag_mod, "_WARNED")


# -- m7: deterministic orders (edges / __repr__) --------------------------------

def test_m7_repr_is_deterministic_across_processes():
    code = ("from BayesForge.Causal.dag import DAG; "
            "print(repr(DAG('a -> b, a -> c, a -> d, a -> e')))")
    envs = [{"PYTHONHASHSEED": "1"}, {"PYTHONHASHSEED": "2"}]
    reps = []
    for extra in envs:
        env = {"PYTHONPATH": REPO, "PATH": "/usr/bin:/bin", **extra}
        out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env,
                             capture_output=True, text=True)
        reps.append(out.stdout.strip().splitlines()[-1] if out.stdout.strip()
                    else out.stderr.strip())
    assert reps[0] == reps[1], f"repr differs across hash seeds: {reps}"
    assert reps[0] == "DAG(a -> b, a -> c, a -> d, a -> e)"


# -- m8: module docstring example must run as written ---------------------------

def test_m8_module_docstring_example_runs():
    """The module docstring example is a doctest: run it as one."""
    import doctest
    doc = inspect.getdoc(dag_mod)
    assert ">>> g.check(" in doc, "module docstring example is incomplete"
    parser = doctest.DocTestParser()
    test = parser.get_doctest(doc, {"DAG": DAG}, "dag_module_doc", None, 0)
    runner = doctest.DocTestRunner(optionflags=doctest.NORMALIZE_WHITESPACE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.run(test)
    assert runner.failures == 0 and runner.tries >= 5


# -- m9: sorted(candidates) on heterogeneous node types -------------------------

def test_m9_adjustment_sets_mixed_node_types():
    g = _g([("z", "x"), ("z", "y"), ("x", "y"), (2, "x")])
    try:
        sets = g.adjustment_sets("x", "y")
    except TypeError:
        pytest.fail("adjustment_sets raised TypeError on mixed int/str nodes")
    assert {"z"} in sets


# -- m10: refused edge must not leave orphan nodes ------------------------------

def test_m10_refused_edge_leaves_no_orphan_nodes():
    g = _g("a -> b")
    before = set(g.nodes)
    with pytest.raises(ValueError):
        g.add_edge("b", "a")      # would close a 2-cycle
    assert g.nodes == before      # the refused edge must not mutate the graph
    h = DAG()
    with pytest.raises(ValueError):
        h.add_edge("q", "q")      # self-loop
    assert h.nodes == set()


# -- m11: plot must not force plt.show(); show=True opts in ---------------------

def test_m11_plot_does_not_force_show(monkeypatch):
    import matplotlib.pyplot as plt
    calls = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: calls.append(1))
    g = _g("z -> x, z -> y, x -> y")
    ax = g.plot(x="x", y="y", adjust={"z"})
    assert ax is not None and not calls
    g.plot(show=True)
    assert calls
    plt.close("all")


# -- m12: dead branch in the edge-curvature logic removed -----------------------

def test_m12_dead_curvature_branch_removed():
    src = inspect.getsource(DAG.plot)
    assert "same_row" not in src
    assert "else 0.4" not in src
