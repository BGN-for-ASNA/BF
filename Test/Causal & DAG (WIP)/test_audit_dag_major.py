"""Regression tests for RELEASE_AUDIT.md, module Causal, majors M1-M5."""

import warnings

import matplotlib
import pytest

from BayesForge.Causal.dag import DAG

matplotlib.use("Agg")


def _g(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DAG(spec)


# -- M1: plot() on the empty graph -------------------------------------------

def test_m1_plot_empty_dag_does_not_raise():
    ax = DAG().plot()
    assert ax is not None


def test_m1_plot_empty_dag_keeps_title():
    ax = DAG().plot(title="empty")
    assert ax.get_title() == "empty"


# -- M2: combinatorial guards --------------------------------------------------

def _many_confounders():
    g = DAG()
    for i in range(20):
        g.add_edge(f"c{i}", "x")
        g.add_edge(f"c{i}", "y")
    g.add_edge("x", "y")
    return g


def test_m2_adjustment_sets_default_guard_raises():
    g = _many_confounders()
    with pytest.raises(ValueError, match="max_size"):
        g.adjustment_sets("x", "y")


def test_m2_adjustment_sets_with_max_size_returns_quickly():
    g = _many_confounders()
    assert g.adjustment_sets("x", "y", max_size=1) == []


# -- M3: string node name vs iterable of nodes ---------------------------------

def test_m3_d_separated_string_given_is_one_node():
    g = _g("a -> age, age -> b")
    assert g.d_separated("a", "b", given="age") is True
    assert g.d_separated("a", "b", given="age") == g.d_separated("a", "b", given={"age"})


def test_m3_check_string_given_matches_set():
    g = _g("a -> age, age -> b")
    assert g.check("a", "b", given="age") == g.check("a", "b", given={"age"})


# -- M4: DAG(iterable) element validation ---------------------------------------

def test_m4_string_without_arrow_raises():
    with pytest.raises((ValueError, TypeError)):
        DAG(["ab"])


def test_m4_string_edges_are_parsed():
    g = _g(["z -> x", "x -> y"])
    assert ("z", "x") in g.edges
    assert ("x", "y") in g.edges


def test_m4_non_pair_element_raises_type_error():
    with pytest.raises(TypeError):
        DAG([1])


def test_m4_pair_elements_still_work():
    g = _g([("z", "x"), ("x", "y")])
    assert set(g.edges) == {("z", "x"), ("x", "y")}


# -- M5: long chains without RecursionError -------------------------------------

def test_m5_long_chain_builds_and_traverses():
    g = DAG()
    for i in range(2999):
        g.add_edge(f"n{i}", f"n{i + 1}")
    assert len(g.descendants("n0")) == 2999
