"""Regression tests for RELEASE_AUDIT.md, module 14 (Causal), critical C1/C2."""

import warnings
import pytest

from BayesForge.Causal.dag import DAG


def _g(spec):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DAG(spec)


@pytest.mark.parametrize("spec", ["x -> m <- w", "x <-> y", "a -> b -> c", "x -> ", "<- y"])
def test_c1_malformed_chunks_raise(spec):
    with pytest.raises(ValueError):
        _g(spec)


def test_c1_valid_specs_still_parse():
    g = _g("z -> x; z -> y\n y <- x")
    assert set(g.edges) == {("z", "x"), ("z", "y"), ("x", "y")}


@pytest.mark.parametrize("call", [
    lambda g: g.d_separated("x", "w"),
    lambda g: g.d_separated("x", "y", given={"z_typo"}),
    lambda g: g.adjustment_sets("x", "typo"),
    lambda g: g.check("x", "y", given={"z_typo"}),
    lambda g: g.backdoor_paths("typo", "y"),
    lambda g: g.is_valid_adjustment_set("x", "y", {"nope"}),
])
def test_c2_unknown_nodes_raise(call):
    g = _g("z -> x, z -> y, x -> y")
    with pytest.raises(ValueError):
        call(g)


def test_c2_known_nodes_unchanged():
    g = _g("z -> x, z -> y, x -> y")
    assert g.d_separated("x", "y", given={"z"}) is False
    assert g.adjustment_sets("x", "y") == [{"z"}]
    assert g.check("x", "y", given={"z"}) == []
