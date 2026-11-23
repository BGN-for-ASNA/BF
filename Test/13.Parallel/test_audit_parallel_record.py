"""Regression test for RELEASE_AUDIT.md, module 10 (Parallel/simulate.py), critical C1."""
import threading

from BayesForge.Parallel.simulate import run_simulations


def _sim(sim, seed, **_):
    if sim % 2 == 0:
        return {"x": float(sim), "obj": threading.Lock()}  # result cannot be pickled
    return {"x": float(sim)}


def test_c1_unpicklable_result_is_recorded_not_fatal():
    df = run_simulations(_sim, 6, workers=2, progress=False, on_error="record")
    assert len(df) == 6
    failed = df[df["error"].notna()]
    assert sorted(failed["sim"].tolist()) == [0, 2, 4]
    ok = df[df["error"].isna()]
    assert sorted(ok["x"].tolist()) == [1.0, 3.0, 5.0]
