"""Regression tests for RELEASE_AUDIT.md, module Parallel, minors m1-m12.

One test per audit point. m1, m8 and m9b are source-level checks (docstring,
call structure, ordering); m3a checks the devnull handle is closed at exit;
the others exercise behaviour. All tests use tiny functions, at most two
workers and progress=False so the module's default (workers = ncpus - 1,
rich progress) never slows them down.
"""

import inspect
import multiprocessing
import os
import sys
import threading
from unittest import mock

import pandas as pd
import pytest

from BayesForge.Parallel import grid, run_simulations
from BayesForge.Parallel.simulate import (
    _check_int,
    _env_override,
    _merge_xla_flags,
    _normalise_result,
    _normalise_scenarios,
    _worker_init,
    _parallel,
    _pin_worker,
    _plan,
    worker_stdout,
)


# ---------------------------------------------------------------------------
# m1 — _pin_worker block-overlap limitation documented
# ---------------------------------------------------------------------------

def test_m1_pin_worker_block_overlap_documented():
    doc = inspect.getdoc(_pin_worker) or ""
    assert "Known limitation" in doc
    assert "submission order" in doc


# ---------------------------------------------------------------------------
# m2 — XLA_FLAGS merged, not overwritten
# ---------------------------------------------------------------------------

def test_m2_merge_xla_flags_preserves_other_user_flags():
    merged = _merge_xla_flags(
        "--xla_dump_to=/tmp/keepme --foo=1",
        "--xla_force_host_platform_device_count=2",
    )
    assert "--xla_force_host_platform_device_count=2" in merged
    assert "--xla_dump_to=/tmp/keepme" in merged
    assert "--foo=1" in merged


def test_m2_env_override_keeps_user_xla_flags():
    with mock.patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/keepme"}):
        with _env_override({"XLA_FLAGS": "--xla_force_host_platform_device_count=1"}):
            assert os.environ["XLA_FLAGS"] == (
                "--xla_dump_to=/tmp/keepme --xla_force_host_platform_device_count=1"
            )
        assert os.environ["XLA_FLAGS"] == "--xla_dump_to=/tmp/keepme"


def test_m2_spawn_child_inherits_merged_xla_flags():
    ctx = multiprocessing.get_context("spawn")
    with mock.patch.dict(os.environ, {"XLA_FLAGS": "--xla_dump_to=/tmp/keepme"}):
        with _env_override({"XLA_FLAGS": "--xla_force_host_platform_device_count=2"}):
            with ctx.Pool(1) as pool:
                child_flags = pool.apply(os.getenv, ("XLA_FLAGS",))
    assert "--xla_dump_to=/tmp/keepme" in child_flags
    assert "--xla_force_host_platform_device_count=2" in child_flags


# ---------------------------------------------------------------------------
# m3 — devnull closed at exit; worker_stdout exported from the module
# ---------------------------------------------------------------------------

def test_m3_devnull_handle_is_closed_at_exit():
    src = inspect.getsource(_worker_init)
    assert "atexit.register(devnull.close)" in src
    assert '_STATE["devnull"]' in src


def test_m3_worker_stdout_exported_from_simulate():
    import BayesForge.Parallel.simulate as sim_mod

    assert "worker_stdout" in sim_mod.__all__
    assert sim_mod.worker_stdout is worker_stdout
    # outside a worker the context manager is a no-op redirect
    with worker_stdout():
        print("in main thread")


# ---------------------------------------------------------------------------
# m4 — scenario validation before the dict(s) copy
# ---------------------------------------------------------------------------

def test_m4_string_scenario_rejected_with_element_index():
    with pytest.raises(TypeError, match=r"element 1 is str: 'abc'"):
        _normalise_scenarios([{"x": 1}, "abc"])
    with pytest.raises(TypeError, match=r"element 0 is str: 'b'"):
        _normalise_scenarios(["b"])
    # a plain string is not an iterable of dicts either
    with pytest.raises(TypeError, match="scenarios must be an int or an iterable of dicts"):
        _normalise_scenarios("abc")


def test_m4_non_int_reps_rejected():
    with pytest.raises(TypeError, match="reps must be an integer"):
        grid(reps="3")
    with pytest.raises(TypeError, match="reps must be an integer"):
        grid(reps=2.5)
    # integral floats pass (R/Julia JSON bridges send 2.0)
    assert len(grid(reps=2.0)) == 2


# ---------------------------------------------------------------------------
# m5 — deterministic scenario order (sorted, not set iteration)
# ---------------------------------------------------------------------------

def test_m5_scenario_order_is_deterministic():
    expected = ["a", "b", "c"]
    for _ in range(5):
        scens = _normalise_scenarios(grid(x={"a", "b", "c"}, reps=1))
        assert [s["x"] for s in scens] == expected


def test_m5_run_simulations_rows_follow_sorted_order():
    def fn(x, **_):
        return {"x": x}

    out = run_simulations(
        fn, grid(x=["a", "b"], reps=1), workers=1, backend="serial", progress=False,
    )
    assert out["x"].tolist() == ["a", "b"]


# ---------------------------------------------------------------------------
# m6 — reserved column names reported as a recorded error, not a crash
# ---------------------------------------------------------------------------

def test_m6_reserved_column_names_recorded_not_raised():
    def fn(**_):
        return {"sim": 99, "ok": 1}

    out = run_simulations(
        fn, [{"id": 0}], workers=1, backend="serial",
        progress=False, on_error="record",
    )
    assert len(out) == 1
    assert isinstance(out["error"].iloc[0], str)
    assert "reserved" in out["error"].iloc[0]
    assert "sim" in out["error"].iloc[0]


def test_m6_reserved_column_names_raise_under_on_error_raise():
    def fn(**_):
        return {"seed": 99}

    with pytest.raises(RuntimeError, match="reserved column names"):
        run_simulations(
            fn, [{"id": 0}], workers=1, backend="serial",
            progress=False, on_error="raise",
        )


# ---------------------------------------------------------------------------
# m7 — _check_int: TypeError for wrong type, ValueError for out of range
# ---------------------------------------------------------------------------

def test_m7_check_int_type_and_range_errors():
    with pytest.raises(TypeError):
        _check_int("workers", "2")
    with pytest.raises(ValueError):
        _check_int("workers", 0)
    # integral floats are accepted
    assert _check_int("workers", 2.0) == 2
    assert _check_int("workers", None, allow_none=True) is None


def test_m7_plan_errors_are_typed():
    with pytest.raises(TypeError):
        _plan(1, workers=1.5)
    with pytest.raises(ValueError):
        _plan(1, workers=0)
    # integral floats pass (workers is capped at the scenario count)
    plan = _plan(3, workers=2.0, cores_per_worker=1.0)
    assert plan["workers"] == 2 and plan["cores_per_worker"] == 1


# ---------------------------------------------------------------------------
# m8 — _plan built once in run_simulations; _parallel receives env/block_size
# ---------------------------------------------------------------------------

def test_m8_parallel_takes_env_and_block_size_as_required_params():
    params = inspect.signature(_parallel).parameters
    assert list(params)[-2:] == ["env", "block_size"]
    assert params["env"].default is inspect.Parameter.empty
    assert params["block_size"].default is inspect.Parameter.empty
    # _parallel no longer plans for itself
    assert "_plan(" not in inspect.getsource(_parallel)


def test_m8_run_simulations_passes_plan_values_to_parallel():
    src = inspect.getsource(run_simulations)
    assert 'plan["env"], plan["block_size"]' in src


# ---------------------------------------------------------------------------
# m9 — BaseException path closes the display; close() is idempotent
# ---------------------------------------------------------------------------

def test_m9_display_closed_on_base_exception_path():
    from BayesForge.Parallel.simulate import _WorkerDisplay

    display = _WorkerDisplay(total=1, enabled=False)
    display.close()
    display.close()  # idempotent: no error on the second call

    # the KeyboardInterrupt path in _parallel closes the display before
    # re-raising (source-level check: close() precedes the bare re-raise)
    src = inspect.getsource(_parallel)
    body = src.split("except BaseException:", 1)[1].split("finally:", 1)[0]
    assert "display.close()" in body
    assert "raise" in body


def test_m9_keyboard_interrupt_propagates_and_leaves_no_worker_pools():
    import concurrent.futures

    def fn(**_):
        return {"x": 1}

    def _interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    with mock.patch.object(
        concurrent.futures, "ProcessPoolExecutor", side_effect=_interrupt
    ):
        with pytest.raises(KeyboardInterrupt):
            run_simulations(
                fn, [{"id": i} for i in range(4)], workers=2,
                backend="process", progress=False,
            )
    # no worker pools left behind by the aborted dispatch
    assert sum(
        1 for t in threading.enumerate() if "QueueFeederThread" in t.name
    ) == 0


# ---------------------------------------------------------------------------
# m10 — manager failure falls back to progress=False, run still completes
# ---------------------------------------------------------------------------

def test_m10_manager_failure_falls_back_to_no_progress():
    def _fallback_fn(seed=0, **kw):
        return {"ok": True}

    real_manager = multiprocessing.context.BaseContext.Manager

    def broken_manager(self, *args, **kwargs):
        raise OSError("manager down")

    with mock.patch.object(
        multiprocessing.context.BaseContext, "Manager", broken_manager
    ):
        with pytest.warns(UserWarning, match="progress manager"):
            out = run_simulations(
                _fallback_fn, [{"id": i} for i in range(4)], workers=2,
                backend="process", progress=True,
            )
    assert len(out) == 4
    assert out["error"].isna().all()
    # the fallback must not leave the manager's feeder threads behind
    assert sum(
        1 for t in threading.enumerate() if "QueueFeederThread" in t.name
    ) == 0
    # sanity: the real Manager was reachable before patching
    assert callable(real_manager)


# ---------------------------------------------------------------------------
# m11 — meta columns first, then scenario keys, then fn keys
# ---------------------------------------------------------------------------

def test_m11_meta_columns_come_first():
    def fn(a, b, c, d, **_):
        return {"d": d, "c": c, "b": b, "a": a}

    out = run_simulations(
        fn, [{"d": 4, "c": 3, "b": 2, "a": 1}], workers=1,
        backend="serial", progress=False,
    )
    cols = list(out.columns)
    assert cols[:4] == ["sim", "seed", "error", "worker"]
    # scenario keys keep their first-seen order, then the remaining meta
    assert cols[4:] == ["d", "c", "b", "a", "rep", "elapsed_s"]


# ---------------------------------------------------------------------------
# m12 — _normalise_result rejects MultiIndex frames and duplicate columns
# ---------------------------------------------------------------------------

def test_m12_multiindex_frame_rejected():
    frame = pd.DataFrame(
        [[1, 2]], columns=pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
    )
    with pytest.raises(ValueError, match="MultiIndex"):
        _normalise_result(frame)


def test_m12_duplicate_columns_rejected():
    frame = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicated column names"):
        _normalise_result(frame)
