"""Regression tests for RELEASE_AUDIT.md, Integration X2 and X5.

X2: an unpicklable return value must be recorded as an error row in BOTH
backends -- serial must not accept what the process backend rejects, and the
process run must not crash. X5: workers must not inherit JAX's default VRAM
preallocation.

Style follows Test/13.Parallel/test_audit_parallel_major.py: module-level
simulation functions (picklable by reference for the spawned workers), tiny
scenario counts, real workers kept to 2.
"""
import threading

import pytest

from BayesForge.Parallel.simulate import (
    _n_cpus,
    _worker_env,
    run_simulations,
)


def _lock_result(sim, **_):
    """Return value no worker can pickle (X2): an RLock is not picklable."""
    return {"lock": threading.RLock()}


def test_x2_serial_records_unpicklable_result():
    # Serial keeps rows in-process, so without the X2 check this run would
    # "succeed" while the process backend dies on the same fn.
    df = run_simulations(_lock_result, 3, backend="serial", progress=False)
    assert len(df) == 3
    assert "error" in df.columns
    for msg in df["error"]:
        assert isinstance(msg, str) and "not picklable" in msg


def test_x2_process_records_unpicklable_result():
    if _n_cpus() < 2:
        pytest.skip("needs at least 2 CPUs for workers=2")
    # on_error="record" (default): the run completes and every row carries
    # the error, instead of the dumps failure killing the whole run.
    df = run_simulations(_lock_result, 2, workers=2, progress=False)
    assert len(df) == 2
    assert "error" in df.columns
    for msg in df["error"]:
        assert isinstance(msg, str) and "not picklable" in msg


def test_x5_worker_env_disables_prealloc(monkeypatch):
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    assert _worker_env(2)["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_x5_worker_env_keeps_user_value(monkeypatch):
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    assert _worker_env(2)["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
