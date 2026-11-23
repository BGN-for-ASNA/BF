"""Regression tests for RELEASE_AUDIT.md, module Parallel, majors M1-M7.

Setup follows Test/13.Parallel/test_audit_parallel_record.py (critical C1):
module-level simulation functions, real spawned workers kept tiny (2 workers,
few scenarios).

Coverage notes:
- M1, M2, M3, M7 are covered directly (pure helpers or tiny in-process runs).
- M4 uses a real worker that calls os._exit(1) -- the OOM-killer stand-in --
  with on_error="record", 2 workers and 3 tiny scenarios.
- M5 and M6 are covered through their pure helpers only (_interrupt_message,
  _manager_failure_msg): a real test would have to deliver SIGINT to a parent
  mid-run, or run an unguarded script that spawns a manager, both of which
  risk killing or hanging the pytest session itself. The process-level
  behaviour (cancel_futures without joining, friendly manager error) is
  therefore NOT exercised here.
"""
import os
import time
import warnings

import pytest

from BayesForge.Parallel.simulate import (
    _DEAD_WORKER_MSG,
    _gpu_env_for_worker,
    _interrupt_message,
    _manager_failure_msg,
    _n_cpus,
    _plan,
    _visible_gpus,
    grid,
    run_simulations,
)


def _noop(sim, **_):
    return {"x": 1.0}


# ---------------------------------------------------------------------------
# M1 -- affinity-aware CPU budget, oversubscription warning, pinning blocks
# ---------------------------------------------------------------------------

def test_m1_oversubscription_warns_under_affinity_limit(monkeypatch):
    # taskset -c 0: one usable CPU, two workers asked for -> must warn+cap.
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0})
    with pytest.warns(UserWarning, match="exceeds the 1 available cores"):
        df = run_simulations(_noop, 2, workers=2, progress=False)
    assert len(df) == 2          # capped to 1 worker, run still completes


def test_m1_plan_caps_on_affinity_not_cpu_count(monkeypatch):
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {0, 1})
    plan = _plan(10, workers=4, cores_per_worker=1)
    assert plan["workers"] == 2                      # 2 CPUs, not os.cpu_count()
    assert plan["block_size"] == 1
    assert any("exceeds" in w for w in plan["warnings"])


def test_m1_pinning_blocks_cut_from_the_affinity_set(monkeypatch):
    from BayesForge.Parallel import simulate as sim

    pinned = {}
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: {4, 5, 6, 7})
    monkeypatch.setattr(os, "sched_setaffinity",
                        lambda pid, cpus: pinned.setdefault("cpus", set(cpus)))
    saved = dict(sim._STATE)
    sim._STATE.pop("cpus", None)
    try:
        cpus = sim._pin_worker(1, 2)                 # second block of two
    finally:
        sim._STATE.clear()
        sim._STATE.update(saved)
    # Blocks come from the affinity set {4,5,6,7}, not range(os.cpu_count()).
    assert cpus == [6, 7]
    assert pinned["cpus"] == {6, 7}


# ---------------------------------------------------------------------------
# M2 -- workers=1 in-process fallback: warn, honour quiet
# ---------------------------------------------------------------------------

def test_m2_workers_1_warns_and_honours_quiet(capsys):
    def noisy(sim, **_):
        print("bf-noise")
        return {"x": float(sim)}

    with pytest.warns(UserWarning, match="runs in-process"):
        df = run_simulations(noisy, 2, workers=1, quiet=True, progress=False)
    assert len(df) == 2
    assert "bf-noise" not in capsys.readouterr().out

    with pytest.warns(UserWarning):
        df = run_simulations(noisy, 1, workers=1, quiet=False, progress=False)
    assert "bf-noise" in capsys.readouterr().out


def test_m2_explicit_serial_does_not_warn(capsys):
    def noisy(sim, **_):
        print("bf-noise")
        return {"x": float(sim)}

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning becomes an error
        df = run_simulations(noisy, 1, backend="serial", progress=False)
    assert len(df) == 1
    # backend='serial' keeps stdout open for debugging.
    assert "bf-noise" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# M3 -- duplicate sim ids rejected
# ---------------------------------------------------------------------------

def test_m3_duplicate_sim_raises():
    # Two grids, each numbered from 0 -- the natural composition footgun.
    scenarios = grid(N=[1, 2], reps=2) + grid(N=[3], reps=2)
    with pytest.raises(ValueError, match="duplicate 'sim'"):
        run_simulations(_noop, scenarios, workers=1, progress=False)

    # Unique sims still plan fine.
    plan = _plan(grid(N=[1, 2], reps=2))
    assert [s["sim"] for s in plan["scenarios"]] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# M4 -- BrokenProcessPool: keep collected rows, record the dead ones
# ---------------------------------------------------------------------------

def _die_once(sim, **_):
    if sim == 2:
        time.sleep(0.5)     # let the other workers' results reach the parent
        os._exit(1)         # OOM-killer stand-in: no cleanup, no sentinel
    return {"x": float(sim)}


def test_m4_dead_worker_keeps_other_results():
    if _n_cpus() < 2:
        pytest.skip("needs at least 2 CPUs for workers=2")
    with pytest.warns(UserWarning, match="worker process died"):
        df = run_simulations(_die_once, 3, workers=2, progress=False,
                             on_error="record")
    assert len(df) == 3                       # nothing lost
    dead = df[df["error"].notna()]
    assert dead["sim"].tolist() == [2]        # sims 0 and 1 finished first
    assert dead["error"].iloc[0] == _DEAD_WORKER_MSG
    ok = df[df["error"].isna()]
    assert sorted(ok["x"].tolist()) == [0.0, 1.0]


# ---------------------------------------------------------------------------
# M5 -- KeyboardInterrupt (pure helper only, see module docstring)
# ---------------------------------------------------------------------------

def test_m5_interrupt_message_reports_rows():
    msg = _interrupt_message(3, 10)
    assert "3/10" in msg
    assert ".rows" in msg


# ---------------------------------------------------------------------------
# M6 -- unguarded script + progress=True (pure helper only, see docstring)
# ---------------------------------------------------------------------------

def test_m6_manager_failure_message_names_the_guard():
    msg = _manager_failure_msg(RuntimeError("manager process failed to start"))
    assert "__main__" in msg
    assert "guard" in msg
    assert "manager process failed to start" in msg   # underlying error kept
    assert "progress=False" in msg


# ---------------------------------------------------------------------------
# M7 -- per-worker CUDA_VISIBLE_DEVICES
# ---------------------------------------------------------------------------

def test_m7_gpu_env_round_robin():
    assert _gpu_env_for_worker(0, ["0", "1", "2"]) == {"CUDA_VISIBLE_DEVICES": "0"}
    assert _gpu_env_for_worker(1, ["0", "1", "2"]) == {"CUDA_VISIBLE_DEVICES": "1"}
    assert _gpu_env_for_worker(3, ["0", "1", "2"]) == {"CUDA_VISIBLE_DEVICES": "0"}
    assert _gpu_env_for_worker(5, ["2", "7"]) == {"CUDA_VISIBLE_DEVICES": "7"}
    assert _gpu_env_for_worker(0, []) == {}


def test_m7_visible_gpus_reads_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 5")
    assert _visible_gpus() == ["2", "5"]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert _visible_gpus() == []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "none")
    assert _visible_gpus() == []
