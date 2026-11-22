"""Run independent simulation scenarios in parallel worker processes.

A simulation study -- parameter recovery, a scenario sweep, prior sensitivity --
is a loop of *independent* fits. ``fit`` already parallelises what happens
*inside* one fit (``num_chains`` with ``chain_method='parallel'``, and the
experimental data sharding behind ``fit(shard=True)``), but the loop over
datasets stays serial, so one core works while the rest idle.

:func:`run_simulations` spreads that loop over worker processes. Each worker is
a fresh Python process with its own JAX backend, so each simulation gets its own
``bf`` instance, its own device budget and its own RNG -- exactly the isolation
a per-fit ``cores=`` argument could never provide, because the CPU device count
is written into ``XLA_FLAGS`` once per process (see
``BayesForge/SetDevice/set.py``) and is ignored on any later change.

The scenario parameters travel with the results: every row of the returned
DataFrame carries the settings that produced it, so joining true against
estimated values needs no bookkeeping of your own.

This module deliberately imports neither ``jax`` nor ``BayesForge`` at module
level -- worker processes must be able to set their XLA environment before the
backend is initialised.
"""

import atexit
import hashlib
import itertools
import numbers
import os
import shutil
import sys
import time
import traceback
import warnings
from contextlib import contextmanager

__all__ = ["grid", "run_simulations", "worker_stdout"]


_GUARD_MSG = (
    "run_simulations uses 'spawn' worker processes, which re-import the "
    "calling script. Module-level code therefore runs again in every worker. "
    "Put the call behind a main guard:\n\n"
    "    if __name__ == '__main__':\n"
    "        res = bf.run_simulations(one_sim, scenarios, workers=8)\n\n"
    "Notebooks and the REPL are unaffected; this applies to .py scripts. Use "
    "backend='serial' to run without worker processes."
)

# Failure text for a worker killed outright (M4): recorded in the row, so the
# DataFrame says why a simulation has no result.
_DEAD_WORKER_MSG = "worker process died (BrokenProcessPool: OOM or killed)"

_DEAD_POOL_MSG = (
    "run_simulations: a worker process died.\n\n" + _GUARD_MSG + "\n\n"
    "If the script already has that guard, the worker was most likely "
    "killed by the OS (out of memory) -- lower `workers`."
)

_DEAD_POOL_HINT = (
    "run_simulations: a worker process died (BrokenProcessPool). Rows already "
    "collected are kept; the simulations that had not finished are recorded "
    "as failed. If the script has no `if __name__ == '__main__':` guard, "
    "that is the cause; otherwise the worker was most likely killed by the OS "
    "(out of memory) -- lower `workers`."
)


def _manager_failure_msg(exc):
    """Friendly text for a progress manager that never came up (M6).

    Without a __main__ guard the manager's spawned server re-imports the
    calling script, re-runs the call and dies on the recursion guard before
    it can report an address -- the parent only ever sees an opaque pipe or
    timeout error. Name the guard instead.
    """
    return (
        "run_simulations: the progress manager process failed to start. The "
        "most common cause is a .py script without an "
        "`if __name__ == '__main__':` guard: the spawned helper processes "
        "re-import the script, which re-runs the call.\n\n"
        + _GUARD_MSG +
        f"\n\nUnderlying error: {exc!r}\n"
        "If the script does have the guard, rerun with progress=False to "
        "skip the manager."
    )


def _interrupt_message(n_done, total):
    """Line printed when Ctrl+C stops a run (M5)."""
    return (f"run_simulations: interrupted -- {n_done}/{total} simulation(s) "
            "finished; the rows collected so far are attached to the "
            "KeyboardInterrupt as .rows")


# --------------------------------------------------------------------------
# Scenario construction
# --------------------------------------------------------------------------

def _grid_values(v):
    """Expand one ``grid`` parameter into its list of values (m5).

    Anything iterable except strings expands: lists, tuples, ``range``,
    ``set`` (sorted, for a deterministic order), numpy arrays, generators.
    Strings and bytes stay scalars -- iterating them would explode a label
    into its characters. A 0-d numpy array is a scalar too.
    """
    if isinstance(v, (str, bytes)):
        return [v]
    if getattr(v, "ndim", None) == 0:          # 0-d numpy array
        return [v.item()]
    if isinstance(v, (set, frozenset)):
        return sorted(v)                       # sets have no order; sort
    if hasattr(v, "__iter__"):
        return list(v)
    return [v]


def grid(reps=1, **params):
    """Build a list of scenario dicts from the cartesian product of ``params``.

    Args:
        reps: Number of replicates of each parameter combination.
        **params: Named parameters to cross. Any iterable except a string
            (list, tuple, range, set, numpy array, generator) is expanded;
            a set is expanded in sorted order. Scalars are treated as
            single-element lists.

    Returns:
        list[dict]: One dict per simulation, each carrying its ``sim`` (global
        index) and ``rep`` (replicate index within its cell) alongside the
        parameter values.

    Examples:
        >>> grid(N=[50, 100], rho=[0.0, 0.3], reps=2)   # doctest: +SKIP
        [{'sim': 0, 'rep': 0, 'N': 50, 'rho': 0.0},
         {'sim': 1, 'rep': 1, 'N': 50, 'rho': 0.0}, ...]   # 8 dicts
    """
    reps = _check_int("reps", reps)

    names = list(params)
    values = [_grid_values(v) for v in params.values()]

    scenarios = []
    for combo in itertools.product(*values) if names else [()]:
        for rep in range(reps):
            cell = dict(zip(names, combo))
            cell["sim"] = len(scenarios)
            cell["rep"] = rep
            scenarios.append(cell)
    return scenarios


def _normalise_scenarios(scenarios):
    """Accept an int, a list of dicts, or the output of :func:`grid`."""
    if isinstance(scenarios, int):
        if scenarios < 1:
            raise ValueError(f"scenarios must be >= 1, got {scenarios}")
        scenarios = [{} for _ in range(scenarios)]
    else:
        scenarios = list(scenarios)
        # Validate before copying (m4): dict(s) on a non-mapping raises its
        # own confusing error before this check is ever reached. Name the
        # type and the offending index -- "cannot convert dictionary update
        # sequence element N to a sequence" names neither.
        bad = [i for i, s in enumerate(scenarios) if not isinstance(s, dict)]
        if bad:
            i = bad[0]
            raise TypeError(
                "scenarios must be an int or an iterable of dicts: element "
                f"{i} is {type(scenarios[i]).__name__}: {scenarios[i]!r}"
                + (f" -- {len(bad)} elements in total are not dicts"
                   if len(bad) > 1 else "")
            )
        scenarios = [dict(s) for s in scenarios]

    for i, s in enumerate(scenarios):
        s.setdefault("sim", i)
        s.setdefault("rep", 0)

    # Seeds derive from (base, sim), so a duplicated sim silently draws the
    # same data twice (M3). Composed grids are the usual culprit: every
    # grid() numbers its cells from 0.
    seen = set()
    dups = []
    for s in scenarios:
        sim = s["sim"]
        if sim in seen and sim not in dups:
            dups.append(sim)
        seen.add(sim)
    if dups:
        raise ValueError(
            "scenarios contain duplicate 'sim' ids: "
            f"{', '.join(str(d) for d in dups)} -- every simulation needs a "
            "unique sim, because its seed is derived from it and duplicates "
            "draw identical data. Build one grid(...) instead of adding "
            "several, or renumber: for i, s in enumerate(scenarios): "
            "s['sim'] = i"
        )
    return scenarios


def _derive_seed(base, sim):
    """Deterministic per-simulation seed.

    Uses sha256 rather than ``hash()`` because Python salts string hashes per
    process -- workers must agree with the parent and with each other.
    """
    digest = hashlib.sha256(f"{int(base)}:{int(sim)}".encode()).hexdigest()
    return int(digest, 16) & 0x7FFFFFFF


def _n_cpus():
    """CPU budget: sched_getaffinity (respects taskset/cgroups), else cpu_count.

    Every core budget in this module goes through here (M1): the worker cap,
    the oversubscription check and the pinning blocks must agree on one
    number, or ``taskset -c 0-7`` lets 24 workers pile onto 8 CPUs while the
    cap, computed on the physical count, sees nothing wrong.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            n = len(os.sched_getaffinity(0))
        except OSError:
            n = 0
        if n >= 1:
            return n
    return os.cpu_count() or 1


# --------------------------------------------------------------------------
# Result normalisation
# --------------------------------------------------------------------------

def _looks_like_bf(obj):
    cls = type(obj)
    return (
        cls.__name__ in ("bf", "BF")
        and getattr(cls, "__module__", "").startswith("BayesForge")
    )


def _normalise_result(ret):
    """Turn whatever the user's function returned into a list of row dicts."""
    if _looks_like_bf(ret):
        raise TypeError(
            "run_simulations: the simulation function returned a `bf` object. "
            "A fitted bf cannot cross a process boundary -- m.diag holds "
            "closures over m, and m.dist is an unpicklable proxy. Return "
            "summaries instead, e.g. m.summary(), m.sampler_stats(), or a dict "
            "of the numbers you need."
        )

    if ret is None:
        return [{}]

    # pandas objects, imported lazily so this module stays import-light.
    mod = type(ret).__module__ or ""
    if mod.startswith("pandas"):
        import pandas as pd
        if isinstance(ret, pd.DataFrame):
            # m12: to_dict("records") would silently merge duplicated column
            # names (last wins) and mangle a MultiIndex -- refuse instead.
            if isinstance(ret.columns, pd.MultiIndex):
                raise ValueError(
                    "run_simulations: fn returned a DataFrame with MultiIndex "
                    "columns; they cannot become row keys. Flatten them "
                    "first, e.g. df.columns = ['_'.join(map(str, c)) "
                    "for c in df.columns]"
                )
            seen_cols, dups = set(), []
            for c in ret.columns:
                key = str(c)
                if key in seen_cols and key not in dups:
                    dups.append(key)
                seen_cols.add(key)
            if dups:
                raise ValueError(
                    "run_simulations: fn returned a DataFrame with duplicated "
                    f"column names: {', '.join(dups)} -- to_dict would keep "
                    "only one column per name and silently drop the rest. "
                    "Rename the duplicates before returning."
                )
            return ret.to_dict("records")
        if isinstance(ret, pd.Series):
            return [{str(k): v for k, v in ret.items()}]

    if isinstance(ret, dict):
        return [ret]

    if isinstance(ret, (list, tuple)):
        if not ret:
            return [{}]
        if all(isinstance(r, dict) for r in ret):
            return [dict(r) for r in ret]
        raise TypeError(
            "run_simulations: a list return value must contain dicts, got "
            f"{type(ret[0]).__name__}"
        )

    return [{"value": ret}]


# --------------------------------------------------------------------------
# Worker side
# --------------------------------------------------------------------------

_STATE = {}


def _worker_env(cores_per_worker, quiet=True):
    """Environment for one worker.

    This governs the JAX *device* count and the BLAS/OpenMP pools that numpy and
    friends consult. It does NOT bound XLA's own CPU thread pool -- that follows
    the affinity mask, so :func:`_pin_worker` is what keeps workers off each
    other's cores.

    ``BF_QUIET`` must be in the environment *inherited at spawn*, not merely set
    by the pool initializer: BayesForge prints its banner while the child
    imports this module to find the initializer, which is strictly earlier.

    ``XLA_PYTHON_CLIENT_PREALLOCATE`` defaults to ``false`` (an explicit user
    value wins) so N workers on one GPU do not each grab ~90 % of the VRAM at
    the first call and OOM the second worker; it is inert on CPU.
    """
    return {
        "BF_QUIET": "1" if quiet else "0",
        "XLA_FLAGS": f"--xla_force_host_platform_device_count={cores_per_worker}",
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get(
            "XLA_PYTHON_CLIENT_PREALLOCATE", "false"),
        "BF_MAX_CORES": str(cores_per_worker),
        "OMP_NUM_THREADS": str(cores_per_worker),
        "MKL_NUM_THREADS": str(cores_per_worker),
        "OPENBLAS_NUM_THREADS": str(cores_per_worker),
        "NUMEXPR_NUM_THREADS": str(cores_per_worker),
        # RELEASE_AUDIT X7: workers start with the parent's x64 setting; a
        # bf() created in the worker still applies its own float_precision.
        "JAX_ENABLE_X64": "1" if _parent_x64() else "0",
    }


def _parent_x64():
    """True when the parent process runs JAX with jax_enable_x64 on."""
    import jax
    return bool(jax.config.jax_enable_x64)


def _visible_gpus():
    """Visible CUDA device ids, [] when no GPU is visible.

    Reads the environment and /proc only -- this module must not import jax
    (see the module docstring), so the GPU count is detected without bringing
    up a backend. An explicit ``CUDA_VISIBLE_DEVICES`` wins; ``"none"`` or
    ``""`` means the user hid every GPU and that is honoured.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cvd = cvd.strip()
        if not cvd or cvd.lower() == "none":
            return []
        return [t.strip() for t in cvd.split(",") if t.strip()]
    try:
        n = len(os.listdir("/proc/driver/nvidia/gpus"))
    except OSError:
        return []
    return [str(i) for i in range(n)]


def _gpu_env_for_worker(i, visible):
    """CUDA_VISIBLE_DEVICES for worker i: one device, round-robin (M7).

    Empty dict when no GPU is visible, so the worker inherits the parent's
    environment untouched.
    """
    if not visible:
        return {}
    return {"CUDA_VISIBLE_DEVICES": str(visible[i % len(visible)])}


_XLA_DEVICE_FLAG = "--xla_force_host_platform_device_count"


def _merge_xla_flags(inherited, override):
    """Merge an inherited XLA_FLAGS with our override (m2).

    The user's inherited flags are kept; only an inherited device-count
    token is replaced by ours (the override always carries one).
    """
    kept = [t for t in inherited.split()
            if not t.startswith(_XLA_DEVICE_FLAG + "=")]
    kept += [t for t in override.split() if t not in kept]
    return " ".join(kept)


@contextmanager
def _env_override(overrides):
    """Temporarily set env vars, restoring the previous values afterwards.

    Applied in the *parent* around the pool's lifetime: ``spawn`` children copy
    ``os.environ`` at creation, and ProcessPoolExecutor creates them lazily as
    tasks are submitted, so the override has to outlive the submissions.

    ``XLA_FLAGS`` is *merged* with the inherited value rather than replacing
    it, so a user-set flag such as ``--xla_cpu_multi_thread_eigen=false``
    survives; only the device-count token is ours.

    Not thread-safe: ``os.environ`` is process-wide, so two concurrent
    :func:`run_simulations` calls in two threads of one process would
    interleave their save/restore. Run them in separate processes.
    """
    overrides = dict(overrides)
    inherited = os.environ.get("XLA_FLAGS")
    if inherited and "XLA_FLAGS" in overrides:
        overrides["XLA_FLAGS"] = _merge_xla_flags(inherited,
                                                  overrides["XLA_FLAGS"])
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _pin_worker(block, block_size):
    """Confine this worker to its own ``block_size`` CPUs, once.

    This is the part that actually limits XLA. Its CPU thread pool is sized from
    the process's affinity mask -- not from OMP_NUM_THREADS, and not from any
    XLA_FLAGS thread setting. Unpinned, every worker starts a pool sized for the
    whole machine and N workers oversubscribe it N-fold (measured: ~104 threads
    per worker on a 24-core host, against 27 when pinned).

    The block index comes from the worker's *first* task rather than a shared
    counter: XLA sizes its thread pool once, when the backend comes up, so only
    the first pin can affect it -- and avoiding the shared counter avoids
    handing every worker a semaphore to unpickle, which races with pool
    teardown.

    Known limitation (m1): ProcessPoolExecutor does *not* guarantee that each
    worker picks up its first task in submission order. A worker that finishes
    its first task before a slow-starting sibling has pulled one can take a
    later task whose ``i % workers`` class matches a block another worker has
    already claimed -- two workers then share a block and one goes unused.
    This is a performance-only effect (cores are still bounded, nothing is
    incorrect), and the submission order -- one task per class before any
    second task -- makes it rare; a deterministic per-worker block
    would need a shared counter, which the teardown race above rules out.

    ``block_size`` is the machine divided by the worker count, NOT
    ``cores_per_worker`` -- the latter is a JAX *device* count. Pinning to
    ``cores_per_worker`` would strand every core beyond
    ``workers * cores_per_worker`` (measured: with 12 workers at 1 core each on
    a 24-core host, half the machine sat idle and the run was slower than no
    pinning at all).

    Returns the assigned CPU list, or None where affinity is unavailable
    (non-Linux) or there are too few CPUs to divide.
    """
    if "cpus" in _STATE:                      # already pinned; never re-pin
        return _STATE["cpus"]

    _STATE["cpus"] = None
    if not hasattr(os, "sched_setaffinity"):
        return None

    # sched_getaffinity, not cpu_count: respects taskset and cgroup limits.
    cpus = sorted(os.sched_getaffinity(0))
    if block_size < 1 or len(cpus) < block_size:
        return None

    start = (block * block_size) % len(cpus)
    mine = {cpus[(start + k) % len(cpus)] for k in range(block_size)}
    try:
        os.sched_setaffinity(0, mine)
    except OSError:
        return None

    _STATE["cpus"] = sorted(mine)
    return _STATE["cpus"]


def _worker_init(env, fn_bytes, fn_kwargs_bytes, quiet, block_size, queue,
                 gpus=()):
    """Pool initializer: set the environment, then unpack the payloads."""
    env = dict(env)
    inherited = os.environ.get("XLA_FLAGS")
    if inherited and "XLA_FLAGS" in env:
        # m2: the child inherited the parent's merged XLA_FLAGS; merge again
        # (idempotent) so user flags survive this update() too.
        env["XLA_FLAGS"] = _merge_xla_flags(inherited, env["XLA_FLAGS"])
    os.environ.update(env)
    _STATE["block_size"] = block_size
    _STATE["queue"] = queue
    _STATE["gpus"] = list(gpus)

    if quiet:
        # bf.__init__ prints its device count and BayesForge prints a banner on
        # import; times a hundred simulations that is pure noise. The handle is
        # kept so it is closed when the worker exits (m3), and the real stdout
        # stays reachable through worker_stdout() for fn that wants to log.
        devnull = open(os.devnull, "w")
        atexit.register(devnull.close)
        _STATE["real_stdout"] = sys.stdout
        _STATE["devnull"] = devnull
        sys.stdout = devnull

    # Deliberately NOT deserialised here: fn_kwargs may hold jax arrays, and
    # rebuilding those brings the XLA backend up -- which must happen after the
    # worker is pinned, or the thread pool is sized for the whole machine.
    _STATE["fn_bytes"] = fn_bytes
    _STATE["fn_kwargs_bytes"] = fn_kwargs_bytes
    _STATE["worker"] = os.getpid()


@contextmanager
def worker_stdout():
    """Temporarily restore the real stdout inside ``fn`` (m3).

    With ``quiet=True`` (the default) a worker's stdout is sent to
    ``os.devnull``. Wrap the few lines that must be seen in this context
    manager to write them to the terminal instead::

        def one_sim(N, seed, **_):
            ...
            with worker_stdout():
                print(f"sim {seed}: divergent chains")

    Outside a quiet worker this is a no-op: stdout is already the real one.
    """
    real = _STATE.get("real_stdout")
    if real is None:
        yield sys.stdout
        return
    sys.stdout = real
    try:
        yield real
    finally:
        sys.stdout = _STATE.get("devnull", sys.stdout)


def _worker_entry(payload_bytes, block):
    """Run one scenario. Returns cloudpickled ``(rows, error)``."""
    import cloudpickle

    _pin_worker(block, _STATE.get("block_size", 1))

    gpus = _STATE.get("gpus")
    if gpus and "gpu" not in _STATE:
        # One device per worker, round-robin (M7) -- set before the fn unpack
        # below brings the XLA/CUDA backend up, the same window _pin_worker
        # uses to size the thread pool. Set once: later tasks of this worker
        # must not migrate between GPUs.
        _STATE["gpu"] = _gpu_env_for_worker(block, gpus)
        os.environ.update(_STATE["gpu"])

    if "fn" not in _STATE:                    # first task: unpack, now pinned
        fn = cloudpickle.loads(_STATE["fn_bytes"])
        blob = _STATE["fn_kwargs_bytes"]
        # Set both only once both loaded, so a failed unpack is retried (and
        # reported) by the next task instead of leaving half a state behind.
        _STATE["fn_kwargs"] = cloudpickle.loads(blob) if blob else {}
        _STATE["fn"] = fn

    scenario = cloudpickle.loads(payload_bytes)
    pid = _STATE.get("worker", os.getpid())
    sim = scenario.get("sim")

    # Tell the parent this simulation is now running, so its display can show
    # in-flight work rather than only completions. Never let the progress
    # channel break a simulation.
    _notify(("start", pid, sim, time.time()))
    started = time.time()
    rows, error = _run_one(_STATE["fn"], scenario, _STATE["fn_kwargs"], pid)
    _notify(("done", pid, sim, time.time() - started, error is not None))

    # X2: the return value itself must survive the trip back. An unpicklable
    # result (sampler, figure, lock) is recorded like any other error instead
    # of raising here and taking the whole run -- and every row already
    # computed -- down with it.
    try:
        return cloudpickle.dumps((rows, error))
    except Exception as exc:
        error = (f"result of fn is not picklable ({type(exc).__name__}: {exc}); "
                 "return plain data (numbers, arrays, DataFrames), "
                 "not samplers or figures")
        return cloudpickle.dumps(([dict(scenario, error=error)], error))


def _notify(msg):
    """Best-effort progress message to the parent."""
    q = _STATE.get("queue")
    if q is None:
        return
    try:
        q.put(msg)
    except Exception:
        _STATE["queue"] = None            # parent gone; stop trying


def _run_one(fn, scenario, fn_kwargs, worker):
    """Execute one simulation and shape its output into rows.

    Runs identically in-process (``backend='serial'``) and in a worker, so the
    serial path is a genuine debugging equivalent rather than a second code
    path that can drift.
    """
    started = time.time()
    call = dict(fn_kwargs)
    call.update(scenario)

    try:
        ret = fn(**call)
        rows = _normalise_result(ret)
        # m6: fn must not overwrite the runner's own columns. A returned
        # "error" key would fake an error row in the DataFrame while the
        # progress counter (which follows real exceptions) stayed clean; a
        # returned "sim" breaks the sort and the seed<->row association.
        clash = sorted({k for r in rows for k in r
                        if k in ("sim", "seed", "error", "worker")})
        if clash:
            raise ValueError(
                "run_simulations: fn returned rows with reserved column "
                f"names: {', '.join(clash)} -- sim, seed, error and worker "
                "are set by the runner. Rename them in the returned data."
            )
        error = None
    except Exception:
        rows = [{}]
        error = traceback.format_exc()

    elapsed = time.time() - started
    meta = dict(scenario)
    meta.update(worker=worker, elapsed_s=round(elapsed, 3), error=error)

    # Scenario/meta first so a result key of the same name wins -- the
    # simulation's own output is the more specific value.
    out = []
    for row in rows:
        merged = dict(meta)
        merged.update(row)
        out.append(merged)
    return out, error


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _check_int(name, value, minimum=1, allow_none=False):
    """Return ``int(value)``, or raise for a value that is not an integer.

    TypeError for the wrong type (``workers='4'``, ``workers=2.5``, a bool --
    int() would silently truncate the float and ``True`` would mean one
    worker), ValueError for a well-formed integer below ``minimum``. Same
    split as ``BayesForge.Main.main`` uses for its own argument checks.
    Integral floats are accepted: the R and Julia bridges hand over JSON
    numbers, which arrive as 4.0.
    """
    if value is None:
        if allow_none:
            return None
        raise TypeError(f"{name} must be an integer, got None")
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        # numbers.Real covers float and numpy integer/float scalars without
        # importing numpy here.
        if (isinstance(value, numbers.Real) and not isinstance(value, bool)
                and float(value).is_integer()):
            value = int(value)
        else:
            raise TypeError(f"{name} must be an integer, got {value!r}")
    else:
        value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _plan(scenarios, workers=None, num_chains=None, cores_per_worker=None,
          seed=0, quiet=True):
    """Everything run_simulations decides before running anything.

    Normalises the scenarios, derives each one's seed, sets the core budget and
    the worker environment. Shared with the R package (BayesForge for R calls it
    through reticulate), so both languages split cores and seed simulations
    identically. Warnings are returned, not raised, so each caller reports them
    its own way.

    Returns:
        dict with keys ``scenarios`` (list of dicts, each with sim, rep, seed and,
        when num_chains is given, num_chains), ``workers`` (int),
        ``cores_per_worker`` (int), ``block_size`` (int, CPUs each worker is
        pinned to), ``env`` (dict from _worker_env) and ``warnings`` (list of str).
    """
    notes = []
    scenarios = _normalise_scenarios(scenarios)
    n = len(scenarios)

    workers = _check_int("workers", workers, allow_none=True)
    cores_per_worker = _check_int("cores_per_worker", cores_per_worker,
                                  allow_none=True)
    num_chains = _check_int("num_chains", num_chains, allow_none=True)

    for s in scenarios:
        s.setdefault("seed", _derive_seed(seed, s["sim"]))

    # num_chains is the single knob for multi-chain fits: it reserves one core
    # per chain in every worker and is handed to fn so the fit uses the same
    # number. Asking for 4 chains on a 24-core host therefore gives each worker
    # 4 cores and runs 6 simulations at a time.
    if num_chains is not None:
        if cores_per_worker is None:
            cores_per_worker = num_chains
        elif cores_per_worker < num_chains:
            notes.append(
                f"cores_per_worker={cores_per_worker} is below num_chains="
                f"{num_chains}: chain_method='parallel' needs one device per "
                "chain, so numpyro will draw the chains sequentially. Drop "
                "cores_per_worker to let num_chains set it, or use "
                "chain_method='vectorized' in the fit."
            )
        for s in scenarios:
            s.setdefault("num_chains", num_chains)

    cores_per_worker = 1 if cores_per_worker is None else cores_per_worker
    # sched_getaffinity, not cpu_count (M1): the cap and the oversubscription
    # check must see the same budget _pin_worker pins to, or a taskset/cgroup
    # limit lets N workers pile onto M < N CPUs with no warning at all.
    n_cpu = _n_cpus()
    if workers is None:
        workers = min(n, max(1, n_cpu // cores_per_worker))
    workers = max(1, min(workers, n))

    if workers * cores_per_worker > n_cpu:
        capped = max(1, n_cpu // cores_per_worker)
        notes.append(
            f"workers*cores_per_worker = {workers}*{cores_per_worker} = "
            f"{workers * cores_per_worker} exceeds the {n_cpu} available cores; "
            f"capping workers to {capped}. Oversubscribing makes every "
            f"simulation slower, not faster.",
        )
        workers = capped

    # n_cpu is already the affinity-aware budget (M1); the pinning blocks are
    # cut from the same set _pin_worker uses (sched_getaffinity there too).
    block_size = max(cores_per_worker, n_cpu // workers)

    env = _worker_env(cores_per_worker, quiet)

    return {
        "scenarios": scenarios,
        "workers": workers,
        "cores_per_worker": cores_per_worker,
        "block_size": block_size,
        "env": env,
        "warnings": notes,
    }


def run_simulations(
    fn,
    scenarios,
    workers=None,
    num_chains=None,
    cores_per_worker=None,
    fn_kwargs=None,
    seed=0,
    on_error="record",
    quiet=True,
    progress=True,
    backend="process",
):
    """Run independent simulations in parallel and collect them into a DataFrame.

    Each scenario dict is passed to ``fn`` as keyword arguments, together with
    an injected ``sim`` (index), ``rep`` (replicate) and ``seed``. Thread that
    ``seed`` into ``bf(rand_seed=seed)`` -- ``bf.__init__`` seeds the global
    numpy/random RNGs and the shared distribution singletons from it, so
    workers that share a seed draw identical data.

    Args:
        fn: The simulation function. Called as ``fn(**scenario, **fn_kwargs)``;
            give it a ``**_`` catch-all so added keys do not break it. It may
            return a dict, a list of dicts, a ``pandas`` DataFrame or Series, or
            None. It must NOT return a ``bf`` instance.
        scenarios: An int (that many bare simulations), a list of dicts, or the
            output of :func:`grid`.
        workers: Number of worker processes, i.e. how many simulations run at
            once. Defaults to ``min(len(scenarios), n_cpu // cores_per_worker)``
            -- so it follows ``num_chains`` automatically. See
            "Chains, workers and cores".
        num_chains: Chains each fit will draw, for fits using
            ``chain_method='parallel'`` (setup B). Reserves one core and one JAX
            device per chain in every worker, and is passed to ``fn`` as a
            keyword so the core budget and the fit cannot disagree. On 24 cores,
            ``num_chains=4`` means 4 cores per worker and 6 simulations at a
            time. Omit it for single-chain fits (A) and for
            ``chain_method='vectorized'`` (C), which needs no reserved cores.
        cores_per_worker: JAX *devices* per worker, to set the budget directly
            instead of via ``num_chains``. Defaults to ``num_chains``, else 1.
            It does not decide how many CPUs a worker gets -- the machine's
            cores are always divided evenly across ``workers`` so none sit idle
            -- only how many devices JAX exposes inside one worker. Setting it
            below ``num_chains`` warns, because parallel chains would then fall
            back to sequential.
        fn_kwargs: Constants passed to every call. Serialised once per worker
            rather than once per scenario, so this is where large shared arrays
            belong.
        seed: Base seed. Per-simulation seeds are derived from it
            deterministically, so a rerun reproduces regardless of how many
            workers ran or in what order they finished.
        on_error: ``'record'`` (default) keeps going and puts the traceback in
            the row's ``error`` column; ``'raise'`` aborts the run -- the rows
            collected so far are attached to the raised exception as
            ``.rows``, so nothing already computed is lost.
        quiet: Suppress worker stdout; when ``workers=1`` runs in-process it
            also silences ``fn``'s stdout (``backend='serial'`` keeps stdout
            open for debugging).
        progress: Live progress. ``True`` (default) draws a panel showing every
            worker -- which simulation it is on, for how long, how many it has
            finished and its average -- above an overall bar with elapsed time
            and ETA, redrawn in place on a terminal::

                run_simulations  14/100 [###-------]  14%  elapsed 2m14s  eta 8m03s
                  worker 12345   sim 27    running    14.2s  done 3    avg  41.1s
                  worker 12346   sim 31    running     2.1s  done 4    avg  38.7s
                  worker 12347   idle                        done 3    avg  40.0s

            Off a terminal (redirected to a log) it degrades to about ten
            one-line updates plus a per-worker summary at the end. ``'plain'``
            forces that simpler form even on a terminal; ``False`` is silent.
        backend: ``'process'`` (default) or ``'serial'``. The serial path runs
            in-process, so ``pdb`` and full tracebacks work while debugging.

    Returns:
        pandas.DataFrame: One row per dict returned by ``fn``, carrying the
        scenario columns plus ``sim``, ``rep``, ``seed``, ``worker``,
        ``elapsed_s`` and ``error``. Sorted by ``sim``.

    Precision:
        Workers inherit the parent's ``jax_enable_x64`` through the
        ``JAX_ENABLE_X64`` environment variable, but a ``bf()`` created inside
        ``fn`` applies its own ``float_precision`` (default 64 for numpyro,
        32 for tfp). Pass ``float_precision`` explicitly in ``fn`` to control
        the precision of the simulations (RELEASE_AUDIT X7).

    Chains, workers and cores:
        Chains and simulations compete for the same cores, so ``num_chains``,
        ``workers`` and the fit's ``chain_method`` have to be chosen together.
        There are three sensible setups. All three produce the same number of
        genuine chains and equivalent ESS -- they differ only in speed and in
        how many simulations are in flight at once.

        **A. Single-chain fits** -- the default, and the fastest way through a
        large batch when you do not need per-fit R-hat::

            def one_sim(N, seed, **_):
                m = bf(rand_seed=seed, print_devices_found=False)
                m.fit(model, obs=dgp(N, seed, m), num_chains=1,
                      progress_bar=False)
                ...

            run_simulations(one_sim, scenarios)          # workers = every core

        One core per worker, 24 simulations at a time on a 24-core host.

        **B. Chains reserved per worker** -- ``chain_method='parallel'``, which
        places one chain on one JAX device and so needs a device per chain.
        Pass ``num_chains`` here and take it back in ``fn``, so the core budget
        and the fit can never disagree::

            def one_sim(N, seed, num_chains, **_):
                m = bf(rand_seed=seed, print_devices_found=False)
                m.fit(model, obs=dgp(N, seed, m),
                      num_chains=num_chains, chain_method="parallel",
                      progress_bar=False)
                ...

            run_simulations(one_sim, scenarios, num_chains=4)

        ``num_chains=4`` gives each worker 4 cores and 4 JAX devices, sets
        ``workers = n_cpu // 4``, and pins each worker to its own disjoint block
        of 4 CPUs. On 24 cores: 6 workers, so **6 simulations run at a time**
        instead of 24. Simulation concurrency is traded for chain concurrency.

        **C. Chains vectorized** -- ``chain_method='vectorized'`` vmaps the
        chains inside a *single* device, so no cores are reserved for them and
        every core stays free for a different simulation::

            def one_sim(N, seed, **_):
                m = bf(rand_seed=seed, print_devices_found=False)
                m.fit(model, obs=dgp(N, seed, m), num_chains=4,
                      chain_method="vectorized", progress_bar=False)
                ...

            run_simulations(one_sim, scenarios)   # no num_chains needed here

        Do NOT pass ``num_chains`` for this one: it would reserve cores that
        vectorized chains never use, cutting simulation concurrency for nothing.

        Measured, 24 simulations over 24 cores, each drawing 4 chains at equal
        ESS (~1900)::

            C  workers=24  cpw=1  vectorized   17.5s   <- fastest
            B  workers=6   cpw=4  parallel     21.9s   <- 6 sims at a time
               workers=6   cpw=4  vectorized   29.1s   <- reserves cores, wastes them

        Summary:

        ==================  ===========  ==========  ============  ============
        fit's chain_method  num_chains=  workers     cores/worker  sims at once
        ==================  ===========  ==========  ============  ============
        (1 chain)           omit         n_cpu       1             n_cpu
        'parallel'          N            n_cpu // N  N             n_cpu // N
        'vectorized'        omit         n_cpu       1             n_cpu
        ==================  ===========  ==========  ============  ============

        The failure mode to know: ``chain_method='parallel'`` with only one
        device finds one device, warns once per fit, and draws the chains
        **sequentially** -- correct results, needlessly slow. That is what
        happens if you ask a fit for 4 parallel chains without telling the
        runner. Passing ``num_chains`` prevents it; passing a
        ``cores_per_worker`` below ``num_chains`` warns.

    Notes:
        In a **.py script**, put the call behind ``if __name__ == '__main__':``.
        Workers are spawned, so they re-import the calling script and would
        otherwise re-run its module level. Notebooks and the REPL need no guard.

        Ctrl+C cancels the queued simulations and re-raises the
        ``KeyboardInterrupt`` with the rows collected so far attached to it as
        ``.rows``; a worker killed outright (out of memory) breaks the pool,
        so its unfinished simulations are recorded with an ``error`` while
        the completed rows are kept.

        Each worker JIT-compiles the model on its first fit. If the model
        function is built fresh inside ``fn`` (a closure), *every* simulation
        pays full compilation because the JIT cache key changes each time.
        Defining the model at module level lets a worker reuse its compiled
        executable across all the simulations it handles -- usually the single
        largest speedup available here.

    Examples:
        >>> def one_sim(N, rho, seed, **_):            # doctest: +SKIP
        ...     m = bf(cores=1, rand_seed=seed, print_devices_found=False)
        ...     m.fit(model, obs=dgp(N, rho, m), num_chains=1, progress_bar=False)
        ...     s = m.summary()
        ...     return {"rho_hat": s.loc["rho", "mean"], "rho_true": rho}
        >>> res = run_simulations(                      # doctest: +SKIP
        ...     one_sim, grid(N=[50, 100, 500], rho=[0.0, 0.3], reps=10), workers=12)
        >>> res.groupby(["N", "rho"]).rho_hat.mean()    # doctest: +SKIP
    """
    import pandas as pd

    if on_error not in ("record", "raise"):
        raise ValueError(f"on_error must be 'record' or 'raise', got {on_error!r}")
    if backend not in ("process", "serial"):
        raise ValueError(f"backend must be 'process' or 'serial', got {backend!r}")
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}")

    plan = _plan(scenarios, workers=workers, num_chains=num_chains,
                 cores_per_worker=cores_per_worker, seed=seed, quiet=quiet)
    for msg in plan["warnings"]:
        warnings.warn(msg, stacklevel=2)
    scenarios = plan["scenarios"]
    workers = plan["workers"]
    cores_per_worker = plan["cores_per_worker"]
    fn_kwargs = dict(fn_kwargs or {})

    if backend == "serial" or workers == 1:
        if backend == "process":
            # workers == 1: the requested process backend is silently
            # replaced by an in-process run (M2) -- say so once. With
            # backend='serial' the user chose in-process on purpose.
            warnings.warn(
                "simulate(workers=1) runs in-process: no environment "
                "isolation; pass workers>=2 for isolated worker processes.",
                UserWarning, stacklevel=2,
            )
        # quiet is honoured in-process too on the fallback path (M2);
        # backend='serial' stays un-silenced so pdb prompts and prints reach
        # the terminal while debugging.
        rows = _serial(fn, scenarios, fn_kwargs, on_error, progress,
                       quiet=(quiet and backend == "process"))
    else:
        rows = _parallel(
            fn, scenarios, fn_kwargs, workers, cores_per_worker,
            on_error, quiet, progress, plan["env"], plan["block_size"],
        )

    # m11: deterministic column order -- the metadata columns (sim, seed,
    # error, worker) come first in a fixed order, then the function's own
    # keys in first-seen order; rows are sorted by sim id.
    seen = []
    for row in rows:
        for k in row:
            if k not in seen:
                seen.append(k)
    _meta = ("sim", "seed", "error", "worker")
    columns = [c for c in _meta if c in seen]
    columns += [k for k in seen if k not in _meta]
    df = pd.DataFrame(({col: row.get(col) for col in columns} for row in rows),
                      columns=columns)
    if "sim" in df.columns:
        df = df.sort_values("sim", kind="stable").reset_index(drop=True)

    if progress:
        # The live display already reported the counts; only the pointer to the
        # failed rows is worth adding.
        n_failed = int(df["error"].notna().sum()) if "error" in df.columns else 0
        if n_failed:
            print(f"run_simulations: {n_failed} row(s) carry an error -- "
                  "see df[df.error.notna()]")
    return df


def _fmt_secs(s):
    if s is None:
        return "--"
    if s < 10:
        return f"{s:.1f}s"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class _WorkerDisplay:
    """Live per-worker progress, redrawn in place on a terminal.

    Shows what every worker is doing right now -- which simulation it picked
    up, how long it has been on it, how many it has finished -- rather than a
    single aggregate counter. On a long study that is the difference between
    "something is running" and "worker 5 has been stuck on sim 63 for 20
    minutes".

    Falls back to periodic one-line updates when stdout is not a terminal
    (redirected to a log), and prints a per-worker summary at the end either
    way.
    """

    BAR = 24

    def __init__(self, total, enabled=True, rich=None):
        self.total = total
        self.enabled = enabled
        self.done = 0
        self.failed = 0
        self.started = time.time()
        self.running = {}                 # pid -> (sim, start_time)
        self.stats = {}                   # pid -> [count, total_seconds]
        self._lines = 0                   # lines painted, for cursor rewind
        self._last_paint = 0.0
        self._closed = False

        if rich is None:
            try:
                rich = enabled and sys.stdout.isatty()
            except Exception:
                rich = False
        self.rich = rich

    # -- event intake ----------------------------------------------------
    def handle(self, msg):
        kind = msg[0]
        if kind == "start":
            _, pid, sim, t = msg
            self.running[pid] = (sim, t)
            self.stats.setdefault(pid, [0, 0.0])
        elif kind == "done":
            _, pid, sim, elapsed, failed = msg
            self.running.pop(pid, None)
            st = self.stats.setdefault(pid, [0, 0.0])
            st[0] += 1
            st[1] += elapsed
            if failed:
                self.failed += 1

    def completed(self, n):
        self.done = n

    # -- rendering -------------------------------------------------------
    def _bar(self, frac):
        filled = int(round(self.BAR * frac))
        return "#" * filled + "-" * (self.BAR - filled)

    def _header(self):
        frac = self.done / self.total if self.total else 1.0
        elapsed = time.time() - self.started
        eta = (elapsed / self.done) * (self.total - self.done) if self.done else None
        msg = (f"run_simulations  {self.done}/{self.total} [{self._bar(frac)}] "
               f"{frac * 100:3.0f}%  elapsed {_fmt_secs(elapsed)}  "
               f"eta {_fmt_secs(eta)}")
        if self.failed:
            msg += f"  failed {self.failed}"
        return msg

    def _worker_lines(self):
        now = time.time()
        out = []
        for pid in sorted(self.stats):
            n, tot = self.stats[pid]
            avg = tot / n if n else None
            if pid in self.running:
                sim, t0 = self.running[pid]
                state = f"sim {sim:<5} running {_fmt_secs(now - t0):>7}"
            else:
                state = f"{'idle':<21}"
            out.append(f"  worker {pid:<7} {state}  done {n:<4} avg {_fmt_secs(avg):>6}")
        return out

    def paint(self, force=False):
        if not self.enabled:
            return
        if not self.rich:
            self._paint_plain()
            return
        now = time.time()
        if not force and now - self._last_paint < 0.25:
            return                        # cap redraws; the queue can be chatty
        self._last_paint = now

        try:
            width = shutil.get_terminal_size().columns - 1
        except Exception:
            width = 100
        lines = [self._header()] + self._worker_lines()
        lines = [ln[:width] for ln in lines]

        buf = []
        if self._lines:
            buf.append(f"\033[{self._lines}A")     # rewind over the last frame
        for ln in lines:
            buf.append("\033[2K" + ln + "\n")      # clear then write
        sys.stdout.write("".join(buf))
        sys.stdout.flush()
        self._lines = len(lines)

    def _paint_plain(self):
        step = max(1, self.total // 10)
        if self.done and (self.done == self.total or self.done % step == 0):
            if self.done != getattr(self, "_last_plain", None):
                self._last_plain = self.done
                print(self._header(), flush=True)

    def close(self):
        if not self.enabled or getattr(self, "_closed", False):
            return
        self._closed = True
        self.running.clear()
        self.paint(force=True)          # emits the final header either way
        if not self.rich:
            for ln in self._worker_lines():
                print(ln, flush=True)


def _report(progress, done, total):
    """Progress line: rewritten in place on a terminal, thinned in a log file.

    ``\\r`` only overwrites on a TTY; redirected to a file it concatenates every
    update onto one enormous line, so there we emit occasional real lines
    instead.
    """
    if not progress:
        return

    try:
        tty = sys.stdout.isatty()
    except Exception:                        # stdout replaced (e.g. quiet worker)
        tty = False

    if tty:
        print(f"\rrun_simulations: {done}/{total} complete", end="", flush=True)
        if done == total:
            print()
        return

    step = max(1, total // 10)
    if done == total or done % step == 0:
        print(f"run_simulations: {done}/{total} complete", flush=True)


def _serial(fn, scenarios, fn_kwargs, on_error, progress, quiet=False):
    import cloudpickle

    rows = []
    worker = os.getpid()
    devnull = open(os.devnull, "w") if quiet else None
    saved_stdout = sys.stdout
    try:
        for i, scenario in enumerate(scenarios, 1):
            if devnull is not None:
                sys.stdout = devnull      # honour quiet in-process (M2)
            try:
                out, error = _run_one(fn, scenario, fn_kwargs, worker)
            finally:
                sys.stdout = saved_stdout
            if error is None:
                # X2: the worker path pickles its (rows, error) before sending,
                # so an unpicklable return is recorded as an error there. Do
                # the same here: serial must not silently accept what the
                # process backend rejects.
                try:
                    cloudpickle.dumps(out)
                except Exception as exc:
                    error = (f"result of fn is not picklable "
                             f"({type(exc).__name__}: {exc}); return plain "
                             "data (numbers, arrays, DataFrames), not "
                             "samplers or figures")
                    out = [dict(scenario, error=error)]
            if error is not None and on_error == "raise":
                err = RuntimeError(
                    f"run_simulations: simulation {scenario.get('sim')} failed:\n{error}"
                )
                err.rows = rows       # keep what finished (never lose results)
                raise err
            rows.extend(out)
            _report(progress, i, len(scenarios))
    finally:
        if devnull is not None:
            devnull.close()
    return rows


def _parallel(fn, scenarios, fn_kwargs, workers, cores_per_worker,
              on_error, quiet, progress, env, block_size):
    import multiprocessing
    import queue as _queue
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from concurrent.futures.process import BrokenProcessPool

    import cloudpickle

    if multiprocessing.parent_process() is not None:
        # We are inside a spawned child that re-executed its parent script's
        # module level -- i.e. the script has no __main__ guard. Stop here
        # rather than recursing into a pool of pools.
        raise RuntimeError(_GUARD_MSG)

    # cloudpickle, not pickle: model and DGP functions are routinely closures
    # over a bf instance, and spawn's default serialiser cannot carry those.
    fn_bytes = cloudpickle.dumps(fn)
    fn_kwargs_bytes = cloudpickle.dumps(fn_kwargs) if fn_kwargs else b""
    payloads = [cloudpickle.dumps(s) for s in scenarios]

    # env and block_size come from _plan, so the R and Julia paths and this one
    # cannot drift apart: a second _worker_env call here would re-read the
    # environment and a second core split could disagree with the one the
    # caller was warned about.
    # spawn, not fork: setup_device must configure XLA before the backend comes
    # up, and forking a process that has already initialised JAX is unsafe.
    ctx = multiprocessing.get_context("spawn")
    # spawn reads __main__.__spec__ to rebuild the parent's main module. Some
    # kernels (e.g. the Jupyter kernel quarto starts) leave it unset rather
    # than None, and spawn then fails with AttributeError. None means "no
    # importable main module", which is what such a kernel is.
    import __main__
    if not hasattr(__main__, "__spec__"):
        __main__.__spec__ = None

    # GPU budget (M7): with several devices visible, each worker gets its own
    # CUDA_VISIBLE_DEVICES (round-robin over the visible ids), applied in the
    # worker before its backend comes up. One GPU cannot be split across
    # workers -- warn instead of letting the per-process preallocations
    # collide and kill the pool.
    gpus = _visible_gpus()
    if len(gpus) == 1 and workers > 1:
        warnings.warn(
            f"run_simulations: {workers} workers share the single visible "
            "GPU -- JAX preallocates most of its memory per process, so a "
            "second worker may fail to initialise. Use workers=1 on this "
            "machine, or expose more devices via CUDA_VISIBLE_DEVICES.",
            stacklevel=3,
        )
        gpus = []
    elif len(gpus) < 2:
        gpus = []

    display = _WorkerDisplay(len(scenarios), enabled=bool(progress),
                             rich=(None if progress is True else False))

    state = {"manager": None, "q": None}

    def drain():
        q = state["q"]
        if q is None:
            return
        while True:
            try:
                display.handle(q.get_nowait())
            except _queue.Empty:
                break
            except Exception:
                break

    rows = []
    done_ids = set()          # id() of scenarios already turned into rows
    warned_dead = False
    try:
        with _env_override(env):
            # Created inside the env override so the manager process inherits
            # BF_QUIET: it is spawned too, and re-imports the calling script.
            # A manager-backed queue carries per-simulation start/finish events
            # for the live display; manager proxies pickle cleanly into spawned
            # children, unlike a raw shared Value.
            if progress:
                try:
                    manager = ctx.Manager()
                    try:
                        state["q"] = manager.Queue()
                    except Exception:
                        manager.shutdown()
                        raise
                    state["manager"] = manager
                except Exception as exc:
                    # M6: without a __main__ guard the manager's spawned
                    # server re-runs the call and dies during startup. The
                    # queue only feeds the live display, so say so and run
                    # without it rather than failing simulations that would
                    # otherwise have completed.
                    warnings.warn(_manager_failure_msg(exc), stacklevel=3)
                    progress = False
                    display.enabled = False
            q = state["q"]

            # No `with` on the pool: its __exit__ calls shutdown(wait=True),
            # which joins in-flight fits and turns Ctrl+C into an apparent
            # hang (M5). Shutdown is driven explicitly below instead.
            pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(env, fn_bytes, fn_kwargs_bytes, quiet, block_size,
                          q, gpus),
            )
            try:
                futures = {pool.submit(_worker_entry, p, i % workers): s
                           for i, (p, s) in enumerate(zip(payloads, scenarios))}
                pending = set(futures)
                completed = 0
                display.paint(force=True)

                while pending:
                    # Short timeout rather than as_completed: the display has to
                    # keep ticking while long fits are still in flight.
                    finished, pending = wait(pending, timeout=0.25,
                                             return_when=FIRST_COMPLETED)
                    drain()
                    for future in finished:
                        scenario = futures[future]
                        dead = False
                        try:
                            out, error = cloudpickle.loads(future.result())
                        except BrokenProcessPool:
                            # M4: one killed worker breaks the whole pool --
                            # every unfinished future fails this way. Record
                            # the death like any other error instead of
                            # losing the rows already collected.
                            dead = True
                            error = _DEAD_WORKER_MSG
                            out = [dict(scenario, error=error)]
                        except Exception:
                            # Failure outside fn's body (unpicklable result,
                            # fn_kwargs that cannot be rebuilt): record it
                            # like any other error instead of losing every
                            # row already collected.
                            error = traceback.format_exc()
                            out = [dict(scenario, error=error)]
                        if error is not None and on_error == "raise":
                            # cancel_futures drops the queued work so the pool's
                            # exit only waits on what is already running.
                            pool.shutdown(wait=False, cancel_futures=True)
                            display.close()
                            if dead:
                                err = RuntimeError(_DEAD_POOL_MSG)
                            else:
                                err = RuntimeError(
                                    f"run_simulations: simulation "
                                    f"{scenario.get('sim')} failed:\n{error}"
                                )
                            err.rows = rows   # partial results survive (M4)
                            raise err
                        if dead and not warned_dead:
                            warned_dead = True
                            warnings.warn(_DEAD_POOL_HINT, stacklevel=3)
                        rows.extend(out)
                        done_ids.add(id(scenario))
                        completed += 1
                    display.completed(completed)
                    display.paint()
                drain()
                display.completed(completed)
                display.close()
            except KeyboardInterrupt as exc:
                # M5: drop the queued work without joining the fits already
                # running -- the workers exit once their current task ends.
                pool.shutdown(wait=False, cancel_futures=True)
                drain()
                display.close()
                print(_interrupt_message(len(done_ids), len(scenarios)),
                      flush=True)
                exc.rows = rows       # keep what finished (M5)
                raise
            except BaseException:
                # Any other exit (on_error='raise', a pool broken at submit
                # time, an unexpected error): never block on the workers, and
                # close the display so a half-painted frame is not left on the
                # terminal above the traceback.
                pool.shutdown(wait=False, cancel_futures=True)
                display.close()
                raise
            else:
                pool.shutdown(wait=True)
    except BrokenProcessPool as exc:
        # M4: the pool died before or during submission (a worker killed at
        # startup). Keep every row already collected, record the rest with
        # the same shape on_error='record' failures take.
        for s in scenarios:
            if id(s) not in done_ids:
                rows.append(dict(s, error=_DEAD_WORKER_MSG))
        display.close()
        if on_error == "raise":
            err = RuntimeError(_DEAD_POOL_MSG)
            err.rows = rows
            raise err from exc
        warnings.warn(_DEAD_POOL_HINT, stacklevel=3)
        return rows
    finally:
        if state["manager"] is not None:
            state["manager"].shutdown()
    return rows
