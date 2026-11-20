"""Regression tests for RELEASE_AUDIT.md, module SetDevice, majors M1-M3."""

import os
import subprocess
import sys
import textwrap
import pytest


def _run(code, env=None, check=True):
    """Run code in a subprocess with JAX_PLATFORMS=cpu (never in-process).

    check=False for the error cases: the subprocess prints the exception
    type and message itself instead of exiting non-zero.
    """
    full_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    full_env.pop("BF_MAX_CORES", None)  # never inherit a parent cap into a test
    if env:
        full_env.update(env)
    r = subprocess.run([sys.executable, "-W", "always", "-c", textwrap.dedent(code)],
                       capture_output=True, text=True, timeout=300, env=full_env)
    if check:
        assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout + r.stderr


# ---------------------------------------------------------------- M1

def test_m1a_uppercase_platform_cpu_works():
    out = _run('''
        from BayesForge.SetDevice.set import setup_device
        setup_device(platform='CPU', print_devices_found=False)
        import jax
        print("BACKEND", jax.default_backend())
    ''')
    assert "BACKEND cpu" in out


def test_m1b_invalid_platform_raises_value_error():
    out = _run('''
        from BayesForge.SetDevice.set import setup_device
        try:
            setup_device(platform='banana', print_devices_found=False)
            print("NO_ERROR")
        except ValueError as e:
            print("ERR", type(e).__name__, e)
    ''', check=False)
    assert "NO_ERROR" not in out
    assert "ERR ValueError" in out
    assert "platform" in out


def test_m1c_gpu_without_gpu_raises_runtime_error():
    out = _run('''
        import jax
        try:
            gpus = jax.devices("gpu")
        except Exception:
            gpus = []
        if gpus:
            print("HAS_GPU")
        else:
            from BayesForge.SetDevice.set import setup_device
            try:
                setup_device(platform='gpu', gpu_index=0, print_devices_found=False)
                print("NO_ERROR")
            except RuntimeError as e:
                print("ERR", type(e).__name__, e)
            print("CFG", jax.config.jax_platform_name)
    ''', check=False)
    if "HAS_GPU" in out:
        pytest.skip("a GPU backend is available in this environment")
    assert "NO_ERROR" not in out
    assert "ERR RuntimeError" in out
    assert "platform='cpu'" in out
    # jax_platform_name must never stay set to the unavailable backend
    assert "CFG gpu" not in out


# ---------------------------------------------------------------- M2

def test_m2_affinity_limits_cpu_device_count():
    out = _run('''
        import os
        if not hasattr(os, "sched_setaffinity"):
            print("SKIP")
        else:
            os.sched_setaffinity(0, {0})
            from BayesForge.SetDevice.set import setup_device
            setup_device(print_devices_found=False)
            import jax
            print("NDEV", len(jax.devices("cpu")))
    ''')
    if "SKIP" in out:
        pytest.skip("os.sched_setaffinity not available on this platform")
    assert "NDEV 1" in out


# ---------------------------------------------------------------- M3

def test_m3_invalid_cores_raise_value_error():
    out = _run('''
        from BayesForge.SetDevice.set import setup_device
        for value in (0, -1, 2.5, True):
            try:
                setup_device(cores=value, print_devices_found=False)
                print("NO_ERROR", repr(value))
            except ValueError as e:
                print("ERR", repr(value), type(e).__name__, e)
    ''', check=False)
    assert "NO_ERROR" not in out
    for value in ("0", "-1", "2.5", "True"):
        assert f"ERR {value} ValueError" in out
    assert "cores must be a positive integer" in out


@pytest.mark.parametrize("raw", ["0", "auto"])
def test_m3_invalid_bf_max_cores_raise_value_error(raw):
    out = _run('''
        from BayesForge.SetDevice.set import setup_device
        try:
            setup_device(print_devices_found=False)
            print("NO_ERROR")
        except ValueError as e:
            print("ERR", type(e).__name__, e)
    ''', env={"BF_MAX_CORES": raw}, check=False)
    assert "NO_ERROR" not in out
    assert "ERR ValueError" in out
    assert "BF_MAX_CORES" in out


def test_m3_whole_float_cores_works():
    out = _run('''
        from BayesForge.SetDevice.set import setup_device
        setup_device(cores=2.0, print_devices_found=False)
        import jax
        print("NDEV", len(jax.devices("cpu")))
    ''')
    assert "NDEV 2" in out
