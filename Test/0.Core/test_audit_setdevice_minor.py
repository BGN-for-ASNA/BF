"""Regression tests for RELEASE_AUDIT.md, module SetDevice, minors m1-m8."""

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


# ---------------------------------------------------------------- m1

def test_m1_invalid_gpu_index_raises_value_error():
    out = _run('''
        from BayesForge.SetDevice.set import setup_device
        for value in (-3, True):
            try:
                setup_device(platform='cpu', gpu_index=value, print_devices_found=False)
                print("NO_ERROR", repr(value))
            except ValueError as e:
                print("ERR", repr(value), type(e).__name__, e)
    ''', check=False)
    assert "NO_ERROR" not in out
    for value in ("-3", "True"):
        assert f"ERR {value} ValueError" in out
    assert "gpu_index" in out


def test_m1_comma_separated_gpu_index_accepted():
    # platform='cpu': validation runs but no GPU probe and no CUDA write.
    out = _run('''
        import os
        from BayesForge.SetDevice.set import setup_device, _validate_gpu_index
        print("VALIDATED", _validate_gpu_index("0,1"))
        before = os.environ.get("CUDA_VISIBLE_DEVICES")
        setup_device(platform='cpu', gpu_index="0,1", print_devices_found=False)
        print("UNCHANGED", os.environ.get("CUDA_VISIBLE_DEVICES") == before)
    ''')
    assert "VALIDATED 0,1" in out
    assert "UNCHANGED True" in out


# ---------------------------------------------------------------- m2

def test_m2_deallocate_docstring_and_frozen_warning():
    out = _run('''
        import inspect
        from BayesForge.SetDevice.set import setup_device, _warn_frozen_settings
        print("DOC>>>" + inspect.getdoc(setup_device) + "<<<DOC")
        print("SRC>>>" + inspect.getsource(_warn_frozen_settings) + "<<<SRC")
    ''')
    doc = out.split("DOC>>>")[1].split("<<<DOC")[0]
    src = out.split("SRC>>>")[1].split("<<<SRC")[0]
    # the deallocate parameter describes the real effect, not "free devices"
    assert "allocator (no preallocation)" in doc
    assert "It does not free" in doc
    assert "cannot be undone in the same process" in doc
    assert "Whether to deallocate any existing devices" not in doc
    # the frozen-settings warning covers deallocate (already fixed with C1)
    assert "deallocate" in src


# ---------------------------------------------------------------- m3

def test_m3_docstring_backend_init_contract():
    out = _run('''
        import inspect
        from BayesForge.SetDevice.set import setup_device
        print("DOC>>>" + inspect.getdoc(setup_device) + "<<<DOC")
    ''')
    doc = out.split("DOC>>>")[1].split("<<<DOC")[0]
    assert "must be called before any JAX imports" not in doc
    assert "take effect only" in doc
    assert "the JAX backend is initialised" in doc
    assert "later calls warn and change nothing" in doc


# ---------------------------------------------------------------- m4

def test_m4_existing_xla_flags_value_warns():
    out = _run('''
        import warnings
        from BayesForge.SetDevice.set import setup_device
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            setup_device(cores=2, print_devices_found=False)
        for w in caught:
            print("WARN", w.message)
    ''', env={"XLA_FLAGS": "--xla_force_host_platform_device_count=3"})
    warns = [l for l in out.splitlines() if l.startswith("WARN")]
    assert warns, out
    # the message names both the replaced value (3) and the new one (2)
    assert any("3" in l and "2" in l and "xla_force_host_platform_device_count" in l
               for l in warns)


def test_m4_matching_xla_flags_value_does_not_warn():
    out = _run('''
        import warnings
        from BayesForge.SetDevice.set import setup_device
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            setup_device(cores=2, print_devices_found=False)
        for w in caught:
            print("WARN", w.message)
        print("DONE")
    ''', env={"XLA_FLAGS": "--xla_force_host_platform_device_count=2"})
    assert "DONE" in out
    assert not [l for l in out.splitlines() if l.startswith("WARN")]


# ---------------------------------------------------------------- m5

def test_m5_float_precision_docstring_process_global():
    out = _run('''
        import inspect
        from BayesForge.SetDevice.set import setup_device
        print("DOC>>>" + inspect.getdoc(setup_device) + "<<<DOC")
    ''')
    doc = out.split("DOC>>>")[1].split("<<<DOC")[0]
    assert "x64 is process-global" in doc
    assert "mixes float32" in doc


# ---------------------------------------------------------------- m6

def test_m6_docstring_not_thread_safe():
    out = _run('''
        import inspect
        from BayesForge.SetDevice.set import setup_device
        print("DOC>>>" + inspect.getdoc(setup_device) + "<<<DOC")
    ''')
    doc = out.split("DOC>>>")[1].split("<<<DOC")[0]
    assert "os.environ" in doc
    assert "jax.config" in doc
    assert "not thread-safe" in doc


# ---------------------------------------------------------------- m7

def test_m7_print_names_requested_platform():
    out = _run('''
        from BayesForge.SetDevice.set import setup_device
        setup_device(platform='cpu', cores=2, print_devices_found=True)
    ''')
    assert "jax.local_device_count(cpu) = 2" in out


# ---------------------------------------------------------------- m8

def test_m8_setup_multi_device_cores_same_error_as_setup_device():
    out = _run('''
        from BayesForge.SetDevice.set import setup_device, setup_multi_device
        msgs = {}
        try:
            setup_device(cores="abc", print_devices_found=False)
            print("NO_ERROR setup_device")
        except ValueError as e:
            msgs["setup_device"] = str(e)
        try:
            setup_multi_device(cores="abc")
            print("NO_ERROR setup_multi_device")
        except ValueError as e:
            msgs["setup_multi_device"] = str(e)
        print("ERR setup_device:", msgs.get("setup_device"))
        print("ERR setup_multi_device:", msgs.get("setup_multi_device"))
        print("SAME", msgs.get("setup_device") == msgs.get("setup_multi_device"))
    ''', check=False)
    assert "NO_ERROR" not in out
    assert "SAME True" in out
    assert "cores must be a positive integer" in out
