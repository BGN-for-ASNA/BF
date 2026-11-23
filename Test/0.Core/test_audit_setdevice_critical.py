"""Regression tests for RELEASE_AUDIT.md, module 18 (SetDevice), critical C1."""

import subprocess, sys, textwrap
import pytest


def _run(code):
    r = subprocess.run([sys.executable, "-W", "always", "-c", textwrap.dedent(code)],
                       capture_output=True, text=True, timeout=300,
                       env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"})
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout + r.stderr


def test_c1_second_call_with_other_cores_warns():
    out = _run('''
        from BayesForge.SetDevice.set import setup_device
        setup_device(cores=2, print_devices_found=False)
        import jax; jax.devices()
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            setup_device(cores=4, print_devices_found=False)
        print("WARN", any("no effect" in str(x.message) and "cores=4" in str(x.message) for x in w))
        print("NDEV", len(jax.devices("cpu")))
    ''')
    assert "WARN True" in out
    assert "NDEV 2" in out


def test_c1_same_settings_do_not_warn():
    out = _run('''
        from BayesForge.SetDevice.set import setup_device
        setup_device(cores=2, print_devices_found=False)
        import jax; jax.devices()
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            setup_device(cores=2, print_devices_found=False)
        print("WARN", any("no effect" in str(x.message) for x in w))
    ''')
    assert "WARN False" in out


def test_c1_first_call_does_not_warn():
    out = _run('''
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from BayesForge.SetDevice.set import setup_device
            setup_device(cores=3, print_devices_found=False)
        import jax
        print("WARN", any("no effect" in str(x.message) for x in w))
        print("NDEV", len(jax.devices("cpu")))
    ''')
    assert "WARN False" in out
    assert "NDEV 3" in out
