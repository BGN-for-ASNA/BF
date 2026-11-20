"""RELEASE_AUDIT X9: BayesForge must import whatever was imported before it."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure BayesForge imports whatever was imported before it
@pytest.mark.parametrize("first", ["arviz", "numpyro", "jax"])
def test_import_after_other_libraries(first):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=ROOT)
    code = f"import {first}; import BayesForge; print('IMPORT_OK')"
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0 and "IMPORT_OK" in r.stdout, r.stderr[-2000:]

# Ensure Arviz pin has an upper bound
def test_arviz_pin_has_an_upper_bound():
    with open(os.path.join(ROOT, "pyproject.toml")) as f:
        line = next(l for l in f if l.strip().startswith('"arviz'))
    assert "<" in line
