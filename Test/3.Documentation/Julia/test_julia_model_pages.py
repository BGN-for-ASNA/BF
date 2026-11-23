"""Execute the Julia code blocks of every model page of the documentation.

These pages fit real models (minutes per page): marker ``pages``, as in
../Python/test_model_pages.py.

Every ```julia block of a page is concatenated (blocks build on each other)
and run with ``julia`` in a fresh process, so a page fails as soon as a copy-
pasted Julia example would fail for a reader. Blocks fenced as
```{.julia .no-test} or containing ``# doc-test: skip`` are left out
(install commands, GPU code).

Needs Julia >= 1.12 and a Julia project with PyBayesForge, DataFrames, CSV
and MCMCChains, chosen with $BF_JULIA_PROJECT (default: Julia's default
environment). The Python side is the interpreter running pytest, with this
repository's BayesForge first on PYTHONPATH: PythonCall is pointed at it and
CondaPkg is switched off, unless $JULIA_PYTHONCALL_EXE is already set.
Skipped when no ``julia`` is found; set $BF_JULIA to use another one.

Run a single page with:  pytest Test/3.Documentation/Julia/test_julia_model_pages.py -m pages -k "Poisson"
"""
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from doc_blocks import MODEL_PAGES, ROOT, blocks, copy_data  # noqa: E402


def _find_julia():
    if os.environ.get("BF_JULIA"):
        return os.environ["BF_JULIA"]
    found = shutil.which("julia")
    juliaup = pathlib.Path.home() / ".juliaup" / "bin" / "julia"
    return found or (str(juliaup) if juliaup.exists() else None)


JULIA = _find_julia()

pytestmark = pytest.mark.skipif(JULIA is None, reason="julia not found")

# Headless plotting for matplotlib (used by m.diag.*).
PRELUDE = 'ENV["MPLBACKEND"] = "Agg"\n'


@pytest.mark.pages
@pytest.mark.parametrize("page", MODEL_PAGES, ids=[p.stem for p in MODEL_PAGES])
def test_model_page_julia_examples_run(page, tmp_path):
    code = blocks(page, "julia")
    if not code:
        pytest.skip("no Julia blocks")
    script = tmp_path / "page.jl"
    copy_data(tmp_path)
    script.write_text(PRELUDE + "\n\n".join(code) + "\n")
    env = dict(os.environ, JAX_PLATFORMS="cpu", MPLBACKEND="Agg",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
    if "JULIA_PYTHONCALL_EXE" not in env:
        env.update(JULIA_PYTHONCALL_EXE=sys.executable, JULIA_CONDAPKG_BACKEND="Null")
    cmd = [JULIA]
    if os.environ.get("BF_JULIA_PROJECT"):
        cmd.append(f"--project={os.environ['BF_JULIA_PROJECT']}")
    proc = subprocess.run(
        cmd + [str(script)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, f"{page.name} failed:\n{proc.stderr[-4000:]}"
