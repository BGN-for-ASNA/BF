"""Execute the Python code blocks of the "Get started" documentation pages.

Every ```python block of a page is concatenated (blocks build on each other)
and run in a fresh interpreter, so a page fails the test as soon as a copy-
pasted example would fail for a reader. Blocks fenced as ```{.python .no-test}
or containing ``# doc-test: skip`` are left out (e.g. GPU-only code).

Run a single page with:  pytest Test/3.Documentation/Python -k "seeds"
"""
import os
import subprocess
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from doc_blocks import ROOT, START_PAGES, blocks  # noqa: E402


def python_blocks(page):
    return blocks(page, "python")


@pytest.mark.parametrize("page", START_PAGES, ids=[p.stem for p in START_PAGES])
def test_page_python_examples_run(page, tmp_path):
    code = python_blocks(page)
    if not code:
        pytest.skip("no python blocks")
    script = tmp_path / "page.py"
    # Headless plotting; never open windows from a test run.
    # Under a main guard, as a reader's script must be: run_simulations spawns
    # worker processes that re-import the script.
    body = textwrap.indent("\n\n".join(code), "    ")
    script.write_text("import matplotlib\nmatplotlib.use('Agg')\n"
                      'if __name__ == "__main__":\n' + body)
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=1200,
    )
    assert proc.returncode == 0, f"{page.name} failed:\n{proc.stderr[-4000:]}"
