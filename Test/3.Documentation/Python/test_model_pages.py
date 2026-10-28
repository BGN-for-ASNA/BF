"""2.5 -- execute the Python examples of every model page of the documentation.

Same rule as test_start_pages.py (Get started pages): all ```python
blocks of a page are concatenated and run in a fresh interpreter, so a page
fails as soon as a copy-pasted example would fail for a reader. Blocks fenced
as ```{.python .no-test} or containing ``# doc-test: skip`` are left out.

These pages fit real models (minutes per page), so the test has its own
marker, ``pages``: the pull-request CI skips it (-m "not pages") and the
weekly workflow .github/workflows/doc-pages.yml runs all pages. Locally,
pick pages with -k, e.g.
    pytest Test/3.Documentation/Python/test_model_pages.py -m pages -k "Poisson"
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from doc_blocks import MODEL_PAGES, blocks, copy_data  # noqa: E402


@pytest.mark.pages
@pytest.mark.parametrize("page", MODEL_PAGES, ids=[p.stem for p in MODEL_PAGES])
def test_model_page_python_examples_run(page, tmp_path):
    code = blocks(page, "python")
    if not code:
        pytest.skip("no python blocks")
    script = tmp_path / "page.py"
    copy_data(tmp_path)
    script.write_text("import matplotlib\nmatplotlib.use('Agg')\n" + "\n\n".join(code))
    env = dict(os.environ, JAX_PLATFORMS="cpu", MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=3600)
    assert proc.returncode == 0, f"{page.name} failed:\n{proc.stderr[-4000:]}"
