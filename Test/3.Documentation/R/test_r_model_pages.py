"""Execute the R code blocks of every model page of the documentation.

These pages fit real models (minutes per page): marker ``pages``, as in
../Python/test_model_pages.py.

Every ```r block of a page is concatenated (blocks build on each other) and
run with ``Rscript`` in a fresh session, so a page fails as soon as a copy-
pasted R example would fail for a reader. Blocks fenced as ```{.r .no-test}
or containing ``# doc-test: skip`` are left out (install commands, GPU code).

Needs R with the BayesForge R package (BFR) and its 'BFR' Python environment
(``setup_env()``). The Python side runs this repository's BayesForge: the
repository root is put first on PYTHONPATH. Skipped when ``Rscript`` is not
found; set $BF_RSCRIPT to use another R.

Run a single page with:  pytest Test/3.Documentation/R/test_r_model_pages.py -m pages -k "Poisson"
"""
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from doc_blocks import MODEL_PAGES, ROOT, blocks, copy_data  # noqa: E402

RSCRIPT = os.environ.get("BF_RSCRIPT") or shutil.which("Rscript")

pytestmark = pytest.mark.skipif(RSCRIPT is None, reason="Rscript not found")

# Headless plotting, on both sides: R graphics and matplotlib.
PRELUDE = 'pdf(NULL)\nSys.setenv(MPLBACKEND = "Agg")\n'


@pytest.mark.pages
@pytest.mark.parametrize("page", MODEL_PAGES, ids=[p.stem for p in MODEL_PAGES])
def test_model_page_r_examples_run(page, tmp_path):
    code = blocks(page, "r")
    if not code:
        pytest.skip("no R blocks")
    script = tmp_path / "page.R"
    copy_data(tmp_path)
    script.write_text(PRELUDE + "\n\n".join(code) + "\n")
    env = dict(os.environ, JAX_PLATFORMS="cpu", MPLBACKEND="Agg",
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
    proc = subprocess.run(
        [RSCRIPT, str(script)], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, f"{page.name} failed:\n{proc.stderr[-4000:]}"
