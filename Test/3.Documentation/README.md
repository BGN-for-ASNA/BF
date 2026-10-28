# Test/3.Documentation: the documentation examples are tests

| Test | Checks |
|---|---|
| `test_language_coverage.py` | Every code example of the "Get started" pages (`Documentation/start/`) is in a Python / R / Julia tabset, with code in all three tabs |
| `Python/test_python_start_pages.py` | The Python blocks of each "Get started" page run |
| `Python/test_model_pages.py` | The Python blocks of each model page run (marker `pages`, weekly CI) |
| `R/test_r_model_pages.py` | The R blocks of each model page run (marker `pages`) |
| `Julia/test_julia_model_pages.py` | The Julia blocks of each model page run (marker `pages`) |
| `R/test_r_start_pages.py` | The R blocks of each "Get started" page run with `Rscript` |
| `Julia/test_julia_start_pages.py` | The Julia blocks of each "Get started" page run with `julia` |

The blocks of a page are concatenated in page order (they build on each
other) and run in a fresh process, so a page fails as soon as a copy-pasted
example would fail for a reader. `doc_blocks.py` extracts them.

## Writing examples

* Put every example in a tabset with the three languages:

  ````markdown
  ::: {.panel-tabset group="language"}
  ## Python
  ```python
  ...
  ```

  ## R
  ```r
  ...
  ```

  ## Julia
  ```julia
  ...
  ```
  :::
  ````

* Fence a block that must not run with `.no-test`: ```` ```{.r .no-test} ````
  (install commands, GPU code, placeholders such as `<your model>`). A tested
  block that installs or removes packages (`install.packages()`,
  `setup_env()`, `Pkg.add()`, `pip install`, ...) makes the test fail
  without running it.

* An example that reads a data file by name (e.g. `open('BNN.json')`) finds it in the
  working directory: put the file in `Test/3.Documentation/data/`, the tests copy that folder
  into the directory each page runs in.

## Running

The Python side is always this checkout of BayesForge (the repository root
is put first on `PYTHONPATH`).

```bash
python -m pytest Test/3.Documentation/test_language_coverage.py Test/3.Documentation/Python -q

# R: the BayesForge R package (BFR) and its Python environment.
# R_LIBS=<lib> to test another build of BFR, e.g. its dev branch.
R_LIBS=/path/to/rlib python -m pytest Test/3.Documentation/R -q

# Julia: a project with PyBayesForge, PythonCall, DataFrames, CSV and MCMCChains.
# PythonCall uses the Python running pytest; CondaPkg is switched off.
BF_JULIA_PROJECT=/path/to/project python -m pytest Test/3.Documentation/Julia -q
```

R and Julia tests are skipped when `Rscript` / `julia` are not found
(`$BF_RSCRIPT`, `$BF_JULIA` to point at them).
