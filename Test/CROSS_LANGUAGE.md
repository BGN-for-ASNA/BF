# Cross-language model cases

## What a case is
One markdown file holding **the same model three times**, in fenced blocks
` ```python `, ` ```r ` and ` ```julia `. The harness
(`Test/lang_harness.py`) runs each block in a fresh process, appends an epilogue
so every language writes `m.summary()` to `bf_summary.csv`, and checks that R and
Julia agree with Python.

## Where cases live
```
Test/<topic>/cases/<name>.md
Test/<topic>/cases/data/          # optional CSVs, copied next to the script
```
Anything matching `Test/**/cases/*.md` is collected (folders with `(WIP)` in the
name are ignored). The test id is `<topic>/<name>`.

## Rules
* The BF object is always named `m` (the epilogue calls `m.summary()`).
* Every block loads its data and calls `m.fit`.
* The three blocks must describe the *same* model: same data, same priors,
  same number of samples/warmup and the same `seed`.
* Use a built-in dataset (`m.load.*`) or a CSV in `cases/data/` — no absolute
  paths, no downloads.
* Comparison against Stan and parameter-recovery checks stay in the Python
  scripts; this harness only checks Python vs R vs Julia.
* Skip a block with `# doc-test: skip` or a `.no-test` fence tag if it must not
  run (the language is then skipped, not failed).

## Front matter
Optional, at the top of the file between two `---` lines:

| key | default | meaning |
| --- | --- | --- |
| `tol` | `0.25` | allowed `|mean_r/julia - mean_python|` in units of the python `sd` |
| `timeout` | `1800` | seconds allowed per language |

## How to run
```bash
pytest Test/test_cross_language.py -m lang                     # everything
pytest Test/test_cross_language.py -m lang -k "Continuous"     # one case
pytest Test/test_cross_language.py -m lang -k julia             # one language
BF_TEST_LANGS=python,r pytest Test/test_cross_language.py -m lang   # only Python vs R
BF_JULIA_PROJECT=~/.julia/environments/bf-tests pytest Test/test_cross_language.py -m lang
BF_RSCRIPT=/usr/lib/R/bin/Rscript pytest Test/test_cross_language.py -m lang
```
`BF_JULIA` / `BF_RSCRIPT` point at the interpreters when they are not on `PATH`;
a missing interpreter skips that language instead of failing. `BF_JULIA_PROJECT`
points at a Julia project containing PyBayesForge, PythonCall, DataFrames, CSV,
MCMCChains and Distributions. `BF_TEST_LANGS` is a comma-separated list
(subset of `python,r,julia`) restricting which languages are run. CI runs the
cases in the docs-r and docs-julia jobs of .github/workflows/tests.yml, one
language per job.

## What the test checks
The block runs (returncode 0), `bf_summary.csv` exists, and for every parameter
of the Python summary the label exists in the other summary and
`|diff of means| <= tol * sd(python)`. Problems are listed one per line.
