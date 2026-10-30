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
| `params` | all | comma-separated parameter names to compare (base name, so `a` covers `a[0]`, `a[1]`, ...) |

## How to run
```bash
pytest Test/test_cross_language.py -m lang                     # everything
pytest Test/test_cross_language.py -m lang -k "Continuous"     # one case
pytest Test/test_cross_language.py -m lang -k julia             # one language
BF_TEST_LANGS=python,r pytest Test/test_cross_language.py -m lang   # only Python vs R
BF_JULIA_PROJECT=~/.julia/environments/bf-tests pytest Test/test_cross_language.py -m lang
BF_RSCRIPT=/usr/lib/R/bin/Rscript pytest Test/test_cross_language.py -m lang
```

Locally, point R and Julia at the same Python as the reference run, as CI does, otherwise they
may load another jax/numpyro and drift:

```bash
RETICULATE_PYTHON=$(which python) JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$(which python) pytest Test/test_cross_language.py -m lang
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

## Python, R and Julia: what is identical and what is not

R and Python give bit-identical chains for the same case when R uses the same Python
environment (`RETICULATE_PYTHON`): on the SRM case the max |diff of means| is 0.0.
Locally, reticulate may pick another Python (e.g. `~/.virtualenvs/BFR` with jax 0.11.1 /
numpyro 0.21 instead of 0.12 / 0.22); then R chains drift too. CI sets
`RETICULATE_PYTHON` and `JULIA_PYTHONCALL_EXE` to the same Python.

Julia gives close but not identical chains. On a trivial model (alpha ~ Normal(0,1),
s ~ Exponential(1), y ~ Normal(alpha, s), 5 obs, 200/200 draws, seed 0), the posterior
mean of alpha is:

| run | alpha mean |
| --- | --- |
| `python repro.py` | 0.37562987 |
| `LD_PRELOAD=<julia>/lib/julia/libopenlibm.so.4 python repro.py` | 0.37563018 |
| `julia repro.jl` (same model through PyBayesForge) | 0.37563018 |
| the Python script run inside the Julia process (`pyexec`) | 0.37563018 |
| python + Julia's OPENBLAS_* env vars, or LD_PRELOAD of Julia's libunwind / libLLVM | 0.37562987 (no effect) |

The cause: the Julia process loads its own libm, **openlibm** (`libopenlibm.so`), and
the embedded Python/JAX resolves math functions to it instead of glibc's libm. Some
results differ in the last bit; HMC amplifies these differences along the chain. This
is not a PyBayesForge translation bug: plain Python code run inside Julia gives the
Julia result. Proof on a real case: on the SRM case, Python with `LD_PRELOAD` of
openlibm vs Julia gives max |diff of means| = 0.0 (dr_sigma[0] = 3.695815 in both);
without it, dr_sigma[0] differs by 0.27 sd (ESS ~28).

### Reproduce

```bash
cd Test/0.Core/libm_julia
python repro.py
LD_PRELOAD=$(julia -e 'print(joinpath(Sys.BINDIR, "..", "lib", "julia", "libopenlibm.so.4"))') python repro.py
JULIA_CONDAPKG_BACKEND=Null JULIA_PYTHONCALL_EXE=$(which python) julia --project=$BF_JULIA_PROJECT repro.jl
```

Expected: the first line differs from the other two in the 7th decimal.

### What it means for writing a case

* Make chains mix: check `ess_bulk` in the Python summary. On a well-mixed chain the
  libm drift is far below 0.25 sd (the 13 Rethinking cases pass).
* Use `params:` to compare model-level parameters when there are many weakly
  identified ones.
* A Julia "disagreement" of a few tenths of sd with low ESS is Monte Carlo drift, not
  a translation bug; a larger `tol` is acceptable only with a comment pointing to this
  section. See `6.Network/1.Sender receiver` for an example.
