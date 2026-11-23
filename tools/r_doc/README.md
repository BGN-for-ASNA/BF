# tools/r_doc — R wrapper & docs pipeline (dev-only)

Tools that generate the R wrapper functions and their roxygen documentation
for every BayesForge distribution. They are development utilities only:
none of them is shipped in the Python wheel or package.

Pipeline, in order:

1. **bf_doc.py** — extracts the Python docstrings of every distribution
   exposed on `bf().dist` into `pythonDoc.json` (default: next to the
   script, not the CWD):
   `python tools/r_doc/bf_doc.py [out.json]` (needs BayesForge importable).
2. **LLM step** — converts `pythonDoc.json` into roxygen-style R docs (see
   "Provenance" below); the result is saved as `RDoc.json` (next to the
   scripts, the checked-in copy).
3. **fill_params.py** — deterministic post-processing of the LLM output:
   drops the non-distribution members (`biject_to`, `mask`, `plate`, `unit`)
   and null entries, and replaces every `@param <name> None` with the
   description parsed from the Python docstring (`(undocumented)` when the
   docstring has none). Must be rerun after **any** LLM regeneration:
   `python tools/r_doc/fill_params.py pythonDoc.json RDoc.json [out.json]`
   (pure stdlib, no BayesForge import; prints a summary).
4. **r_wrapper.R** — generates one R function file per distribution through
   reticulate, pairing each function with its roxygen doc:
   `BF_R_OUT=./R Rscript tools/r_doc/r_wrapper.R`
   (reads `RDoc.json` next to the script unless `BF_DOC_JSON` overrides it,
   and stops with an error if that file does not exist; needs a Python
   environment with BayesForge visible to reticulate).

## Provenance

The roxygen text in `RDoc.json` was produced by a **language model**, not by
code: per the comment that used to sit in `bf_doc.py`, the prompt below was
run against **Gemini 2 Pro** ("formated output" option) on the JSON generated
by `bf_doc.py`. No model version, date, temperature or conversation id was
recorded, and no human review was logged. **This step is therefore not
reproducible**: re-running it can produce different wording, and the exact
delivered text cannot be regenerated bit-for-bit.

The prompt, kept verbatim as the only specification of the conversion:

```text
You are an expert code conversion assistant. Your task is to convert Python docstrings (in Google style) into R's roxygen2 documentation format.
Instructions:
For each key-value pair in the input dictionary:
The key is the function's name.
The value is the Python docstring.
Convert the docstring into a complete roxygen2 comment block, where each line begins with #'.
Use the following mapping:
Python (Summary Line) -> R (@title)
Python (Description Body) -> R (@details)
Python (Args:) -> R (@param name description) for each argument.
Python (Returns:) -> R (@return description)
The @examples tag is mandatory and must be constructed as follows:
bf.dist.[function_name]([parameter_names], sample = TRUE)
Replace [function_name] with the key from the dictionary.
Replace [parameter_names] with a comma-separated list of the actual parameter names from the @param section.
```

The LLM failed to extract descriptions that were not prefixed by a parenthesised
type, leaving `@param <name> None` in most entries. **`fill_params.py` is the
deterministic post-processing that repairs this** (and drops the
non-distribution members the prompt wrongly documented as distributions);
rerun it after any LLM regeneration, before `r_wrapper.R`.

## Two sources of truth (known limitation)

The R examples in `Documentation/*.qmd` (the "R" tabs, e.g. `m$data(...)`,
`bf.dist.exponential(...)`) are written by hand. They are **not** generated
from `RDoc.json`, and no test checks that the two agree: the `lang:` tests
run the examples, they do not compare them with the roxygen help. After a
distribution signature changes, update both the qmd pages and `RDoc.json`
(rerun steps 1-3 above). RELEASE_AUDIT R-m4.
