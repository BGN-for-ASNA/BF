"""Regression tests for RELEASE_AUDIT.md, module R (tools/r_doc), majors R-M2..R-M4."""
import importlib.util
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
RDOC_DIR = ROOT / "tools" / "r_doc"


def _load_fill_params():
    spec = importlib.util.spec_from_file_location(
        "fill_params", RDOC_DIR / "fill_params.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # no side effect at import
    return mod


GOOGLE_DOC = '''"""Normal distribution.

Args:
    loc (float): Location parameter of the distribution.
    scale (float): Scale parameter of the distribution.

Returns:
    A NumPyro distribution object.
"""'''

# Verbatim excerpt of pythonDoc.json["normal"]: Markdown, not Google style.
MARKDOWN_DOC = '''### Normal 
        Samples from a Normal (Gaussian) distribution.
        ...
        #### Args:
        - *loc* (jnp.ndarray): The mean of the distribution.
        
        - *scale* (jnp.ndarray): The standard deviation of the distribution.
        
        - *shape* (tuple): A multi-purpose argument for shaping. When `sample=False` (model building), this is used with `shape` to set the distribution's batch shape. When `sample=True` (direct sampling), this is used as `sample_shape` to draw a raw JAX array of the given shape.
        
        - *event* (int): The number of batch dimensions to reinterpret as event dimensions (used in model building).
        
        - *mask* (jnp.ndarray, bool): Optional boolean array to mask observations.
        
        #### Returns:
        - *dist*: A Normal distribution instance.
'''


def test_parse_args_section_google_style():
    mod = _load_fill_params()
    args = mod.parse_args_section(GOOGLE_DOC)
    assert args["loc"] == "Location parameter of the distribution."
    assert args["scale"] == "Scale parameter of the distribution."
    assert "Returns" not in args and "A NumPyro" not in "".join(args.values())


def test_parse_args_section_markdown_style():
    mod = _load_fill_params()
    args = mod.parse_args_section(MARKDOWN_DOC)
    assert set(args) == {"loc", "scale", "shape", "event", "mask"}
    assert args["loc"] == "The mean of the distribution."
    assert args["scale"] == "The standard deviation of the distribution."
    assert args["shape"].startswith("A multi-purpose argument for shaping.")
    assert args["shape"].endswith("draw a raw JAX array of the given shape.")
    assert args["event"] == ("The number of batch dimensions to reinterpret "
                             "as event dimensions (used in model building).")
    assert args["mask"] == "Optional boolean array to mask observations."
    # The "#### Returns:" heading ends the section: nothing leaks into mask.
    assert "dist" not in args and "Normal distribution instance" not in args["mask"]


def test_parse_args_section_markdown_continuation():
    mod = _load_fill_params()
    doc = ("#### Args:\n"
           "    - *seed* (int): Random seed.\n"
           "        Passed to jax.random.\n"
           "        Two lines.\n"
           "\n"
           "    - *n* (int): Number of draws.\n")
    args = mod.parse_args_section(doc)
    assert args["seed"] == "Random seed. Passed to jax.random. Two lines."
    assert args["n"] == "Number of draws."


def test_fill_drops_and_repairs():
    mod = _load_fill_params()
    rdoc = {
        "normal": ("#' @title Normal distribution wrapper.\n"
                   "#' @param loc None\n"
                   "#' @param scale None\n"
                   "#' @examples\n"
                   "#' bf.dist.normal(loc, scale, sample = TRUE)"),
        "mask": None,
        "unit": "#' @title Unit distribution wrapper.\n#' @param log_factor None",
    }
    pydoc = {"normal": GOOGLE_DOC}
    filled, stats = mod.fill(rdoc, pydoc)
    assert set(filled) == {"normal"}
    assert "None" not in filled["normal"]
    assert "@param loc Location parameter of the distribution." in filled["normal"]
    assert "@param scale Scale parameter of the distribution." in filled["normal"]
    assert stats["entries_dropped"] == 2
    assert stats["params_filled"] == 2
    assert stats["params_undocumented"] == 0


def test_fill_undocumented_marker():
    mod = _load_fill_params()
    rdoc = {"normal": "#' @param loc None\n#' @param seed None"}
    filled, stats = mod.fill(rdoc, {"normal": "No args section here."})
    assert "@param loc (undocumented)" in filled["normal"]
    assert "@param seed (undocumented)" in filled["normal"]
    assert "None" not in filled["normal"]
    assert stats["params_filled"] == 0
    assert stats["params_undocumented"] == 2


def test_fill_replaces_bare_literal_defaults():
    # Real RDoc.json["normal"]: @param lines carrying only the default value.
    mod = _load_fill_params()
    rdoc = {"normal": ("#' @title Normal distribution wrapper.\n"
                       "#' @param loc 0.0\n"
                       "#' @param scale 1.0\n"
                       "#' @param validate_args None\n"
                       "#' @param shape (tuple): A multi-purpose argument "
                       "for shaping. When `sample=False` ...\n")}
    filled, stats = mod.fill(rdoc, {"normal": MARKDOWN_DOC})
    assert "@param loc The mean of the distribution." in filled["normal"]
    assert "@param scale The standard deviation of the distribution." in filled["normal"]
    assert "@param validate_args (undocumented)" in filled["normal"]
    # A real description (with spaces) is left untouched.
    assert "@param shape (tuple): A multi-purpose argument for shaping." in filled["normal"]
    assert stats["params_filled"] == 2
    assert stats["params_undocumented"] == 1


def test_fill_literal_default_without_pydoc():
    mod = _load_fill_params()
    rdoc = {"categorical": "#' @param mode 'marginal'\n"
                           "#' @param seed True\n"
                           "#' @param reps 10\n"}
    filled, stats = mod.fill(rdoc, {})
    assert "@param mode (undocumented; default 'marginal')" in filled["categorical"]
    assert "@param seed (undocumented; default True)" in filled["categorical"]
    assert "@param reps (undocumented; default 10)" in filled["categorical"]
    assert stats["params_filled"] == 0
    assert stats["params_undocumented"] == 3


def test_checked_in_rdoc_is_clean():
    # Expected to FAIL until RDoc.json is regenerated with fill_params.py
    # (RELEASE_AUDIT.md R-M3); it locks the fixed state in.
    mod = _load_fill_params()
    with open(RDOC_DIR / "RDoc.json", encoding="utf-8") as f:
        rdoc = json.load(f)
    assert not (set(rdoc) & mod.NON_DISTRIBUTIONS)
    assert all(v is not None for v in rdoc.values())
    bad = [k for k, v in rdoc.items()
           if isinstance(v, str) and re.search(r"@param\s+\w+\s+None\s*$", v, re.M)]
    assert bad == []


def test_r_m2_paths_are_script_relative_and_named_consistently():
    bf_src = (RDOC_DIR / "bf_doc.py").read_text(encoding="utf-8")
    assert "Path(__file__).with_name(\"pythonDoc.json\")" in bf_src
    r_src = (RDOC_DIR / "r_wrapper.R").read_text(encoding="utf-8")
    assert "RDoc.json" in r_src
    assert '"Doc_R.json"' not in r_src
    assert "commandArgs(trailingOnly = FALSE)" in r_src
    assert "file.exists(doc_json)" in r_src


def test_r_m4_prompt_moved_out_of_bf_doc():
    bf_src = (RDOC_DIR / "bf_doc.py").read_text(encoding="utf-8")
    assert "gemini" not in bf_src.lower()
    assert "ROXYGEN_PROMPT" not in bf_src
    readme = (RDOC_DIR / "README.md").read_text(encoding="utf-8")
    assert "You are an expert code conversion assistant." in readme
    assert "fill_params.py" in readme
