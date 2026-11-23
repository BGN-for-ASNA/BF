"""Regression tests for RELEASE_AUDIT.md, module Build, minors T-7, N-9, N-10, B-5, B-6.

Scope: the minors of the RELEASE_AUDIT.md sections "Module Build" — TFP (T-7),
numpyro (N-9, N-10) and the test generator (B-5, B-6). The audit line numbers
predate the generator rewrites (1bc8055b, 5d666ffd), so every check reads the
current source with ``ast``/``tokenize`` or execs a single pure helper in
isolation; the generators are never imported and never run. What the tests pin:

* T-7  the "Default = ..." docstring lines go through ``format_default``
       (classes/dtypes by qualified name, strings quoted, numbers as is);
* N-9  every ``open``/``read_text``/``write_text`` in the three generators
       declares ``encoding="utf-8"``;
* N-10 no ``# %%`` notebook markers and no stray spaces in code (checked on
       the token stream, so prose inside the LLM prompt strings is exempt);
       the ``conjugate.NegativeBinomial`` line stays, with a comment saying
       why (private submodule, public alias missing);
* B-5  Build_test_file.py resolves its paths from ``Path(__file__).resolve()``
       (repo root = parents[2]) and the generated file's template does the
       same from its own location;
* B-6  ``collect_dist_methods`` keeps only public callables outside the
       plumbing names (``mask``, ``plate``, ``normalize_seed``), a method
       without a docstring is skipped with a notice instead of sending
       ``"None"`` to the LLM, and importing Build_test_file does not start
       JAX (``bf(platform='cpu')`` lives in ``main()``).
"""

import ast
import inspect
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BUILD_DIR = REPO_ROOT / "BayesForge" / "Build"

NP_GEN = BUILD_DIR / "Build_NP_light.py"
TFP_GEN = BUILD_DIR / "Build_TFP_light.py"
TEST_GEN = BUILD_DIR / "Build_test_file.py"

GENERATORS = ("Build_NP_light.py", "Build_TFP_light.py", "Build_test_file.py")

# The generators import numpyro, tensorflow_probability and ollama, so they
# are never imported here: parsing the source keeps these tests offline.
SOURCES = {path.name: path.read_text(encoding="utf-8") for path in (NP_GEN, TFP_GEN, TEST_GEN)}
TREES = {name: ast.parse(src) for name, src in SOURCES.items()}


def _source(name):
    return SOURCES[name]


def _tree(name):
    return TREES[name]


def _functions(name):
    return {node.name: node for node in ast.walk(_tree(name)) if isinstance(node, ast.FunctionDef)}


def _body(name, func):
    node = _functions(name).get(func)
    assert node is not None, f"{name} defines no function {func}()"
    return ast.get_source_segment(_source(name), node) or ""


def _module_level_imports(name):
    modules = set()
    for node in _tree(name).body:
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def _exec_defs(name, wanted, extra_globals=None):
    """Exec selected top-level definitions of a generator, without importing it.

    Same isolation contract as the major test module: the generator module is
    never executed, so its heavy imports stay untouched; stdlib names the
    extracted chunks rely on are injected by the caller.
    """
    source = _source(name)
    chunks = []
    found = set()
    for node in ast.parse(source).body:
        key = getattr(node, "name", None)
        if key is None and isinstance(node, ast.Assign):
            targets = node.targets
            key = targets[0].id if len(targets) == 1 and isinstance(targets[0], ast.Name) else None
        if key in wanted:
            chunks.append(ast.get_source_segment(source, node))
            found.add(key)
    assert found == wanted, f"{name} defines none of {sorted(wanted - found)} at module level"
    isolated = {
        "__name__": f"{name.rsplit('.', 1)[0]}_isolated",
        "__file__": str(BUILD_DIR / name),
    }
    isolated.update(extra_globals or {})
    exec("\n".join(chunk for chunk in chunks if chunk), isolated)
    return isolated


# --------------------------------------------------------------------------
# T-7 — docstring defaults rendered by the clean formatter
# --------------------------------------------------------------------------

def test_tfp_docstring_uses_the_default_formatter():
    """The 'Default = ...' lines must come from format_default, not from repr."""
    body = _body("Build_TFP_light.py", "generate")
    assert "Default = " in body, "the docstring builder lost its Default lines"
    assert "format_default(param.default)" in body, (
        "RELEASE_AUDIT T-7: the docstring must render defaults with "
        "format_default, not with the raw repr of the default value"
    )
    assert not re.search(r"Default = \{param\.default\}", body), (
        "RELEASE_AUDIT T-7: f'... Default = {param.default}' pastes the repr "
        "(<class 'jax.numpy.int32'>) back into the docstring"
    )


def test_default_formatter_renders_classes_strings_and_numbers():
    """Unit test of format_default (T-7) on representative values."""
    pytest.importorskip("jax")
    import jax.numpy as jnp

    isolated = _exec_defs(
        "Build_TFP_light.py", {"format_default"}, {"jnp": jnp, "inspect": inspect}
    )
    fmt = isolated["format_default"]

    class Sample:
        pass

    assert fmt(jnp.int32) == "jnp.int32"
    # the rendered default must be valid Python that names the same object
    assert eval(fmt(jnp.int32), {"jnp": jnp}) is jnp.int32
    assert fmt("x") == "'x'", "strings stay quoted"
    assert fmt(1.0) == "1.0" and fmt(0) == "0", "numbers render as is"
    assert fmt(None) == "None"
    assert fmt(int) == "builtins.int", "classes render by qualified name"
    assert fmt(Sample) == f"{Sample.__module__}.{Sample.__qualname__}"


# --------------------------------------------------------------------------
# N-9 — every file access declares utf-8
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", GENERATORS)
def test_no_open_without_encoding(name):
    """RELEASE_AUDIT N-9: open/read_text/write_text must pass encoding='utf-8'."""
    offenders = []
    for node in ast.walk(_tree(name)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        call_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if call_name not in {"open", "read_text", "write_text"}:
            continue
        encoding = next((kw for kw in node.keywords if kw.arg == "encoding"), None)
        if encoding is None or not (
            isinstance(encoding.value, ast.Constant) and encoding.value.value == "utf-8"
        ):
            offenders.append(ast.unparse(node))
    assert not offenders, (
        f"RELEASE_AUDIT N-9: {name} reads or writes text without encoding='utf-8': {offenders}"
    )


# --------------------------------------------------------------------------
# N-10 — notebook markers, stray spaces, documented private-submodule import
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", GENERATORS)
def test_no_notebook_cell_markers(name):
    markers = [line for line in _source(name).splitlines() if re.match(r"^#\s*%%", line)]
    assert not markers, f"RELEASE_AUDIT N-10: {name} still has # %% markers: {markers}"


# Keywords whose name and operand must stay single-spaced in code.
_CONTROL_KEYWORDS = frozenset({
    "if", "elif", "else", "for", "while", "return", "and", "or", "not",
    "in", "is", "with", "lambda", "yield", "assert", "raise", "import",
})


def _stray_space_offenses(source):
    """Token-level check for 'split    (', 'str   (', 'if    mask' style gaps.

    Done on the token stream so the double spaces that live inside the LLM
    prompt strings (prose) are not reported: only real code gaps count.
    """
    offenses = []
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for previous, token in zip(tokens, tokens[1:]):
        if previous.end[0] != token.start[0]:
            continue  # same physical line only
        gap = token.start[1] - previous.end[1]
        if gap < 2:
            continue
        line = source.splitlines()[token.start[0] - 1].strip()
        if token.type == tokenize.OP and token.string in {"(", "["}:
            offenses.append(line)
        elif (
            previous.type == tokenize.NAME
            and previous.string in _CONTROL_KEYWORDS
            and token.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
                                   tokenize.INDENT, tokenize.DEDENT)
        ):
            offenses.append(line)
    return offenses


@pytest.mark.parametrize("name", GENERATORS)
def test_no_stray_spaces_in_code(name):
    offenses = _stray_space_offenses(_source(name))
    assert not offenses, (
        f"RELEASE_AUDIT N-10: {name} has stray spaces in code: {offenses}"
    )


def test_negative_binomial_private_submodule_is_documented():
    """The private-submodule import stays (the class has no public alias in
    the targeted numpyro version) but must say so (RELEASE_AUDIT N-10)."""
    lines = _source("Build_NP_light.py").splitlines()
    hits = [i for i, line in enumerate(lines) if "conjugate.NegativeBinomial" in line]
    assert len(hits) == 1, "the NegativeBinomial collector line must be kept, once"
    context = "\n".join(lines[max(0, hits[0] - 6):hits[0]]).lower()
    assert "#" in context, "RELEASE_AUDIT N-10: the NegativeBinomial line needs a comment"
    assert "private" in context and "public" in context, (
        "RELEASE_AUDIT N-10: the comment must explain that the class lives in "
        "a private submodule because the public alias is missing"
    )


# --------------------------------------------------------------------------
# B-5 — paths resolved from __file__, in the script and in the template
# --------------------------------------------------------------------------

def test_test_file_paths_resolve_from_dunder_file():
    source = _source("Build_test_file.py")
    assert "Path(__file__).resolve()" in source, (
        "RELEASE_AUDIT B-5: paths must come from the script's own location"
    )
    assert "os.path.dirname" not in source and "abspath" not in source, (
        "RELEASE_AUDIT B-5: the old dirname(abspath('')) logic must be gone"
    )
    module_lines = [
        ast.unparse(node)
        for node in _tree("Build_test_file.py").body
        if isinstance(node, ast.Assign)
    ]
    assert any("parents[2]" in line for line in module_lines), (
        "RELEASE_AUDIT B-5: the script sits at <repo>/BayesForge/Build/, so "
        "the repo root is Path(__file__).resolve().parents[2]"
    )


def test_test_file_repo_root_computes_the_repository_root():
    isolated = _exec_defs(
        "Build_test_file.py",
        {"_HERE", "BUILD_DIR", "REPO_ROOT", "TEST_PATH"},
        {"Path": Path},
    )
    assert isolated["REPO_ROOT"] == REPO_ROOT
    assert isolated["BUILD_DIR"] == BUILD_DIR
    assert isolated["TEST_PATH"] == REPO_ROOT / "Test/1.Distributions/test_generated_dists.py"


def test_generated_template_resolves_root_from_its_own_file():
    """The generated test sits at Test/1.Distributions/, two levels below the root."""
    isolated = _exec_defs("Build_test_file.py", {"HEADER"}, {"Path": Path})
    header = isolated["HEADER"]
    assert "Path(__file__).resolve().parents[2]" in header, (
        "RELEASE_AUDIT B-5: the generated file must compute the repo root "
        "from its own __file__, not from the CWD"
    )
    assert "sys.path.insert" in header


# --------------------------------------------------------------------------
# B-6 — dir(m.dist) filter, None docstrings, no JAX at import time
# --------------------------------------------------------------------------

def test_collect_dist_methods_keeps_only_public_callables():
    isolated = _exec_defs(
        "Build_test_file.py",
        {"EXCLUDED", "NON_DISTRIBUTION", "collect_dist_methods"},
    )

    class FakeDist:
        def normal(self, **kwargs):
            pass

        def mask(self, mask):
            pass

        def plate(self, name, shape):
            pass

        def normalize_seed(self, seed):
            pass

        def _helper(self):
            pass

        version = "1.0"

        class Distribution:
            pass

    methods = isolated["collect_dist_methods"](FakeDist())
    assert "normal" in methods
    for name in ("mask", "plate", "normalize_seed", "_helper", "version", "Distribution"):
        assert name not in methods, (
            f"RELEASE_AUDIT B-6: {name!r} is not a distribution factory and "
            "must not be sent to the LLM"
        )


def test_build_examples_skips_methods_without_a_docstring(capsys):
    calls = []

    def fake_llm(doc):
        calls.append(doc)
        return "x = 1"

    isolated = _exec_defs(
        "Build_test_file.py", {"build_examples"}, {"generate_code_example": fake_llm}
    )

    def documented():
        """doc here"""

    def undocumented():
        pass

    undocumented.__doc__ = None
    examples = isolated["build_examples"]({"documented": documented, "undocumented": undocumented})

    assert "documented" in examples and "undocumented" not in examples
    assert calls == ["doc here"], "RELEASE_AUDIT B-6: the LLM must never receive None"
    assert None not in calls
    printed = capsys.readouterr().out
    assert "undocumented" in printed, (
        "RELEASE_AUDIT B-6: a skipped method must be reported, not dropped in silence"
    )


def test_bf_instance_is_created_in_main_only():
    tree = _tree("Build_test_file.py")
    assert "BayesForge" not in _module_level_imports("Build_test_file.py"), (
        "RELEASE_AUDIT B-6: importing the script must not import BayesForge (and JAX)"
    )
    module_calls = [
        ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.Expr)) and isinstance(node.value, ast.Call)
    ]
    assert not any("bf(" in call for call in module_calls), (
        "RELEASE_AUDIT B-6: bf(platform='cpu') must live in main(), not at module level"
    )
    body = _body("Build_test_file.py", "main")
    assert "from BayesForge import bf" in body and "bf(platform='cpu')" in body


def test_importing_build_test_file_does_not_import_jax():
    """B-6 in a fresh interpreter: loading the script must not start JAX.

    The module file is loaded directly from its path: ``import
    BayesForge.Build.Build_test_file`` would first run the package
    ``__init__``, which imports JAX by design (compatibility monkey-patch),
    and that has nothing to do with what this script imports on its own.
    """
    code = (
        "import sys\n"
        "from importlib import util\n"
        f"spec = util.spec_from_file_location('build_test_file', {str(TEST_GEN)!r})\n"
        "module = util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print('jax' in sys.modules)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        if "ollama" in proc.stderr.lower() or "api key" in proc.stderr.lower():
            pytest.skip("Build_test_file needs an LLM at import time on this machine")
        pytest.fail(f"importing Build_test_file.py failed:\n{proc.stderr}")
    assert proc.stdout.strip().splitlines()[-1] == "False", (
        "RELEASE_AUDIT B-6: importing Build_test_file must not import jax"
    )
