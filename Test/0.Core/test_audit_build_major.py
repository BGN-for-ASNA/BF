"""
Regression tests for RELEASE_AUDIT.md, module Build, majors (static checks: the generators are never executed in tests).

Scope: the majors of the RELEASE_AUDIT.md sections "Module Build" — TFP
(T-4, T-5, T-6), numpyro (N-5, N-6, N-7, N-8) and the test generator
(B-3, B-4).

The generators are code writers: importing them pulls in numpyro, jax and
tensorflow_probability, and running them overwrites production files and calls
a local LLM. Both are forbidden here, so every check inspects the source with
``ast`` (never a line-number snapshot) or execs a single pure helper in
isolation. What the audit asked for and what these tests pin:

* T-4  the collector keeps ``tfd`` distribution classes only, never ``dir(tfd)``;
* T-5  a method name defined twice raises ``ValueError`` before anything is written;
* T-6  the wrapper's ``seed`` defaults to ``None``, not to a fixed key 0;
* N-5  the generator runs without ollama: the committed doc cache is the default
       source, ollama is imported lazily, and a missing class is reported
       instead of silently dropped;
* N-6  the generated file is compile-checked before it is written;
* N-7  the required set is the production ``UnifiedDist`` method list, read with
       ``ast`` (the module is never imported), and a name that would be lost
       stops the run instead of being skipped;
* N-8  the generated header records the generator, the date and the installed
       numpyro / jax / TFP versions instead of hardcoded strings;
* B-3  the test generator resolves its output from its own location and writes
       into ``Test/1.Distributions``;
* B-4  every generated example is executed by a test that asserts on it, and the
       file says plainly that these are smoke tests, not behavioural validation.
"""

import ast
import json
import re
import textwrap
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
BUILD_DIR = REPO_ROOT / "BayesForge" / "Build"

NP_GEN = BUILD_DIR / "Build_NP_light.py"
TFP_GEN = BUILD_DIR / "Build_TFP_light.py"
TEST_GEN = BUILD_DIR / "Build_test_file.py"
DOC_CACHE = BUILD_DIR / "Doc_python.json"
PRODUCTION_NP = REPO_ROOT / "BayesForge" / "Distributions" / "np_dists.py"

GENERATORS = ("Build_NP_light.py", "Build_TFP_light.py", "Build_test_file.py")

# The generators import numpyro, tensorflow_probability and ollama, so they are
# never imported here: parsing the source keeps these tests offline and safe.
SOURCES = {path.name: path.read_text() for path in (NP_GEN, TFP_GEN, TEST_GEN)}
TREES = {name: ast.parse(src) for name, src in SOURCES.items()}

# GET SOURCE CODE
def _source(name):
    return SOURCES[name]

# GET TREE
def _tree(name):
    return TREES[name]

# GET FUNCTIONS FROM TREE
def _functions(name):
    return {node.name: node for node in ast.walk(_tree(name)) if isinstance(node, ast.FunctionDef)}

# GET BODY OF FUNCTION
def _body(name, func):
    node = _functions(name).get(func)
    assert node is not None, f"{name} defines no function {func}()"
    return ast.get_source_segment(_source(name), node) or ""

# GET MODULE MODULES IMPORTED
def _module_level_imports(name):
    modules = set()
    for node in _tree(name).body:
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules

# GET STRING CONSTANTS FROM EACH NODES
def _string_constants(node):
    return {
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    }

# GET EXPLICIT WRITTEN PATHS
def _written_paths(name):
    """Paths handed to an ``open(..., "w")`` call, as written in the source."""
    paths = set()
    for node in ast.walk(_tree(name)):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "open" or len(node.args) < 2:
            continue
        mode = node.args[1]
        if not (isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "w" in mode.value):
            continue
        target = node.args[0]
        if isinstance(target, ast.Name):
            paths.add(target.id)
    return paths

# CHECK WETHER A FUNCTION RAISES A VALUE ERROR 
def _raises_value_error(name, func, needle):
    """True when ``func`` raises ValueError with a message containing ``needle``."""
    node = _functions(name).get(func)
    assert node is not None, f"{name} defines no function {func}()"
    for child in ast.walk(node):
        if not isinstance(child, ast.Raise) or not isinstance(child.exc, ast.Call):
            continue
        if not isinstance(child.exc.func, ast.Name) or child.exc.func.id != "ValueError":
            continue
        if needle in ast.unparse(child):
            return True
    return False

# EXECUTE A GIVEN MODULE ISOLATED
def _run_isolated(name, wanted, extra_globals=None):
    """Exec selected top-level definitions of a generator, without importing it.

    Only pure helpers are reachable this way: the module is never executed, so
    numpyro, tensorflow_probability and ollama stay untouched. The generator's
    own module-level imports are not executed either, so the small set of
    stdlib names they provide (re, textwrap, json, Path, ast) is injected here.
    """
    source = _source(name)
    tree = _tree(name)
    chunks = []
    for node in tree.body:
        key = getattr(node, "name", None)
        if key is None and isinstance(node, ast.Assign):
            targets = node.targets
            key = targets[0].id if len(targets) == 1 and isinstance(targets[0], ast.Name) else None
        if key in wanted:
            chunks.append(ast.get_source_segment(source, node))
    found = {
        getattr(node, "name", None)
        for node in tree.body
        if getattr(node, "name", None) in wanted
    }
    found |= {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in wanted
    }
    assert found == wanted, f"{name} defines none of {sorted(wanted - found)}"
    # The exec'd chunks are module-level code: without __name__ they would run
    # the generator's argparse main(), and without __file__ its paths would
    # resolve against this test file.
    isolated_globals = {
        "__name__": f"{name.rsplit('.', 1)[0]}_isolated",
        "__file__": str(BUILD_DIR / name),
    }
    isolated_globals.update({"re": re, "textwrap": textwrap, "json": json, "Path": Path, "ast": ast})
    isolated_globals.update(extra_globals or {})
    exec("\n".join(chunk for chunk in chunks if chunk), isolated_globals)
    return isolated_globals


# --------------------------------------------------------------------------
# T-4 — the TFP collector keeps tfd distribution classes only
# --------------------------------------------------------------------------
# ENSURE TFP COLLECTS CLASSES ONLY
def test_tfp_light_collects_classes_not_dir_entries():
    """``dir(tfd)`` is only a starting point: the loop must keep classes only."""
    source = _source("Build_TFP_light.py")
    loops = [
        node
        for node in ast.walk(_tree("Build_TFP_light.py"))
        if isinstance(node, ast.For) and "dir(tfd)" in ast.unparse(node.iter)
    ]
    assert loops, "the collector no longer walks dir(tfd)"
    kept = False
    for loop in loops:
        body = "\n".join(ast.unparse(child) for child in loop.body)
        if "issubclass" in body and "tfd.Distribution" in body and "inspect.isclass" in body:
            kept = True
    assert kept, (
        "RELEASE_AUDIT T-4: every dir(tfd) entry became a wrapper; the loop must keep "
        "tfd.Distribution subclasses only"
    )
    # A comprehension over dir(tfd) would be the unfiltered collector itself.
    assert not any(
        isinstance(node, ast.comprehension) and "dir(tfd)" in ast.unparse(node.iter)
        for node in ast.walk(_tree("Build_TFP_light.py"))
    ), "RELEASE_AUDIT T-4: unfiltered dict comprehension over dir(tfd)"

# TEST TFP COLLECTOR SKIPS PRIVATE AND BASE CLASS
def test_tfp_light_collector_skips_private_and_base_class():
    body = _body("Build_TFP_light.py", "generate")
    assert 'name.startswith("_")' in body, "private aliases must not become wrappers"
    assert "obj is not tfd.Distribution" in body, "the base class must not be wrapped"


# --------------------------------------------------------------------------
# T-5 — a duplicated method name is an error, not a silent overwrite
# --------------------------------------------------------------------------
# TEST TFP COLLECTOR DETECTS DUPLICATE METHOD NAMES
def test_tfp_light_detects_duplicate_method_names():
    isolated = _run_isolated("Build_TFP_light.py", {"duplicate_definitions"})
    duplicates = isolated["duplicate_definitions"](
        [("beta", "src1", "Beta"), ("gamma", "src2", "Gamma"), ("beta", "src3", "Beta2")]
    )
    assert set(duplicates) == {"beta"}, "a name defined twice must be reported"
    assert duplicates["beta"] == ("Beta", "Beta2"), (
        "the report must name both classes so the rename is actionable"
    )
    assert isolated["duplicate_definitions"]([("beta", "s", "Beta")]) == {}

# TEST TFP COLLECTOR RAISES VALUE ERROR ON DUPLICATE METHOD NAMES
def test_tfp_light_refuses_to_write_a_duplicated_name():
    assert _raises_value_error("Build_TFP_light.py", "generate", "defined twice"), (
        "RELEASE_AUDIT T-5: a duplicated name must raise ValueError before writing"
    )

# TEST TFP COLLECTOR CHECKS DUPLICATES BEFORE OPENING THE OUTPUT
def test_tfp_light_checks_duplicates_before_opening_the_output():
    """The guard has to run before the file is opened: a half-written file is the bug."""
    body = _body("Build_TFP_light.py", "generate")
    guard = body.find("duplicate_definitions(")
    write = body.find('open(out_path, "w"')
    assert guard != -1 and write != -1, "no duplicate guard, or no output write"
    assert guard < write, "the duplicate guard must run before the output is opened"


# --------------------------------------------------------------------------
# T-6 — seed=None, never a fixed key 0
# --------------------------------------------------------------------------
# TEST THAT DEFAULT ARGUMENT FOR SEED IS NONE
@pytest.mark.parametrize("name", ["Build_NP_light.py", "Build_TFP_light.py"])
def test_wrapper_seed_defaults_to_none(name):
    source = _source(name)
    assert '"seed": "None"' in source, (
        "RELEASE_AUDIT T-6: the wrapper's seed argument must default to None"
    )
    assert '"seed": "0"' not in source

# TEST THAT GENERATED SOURCE DOES NOT HAVE FIXED PRNG KEY
@pytest.mark.parametrize("name", ["Build_NP_light.py", "Build_TFP_light.py"])
def test_no_fixed_prng_key_in_generated_source(name):
    """A literal key 0 in the emitted source would make every call identical."""
    offenders = []
    for node in ast.walk(_tree(name)):
        if not isinstance(node, ast.JoinedStr):
            continue
        text = ast.unparse(node)
        if "PRNGKey" in text and re.search(r"PRNGKey\(\s*0\s*\)", text):
            offenders.append(text[:80])
    assert not offenders, f"RELEASE_AUDIT T-6: fixed key 0 in {name}: {offenders}"


# --------------------------------------------------------------------------
# N-5 — the generators run without ollama
# --------------------------------------------------------------------------
# TEST THAT NP LIGHT HAS NO TOP LEVEL OLLAMA IMPORT
def test_np_light_has_no_top_level_ollama_import():
    modules = _module_level_imports("Build_NP_light.py")
    assert "ollama" not in modules, (
        "RELEASE_AUDIT N-5: ollama must be imported lazily, not at module level"
    )

# TEST THAT NP LIGHT IMPORTS OLLAMA LAZILY WITH AN ACTIONABLE MESSAGE
def test_np_light_imports_ollama_lazily_with_an_actionable_message():
    body = _body("Build_NP_light.py", "generate_doc_string")
    assert "import ollama" in body, "the LLM path must import ollama where it is used"
    assert "ImportError" in body, (
        "RELEASE_AUDIT N-5: a missing ollama must say how to proceed (--cache)"
    )

# TEST THAT NP LIGHT READS THE COMMITTED DOC CACHE
def test_np_light_reads_the_committed_doc_cache():
    """The committed cache is the default: a run needs no LLM at all."""
    assert DOC_CACHE.is_file(), f"{DOC_CACHE} is the offline doc source and must exist"
    cache = json.loads(DOC_CACHE.read_text())
    assert isinstance(cache, dict) and cache, "the doc cache must be a non-empty mapping"
    assert all(isinstance(value, str) for value in cache.values())

    body = _body("Build_NP_light.py", "generate")
    assert "read_doc_cache" in body, "generate() must fall back on the committed cache"
    assert "write_doc_cache" in _source("Build_NP_light.py"), (
        "the LLM path must be able to refresh the cache"
    )

# TEST THAT NP LIGHT DOC CACHE PATH IS ABSOLUTE
def test_np_light_doc_cache_path_is_absolute():
    source = _source("Build_NP_light.py")
    assert "DOC_CACHE_PATH" in source, "the cache path must be a module constant"
    assert "__file__" in source and "resolve()" in source, (
        "RELEASE_AUDIT N-5: the cache must be found from the script's own location"
    )
    assert "Doc_python.json" in source

# TEST THAT NP LIGHT REPORTS SKIPPED CLASSES
def test_np_light_reports_skipped_classes():
    body = _body("Build_NP_light.py", "check_coverage")
    assert re.search(r"\bprint\(", body), (
        "RELEASE_AUDIT N-5: skipped classes must be reported, not silently dropped"
    )


# --------------------------------------------------------------------------
# N-6 — the generated file is compile-checked before it is written
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["Build_NP_light.py", "Build_TFP_light.py"])
def test_generated_source_is_compile_checked(name):
    source = _source(name)
    assert re.search(r"\bcompile\(", source), (
        f"RELEASE_AUDIT N-6: {name} must compile() the generated source before writing"
    )
    assert _written_paths(name), f"{name} writes no output file"


# --------------------------------------------------------------------------
# N-7 — the required set is the production method list, not a hardcoded one
# --------------------------------------------------------------------------

def test_np_light_required_set_comes_from_production():
    source = _source("Build_NP_light.py")
    assert "np_dists.py" in source, (
        "RELEASE_AUDIT N-7: the required set must be read from the production facade"
    )
    assert "REQUIRED_EXCLUDED" not in source, (
        "RELEASE_AUDIT N-7: a hardcoded exclusion list is the bug being fixed"
    )


def test_np_light_reads_production_without_importing_it():
    body = _body("Build_NP_light.py", "production_method_names")
    assert "ast" in body, "the production facade must be parsed, never imported"
    assert "import numpyro" not in body and "from BayesForge" not in body


def test_production_method_names_finds_the_facade():
    """Runs the real helper: it must return the UnifiedDist methods of the repo."""
    isolated = _run_isolated("Build_NP_light.py", {"BUILD_DIR", "PRODUCTION_NP_DISTS", "production_method_names"})
    names = isolated["production_method_names"](str(PRODUCTION_NP))
    assert {"normal", "beta", "gamma", "bernoulli"} <= names, sorted(names)[:10]
    assert all(isinstance(name, str) for name in names)
    assert len(names) > 50, f"only {len(names)} methods parsed: the facade was not found"


def test_required_names_is_the_intersection_with_production():
    """A class the production facade does not expose is not required."""
    isolated = _run_isolated(
        "Build_NP_light.py",
        {"BUILD_DIR", "PRODUCTION_NP_DISTS", "camel_to_snake", "production_method_names", "required_names"},
    )
    production = isolated["production_method_names"]()
    collected = {"Normal", "Beta", "SineSkewed", "Dagum", "BetaNegativeBinomial"}
    assert isolated["camel_to_snake"]("BetaNegativeBinomial") not in production
    assert isolated["camel_to_snake"]("Dagum") not in production
    required = isolated["required_names"](collected, production)
    assert {"Normal", "Beta", "SineSkewed"} <= set(required), sorted(required)
    assert not ({"Dagum", "BetaNegativeBinomial"} & set(required)), (
        "RELEASE_AUDIT N-7: only production names are required; new numpyro classes "
        "may be skipped, they are reported by check_coverage"
    )


def test_np_light_missing_required_name_raises_value_error():
    assert _raises_value_error("Build_NP_light.py", "check_coverage", "refusing to write"), (
        "RELEASE_AUDIT N-7: losing a production name must raise ValueError"
    )
    body = _body("Build_NP_light.py", "check_coverage")
    assert "allow_skips" in body, "--allow-skips must stay available as the escape hatch"


def test_np_light_coverage_runs_before_the_output_is_written():
    body = _body("Build_NP_light.py", "generate")
    guard = body.find("check_coverage(")
    write = body.find('open(out_path, "w"')
    assert guard != -1 and write != -1, "no coverage guard, or no output write"
    assert guard < write, "the coverage guard must run before the output is opened"


# --------------------------------------------------------------------------
# N-8 — provenance: generator, date and installed versions
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name", ["Build_NP_light.py", "Build_TFP_light.py", "Build_test_file.py"]
)
def test_generated_files_name_their_generator(name):
    """Every generated file says who wrote it, so a stale file is traceable."""
    literals = _header_texts(_tree(name))
    assert any("Build_" in text and "do not edit" in text for text in literals), (
        f"RELEASE_AUDIT N-8: {name} emits no header naming its generator"
    )


@pytest.mark.parametrize("name", ["Build_NP_light.py", "Build_TFP_light.py"])
def test_generators_read_installed_versions_from_the_metadata(name):
    source = _source(name)
    assert "importlib.metadata" in source, (
        f"RELEASE_AUDIT N-8: {name} must not hardcode dependency versions"
    )
    assert re.search(r"\bversion\(", source), "no package version is recorded"


@pytest.mark.parametrize(
    "name,package",
    [
        ("Build_NP_light.py", "numpyro"),
        ("Build_TFP_light.py", "tensorflow_probability"),
    ],
)
def test_generated_header_records_the_backend_version(name, package):
    source = _source(name)
    assert re.search(rf"{package}.*version_of|version_of\(['\"]{package}", source) or (
        package in source and "_version" in source
    ), f"RELEASE_AUDIT N-8: {name} does not record the {package} version"


def test_np_light_generated_header_names_its_inputs():
    body = _body("Build_NP_light.py", "header_comment")
    assert "Build_NP_light.py" in body, "the generated file must name its generator"
    assert "date.today()" in body, "the generated file must record when it was built"
    assert "numpyro" in body and "jax" in body


def _header_texts(tree):
    """Every string the module can emit, f-strings included, as unparsed text.

    ``{expression}`` parts are kept: the provenance line builds its date and
    versions that way, so dropping them would hide the thing being checked.
    """
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            out.append(ast.unparse(node))
    return out


def _provenance_line(name):
    """The f-string the generator writes first, as one unparsed string.

    The provenance header is not a module-level statement: in the TFP generator
    it is the first ``f.write(...)`` inside the ``with open(out_path, "w")``
    block, so the whole JoinedStr is unparsed rather than looked up as a
    top-level node.
    """
    for text in _header_texts(_tree(name)):
        if "Generated by" in text:
            return text
    return ""


def test_tfp_generated_header_names_its_inputs():
    header = _provenance_line("Build_TFP_light.py")
    assert "Build_TFP_light.py" in header, (
        "RELEASE_AUDIT N-8: the generated file must name its generator"
    )
    assert "tensorflow_probability" in header, "the TFP version must be recorded"
    assert "jax" in header, "the jax version must be recorded"
    assert "date" in header, "the build date must be recorded"


# --------------------------------------------------------------------------
# B-3 — the test generator writes into the test tree it lives next to
# --------------------------------------------------------------------------

def test_test_file_targets_the_test_tree():
    source = _source("Build_test_file.py")
    assert "Test/1.Distributions" in source, (
        "RELEASE_AUDIT B-3: the generated tests must live in Test/1.Distributions"
    )
    assert not re.search(r'"\s*Test/test_generated_dists', source)
    assert "Path(__file__).resolve().parents[" in source, (
        "RELEASE_AUDIT B-3: the output path must come from the script's own location"
    )


def test_test_file_sys_path_points_at_the_repo_root():
    """The generated test sits two directories below the repo root."""
    isolated = _run_isolated("Build_test_file.py", {"HEADER"})
    header = isolated["HEADER"]
    assert "parents[2]" in header, (
        "RELEASE_AUDIT B-3: Test/1.Distributions/x.py needs parents[2] to reach the root"
    )
    assert "sys.path" in header


def test_test_file_creates_its_output_directory():
    body = _body("Build_test_file.py", "write_test_file")
    assert "mkdir" in body, "the target directory must be created before the write"


# --------------------------------------------------------------------------
# B-4 — the generated tests run the examples and say what they prove
# --------------------------------------------------------------------------

def test_test_file_compiles_before_writing():
    body = _body("Build_test_file.py", "write_test_file")
    assert re.search(r"\bcompile\(", body), (
        "RELEASE_AUDIT B-4: the generated test file must be compile-checked"
    )


def test_test_file_declares_what_it_does_not_prove():
    source = _source("Build_test_file.py")
    assert re.search(r"smoke test", source, re.IGNORECASE), (
        "RELEASE_AUDIT B-4: the file must say these are smoke tests"
    )
    assert re.search(r"not.{0,40}behavioural|not.{0,40}behavioral", source, re.IGNORECASE), (
        "RELEASE_AUDIT B-4: the file must say it does not validate behaviour"
    )


def test_generated_test_file_asserts_on_every_example():
    """One test per example, named after it, asserting the example ran."""
    isolated = _run_isolated("Build_test_file.py", {"HEADER", "_literal", "build_test_source"})
    examples = {
        "normal": "d = bf.UnifiedDist.normal(shape=(3,), seed=1)",
        "beta": "d = bf.UnifiedDist.beta(0.5, 0.5, shape=(2,), seed=1)",
    }
    source = isolated["build_test_source"](examples)
    compile(source, "generated_tests.py", "exec")

    for name in examples:
        assert f"def test_{name}()" in source, f"no test for the {name!r} example"
        assert f'assert "{name}" in example' in source, (
            f"RELEASE_AUDIT B-4: test_{name} must assert on its own example"
        )
        assert f"_run_example('{name}', example)" in source, (
            f"RELEASE_AUDIT B-4: test_{name} must run its example, not quote it"
        )
    assert "bf.UnifiedDist.normal(shape=(3,), seed=1)" in source


def test_generated_test_file_drops_markdown_fences():
    """A fenced example would not compile; the renderer must strip the fences."""
    isolated = _run_isolated("Build_test_file.py", {"HEADER", "_literal", "build_test_source"})
    source = isolated["build_test_source"]({"normal": "```python\nx = 1\n```"})
    assert "```" not in source
    compile(source, "generated_tests.py", "exec")


def test_generated_example_runner_checks_what_the_example_produced():
    """The runner cannot assert a value, so it must at least prove the call returned."""
    isolated = _run_isolated("Build_test_file.py", {"HEADER", "_literal", "build_test_source"})
    source = isolated["build_test_source"]({"normal": "d = 1"})
    tree = ast.parse(source)
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_example"
    )
    asserts = [ast.unparse(node) for node in ast.walk(runner) if isinstance(node, ast.Assert)]
    assert asserts, "RELEASE_AUDIT B-4: _run_example asserts nothing about the example"
    assert not any(text.startswith("assert True") for text in asserts), asserts
    assert any("defined" in text for text in asserts), (
        f"RELEASE_AUDIT B-4: an example that defines nothing must fail: {asserts}"
    )


def test_generated_example_runner_rejects_a_none_binding():
    isolated = _run_isolated("Build_test_file.py", {"HEADER", "_literal", "build_test_source"})
    source = isolated["build_test_source"]({"normal": "d = 1"})
    tree = ast.parse(source)
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_example"
    )
    asserts = [ast.unparse(node) for node in ast.walk(runner) if isinstance(node, ast.Assert)]
    assert any("is not None" in text for text in asserts), (
        f"RELEASE_AUDIT B-4: a name bound to None must fail the test: {asserts}"
    )


# --------------------------------------------------------------------------
# the generators themselves stay runnable
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", GENERATORS)
def test_generator_is_valid_python(name):
    compile(_source(name), str(BUILD_DIR / name), "exec")


@pytest.mark.parametrize("name", GENERATORS)
def test_generator_has_a_main_guard(name):
    source = _source(name)
    assert 'if __name__ == "__main__":' in source, (
        f"RELEASE_AUDIT B-3: {name} must be runnable as a script"
    )


def test_np_light_cli_options_document_the_offline_path():
    source = _source("Build_NP_light.py")
    for option in ("--cache", "--llm", "--allow-skips"):
        assert option in source, f"{option} is missing from the CLI"


def test_np_light_generate_accepts_injected_docs():
    """The critical test injects docs; the signature must keep accepting them."""
    isolated = _run_isolated("Build_NP_light.py", {"camel_to_snake"})
    tree = _tree("Build_NP_light.py")
    generate = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "generate"
    )
    args = [arg.arg for arg in generate.args.args] + [arg.arg for arg in generate.args.kwonlyargs]
    assert "out_path" in args and "docs" in args, args
    assert isolated["camel_to_snake"]("BetaProportion") == "beta_proportion"
