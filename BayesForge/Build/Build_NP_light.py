"""
This file wraps NumPyro distribution functions to enable different calls depending on the provided arguments (sampling, model building, etc.). It also generates the documentation for each distribution function.

The generator writes `dists_generated.py`; it never writes the production `np_dists.py`. The production file is maintained by hand and the generated file is only a starting point to diff against. Known issues that may still need hand-editing after generation:
1. The `mask` argument is present twice in some functions, so it is removed.
2. `*args` and `**kwargs` are not the last arguments in some functions.
3. Some function names are changed due to camel case conversion to snake case (e.g., `l_k_j` -> `lkj`).
4. All distribution functions description and pdf have been modified  to ensure consistency and accuracy.

Importing this module has no side effects: run it as a script to rebuild the docs with the local LLM and write `dists_generated.py`, or call `generate(out_path, docs=...)` directly.

The committed `Doc_python.json` cache is the default documentation source, so a
plain run needs no network and no LLM. Ollama is only reached when `--llm` is
passed explicitly, and then only for the entries the cache does not have yet.
"""

import argparse
import ast
import importlib.metadata
import inspect
import json
import os
import re
from datetime import date
from pathlib import Path

import jax.numpy as jnp
import numpyro
from numpyro.distributions import Distribution

# Where this script lives and the artifacts it owns.
BUILD_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DOC_CACHE_PATH = BUILD_DIR / "Doc_python.json"
PRODUCTION_NP_DISTS = BUILD_DIR.parent / "Distributions/np_dists.py"

# Distributions NumPyro exposes but the generator does not wrap: the
# production file writes them by hand, so they have no cache entry either.
DOC_EXCLUDED = frozenset({"SineBivariateVonMises"})

def collect_dist_classes():
    """Return the NumPyro distribution classes and functions to wrap."""
    # --- Filter for actual, instantiable Distribution classes ---
    dist_classes = {}
    for name in dir(numpyro.distributions):
        obj = getattr(numpyro.distributions, name)
        if (
            inspect.isclass(obj) and issubclass(obj, Distribution) and not name.startswith("_")
            and name not in ["Distribution", "ExpandedDistribution", "TransformedDistribution", "IndependentDistribution", "MaskedDistribution"]
        ):
            dist_classes[name] = obj

    # --- Filter for functions in the distributions module ---
    dist_functions = {}
    for name in dir(numpyro.distributions):
        obj = getattr(numpyro.distributions, name)
        if inspect.isfunction(obj) and not name.startswith("_"):
            dist_functions[name] = obj

    dist_classes = {**dist_classes, **dist_functions}
    # NegativeBinomial is not re-exported from the public numpyro.distributions
    # namespace in the version this generator targets, so the dir() loop above
    # misses it; the class only lives in the private conjugate submodule and
    # must be reached there (RELEASE_AUDIT N-10: keep, do not "clean up").
    dist_classes['NegativeBinomial'] = numpyro.distributions.conjugate.NegativeBinomial
    return dist_classes


def camel_to_snake(name):
    name = name.strip()  # Remove trailing/leading spaces
    s1 = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    return s1.lower()

def generate_doc_string(name, dist_class, name_snake):
    """Ask the local LLM for one docstring. Only reached when `--llm` was passed."""
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise ImportError(
            "Build_NP_light only calls ollama when run with --llm, and the "
            "'ollama' package is not installed. Rebuild the missing docstrings "
            "on a machine with ollama, or add them to Doc_python.json by hand."
        ) from exc

    fn_source_code = inspect.getsource(dist_class)
    prompt = f"""
    As an expert Python programmer specializing in the NumPyro library, your task is to generate a Google-style docstring  for the provided Python function source code. **GENERATE ONLY THE DOCSTRING NOTHING ELSE**.

    Follow these instructions precisely:

    1.  **Analyze the Function**: From the `{fn_source_code}` provided below, identify the statistical distribution it  represents and its required arguments. Note that numpyro use jax.numpy.array.

    2.  **Construct the Docstring**: Generate a complete Google-style docstring with the following sections in this exact   order:

        *   **Summary**: A concise,  summary of what the function does (e.g., "Samples from a [Distribution Name]   distribution.") with general description of the distribution.

        *   **Distribution Formula**: Include the LaTeX formula for the probability density function (PDF) or probability   mass function (PMF) of the distribution, formatted within a `.. math::` block.

        *   **Args**: List and describe all arguments in list format with empty line between each arguments. 

            *   **Distribution Args**: First, list the distribution's own parameters (like `loc` or `scale`). 

            *   **Sampling / Modeling Args**: Then, include the following four standard arguments with their exact descriptions:

                *   `shape` (tuple): A multi-purpose argument for shaping. When `sample=False`  (model building), this is used   with `.expand(shape)` to set the distribution's     batch shape. When `sample=True` (direct sampling), this is    used as `sample_shape`    to draw a raw JAX array of the given shape.

                *   `event` (int): The number of batch dimensions to reinterpret as event dimensions    (used in model building).

                *   `mask` (jnp.ndarray, bool): Optional boolean array to mask observations.

                *   `create_obj` (bool): If True, returns the raw NumPyro distribution object instead of creating a sample  site. This is essential for building complex distributions like `MixtureSameFamily`.

        *   **Returns**: 
           *   **When `sample=False`**: A NumPyro {name} distribution object (for model building).

           *   **When `sample=True`**: A JAX array of samples drawn from the {name} distribution (for direct sampling).

           *   **When `create_obj=True`**: The raw NumPyro distribution object (for advanced use cases).

        *   **Example Usage**: Provide a code block showing a simple, direct sampling call to the function. The structure must be:
                from BayesForge import bf
                m = bf(platform='cpu')
                m.dist.{name_snake}(required_args_with_defaults, sample=True)
 
                - Replace only `required_args_with_defaults` with the distribution's essential parameters set to reasonable  default values (e.g., `loc=0.0, scale=1.0`).

        *   **Wrapper of:**: Add a final note pointing to the official NumPyro documentation, using this exact URL structure:  `https://num.pyro.ai/en/stable/distributions.html#function_name_lowercase`
            - Replace `function_name_lowercase` with the distribution's name in PascalCase (e.g., Normal, Bernoulli).

    3.  **Final Output**: Return **only the complete docstring** and nothing else. Do not include any explanatory text or   the original function code in your response.

    **Function Source Code:**
    ```python
    {fn_source_code}
    ```
        """

    response = ollama.chat(
        model='gemma3:12b',
        messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ],
        options={
        'temperature': 0  # lower = more deterministic
        }
    )
    return response['message']['content']
def build_docs(dist_classes, docs=None):
    """Fill the docstring dict for the distributions ``docs`` does not have yet.

    Calls the local LLM (up to 2h to run for a cold cache), so it is only used
    when ``Build_NP_light.py`` is run with ``--llm``.
    """
    Doc = dict(docs or {})
    for name, dist_class in dist_classes.items():
        if name in Doc:
            continue
        if name in DOC_EXCLUDED:
            continue
        print(f"building doc for {name}")
        doc = generate_doc_string(name, dist_class, camel_to_snake(name))
        Doc[name] = doc
    return Doc


def read_doc_cache(path=DOC_CACHE_PATH):
    """Return the committed docstring cache, or ``{}`` when there is none."""
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_doc_cache(docs, path=DOC_CACHE_PATH):
    """Write the docstring cache next to this script, not in the CWD."""
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(docs, f)
    return path


def production_method_names(path=PRODUCTION_NP_DISTS):
    """Return the ``UnifiedDist`` method names of the production file.

    Parsed with ``ast`` so the production package is never imported. Returns
    ``None`` when the file is unavailable, which disables the coverage check.
    """
    path = Path(path)
    if not path.is_file():
        return None
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "UnifiedDist":
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return None


def required_names(dist_classes, production_methods):
    """NumPyro names the generated file must contain (N-7: no silent skip).

    The production file exposes one factory method per distribution on its
    ``UnifiedDist`` facade, named after the NumPyro class in snake_case. A name
    is therefore required when the production file has that method and NumPyro
    still provides the class. Production methods that no NumPyro class maps to
    (plumbing such as ``normalize_seed``, hand-named ones such as ``car`` or
    ``lkj``) are not required, and NumPyro classes the production file never
    exposed are reported as new instead of required.
    """
    if production_methods is None:
        return set()
    return {name for name in dist_classes if camel_to_snake(name) in production_methods}


def version_of(package):
    """Return an installed package version for the generated header (N-8)."""
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# --- Signature helpers (copied from Build_TFP_light.py: no cross import) ---

def format_default(value):
    """Return valid Python source for a parameter default value."""
    name = getattr(value, "__name__", None)
    if name is not None and hasattr(jnp, name) and getattr(jnp, name) is value:
        return f"jnp.{name}"
    if inspect.isclass(value) or inspect.isfunction(value) or inspect.isbuiltin(value):
        module = getattr(value, "__module__", "") or ""
        qualname = getattr(value, "__qualname__", name)
        if module == "jax.numpy" or module.startswith("jax._src.numpy"):
            return f"jnp.{qualname}"
        return f"{module}.{qualname}"
    return repr(value)


def _param_source(param):
    """Render one non-var parameter as 'name' or 'name=default_source'."""
    if param.default is inspect.Parameter.empty:
        return param.name
    return f"{param.name}={format_default(param.default)}"


def build_signature(parameters, extra):
    """Build a valid wrapper signature from an inspect signature plus extras.

    ``parameters`` is an ``inspect.Signature.parameters`` mapping and
    ``extra`` an ordered dict of name -> default *source string*. Extras
    whose name already exists in ``parameters`` are skipped. Output order:
    positional/keyword params, ``*args``, extras, keyword-only params,
    ``**kwargs``. When there is no ``*args`` but keyword-only params exist,
    the extras go before a bare ``*`` marker.
    """
    positional = []
    keyword_only = []
    var_positional = None
    var_keyword = None
    for param in parameters.values():
        if param.name == "self":
            continue
        if param.kind is param.VAR_POSITIONAL:
            var_positional = f"*{param.name}"
        elif param.kind is param.VAR_KEYWORD:
            var_keyword = f"**{param.name}"
        elif param.kind is param.KEYWORD_ONLY:
            keyword_only.append(_param_source(param))
        else:
            positional.append(_param_source(param))

    existing = set(parameters.keys())
    extras = [
        f"{name}={default}" for name, default in extra.items() if name not in existing
    ]

    parts = list(positional)
    if var_positional is not None:
        parts.append(var_positional)
        parts.extend(extras)
    else:
        parts.extend(extras)
        if keyword_only:
            parts.append("*")
    parts.extend(keyword_only)
    if var_keyword is not None:
        parts.append(var_keyword)
    return ", ".join(parts)


def build_call_args(parameters):
    """Build the ``name=name`` argument string for calling the wrapped class."""
    parts = []
    for param in parameters.values():
        if param.name == "self":
            continue
        if param.kind is param.VAR_POSITIONAL:
            parts.append(f"*{param.name}")
        elif param.kind is param.VAR_KEYWORD:
            parts.append(f"**{param.name}")
        else:
            parts.append(f"{param.name}={param.name}")
    return ", ".join(parts)


def clean_llm_docstring(text):
    """Turn an LLM answer into a valid raw docstring literal."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*\n", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    if text.startswith('r"""'):
        text = text[4:]
    elif text.startswith('"""'):
        text = text[3:]
    if text.endswith('"""'):
        text = text[:-3]
    text = text.replace('"""', '\\"\\"\\"')
    return 'r"""' + text + '\n"""'


# --- Generate the wrapper file ---
def header_comment():
    """Provenance line written at the top of the generated file (N-8)."""
    return (
        f"# Generated by BayesForge/Build/Build_NP_light.py on {date.today().isoformat()}"
        f" with numpyro {version_of('numpyro')} and jax {version_of('jax')};"
        " do not edit by hand.\n"
    )


def check_coverage(dist_classes, generated, skipped, allow_skips):
    """Fail on distributions the generated file would silently lose (N-7)."""
    for name, reason in skipped:
        print(f"Skipped '{name}': {reason}")
    print(f"Generated {len(generated)} of {len(dist_classes)} distributions.")
    missing = sorted(
        required_names(dist_classes, production_method_names()) - generated
    )
    if not missing:
        return
    print("Missing distributions the production file has: " + ", ".join(missing))
    if allow_skips:
        print("--allow-skips: writing the incomplete file anyway.")
        return
    raise ValueError(
        "Build_NP_light: refusing to write a file that loses "
        f"{len(missing)} distributions the production file has "
        f"({', '.join(missing)}). Re-run with --llm to build the missing "
        "docstrings, or --allow-skips to write anyway."
    )


def generate(out_path="dists_generated.py", docs=None, allow_skips=False):
    """Write the wrapper file to ``out_path`` and verify that it compiles.

    ``docs`` maps distribution names to raw LLM docstring text; ``None`` reads
    the committed ``Doc_python.json`` cache. A collected distribution that
    ``docs`` has no entry for is reported, never dropped in silence, and
    ``generate`` raises when that distribution is one the production file
    already has, unless ``allow_skips`` is set (N-7). Never writes a
    production file.
    """
    if os.path.basename(out_path) in {"np_dists.py", "tfp_dists.py"}:
        raise ValueError(f"Refusing to overwrite production file: {out_path}")
    docs = read_doc_cache() if docs is None else docs
    dist_classes = collect_dist_classes()

    wrapper_args = {
        "name": "'x'",
        "obs": "None",
        "mask": "None",
        "sample": "False",
        "seed": "None",
        "shape": "()",
        "event": "0",
        "create_obj": "False",
    }

    methods = []
    generated = set()
    skipped = []
    for name, dist_class in dist_classes.items():
        if name not in docs:
            if name not in DOC_EXCLUDED:
                skipped.append((name, "no docstring in the docs cache"))
            continue
        try:
            signature = inspect.signature(dist_class)
            parameters = signature.parameters
            full_signature = build_signature(parameters, wrapper_args)

            method_name = camel_to_snake(name)
            method_str = f"    @staticmethod\n"
            method_str += f"    def {method_name}({full_signature}):\n"

            full_docstring = clean_llm_docstring(docs[name])
            indented_full_docstring = "\n        ".join(full_docstring.split("\n"))
            method_str += f"        \n        {indented_full_docstring}\n           \n"

            arg_str = build_call_args(parameters)
            method_str += f"        d = {dist_class.__module__}.{dist_class.__name__}({arg_str})\n"

            # The `sample` flag is for direct JAX array sampling, completely separate from model building.
            method_str += f"        if sample:\n"
            method_str += f"            seed_key = random.PRNGKey(0 if seed is None else seed)\n"
            method_str += f"            return d.sample(seed_key, sample_shape=shape)\n"

            # This `else` block handles ALL model-building logic (creating sample sites OR objects).
            method_str += f"        else:\n"

            # --- This is the common logic for modifying the distribution object ---
            # It applies to both `create_obj=True` and `create_obj=False`.
            method_str += f"            if shape:\n"
            method_str += f"                d = d.expand(shape)\n"
            method_str += f"            if event > 0:\n"
            method_str += f"                d = d.to_event(event)\n"
            # --- End of common logic ---

            # --- This is the new switch ---
            # If the user wants the raw object, we return it now.
            method_str += f"            if create_obj:\n"
            method_str += f"                return d\n"

            # Otherwise, we proceed with the original behavior: creating a sample site.
            method_str += f"            else:\n"
            method_str += f"                infer_dict = {{'obs_mask': mask}} if mask is not None else None\n"
            method_str += f"                return numpyro.sample(name, d, obs=obs, infer=infer_dict)\n"

        except (ValueError, TypeError) as e:
            skipped.append((name, f"signature could not be read: {e}"))
            print(f"Could not generate wrapper for '{name}': {e}")
            continue
        methods.append(method_str + "\n")
        generated.add(name)

    check_coverage(dist_classes, generated, skipped, allow_skips)

    # N-9: always utf-8 (the locale codec broke the Unicode/LaTeX docstrings).
    with open(out_path, "w", encoding="utf-8") as file:
        file.write(header_comment())
        file.write("import jax\n")
        file.write("from jax import random\n")
        file.write("import numpyro\n\n")
        file.write("class UnifiedDist:\n\n")
        file.write("    def __init__(self):\n")
        file.write("        pass\n\n")

        file.write("    def mask(self,mask):\n")
        file.write("        return numpyro.handlers.mask(mask=mask)\n\n")

        file.write("    def plate(self,name, shape):\n")
        file.write("        return numpyro.plate(name, shape)\n\n")

        for method_str in methods:
            file.write(method_str)

    with open(out_path, encoding="utf-8") as file:
        compile(file.read(), out_path, "exec")
    print(f"Successfully generated '{out_path}'")
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the BayesForge NumPyro wrapper file.",
    )
    parser.add_argument(
        "--out",
        default="dists_generated.py",
        help="file to write (a production file name is refused)",
    )
    parser.add_argument(
        "--cache",
        default=str(DOC_CACHE_PATH),
        help="docstring JSON cache to read, and to update when --llm is set",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="ask ollama for the docstrings the cache does not have (slow)",
    )
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="write the file even when distributions have no docstring",
    )
    args = parser.parse_args(argv)

    dist_classes = collect_dist_classes()
    docs = read_doc_cache(args.cache)
    if args.llm:
        docs = build_docs(dist_classes, docs)
        write_doc_cache(docs, args.cache)
        print(f"Docstring cache updated: {args.cache}")
    else:
        missing = sorted(set(dist_classes) - set(docs) - DOC_EXCLUDED)
        if missing:
            print(
                f"{len(missing)} distributions have no docstring in {args.cache}; "
                "run with --llm to build them, or --allow-skips to skip them."
            )
    generate(args.out, docs=docs, allow_skips=args.allow_skips)


if __name__ == "__main__":
    main()
