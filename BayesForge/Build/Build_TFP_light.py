"""
Development-only generator that wraps TFP distribution classes to enable
different calls depending on the provided arguments.

Running this file as a script writes ``tfp_dists_generated.py`` (or the
``out_path`` given to :func:`generate`). It never writes the production
``tfp_dists.py``: that file is maintained by hand and the generated file is
only a starting point to diff against. Importing this module has no side
effects.
"""

import datetime
import inspect
import os
import re
from importlib.metadata import PackageNotFoundError, version

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

PRODUCTION_FILES = {"tfp_dists.py", "np_dists.py"}


def _version(package):
    """Installed version of ``package``, or ``"unknown"`` when it is not installed.

    ``importlib.metadata`` reads the installed distribution metadata instead of a
    ``__version__`` attribute the package may not expose (TFP does not).
    """
    try:
        return version(package)
    except PackageNotFoundError:
        return "unknown"


def duplicate_definitions(methods):
    """Return ``{method_name: (first_key, second_key)}`` for names defined twice.

    ``methods`` is the list of ``(method_name, source, key)`` tuples the generator is
    about to write. Two distribution classes can map to the same snake_case method name;
    the second ``def`` would silently overwrite the first one in the generated file, so
    the caller refuses to write and asks for a rename instead.
    """
    seen = {}
    duplicates = {}
    for method_name, _source, key in methods:
        if method_name in seen:
            duplicates[method_name] = (seen[method_name], key)
        else:
            seen[method_name] = key
    return duplicates


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


def generate(out_path="tfp_dists_generated.py"):
    """Write the wrapper file to ``out_path`` and verify that it compiles."""
    if os.path.basename(out_path) in PRODUCTION_FILES:
        raise ValueError(f"Refusing to overwrite production file: {out_path}")

    class_dict = {}
    for name in dir(tfd):
        obj = getattr(tfd, name)
        if (
            not name.startswith("_")
            and inspect.isclass(obj)
            and issubclass(obj, tfd.Distribution)
            and obj is not tfd.Distribution
        ):
            class_dict[name] = obj

    wrapper_args = {
        "mask": "None",
        "shape": "()",
        "sample": "False",
        "seed": "None",
        "obs": "None",
        "wrap": "True",
        "create_obj": "False",
    }

    methods = []
    for key, value in class_dict.items():
        try:
            # Use inspect to get the signature of the distribution's __init__ method
            signature = inspect.signature(value)
            parameters = signature.parameters
            full_signature = build_signature(parameters, wrapper_args)

            # Create the method definition string
            method_name = re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()  # Convert CamelCase to snake_case
            method_str = "    @staticmethod\n"
            method_str += f"    def {method_name}({full_signature}):\n"

            # Create a rich docstring
            docstring = f"Wrapper for the tfd.{value.__name__} distribution.\n\n"
            docstring += "    This method can either return a distribution object for use in a\n"
            docstring += "    probabilistic model or directly return samples.\n\n"
            docstring += "    Original TFP Arguments:\n"
            for param in parameters.values():
                if param.name != "self":
                    if param.default is inspect.Parameter.empty:
                        default = "no default"
                    else:
                        default = format_default(param.default)
                    docstring += f"        {param.name}: Default = {default}\n"
            docstring += "\n    Wrapper Arguments:\n"
            docstring += "        shape (tuple): The shape of the sample. Defaults to ().\n"
            docstring += "        sample (bool): If True, draw samples from the distribution. \n"
            docstring += "                       If False, return a Root distribution object. Defaults to False.\n"
            docstring += "        seed (int, optional): PRNG seed for sampling. Defaults to None (session stream).\n"
            docstring += "        mask (array, optional): Entries where mask is False add 0 to the log density (tfd.Masked), like numpyro.handlers.mask.\n"
            docstring += "        obs (array, optional): The value itself is pinned by the sampler (last yield of the model); here it sets the shape, broadcast to the batch shape of obs (its shape minus the event dims), and those batch dims become one event.\n"
            docstring += "        create_obj (bool): If True, return the bare TFP distribution (to build mixtures). Defaults to False.\n"

            # Format and indent the docstring
            indented_docstring = "\n        ".join(docstring.splitlines())
            method_str += f'        """\n        {indented_docstring}\n        """\n'

            # Create the argument string for instantiating the TFP distribution
            # This correctly forwards all the original arguments
            arg_str = build_call_args(parameters)

            # Add the method body
            method_str += f"        dist = tfd.{value.__name__}({arg_str})\n"
            # mask/name only reach _finish when they are facade extras: a
            # class with its own mask (HiddenMarkovModel, LGSSM) keeps it
            # for the TFP constructor, like the hand-converted tfp_dists.py.
            m_arg = "None" if "mask" in parameters else "mask"
            n_arg = "name" if "name" in parameters else "None"
            method_str += f"        return UnifiedDist._finish(dist, shape=shape, sample=sample, seed=seed, obs=obs, wrap=wrap, mask={m_arg}, create_obj=create_obj, name={n_arg})\n"

            methods.append((method_name, method_str, key))
        except (ValueError, TypeError) as e:
            # Some objects in tfp.distributions might not be classes or inspectable
            print(f"Skipping {key} due to inspection error: {e}")

    # A name defined twice would silently keep the last wrapper, so stop before
    # anything is written rather than shipping a file that lost a distribution.
    duplicates = duplicate_definitions(methods)
    if duplicates:
        name, (first, second) = sorted(duplicates.items())[0]
        raise ValueError(
            f"Build_TFP_light: generated name {name!r} defined twice "
            f"({first} and {second}); rename before writing"
        )

    # N-9: always utf-8 (the locale codec broke the Unicode/LaTeX docstrings).
    with open(out_path, "w", encoding="utf-8") as file:
        # Write the header and the import statements
        file.write(f"# Generated by BayesForge/Build/Build_TFP_light.py on {datetime.date.today().isoformat()} with tensorflow_probability {_version('tensorflow_probability')} and jax {_version('jax')}; do not edit by hand.\n")
        file.write("import warnings\n")
        file.write("import jax\n")
        # Some TFP defaults are rendered with their full module path.
        file.write("import tensorflow_probability\n")
        file.write("import jax.numpy as jnp\n")
        file.write("from tensorflow_probability.substrates import jax as tfp\n")
        file.write("import tensorflow_probability.substrates.jax.distributions as tfd\n\n")
        file.write("tfb = tfp.bijectors\n")
        file.write("root = tfd.JointDistributionCoroutine.Root\n\n")

        # Write the class definition
        file.write("class UnifiedDist:\n")
        file.write('    """A class that wraps TFP distributions for a unified sampling interface."""\n\n')
        # Shared helpers of the hand-maintained wrapper (session RNG key and
        # the common tail every generated method returns through).
        from BayesForge.Distributions.tfp_dists import UnifiedDist as _Ref
        file.write("    _rng = None\n\n")
        for _helper in ("_draw_key", "_finish"):
            file.write(inspect.getsource(_Ref.__dict__[_helper].__func__) + "\n")

        # Write the generated methods
        for _method_name, method_str, _key in methods:
            file.write(method_str + "\n\n")

    with open(out_path, encoding="utf-8") as file:
        compile(file.read(), out_path, "exec")
    print(f"Successfully generated {out_path}")
    return out_path


if __name__ == "__main__":
    generate()
