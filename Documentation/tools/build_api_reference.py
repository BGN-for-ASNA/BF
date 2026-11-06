"""Generate the API reference pages (Documentation/reference/*.qmd).

The reference is built from the *live* objects a user works with -- ``m``,
``m.dist``, ``m.diag`` (on a fitted model), ``m.effects``, ... -- so methods
attached at run time (most of ``m.diag``) are documented like any other.
Docstrings are parsed with griffe (Google or NumPy style); the Markdown-style
docstrings of the distributions are rendered as written.

Usage (from the repository root):
    python Documentation/tools/build_api_reference.py
"""
import inspect
import logging
import pathlib
import re
import textwrap

import griffe
import jax.numpy as jnp

from BayesForge import bf

logging.getLogger("griffe").setLevel(logging.ERROR)
OUT = pathlib.Path(__file__).resolve().parents[1] / "reference"


def fitted_session():
    m = bf(platform="cpu", rand_seed=0, print_devices_found=False)
    x = jnp.linspace(-1, 1, 20)

    def model(x, y):
        a = m.dist.normal(0, 1, name="a")
        s = m.dist.exponential(1, name="s")
        m.dist.normal(a * x, s, obs=y)

    m.fit(model, obs=dict(x=x, y=2 * x), num_samples=20, num_warmup=20,
          num_chains=1, progress_bar=False)
    return m


# --------------------------------------------------------------- rendering
def docstring_style(doc):
    if re.search(r"^\s*(#{2,}\s*|\*\*)Args", doc, re.M):
        return "markdown"
    if re.search(r"^\s*Parameters\s*\n\s*-{3,}", doc, re.M):
        return "numpy"
    return "google"


def demote_headings(text):
    """Docstring headings would break the page outline: turn them into bold."""
    text = re.sub(r"^(\s*)#{1,6}\s*(.+?):?\s*$", r"\1**\2**", text, flags=re.M)
    return re.sub(r"^(\s*)([^\n#*`-][^\n]*?):?\n\1[=-]{3,}\s*$", r"\1**\2**", text, flags=re.M)


def rst_to_markdown(text):
    """reStructuredText bits used in some docstrings: references and roles."""
    text = re.sub(r"^(\s*)\.\. \[(\d+)\]", r"\1\2.", text, flags=re.M)   # .. [1] -> 1.
    text = re.sub(r":doi:`([^`]+)`", r"doi:[\1](https://doi.org/\1)", text)
    text = re.sub(r":(?:func|class|meth|mod|attr):`~?([^`]+)`", r"`\1`", text)
    return text


def link_definitions_to_links(text):
    """``[1]: url "title"`` definitions clash across functions of one page."""
    return re.sub(r'^(\s*)\[(\d+)\]:\s*(\S+)(?:\s+"([^"]*)")?\s*$',
                  lambda mt: f"{mt.group(1)}- [{(mt.group(4) or mt.group(3)).strip()}]({mt.group(3)})",
                  text, flags=re.M)


def render_docstring(doc):
    doc = demote_headings(link_definitions_to_links(rst_to_markdown(doc)))
    style = docstring_style(doc)
    if style == "markdown":
        return doc
    out = []
    for section in griffe.Docstring(doc, parser=style).parse():
        kind = section.kind.value
        if kind == "text":
            out.append(section.value)
        elif kind in ("parameters", "other parameters"):
            out.append("| Argument | Type | Description |\n|---|---|---|")
            for p in section.value:
                desc = " ".join(str(p.description).split()).replace("|", "\\|")
                ann = str(p.annotation or "").replace("|", "\\|")
                out.append(f"| `{p.name}` | {ann} | {desc} |")
        elif kind in ("returns", "yields"):
            out.append("**Returns**\n")
            for r in section.value:
                ann = f"`{r.annotation}`: " if r.annotation else ""
                out.append(f"- {ann}{' '.join(str(r.description).split())}")
        elif kind == "raises":
            out.append("**Raises**\n")
            for r in section.value:
                out.append(f"- `{r.annotation}`: {' '.join(str(r.description).split())}")
        elif kind == "examples":
            out.append("**Examples**\n")
            for sub_kind, text in section.value:
                if sub_kind.value == "examples":
                    code = "\n".join(re.sub(r"^(>>>|\.\.\.) ?", "", l) for l in text.splitlines())
                    out.append(f"```python\n{code}\n```")
                else:
                    out.append(text)
        elif kind in ("admonition",):
            out.append(f"**{section.title}**\n\n{section.value.description}")
        else:
            out.append(str(getattr(section, "value", "")))
        out.append("")
    return "\n".join(out)


def signature(prefix, name, func):
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return f"{prefix}{name}(...)"
    params = [p for n, p in sig.parameters.items() if n != "self"]
    text = f"{prefix}{name}({', '.join(str(p) for p in params)})"
    return text if len(text) <= 88 else (
        f"{prefix}{name}(\n" + "".join(f"    {p},\n" for p in params) + ")")


# Public names that are implementation details, not user API.
INTERNAL = {"build_model_with_Y_None", "find_obs_in_model", "visit_call", "summary_old",
            "normalize_seed"}


def is_api_function(attr):
    """A BayesForge function or method (not the user's model, not an object)."""
    func = getattr(attr, "__wrapped__", attr)  # jax.jit keeps the original here
    return (callable(attr) and not inspect.isclass(attr)
            and (inspect.isroutine(func) or inspect.isroutine(attr))
            and getattr(func, "__module__", "").startswith("BayesForge"))


def page(slug, title, intro, obj, prefix, include=None, exclude=()):
    names = sorted(
        n for n in dir(obj)
        if not n.startswith("_") and n not in exclude and n not in INTERNAL
        and is_api_function(getattr(obj, n, None))
        and (include is None or include(n, getattr(obj, n)))
    )
    lines = ["---", f'title: "{title}"', "sidebar: api",
             "---", "", "<!-- Generated by tools/build_api_reference.py; do not edit. -->", "",
             intro, ""]
    missing = []
    for n in names:
        func = getattr(obj, n)
        doc = inspect.getdoc(func)
        lines += [f"## `{prefix}{n}` {{#{n}}}", "", "```python", signature(prefix, n, func), "```", ""]
        if doc:
            lines += [render_docstring(doc), ""]
        else:
            missing.append(n)
            lines += ["*Not documented yet.*", ""]
    (OUT / f"{slug}.qmd").write_text("\n".join(lines))
    return len(names), missing


def main():
    OUT.mkdir(exist_ok=True)
    m = fitted_session()
    from BayesForge.Data.manip import manip
    data_methods = {n for n in vars(manip) if not n.startswith("_")}

    pages = [
        ("session", "Session (`m`)", "Methods of the BayesForge object returned by `bf()` (Python), "
         "`importBF()` (R, as `m$method()`) and `importBF()` (Julia).",
         m, "m.", lambda n, f: n not in data_methods, ("dist",)),
        ("data", "Data (`m.data`, `m.scale`, ...)", "Loading and preparing data. `m.df` holds the data "
         "frame; `m.data_on_model` the arrays passed to the model.",
         m, "m.", lambda n, f: n in data_methods, ()),
        ("dist", "Distributions (`m.dist`)", "Every distribution can define a latent parameter "
         "(`name=`), a likelihood (`obs=`), draw samples directly (`sample=True`) or return the "
         "distribution object (`create_obj=True`). In R use `bf.dist.<name>()`.",
         object.__getattribute__(m.dist, "_dist"), "m.dist.", None, ()),
        ("diag", "Diagnostics (`m.diag`)", "Convergence diagnostics, model comparison and posterior "
         "predictive checks. Available after `m.fit()`.",
         m.diag, "m.diag.", None, ()),
        ("effects", "Random effects (`m.effects`)", "Building blocks for varying (random) effects.",
         m.effects, "m.effects.", None, ()),
        ("link", "Link functions (`m.link`)", "Inverse link functions and transforms.",
         m.link, "m.link.", None, ()),
        ("net", "Networks (`m.net`)", "Network models and utilities.",
         m.net, "m.net.", None, ()),
        ("load", "Example data (`m.load`)", "Datasets bundled with BayesForge. "
         "`only_path=True` returns the CSV path instead of a data frame.",
         m.load, "m.load.", None, ()),
    ]
    index = ["---", 'title: "API reference"', "sidebar: api", "---", "",
             "<!-- Generated by tools/build_api_reference.py; do not edit. -->", "",
             "| Section | Functions | Undocumented |", "|---|---|---|"]
    for slug, title, intro, obj, prefix, include, exclude in pages:
        if obj is None:
            continue
        n, missing = page(slug, title, intro, obj, prefix, include, exclude)
        index.append(f"| [{title}]({slug}.qmd) | {n} | {len(missing)} |")
        print(f"{slug:8s} {n:4d} functions, {len(missing)} undocumented"
              + (f": {', '.join(missing)}" if missing else ""))
    (OUT / "index.qmd").write_text("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
