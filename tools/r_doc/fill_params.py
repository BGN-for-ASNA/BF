#!/usr/bin/env python3
"""Deterministic post-processing of the LLM-generated roxygen JSON (R-M3).

The LLM step (see README.md, "Provenance") turned many Google-style parameter
descriptions into the literal text "None" and documented non-distribution
members as distributions. This script repairs the roxygen JSON without any
LLM and without importing BayesForge:

    python tools/r_doc/fill_params.py pythonDoc.json RDoc.json [out.json]

- entries whose key is in NON_DISTRIBUTIONS, or whose value is null, are dropped;
- every "@param <name> <desc>" line whose description is missing is replaced
  by the description parsed from that key's Python docstring in
  pythonDoc.json (Markdown "#### Args:" with "- *name* (type): ..." items,
  Google "Args:", numpy "Parameters", or ":param name:" field lists). A
  description is missing when it is the literal text "None" or only the
  parameter's default value as a bare Python literal (0.0, True, 'marginal',
  an int, ...);
- when no description is found the line becomes "@param <name> (undocumented)"
  for "None", or "@param <name> (undocumented; default <literal>)" when the
  line carried a default value.

out.json defaults to RDoc.json (in place). Pure stdlib.
"""
import ast
import json
import re
import sys

NON_DISTRIBUTIONS = {"biject_to", "mask", "plate", "unit"}

# Section headers that end a parameter section (Google and numpy styles).
_HEADER_RE = re.compile(
    r"^\s*(Args|Arguments|Parameters|Returns|Raises|Yields|Examples?|Notes?"
    r"|References|See Also|Attributes|Warns)\s*:?\s*$"
)
# Google-style entry: "    name (type): description" or "    name: description".
_GOOGLE_PARAM_RE = re.compile(
    r"^(?P<indent>\s+)(?P<name>\w+)\s*(?:\([^)]*\))?\s*:\s+(?P<desc>\S.*)$"
)
# numpy-style entry: "name : type" at column 0, description indented below.
_NUMPY_PARAM_RE = re.compile(r"^(?P<name>\w+)\s*:\s*(?P<type>\S[^:]*)$")
# Markdown docstrings (pythonDoc.json): "#### Args:" heading (any number of
# '#', optional) with "- *name* (type): description" items.
_MD_ARGS_HEADER_RE = re.compile(r"^\s*#*\s*(Args|Arguments)\s*:\s*$")
_MD_ITEM_RE = re.compile(
    r"^\s*-\s+\*(?P<name>\w+)\*\s*(?:\([^)]*\))?\s*:\s*(?P<desc>\S.*)$"
)
# Markdown heading ("#### Returns:"), which ends the Args section.
_MD_HEADING_RE = re.compile(r"^\s*#+\s")


def _strip_fences(docstring):
    """Remove the ```python fence and surrounding triple quotes, if present."""
    text = docstring.strip()
    text = re.sub(r"^```[A-Za-z]*\s*\n", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text)
    text = re.sub(r'^[rR]?"""', "", text.strip())
    text = re.sub(r'"""$', "", text.strip())
    return text.strip("\n")


def _parse_markdown_args(lines):
    """Parse a Markdown "#### Args:" section with "- *name* (type): desc" items.

    The description continues on the following indented lines until the next
    "- *name*" item or the next "#" heading.
    """
    out = {}
    n = len(lines)
    i = 0
    while i < n and not _MD_ARGS_HEADER_RE.match(lines[i]):
        i += 1
    if i >= n:
        return {}
    i += 1
    name = None
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _MD_HEADING_RE.match(line):
            break
        im = _MD_ITEM_RE.match(line)
        if im:
            name = im.group("name")
            out[name] = im.group("desc").strip()
        elif name is not None and line[:1].isspace():
            out[name] += " " + line.strip()
        else:
            break
        i += 1
    return out


def _parse_google(lines):
    """Parse Google-style "Args:" / "Arguments:" sections."""
    out = {}
    i, n = 0, len(lines)
    while i < n:
        if not re.match(r"^\s*(Args|Arguments)\s*:\s*$", lines[i]):
            i += 1
            continue
        i += 1
        base, name = None, None
        while i < n:
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            if _HEADER_RE.match(line) or not line[:1].isspace():
                break
            pm = _GOOGLE_PARAM_RE.match(line)
            if pm and (base is None or len(pm.group("indent")) <= base):
                base = len(pm.group("indent"))
                name = pm.group("name")
                out[name] = pm.group("desc").strip()
            elif name is not None:
                out[name] += " " + line.strip()
            i += 1
    return out


def _parse_numpy(lines):
    """Parse a numpy-style "Parameters" section (name : type, indented text)."""
    out = {}
    n = len(lines)
    i = 0
    while i < n and not re.match(r"^\s*Parameters\s*:?\s*$", lines[i]):
        i += 1
    if i >= n:
        return out
    i += 1
    while i < n and not lines[i].strip():
        i += 1
    if i < n and re.match(r"^\s*-+\s*$", lines[i]):  # ---- underline
        i += 1
    name = None
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if _HEADER_RE.match(line):
            break
        if not line[:1].isspace():
            pm = _NUMPY_PARAM_RE.match(line)
            if not pm:
                break
            name = pm.group("name")
            out.setdefault(name, "")
        elif name is not None:
            out[name] = (out[name] + " " + line.strip()).strip()
        i += 1
    return {k: v for k, v in out.items() if v}


def _parse_param_fields(lines):
    """Parse docutils ":param name: description" field lists."""
    out = {}
    name = None
    for line in lines:
        m = re.match(r"^\s*:param\s+(\w+)\s*:\s*(.*)$", line)
        if m:
            name = m.group(1)
            out[name] = m.group(2).strip()
        elif name is not None and line.strip() and line[:1].isspace():
            out[name] += " " + line.strip()
        elif name is not None and line.strip():
            name = None
    return out


def parse_args_section(docstring):
    """Return {param_name: description} found in a Python docstring.

    Understands Markdown "#### Args:" lists, Google-style
    "Args:"/"Arguments:", numpy-style "Parameters" and docutils
    ":param name:" field lists; returns {} when none is found.
    """
    if not docstring:
        return {}
    lines = _strip_fences(docstring).splitlines()
    for parser in (_parse_markdown_args, _parse_google, _parse_numpy,
                   _parse_param_fields):
        found = parser(lines)
        if found:
            return found
    return {}


def _is_bare_literal(text):
    """True when text is exactly one Python literal with no whitespace
    outside quotes (0.0, True, 'marginal', None, an int): a default value
    written where a description belongs, not a description."""
    text = text.strip()
    if not text:
        return False
    unquoted = re.sub(r"(['\"])(?:\\.|(?!\1).)*\1", "", text)
    if re.search(r"\s", unquoted):
        return False
    try:
        ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False
    return True


def fill(rdoc, pydoc):
    """Repair the roxygen JSON; return (new_rdoc, stats).

    A "@param <name> <desc>" line is missing when <desc> is "None" or a
    bare Python literal (the parameter's default value, see
    _is_bare_literal); such lines are filled from the Python docstring.

    stats keys: entries_dropped, params_filled, params_undocumented.
    """
    stats = {"entries_dropped": 0, "params_filled": 0, "params_undocumented": 0}
    out = {}
    for key, value in rdoc.items():
        if key in NON_DISTRIBUTIONS or value is None:
            stats["entries_dropped"] += 1
            continue
        args = parse_args_section(pydoc.get(key))

        def _replace(match):
            name, desc = match.group(1), match.group(2)
            if not _is_bare_literal(desc):
                return match.group(0)
            py = args.get(name)
            if py:
                stats["params_filled"] += 1
                return "@param %s %s" % (name, py)
            stats["params_undocumented"] += 1
            if desc == "None":
                return "@param %s (undocumented)" % name
            return "@param %s (undocumented; default %s)" % (name, desc)

        out[key] = re.sub(r"@param\s+(\w+)\s+(\S.*?)\s*$", _replace, value,
                          flags=re.MULTILINE)
    return out, stats


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "-h" in argv or "--help" in argv or not 2 <= len(argv) <= 3:
        print(__doc__.strip())
        return 2
    py_path, rdoc_path = argv[0], argv[1]
    out_path = argv[2] if len(argv) > 2 else rdoc_path
    with open(py_path, encoding="utf-8") as f:
        pydoc = json.load(f)
    with open(rdoc_path, encoding="utf-8") as f:
        rdoc = json.load(f)
    filled, stats = fill(rdoc, pydoc)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filled, f, indent=1)
    print("entries dropped: %d" % stats["entries_dropped"])
    print("params filled: %d" % stats["params_filled"])
    print("params still undocumented: %d" % stats["params_undocumented"])
    print("wrote %s (%d entries)" % (out_path, len(filled)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
