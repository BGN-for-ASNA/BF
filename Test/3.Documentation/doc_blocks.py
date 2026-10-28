"""Code blocks of the documentation pages, per language.

Shared by the Python, R and Julia documentation tests (Test/3.Documentation/*/)
and by test_language_coverage.py.

A block is left out of the execution tests when its fence carries the
``.no-test`` class (```{.python .no-test}, ```{.r .no-test},
```{.julia .no-test}) or its code contains ``# doc-test: skip``: install
commands, GPU-only code, placeholders such as ``<your model>``.

Data files that the examples open by name live in ``data/`` next to this
module; ``copy_data`` copies them into the directory a page runs in.
"""
import pathlib
import re
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "Documentation"
START_PAGES = sorted((DOCS / "start").glob("*.qmd"))
_NOT_MODEL_PAGES = {"index", "get started", "0.Introduction"}
MODEL_PAGES = sorted(p for p in DOCS.glob("*.qmd") if p.stem not in _NOT_MODEL_PAGES)

# Data files the examples read from the working directory (e.g. 'BNN.json'):
# copied into the temporary directory each page runs in.
DATA = pathlib.Path(__file__).resolve().parent / "data"


def copy_data(dest):
    """Copy the files of Test/3.Documentation/data into ``dest``."""
    if DATA.is_dir():
        shutil.copytree(DATA, dest, dirs_exist_ok=True)

# Language of a fence info string: ```python, ``` R, ```{r}, ```{.julia .no-test}
LANGS = {"python": "python", "r": "r", "julia": "julia"}
_FENCE_OPEN = re.compile(r"^```+\s*(\{[^}]*\}|[\w.+-]+)?\s*$")
_LANG_OF_INFO = re.compile(r"^\{?\s*\.?([A-Za-z]+)")


def _fence_lang(info):
    """'python' / 'r' / 'julia' for a fence info string, else None."""
    if not info:
        return None
    match = _LANG_OF_INFO.match(info)
    return LANGS.get(match.group(1).lower()) if match else None


def iter_blocks(page):
    """Yield (lang, info, code, line) for every fenced code block of a page.

    ``lang`` is None for blocks that are not Python, R or Julia (shell
    commands, program output, diagrams).
    """
    lines = pathlib.Path(page).read_text().splitlines()
    i = 0
    while i < len(lines):
        match = _FENCE_OPEN.match(lines[i])
        if match:
            info = (match.group(1) or "").strip()
            start = i
            i += 1
            body = []
            while i < len(lines) and not re.match(r"^```+\s*$", lines[i]):
                body.append(lines[i])
                i += 1
            yield _fence_lang(info), info, "\n".join(body), start + 1
        i += 1


def is_tested(info, code):
    return "no-test" not in info and "# doc-test: skip" not in code


# Commands that change the reader's environment: never run them from a test.
_INSTALL = re.compile(r"install\.packages\(|setup_env\(|Pkg\.(add|develop|rm|update)\(|"
                      r"CondaPkg\.(add|rm)|pip install|virtualenv_remove\(|delete_env\(")


def blocks(page, lang):
    """Code of the tested ``lang`` blocks of a page, in page order.

    Raises ValueError when a tested block installs or removes packages: such
    blocks must be fenced with ``.no-test``.
    """
    out = []
    for block_lang, info, code, line in iter_blocks(page):
        if block_lang != lang or not is_tested(info, code):
            continue
        uncommented = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith("#"))
        if _INSTALL.search(uncommented):
            raise ValueError(f"{pathlib.Path(page).name}:{line}: tested {lang} block "
                             "installs or removes packages; fence it with .no-test")
        out.append(code)
    return out


def language_tabs(page):
    """Map each Python / R / Julia block to the language tab it sits in.

    Returns a list of (lang, tab, tabset_line, block_line): ``tab`` is the
    lower-cased heading of the enclosing tab of a ``panel-tabset`` div (None
    when the block is outside any tabset) and ``tabset_line`` the line of that
    tabset (None outside), so blocks can be grouped per tabset.
    """
    lines = pathlib.Path(page).read_text().splitlines()
    out = []
    stack = []          # open divs: [is_tabset, line, current_tab]
    in_code = None
    for n, line in enumerate(lines, 1):
        if in_code is not None:
            if re.match(r"^```+\s*$", line):
                lang, start = in_code
                tabset = next((d for d in reversed(stack) if d[0]), None)
                if lang is not None:
                    out.append((lang, tabset[2] if tabset else None,
                                tabset[1] if tabset else None, start))
                in_code = None
            continue
        fence = _FENCE_OPEN.match(line)
        if fence:
            in_code = (_fence_lang((fence.group(1) or "").strip()), n)
            continue
        if re.match(r"^:::+\s*$", line):
            if stack:
                stack.pop()
        elif re.match(r"^:::+\s*\S", line):
            stack.append([".panel-tabset" in line, n, None])
        elif stack and re.match(r"^##\s+\S", line):
            tabset = next((d for d in reversed(stack) if d[0]), None)
            if tabset is not None:
                tabset[2] = line.lstrip("#").strip().lower()
    return out
