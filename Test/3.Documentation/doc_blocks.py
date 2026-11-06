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
# "Advance" pages of the Get started sidebar (API reference with examples).
# Sync-conflict copies ("<page> (conflicted copy ...).qmd") are not pages.
ADVANCE_PAGES = sorted(p for p in (DOCS / "Advance").glob("*.qmd") if "conflicted copy" not in p.stem)
MODEL_PAGES = MODEL_PAGES + ADVANCE_PAGES

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
    commands, program output, diagrams). Blocks inside ``<!-- ... -->``
    HTML comments are skipped.
    """
    lines = pathlib.Path(page).read_text().splitlines()
    i = 0
    in_comment = False
    while i < len(lines):
        if in_comment:
            if "-->" in lines[i]:
                in_comment = False
            i += 1
            continue
        if re.match(r"^\s*<!--", lines[i]) and "-->" not in lines[i]:
            in_comment = True
            i += 1
            continue
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
    stack = []          # open divs: [is_tabset, line, current_tab, tab_level]
    in_code = None
    in_comment = False  # inside <!-- ... -->: not rendered, not counted
    for n, line in enumerate(lines, 1):
        if in_comment:
            in_comment = "-->" not in line
            continue
        if in_code is not None:
            if re.match(r"^```+\s*$", line):
                lang, start, info = in_code
                tabsets = [d for d in reversed(stack) if d[0]]
                # The enclosing language tab, even when a topic tabset sits
                # inside it (e.g. Python > "Explicit model" / "Built-in model").
                tabset = next((d for d in tabsets
                               if (d[2] or "").split(" (")[0] in LANGS), None)
                tabset = tabset or (tabsets[0] if tabsets else None)
                # ``.lang-only``: a note about one language (e.g. an R indexing caveat).
                if lang is not None and "lang-only" not in info:
                    out.append((lang, tabset[2] if tabset else None,
                                tabset[1] if tabset else None, start))
                in_code = None
            continue
        if re.match(r"^\s*<!--", line) and "-->" not in line:
            in_comment = True
            continue
        fence = _FENCE_OPEN.match(line)
        if fence:
            info = (fence.group(1) or "").strip()
            in_code = (_fence_lang(info), n, info)
            continue
        if re.match(r"^:::+\s*$", line):
            if stack:
                stack.pop()
        elif re.match(r"^:::+\s*\S", line):
            stack.append([".panel-tabset" in line, n, None, None])
        elif stack and re.match(r"^#+\s+\S", line):
            # Like Quarto: the tabs of a tabset are the headings at the level
            # of the first heading inside it (##, ###, #####, ...).
            tabset = next((d for d in reversed(stack) if d[0]), None)
            level = len(line) - len(line.lstrip("#"))
            if tabset is not None:
                if tabset[3] is None:
                    tabset[3] = level
                if level == tabset[3]:
                    tabset[2] = line.lstrip("#").strip().lower()
    return out
