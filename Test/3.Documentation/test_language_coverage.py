"""Every code example of the documentation exists in Python, R and Julia.

Pages: "Get started", the model pages and the "Advance" pages.

Each Python, R or Julia block must sit in a ``panel-tabset group="language"``
tab, and each such tabset must have a Python, an R and a Julia tab, each with
at least one code block. Blocks fenced with ``.no-test`` count too: a page
can show install commands that are not executed, but not in one language only.
A block fenced with ``.lang-only`` (a note about one language, such as an R
indexing caveat) and blocks inside ``<!-- -->`` comments are not counted.
"""
import collections
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from doc_blocks import MODEL_PAGES, START_PAGES, language_tabs  # noqa: E402

REQUIRED = {"python", "r", "julia"}


PAGES = START_PAGES + MODEL_PAGES


@pytest.mark.parametrize("page", PAGES, ids=[str(p.relative_to(p.parents[1])) for p in PAGES])
def test_every_example_in_three_languages(page):
    problems = []
    tabsets = collections.defaultdict(set)
    for lang, tab, tabset_line, line in language_tabs(page):
        if tabset_line is None:
            problems.append(f"line {line}: {lang} block outside a Python/R/Julia tabset")
            continue
        # Variant tabs such as "Python (raw)" / "R (built-in)" count as that language.
        tab = (tab or "").split(" (")[0]
        if tab not in REQUIRED:
            problems.append(f"line {line}: {lang} block under tab '{tab}'")
        elif tab != lang:
            problems.append(f"line {line}: {lang} block under the '{tab}' tab")
        tabsets[tabset_line].add(tab)
    for tabset_line, tabs in sorted(tabsets.items()):
        missing = REQUIRED - tabs
        if missing:
            problems.append(f"tabset at line {tabset_line}: no {', '.join(sorted(missing))} code")
    assert not problems, f"{page.name}:\n  " + "\n  ".join(problems)
