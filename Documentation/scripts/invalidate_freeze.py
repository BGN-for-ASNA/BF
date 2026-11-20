#!/usr/bin/env python3
"""Delete the freeze entries of pages whose BayesForge calls changed.

Run as a quarto pre-render hook (from Documentation/): for every page it finds
the first python cell that imports BayesForge through the last one, looks up
the dependency record
written during the previous render by BayesForge/Utils/doc_deps.py, and hashes
the current source of each recorded function. Any mismatch means the frozen
output is stale: the page's _freeze directory (and its record) is deleted, so
quarto reruns exactly that page.

Stdlib only; doc_deps is loaded by path so BayesForge (and jax) is not
imported.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys

DOCS = pathlib.Path(__file__).resolve().parents[1]
REPO = DOCS.parent

_spec = importlib.util.spec_from_file_location(
    "doc_deps", REPO / "BayesForge" / "Utils" / "doc_deps.py")
_doc_deps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_doc_deps)

cell_key = _doc_deps.cell_key
page_key = _doc_deps.page_key
func_hash = _doc_deps.func_hash
DEPS_DIR = _doc_deps.DEPS_DIR

PATTERNS = ["*.qmd", "Advance/*.qmd", "start/*.qmd"]


def python_cells(page: pathlib.Path):
    """Source of every ```{python} cell of a page, in order."""
    try:
        lines = page.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    cell, i = [], 0
    while i < len(lines):
        if lines[i].rstrip() == "```{python}":
            cell = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                cell.append(lines[i])
                i += 1
            yield "\n".join(cell)
        i += 1


def pages():
    seen = set()
    for pattern in PATTERNS:
        for page in sorted(DOCS.glob(pattern)):
            if page in seen or "conflicted copy" in page.name:
                continue
            seen.add(page)
            yield page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="report stale pages, delete nothing")
    args = parser.parse_args()

    count = 0
    for page in pages():
        cells = list(python_cells(page))
        start = next((i for i, c in enumerate(cells) if "BayesForge" in c), None)
        if start is None:
            continue
        deps_file = DEPS_DIR / f"{page_key(cells[start:])}.json"
        if not deps_file.exists():
            continue  # never recorded: leave the freeze alone
        try:
            functions = json.loads(deps_file.read_text(encoding="utf-8"))["functions"]
        except (OSError, ValueError, KeyError):
            continue
        changed = next((qual for rel, qual, h in functions
                        if func_hash(REPO / rel, qual) != h), None)
        if changed is None:
            continue
        count += 1
        print(f"stale: {page.relative_to(DOCS)}  ({changed})", flush=True)
        if not args.dry_run:
            shutil.rmtree(DOCS / "_freeze" / page.relative_to(DOCS).with_suffix(""),
                          ignore_errors=True)
            try:
                deps_file.unlink()
            except OSError:
                pass

    print(f"{count} pages stale (dry run)" if args.dry_run
          else f"{count} pages invalidated", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
