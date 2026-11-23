"""Record which BayesForge functions a documentation page actually calls.

Quarto's ``freeze: auto`` reruns a page only when its .qmd changes, never when
the BayesForge package changes. During a render this module (enabled by
``BF_TRACK_DEPS=1``, see BayesForge/__init__.py) records, for every executed
cell, the set of BayesForge functions that were called together with a hash of
each function's source. ``Documentation/scripts/invalidate_freeze.py`` then
deletes the freeze entries of the pages whose recorded functions changed.

Module level imports are stdlib only: invalidate_freeze.py loads this file
directly with importlib (importing BayesForge would pull in jax). IPython is
imported inside install().
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import sys
import threading

PKG_DIR = pathlib.Path(__file__).resolve().parents[1]
DOCS = PKG_DIR.parent / "Documentation"
DEPS_DIR = DOCS / "_freeze_deps"
_SELF = str(pathlib.Path(__file__).resolve())

DEF_NODES = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

_installed = False
# Recorded (relpath, qualname) pairs and the key of the cell that imported
# BayesForge; module level so they can be inspected from a test.
_called: set = set()
_state: dict = {"key": None, "cells": 0, "srcs": []}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def cell_key(src: str) -> str:
    """Hash of a cell's code, ignoring quarto directives (#| ...) and padding.

    Used by both the tracker (inside the kernel) and the pre-render script, so
    the two sides always agree on the identity of a cell.
    """
    lines = [line.rstrip() for line in src.splitlines()
             if not line.lstrip().startswith("#|")]
    return _sha1("\n".join(lines).strip())


def page_key(cells: list[str]) -> str:
    """Hash of a whole page: all its cells from the importing one onwards."""
    return _sha1("\n\x00\n".join(cell_key(c) for c in cells))


def _norm_qualname(qualname: str) -> str:
    return ".".join(part for part in qualname.split(".") if part != "<locals>")


def func_hash(path, qualname: str) -> str | None:
    """sha1 of the source of ``qualname`` as defined in ``path``, or None."""
    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    if qualname == "<module>":
        return _sha1(text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, TypeError):
        return None
    target = _norm_qualname(qualname)
    parents = {child: parent
               for parent in ast.walk(tree)
               for child in ast.iter_child_nodes(parent)}

    def dotted(node) -> str:
        parts = []
        while isinstance(node, DEF_NODES):
            parts.append(node.name)
            node = parents.get(node)
        return _norm_qualname(".".join(reversed(parts)))

    for node in ast.walk(tree):
        if isinstance(node, DEF_NODES) and dotted(node) == target:
            segment = ast.get_source_segment(text, node)
            if segment is None:
                return None
            return _sha1(segment)
    return None


def install() -> None:
    """Start recording (only inside an IPython kernel; safe to call twice)."""
    global _installed
    if _installed:
        return
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return
        _installed = True

        called = _called
        pkg = str(PKG_DIR)
        root = PKG_DIR.parent

        def profiler(frame, event, arg):  # never raise: runs on every call
            try:
                if event != "call":
                    return
                code = frame.f_code
                filename = code.co_filename
                if not filename.startswith(pkg) or filename in (_SELF, __file__):
                    return
                rel = pathlib.Path(filename).relative_to(root).as_posix()
                qual = getattr(code, "co_qualname", None) or code.co_name
                called.add((rel, qual))
            except BaseException:
                pass

        sys.setprofile(profiler)
        threading.setprofile(profiler)

        state = _state

        def handler(result):
            try:
                state["cells"] += 1
                # The cell that imports BayesForge is the first one seen; the
                # page is keyed by ALL cells from it on, not only the first.
                state["srcs"].append(result.info.raw_cell)
                key = page_key(state["srcs"])
                # Snapshot first: hashing calls Python code, which the profiler
                # itself is watching (and other threads may still be running).
                items = []
                for _ in range(16):
                    try:
                        items = list(called)
                        break
                    except RuntimeError:
                        pass
                entries = sorted(
                    ([rel, qual, func_hash(root / rel, qual)]
                     for rel, qual in items),
                    key=lambda entry: (entry[0], entry[1]),
                )
                DEPS_DIR.mkdir(parents=True, exist_ok=True)
                (DEPS_DIR / f"{key}.json").write_text(
                    json.dumps({"functions": entries}), encoding="utf-8"
                )
                prev = state["key"]
                if prev is not None and prev != key:
                    # Only the final, complete key survives.
                    try:
                        (DEPS_DIR / f"{prev}.json").unlink(missing_ok=True)
                    except OSError:
                        pass
                state["key"] = key
            except BaseException:
                pass

        shell.events.register("post_run_cell", handler)
    except BaseException:
        pass
