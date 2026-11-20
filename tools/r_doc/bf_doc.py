"""Extract BayesForge distribution docstrings to JSON (input of the R roxygen pipeline).

Dev-only tool, not part of the installed package:
    python tools/r_doc/bf_doc.py [out.json]
Without out.json, writes pythonDoc.json next to this script (not the CWD).
"""
import inspect
import json
import sys
from pathlib import Path


def _is_distribution(obj):
    """Distribution wrappers all take ``obs`` and ``sample``; helpers such as
    ``mask``, ``plate`` or ``biject_to`` do not. A rule instead of a frozen
    blacklist, so new dunders or helpers never leak into the R docs (R-m2)."""
    if not callable(obj):
        return False
    try:
        params = inspect.signature(obj).parameters
    except (TypeError, ValueError):
        return False
    return "obs" in params and "sample" in params


def collect_docs(m=None):
    if m is None:
        from BayesForge import bf
        m = bf(print_devices_found=False)
    docs = {}
    for name in dir(m.dist):
        if name.startswith("_"):
            continue
        obj = getattr(m.dist, name, None)
        if _is_distribution(obj):
            docs[name] = obj.__doc__
    return docs


def main(out_path=None):
    # Default output lives next to this script, never in the CWD (R-M2).
    if out_path is None:
        out_path = Path(__file__).with_name("pythonDoc.json")
    docs = collect_docs()
    # utf-8 + ensure_ascii=False: formulas and accents stay readable in git
    # diffs; sort_keys keeps the order stable between runs (R-m3).
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=1, ensure_ascii=False, sort_keys=True)
    return out_path


if __name__ == "__main__":
    main(*sys.argv[1:2])


# The LLM prompt that converted pythonDoc.json into the roxygen JSON is no
# longer a dead string here: it is documented verbatim, with its provenance,
# in tools/r_doc/README.md ("Provenance"). Deterministic post-processing of
# the LLM output is tools/r_doc/fill_params.py.
