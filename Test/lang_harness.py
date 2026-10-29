"""Cross-language harness for model *cases*.

A case is a single markdown file holding the *same* model three times, in fenced
blocks ```` ```python ````, ```` ```r ```` and ```` ```julia ````::

    Test/<topic>/cases/<name>.md

Optional YAML-ish front matter (plain ``key: value`` lines between two ``---``
lines) configures the comparison::

    ---
    tol: 0.25        # agreement tolerance, in python sd units
    timeout: 1800    # seconds per language
    ---

Rules of the case format (see ``Test/CROSS_LANGUAGE.md``):

* the BF object is always named ``m``;
* every block loads the same data (built-in ``m.load.*`` dataset or a CSV that
  lives in ``Test/<topic>/cases/data/``), the same priors, and calls ``m.fit``;
* the harness appends an epilogue so that each language writes its posterior
  summary to ``bf_summary.csv`` in its own working directory;
* R and Julia are compared against the Python summary.

The markdown blocks themselves are extracted with the existing documentation
helper ``Test/3.Documentation/doc_blocks.py`` (fences marked ``.no-test`` or
containing ``# doc-test: skip`` are ignored).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import pathlib

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "3.Documentation"))

from doc_blocks import blocks, ROOT  # noqa: E402

TEST_DIR = pathlib.Path(__file__).resolve().parent

ALL_LANGS = ("python", "r", "julia")
# BF_TEST_LANGS="python,r" restricts the languages (CI runs one language per job).
LANGS = tuple(l for l in ALL_LANGS
              if l in {s.strip().lower() for s in os.environ.get("BF_TEST_LANGS", ",".join(ALL_LANGS)).split(",")})
CASES_GLOB = "cases/*.md"

RSCRIPT = os.environ.get("BF_RSCRIPT") or shutil.which("Rscript")
R_PRELUDE = 'pdf(NULL)\nSys.setenv(MPLBACKEND = "Agg")\n'


def _find_julia():
    if os.environ.get("BF_JULIA"):
        return os.environ["BF_JULIA"]
    found = shutil.which("julia")
    juliaup = pathlib.Path.home() / ".juliaup" / "bin" / "julia"
    return found or (str(juliaup) if juliaup.exists() else None)


JULIA = _find_julia()
JULIA_PRELUDE = 'ENV["MPLBACKEND"] = "Agg"\n'

EPILOGUE = {
    "python": '\nm.summary().to_csv("bf_summary.csv")\n',
    "julia": '\nm.summary().to_csv("bf_summary.csv")\n',
    "r": '\nbf_sum <- m$summary()\n'
    'if (inherits(bf_sum, "data.frame")) write.csv(bf_sum, "bf_summary.csv") '
    'else bf_sum$to_csv("bf_summary.csv")\n',
}

_SCRIPT_NAME = {"python": "case.py", "r": "case.R", "julia": "case.jl"}
_PRELUDE = {"python": "", "r": R_PRELUDE, "julia": JULIA_PRELUDE}


def find_cases(root: pathlib.Path = TEST_DIR) -> list[pathlib.Path]:
    """Every case markdown file under *root* (``<topic>/cases/<name>.md``)."""
    cases = []
    for path in sorted(root.rglob(CASES_GLOB)):
        parts = [str(p) for p in path.parts]
        if any("(WIP)" in p for p in parts) or "__pycache__" in parts:
            continue
        cases.append(path)
    return cases


def case_id(path: pathlib.Path) -> str:
    """``"<topic folder>/<file stem>"``, e.g. ``"4. Rethinking/1.Continuous variable"``."""
    return f"{path.parent.parent.name}/{path.stem}"


def front_matter(path: pathlib.Path) -> dict:
    """Parse a leading ``---`` delimited ``key: value`` block (no yaml import)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return meta
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return {}


def available(lang: str) -> bool:
    if lang == "python":
        return True
    if lang == "r":
        return RSCRIPT is not None
    if lang == "julia":
        return _find_julia() is not None
    raise ValueError(f"unknown language: {lang}")


def _env() -> dict:
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MPLBACKEND="Agg",
        PYTHONPATH=os.pathsep.join(
            filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])
        ),
    )
    if "JULIA_PYTHONCALL_EXE" not in env:
        env.update(
            JULIA_PYTHONCALL_EXE=sys.executable, JULIA_CONDAPKG_BACKEND="Null"
        )
    return env


def _command(lang: str) -> list[str]:
    if lang == "python":
        return [sys.executable]
    if lang == "r":
        return [RSCRIPT]
    if lang == "julia":
        cmd = [_find_julia()]
        if os.environ.get("BF_JULIA_PROJECT"):
            cmd.append(f"--project={os.environ['BF_JULIA_PROJECT']}")
        return cmd
    raise ValueError(f"unknown language: {lang}")


def run_case(
    case: pathlib.Path,
    lang: str,
    workdir: pathlib.Path,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Run the *lang* block of *case* in *workdir*, writing ``bf_summary.csv``."""
    code = blocks(case, lang)
    if not code:
        raise LookupError(f"{case.name}: no {lang} block")

    data = case.parent / "data"
    if data.is_dir():
        shutil.copytree(data, workdir, dirs_exist_ok=True)

    script = workdir / _SCRIPT_NAME[lang]
    script.write_text(
        _PRELUDE[lang] + "\n\n".join(code) + EPILOGUE[lang], encoding="utf-8"
    )

    if timeout is None:
        timeout = int(front_matter(case).get("timeout", 1800))

    return subprocess.run(
        _command(lang) + [str(script)],
        cwd=workdir,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def read_summary(workdir: pathlib.Path) -> pd.DataFrame:
    return pd.read_csv(workdir / "bf_summary.csv", index_col=0)


def compare(ref: pd.DataFrame, other: pd.DataFrame, tol: float) -> list[str]:
    """Labels/means of *other* (R or Julia) vs *ref* (Python); [] means agreement."""
    problems: list[str] = []
    for p in ref.index:
        if p not in other.index:
            problems.append(f"{p}: missing in other summary")
            continue
        x = float(other.loc[p, "mean"])
        y = float(ref.loc[p, "mean"])
        sd = float(ref.loc[p, "sd"])
        d = abs(x - y)
        if d > tol * max(sd, 1e-8):
            problems.append(
                f"{p}: mean {x:.4g} vs python {y:.4g} "
                f"(|diff| = {d / max(sd, 1e-8):.2f} sd > {tol})"
            )
    return problems
