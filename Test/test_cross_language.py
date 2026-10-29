"""Cross-language model cases: the same model in Python, R and Julia.

Run with::

    pytest Test/test_cross_language.py -m lang                 # every case
    pytest Test/test_cross_language.py -m lang -k "Multinomial and r"

Useful environment variables:

* ``BF_RSCRIPT``        path to ``Rscript`` (default: ``shutil.which("Rscript")``)
* ``BF_JULIA``          path to ``julia`` (default: ``julia`` or ``~/.juliaup/bin/julia``)
* ``BF_JULIA_PROJECT``  julia project passed as ``--project=...``

A missing interpreter skips the matching parameterisation. See
``Test/CROSS_LANGUAGE.md`` for the case format.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from lang_harness import (  # noqa: E402
    LANGS,
    available,
    case_id,
    compare,
    find_cases,
    front_matter,
    read_summary,
    run_case,
)

CASES = find_cases()


@pytest.fixture(scope="session")
def python_ref(tmp_path_factory):
    """Cache of {case path: python summary DataFrame}."""
    return {}


def _python_summary(case, cache, tmp_path_factory):
    if case not in cache:
        workdir = tmp_path_factory.mktemp("py")
        proc = run_case(case, "python", workdir)
        assert proc.returncode == 0, (
            f"{case_id(case)} [python] failed:\n{proc.stderr[-4000:]}"
        )
        cache[case] = read_summary(workdir)
    return cache[case]


@pytest.mark.lang
@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("case", CASES, ids=[case_id(c) for c in CASES])
def test_case(case, lang, tmp_path, tmp_path_factory, python_ref):
    if not available(lang):
        pytest.skip("Rscript not found" if lang == "r" else "julia not found")

    if lang == "python":
        summary = _python_summary(case, python_ref, tmp_path_factory)
        assert "mean" in summary.columns
        assert "sd" in summary.columns
        return

    try:
        ref = _python_summary(case, python_ref, tmp_path_factory)
        proc = run_case(case, lang, tmp_path)
    except LookupError as exc:
        pytest.skip(str(exc))

    assert proc.returncode == 0, (
        f"{case_id(case)} [{lang}] failed:\n{proc.stderr[-4000:]}"
    )

    tol = float(front_matter(case).get("tol", 0.25))
    problems = compare(ref, read_summary(tmp_path), tol)
    assert not problems, (
        f"{case_id(case)} [{lang}] disagrees with python:\n" + "\n".join(problems)
    )
