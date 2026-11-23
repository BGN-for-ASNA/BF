"""Verification report for the statistical tests.

A test records each check through the ``verify`` fixture:

    verify.record(quantity="lam", reference="NUTS + exact enumeration",
                  criterion="|z| < 3", value=z, passed=abs(z) < 3,
                  ref=draws_a, new=draws_b)

At the end of the run the checks are
  * printed as a table in the terminal summary,
  * written to Test/0.Core/_report/ (or $BF_TEST_REPORT):
      - summary.md       table of every check (also appended to the GitHub
                         Actions job summary when $GITHUB_STEP_SUMMARY is set)
      - comparisons.png  posterior densities, reference vs tested, per check
      - z_scores.png     z-scores of the Monte Carlo comparisons, with the
                         |z| < 3 acceptance band
      - checks.csv       the same table, machine-readable
"""
import csv
import os

import numpy as np
import pytest

REPORT_DIR = os.environ.get("BF_TEST_REPORT",
                            os.path.join(os.path.dirname(__file__), "_report"))
_ROWS = []

# cqlss object to record test results
class _Verify:
    def __init__(self, test):
        self.test = test

    def record(self, quantity, reference, criterion, value, passed,
               ref=None, new=None, ref_label="reference", new_label="tested"):
        """Record one check. ``ref`` / ``new``: draws to plot (optional)."""
        _ROWS.append(dict(
            test=self.test, quantity=quantity, reference=reference,
            criterion=criterion, value=float(value), passed=bool(passed),
            ref=None if ref is None else np.ravel(np.asarray(ref, dtype=float)),
            new=None if new is None else np.ravel(np.asarray(new, dtype=float)),
            ref_label=ref_label, new_label=new_label))


@pytest.fixture
def verify(request):
    return _Verify(request.node.name)

# function to format the values
def _fmt(v):
    return f"{v:.3g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.1e}"

# function to create the markdown table
def _table_md(rows):
    lines = ["| Test | Quantity | Reference | Criterion | Value | Result |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['test']}` | {r['quantity']} | {r['reference']} | "
                     f"{r['criterion']} | {_fmt(r['value'])} | "
                     f"{'✅' if r['passed'] else '❌'} |")
    return "\n".join(lines)

# function ot create the density plots
def _plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dens = [r for r in rows if r["ref"] is not None and r["new"] is not None]
    if dens:
        ncol = min(3, len(dens))
        nrow = int(np.ceil(len(dens) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow), squeeze=False)
        for ax, r in zip(axes.ravel(), dens):
            lo = min(r["ref"].min(), r["new"].min())
            hi = max(r["ref"].max(), r["new"].max())
            bins = np.linspace(lo, hi, 50) if hi > lo else 10
            ax.hist(r["ref"], bins=bins, density=True, alpha=0.5, label=r["ref_label"])
            ax.hist(r["new"], bins=bins, density=True, alpha=0.5, label=r["new_label"])
            ax.set_title(f"{r['test'].removeprefix('test_')}\n{r['quantity']}: "
                         f"{r['criterion']} → {_fmt(r['value'])} {'✓' if r['passed'] else '✗'}",
                         fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(dens):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "comparisons.png"), dpi=110)
        plt.close(fig)

    zs = [r for r in rows if r["criterion"].startswith("|z|")]
    if zs:
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(zs) + 1.2))
        labels = [f"{r['test'].removeprefix('test_')} · {r['quantity']}" for r in zs]
        vals = [r["value"] for r in zs]
        ax.axvspan(-3, 3, color="0.9", label="|z| < 3 (pass)")
        ax.barh(labels, vals, color=["tab:blue" if r["passed"] else "tab:red" for r in zs])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlim(min(-4, min(vals) - 0.5), max(4, max(vals) + 0.5))
        ax.set_xlabel("z = (mean tested − mean reference) / combined MCSE")
        ax.legend(fontsize=7, loc="lower right")
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "z_scores.png"), dpi=110)
        plt.close(fig)

# function to print the summary table of the tests in terminal and generate plots
def pytest_terminal_summary(terminalreporter):
    if not _ROWS:
        return
    tr = terminalreporter
    tr.section("verification checks")
    w = max(len(f"{r['test']} · {r['quantity']}") for r in _ROWS)
    for r in _ROWS:
        tr.write_line(f"{'PASS' if r['passed'] else 'FAIL'}  "
                      f"{(r['test'] + ' · ' + r['quantity']).ljust(w)}  "
                      f"{r['criterion']:<14} {_fmt(r['value']):>9}   vs {r['reference']}")
    tr.write_line(f"report: {REPORT_DIR}/summary.md (+ comparisons.png, z_scores.png)")

# function to write the final report
def pytest_sessionfinish(session, exitstatus):
    if not _ROWS:
        return
    os.makedirs(REPORT_DIR, exist_ok=True)
    md = ["# Verification checks", "", _table_md(_ROWS), ""]
    try:
        _plots(_ROWS)
        md += ["![posterior comparisons](comparisons.png)", "", "![z-scores](z_scores.png)", ""]
    except Exception as e:  # a plotting problem must not hide the results
        md += [f"(plots not produced: {e})", ""]
    with open(os.path.join(REPORT_DIR, "summary.md"), "w") as f:
        f.write("\n".join(md))
    with open(os.path.join(REPORT_DIR, "checks.csv"), "w", newline="") as f:
        cols = ["test", "quantity", "reference", "criterion", "value", "passed"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(_ROWS)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write("## Verification checks\n\n" + _table_md(_ROWS) + "\n")
    # a verify.record(passed=False) without a paired assert must not stay green
    if any(not r["passed"] for r in _ROWS):
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
