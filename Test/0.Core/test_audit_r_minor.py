"""Regression tests for RELEASE_AUDIT.md, module R, minors R-m1..R-m5."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "r_doc" / "bf_doc.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("bf_doc_tool", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # must not import BayesForge (R-m1)
    return mod


class _FakeDist:
    """Stand-in for m.dist: two distributions and the non-distribution helpers."""

    def normal(self, loc=0.0, scale=1.0, name="x", obs=None, sample=False):
        """Normal distribution. Formula: μ + σ z (été)."""

    def sineskewed(self, base=None, name="x", obs=None, sample=False):
        """SineSkewed distribution."""

    def mask(self, mask=True):
        """Helper, not a distribution."""

    def plate(self, name, size):
        """Helper, not a distribution."""

    def biject_to(self, constraint):
        """Helper, not a distribution."""


class _FakeModel:
    dist = _FakeDist()


def test_r_m1_tool_has_no_import_side_effect_and_no_cwd_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mod = _load_tool()
    src = TOOL.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__"' in src
    monkeypatch.setattr(mod, "collect_docs", lambda m=None: {"a": "b"})
    out = mod.main()
    assert Path(out).parent == TOOL.parent
    assert not (tmp_path / "pythonDoc.json").exists()
    Path(out).unlink()


def test_r_m2_rule_based_filter_not_frozen_blacklist():
    mod = _load_tool()
    docs = mod.collect_docs(_FakeModel())
    # sineskewed is a distribution like any other: no hard-coded exclusion.
    assert set(docs) == {"normal", "sineskewed"}
    assert '"sineskewed"' not in TOOL.read_text(encoding="utf-8")


def test_r_m3_json_is_utf8_readable_and_stable(tmp_path, monkeypatch):
    mod = _load_tool()
    monkeypatch.setattr(mod, "collect_docs",
                        lambda m=None: {"z": "σ été", "a": "x"})
    out = tmp_path / "doc.json"
    mod.main(out)
    text = out.read_text(encoding="utf-8")
    assert "σ été" in text and "\\u03c3" not in text
    assert "\n" in text  # indented, not one line
    assert text.index('"a"') < text.index('"z"')  # sorted keys
    assert json.loads(text) == {"a": "x", "z": "σ été"}


def test_r_m4_two_sources_of_truth_documented():
    readme = (TOOL.parent / "README.md").read_text(encoding="utf-8")
    assert "Two sources of truth" in readme and "R-m4" in readme


def test_r_m5_manifest_has_no_dead_bf_rule():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include BF " not in manifest
