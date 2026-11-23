"""Regression tests for RELEASE_AUDIT.md, module 19 (R), critical R-C1/R-C2."""
import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_rc1_doc_extractor_lists_distributions():
    spec = importlib.util.spec_from_file_location("bf_doc", ROOT / "tools" / "r_doc" / "bf_doc.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # no side effect at import
    docs = mod.collect_docs()
    assert len(docs) > 50
    assert "normal" in docs and docs["normal"]


def test_rc1_distproxy_dir_lists_methods():
    from BayesForge import bf
    m = bf(print_devices_found=False)
    names = dir(m.dist)
    assert "normal" in names and "half_normal" in names


def test_rc2_no_dev_script_inside_package():
    pkg = ROOT / "BayesForge"
    bad = [p for p in pkg.rglob("*.py") if not p.stem.isidentifier()]
    assert bad == []
