"""Regression tests for RELEASE_AUDIT.md, module Resources, minors m-1..m-11."""
from pathlib import Path

import pandas as pd
from BayesForge.Resources import datasets as datasets_module
from BayesForge.Resources.datasets import load

SRC = Path(datasets_module.__file__).read_text(encoding="utf-8")
DATA = Path(datasets_module.__file__).parent

# (loader method, packaged file, index_col the loader passes to pandas,
#  whether the loader drops the resulting index name)
LOADERS = [
    ("howell1", "Howell1.csv", None, False),
    ("milk", "milk.csv", None, False),
    ("kline", "Kline.csv", None, False),
    ("kline2", "Kline2.csv", None, False),
    ("ucbadmit", "UCBadmit.csv", "case", True),
    ("chimpanzees", "chimpanzees.csv", None, False),
    ("mastectomy", "mastectomy.csv", None, False),
    ("reedfrogs", "reedfrogs.csv", None, False),
    ("tulips", "tulips.csv", None, False),
    ("islands_dist_matrix", "islandsDistMatrix.csv", 0, False),
    ("sim_gamma_poisson", "Sim dat Gamma poisson.csv", None, False),
    ("sim_multinomial", "Sim data multinomial.csv", None, False),
    ("sim_multivariate_normal", "Sim data multivariatenormal.csv", 0, False),
    ("WaffleDivorce", "WaffleDivorce.csv", None, False),
    ("elephants", "elephants.csv", None, False),
    ("iris", "iris.csv", None, False),
    ("NBDA_events", "NBDA_event_data.csv", None, False),
    ("NBDA_network", "NBDA_edge_list.csv", None, False),
    ("phylo_simple", "phylo_data_simple.csv", None, False),
    ("phylo_L_simple", "phylo_L_simple.csv", None, False),
    ("phylo_poisson", "phylo_data_poisson.csv", None, False),
    ("phylo_L_poisson", "phylo_L_simple.csv", None, False),
    ("phylo_repeated", "phylo_data_repeated.csv", None, False),
    ("phylo_L_repeated", "phylo_L_simple.csv", None, False),
    ("phylo_meta", "phylo_data_meta.csv", None, False),
    ("phylo_L_meta", "phylo_L_simple.csv", None, False),
    ("phylo_slopes", "phylo_data_slopes.csv", None, False),
    ("phylo_L_slopes", "phylo_L_slopes.csv", None, False),
    ("sim_nested_effects", "sim_nested_effects.csv", None, False),
    ("panama_ewa", "panama_ewa.csv", None, False),
]


def test_m1_only_path_returns_str_for_every_loader():
    ld = load()
    for name, _fn, _ic, _drop in LOADERS:
        p = getattr(ld, name)(only_path=True)
        assert isinstance(p, str), f"{name}: only_path=True returned {type(p).__name__}"


def test_m2_islands_only_path_does_not_read_the_csv(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("pd.read_csv must not run when only_path=True")
    monkeypatch.setattr(datasets_module.pd, "read_csv", boom)
    p = load().islands_dist_matrix(only_path=True)
    assert isinstance(p, str)


def test_m3_class_docstring_describes_dataframe_default():
    doc = load.__doc__
    assert "Each method returns a Bunch" not in doc
    assert "DataFrame" in doc
    assert "frame=False" in doc


def test_m4_module_docstring_uses_real_import_path():
    doc = datasets_module.__doc__
    assert "from BayesForge.Resources.datasets import load" in doc
    assert "BayesForge.resources.data" not in doc
    assert "load_howell1" not in doc


def test_m5_describe_matches_file_contents():
    ld = load()
    ele = ld.elephants(frame=False).DESCR
    assert "aggression" not in ele.lower()
    assert "survival" not in ele.lower()
    assert "AGE" in ele and "MATINGS" in ele
    wd = ld.WaffleDivorce(frame=False).DESCR
    assert "experiment" not in wd.lower()
    assert "Divorce" in wd and "WaffleHouses" in wd


def test_m6_nan_columns_documented():
    ld = load()
    assert ld.milk()["neocortex.perc"].isna().any()
    for text in (ld.milk.__doc__, ld.milk(frame=False).DESCR):
        assert "neocortex.perc" in text
        assert "NA" in text or "missing" in text.lower()
    assert ld.chimpanzees()["recipient"].isna().any()
    for text in (ld.chimpanzees.__doc__, ld.chimpanzees(frame=False).DESCR):
        assert "recipient" in text
        assert "NA" in text or "missing" in text.lower()
    assert ld.panama_ewa()["RCI"].isna().any()
    for text in (ld.panama_ewa.__doc__, ld.panama_ewa(frame=False).DESCR):
        assert "RCI" in text
        assert "NA" in text or "missing" in text.lower()


def test_m7_packaged_data_cleanup():
    # Datasets used by the documentation stay packaged, so a docs reader can run
    # the examples from an installed package without downloading Resources/.
    assert (DATA / "Howell2.csv").exists()
    assert (DATA / "KosterLeckie.rda").exists()

    # The 24.5 MB NBDA json no loader reads stays in the repo but not in the build.
    assert (DATA / "NBDA_data.json").exists()
    pyproject = (DATA.parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "exclude-package-data" in pyproject
    assert "NBDA_data.json" in pyproject

    # Duplicate and byte-identical copies are gone.
    assert not (DATA / "KosterLeckie copy.rda").exists()
    for fn in ("phylo_L_poisson.csv", "phylo_L_repeated.csv", "phylo_L_meta.csv"):
        assert not (DATA / fn).exists(), f"{fn} must not be packaged (one shared file)"

    # The four phylogenetic models share one kinship Cholesky factor.
    ld = load()
    ref = ld.phylo_L_simple()
    ref_path = ld.phylo_L_simple(only_path=True)
    for name in ("phylo_L_poisson", "phylo_L_repeated", "phylo_L_meta"):
        assert getattr(ld, name)().equals(ref), name
        assert getattr(ld, name)(only_path=True) == ref_path, name


def test_m8_no_unused_imports():
    assert "import os" not in SRC
    assert "import json" not in SRC
    assert "import jax" not in SRC


def test_m9_public_names_unchanged_and_naming_documented():
    ld = load()
    for name in ("WaffleDivorce", "NBDA_events", "NBDA_network", "howell1",
                 "ucbadmit", "elephants", "iris", "phylo_L_simple"):
        assert callable(getattr(ld, name)), name
    doc = load.__doc__
    assert "backward compatibility" in doc.lower()
    assert "snake_case" in doc and "WaffleDivorce" in doc


def test_m10_explicit_sep_frames_identical_to_sniffed():
    assert "sep=None" not in SRC
    assert "engine='python'" not in SRC and 'engine="python"' not in SRC
    for name, fn, index_col, drop_index_name in LOADERS:
        frame = getattr(load(), name)()
        ref = pd.read_csv(DATA / fn, sep=None, engine="python", index_col=index_col)
        if drop_index_name:
            ref.index.name = None
        assert frame.shape == ref.shape, name
        assert list(frame.columns) == list(ref.columns), name
        assert list(frame.dtypes.astype(str)) == list(ref.dtypes.astype(str)), name
        assert frame.equals(ref), name


def test_m11_no_trailing_whitespace_single_data_dir_assignment():
    bad = [i for i, ln in enumerate(SRC.splitlines(), 1) if ln != ln.rstrip()]
    assert bad == []
    assert SRC.count("self.data_dir =") == 1
