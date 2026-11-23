"""Regression tests for RELEASE_AUDIT.md, module Resources, majors M-1..M-7."""
import numpy as np
import pandas as pd
import pytest

from BayesForge.Resources.datasets import Bunch, load


def test_m1_ucbadmit_explicit_header_and_unchanged_frame():
    path = load().ucbadmit(only_path=True)
    df = pd.read_csv(path, sep=";")  # C engine must parse the fixed header
    assert "dept" in df.columns
    assert set(df["dept"]) == {"A", "B", "C", "D", "E", "F"}
    frame = load().ucbadmit()
    assert list(frame.columns) == [
        "dept", "applicant.gender", "admit", "reject", "applications",
    ]
    assert len(frame) == 12


def test_m2_sim_multivariate_normal_honours_frame():
    assert isinstance(load().sim_multivariate_normal(), pd.DataFrame)
    assert isinstance(load().sim_multivariate_normal(frame=False), Bunch)


def test_m3_matrix_loaders_return_float64():
    ld = load()
    assert ld.phylo_L_simple(frame=False).data.dtype == np.float64
    assert ld.islands_dist_matrix(frame=False).data.dtype == np.float64


def test_m4_bunch_schema_unified():
    for name in ("phylo_L_simple", "islands_dist_matrix"):
        bunch = getattr(load(), name)(frame=False)
        assert {"data", "feature_names", "DESCR", "frame", "filename"} <= set(bunch.keys())


def test_m5_object_dtype_warns():
    with pytest.warns(UserWarning, match="not numeric"):
        load().iris(frame=False)


def test_m6_load_roles():
    with pytest.raises(TypeError):
        load()(None)
    with pytest.raises(FileNotFoundError):
        load("does_not_exist.pkl")


def test_m7_security_documented():
    doc = load.__call__.__doc__
    assert "cloudpickle" in doc
    assert "trust" in doc
