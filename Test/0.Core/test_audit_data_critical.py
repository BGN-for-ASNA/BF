"""Regression tests for RELEASE_AUDIT.md, module 17 (Data/manip.py), critical C1-C8."""

import numpy as np
import pandas as pd
import pytest

from BayesForge.Data.manip import manip

# BF object with a default DataFrame
def _m(df=None):
    m = manip()
    if df is None:
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [1, 2, 3, 5], "sex": ["f", "m", "f", "m"]})
    m.data(df)
    return m

# TEST THAT SCALE WITH DEFAULT ARGUMENTS SCALES ALL NUMERICAL COLUMNS
def test_c1_scale_all_default():
    m = _m()
    m.scale()
    assert abs(m.df["y"].mean()) < 1e-12 and abs(m.df["y"].std() - 1) < 1e-12
    assert abs(m.df["x"].mean()) < 1e-12
    assert list(m.df["sex"]) == ["f", "m", "f", "m"]

# TEST THAT SCALE WITH SINGLE COLUMN NAME SCALES ONLY THAT COLUMN
def test_c1_scale_single_name():
    m = _m()
    m.scale("y")
    assert abs(m.df["y"].std() - 1) < 1e-12
    assert list(m.df["x"]) == [1, 2, 3, 5]

# TEST THAT SCALE_VAR RUNS AND SCALES THE INPUT ARRAY
def test_c2_scale_var_runs():
    m = _m()
    out = np.asarray(m.scale_var(np.array([1.0, 2.0, 3.0])))
    np.testing.assert_allclose(out, (np.array([1., 2., 3.]) - 2) / np.std([1., 2., 3.], ddof=1), rtol=1e-6)

# TEST THAT OHE DOES NOT DUPLICATE COLUMNS
def test_c3_ohe_does_not_duplicate_columns():
    m = _m()
    new = m.OHE()
    assert list(m.df.columns) == ["y", "x", "sex", "sex_f", "sex_m"]
    assert list(new.columns) == ["sex_f", "sex_m"]
    m.scale(["y"])  # used to raise on duplicated columns

# TEST THAT TO_INT CASTS TO INTEGER
@pytest.mark.parametrize("cols", ["all", ["x"]])
def test_c4_to_int(cols):
    m = _m(pd.DataFrame({"x": [1.0, 2.0], "z": [3.0, 4.0]}))
    m.to_int(cols)
    assert str(m.df["x"].dtype) == "int32"

# TEST THAT TO_FLOAT CASTS TO FLOAT
@pytest.mark.parametrize("cols", ["all", ["x"]])
def test_c5_to_float_really_casts(cols):
    m = _m(pd.DataFrame({"x": [1, 2], "z": [3, 4]}))
    m.to_float(cols, "float32")
    assert str(m.df["x"].dtype) == "float32"

# TEST THAT pd_to_jax RAISES KeyError WHEN A COLUMN IS MISSING
def test_c6_missing_column_raises():
    m = _m()
    def model(y, missing_col):
        pass
    with pytest.raises(KeyError):
        m.pd_to_jax(model)

# test jax convert with defaults
def test_c7_array_and_none_defaults():
    m = _m()
    def model(y, w=np.ones(3), flag=None, label="a", n=2):
        pass
    out = m.pd_to_jax(model)
    assert np.asarray(out["w"]).shape == (3,)
    assert out["flag"] is None and out["label"] == "a" and out["n"] == 2

# TEST THAT INDEX REJECTS MISSING VALUES
def test_c8_index_rejects_missing_values():
    m = _m(pd.DataFrame({"g": ["a", "b", None, "a"], "y": [1.0, 2.0, 3.0, 4.0]}))
    with pytest.raises(ValueError):
        m.index(["g"])

# TEST THAT INDEX WITHOUT MISSING UNCHANGED
def test_c8_index_without_missing_unchanged():
    m = _m(pd.DataFrame({"g": ["b", "a", "b"], "y": [1.0, 2.0, 3.0]}))
    m.index(["g"])
    assert list(m.df["index_g"]) == [1, 0, 1]
    assert m.index_map["g"] == {0: "a", 1: "b"}
