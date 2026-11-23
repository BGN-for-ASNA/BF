"""Regression tests for RELEASE_AUDIT.md, module Data, majors M1-M6."""

import warnings

import numpy as np
import pandas as pd
import jax
import pytest

from BayesForge.Data.manip import manip

# BF object with a default DataFrame
def _m(df=None):
    m = manip()
    if df is None:
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [1, 2, 3, 5], "sex": ["f", "m", "f", "m"]})
    m.data(df)
    return m


# ---------------------------------------------------------------- M1
# TEST THAT SCALE WITH CONSTANT COLUMN RAISES ERROR
def test_m1_constant_column_raises():
    m = _m(pd.DataFrame({"y": [5.0, 5.0, 5.0], "x": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError, match="zero or undefined standard deviation"):
        m.scale(["y"])
    # nothing was written: the other column is untouched and nothing logged
    assert list(m.df["y"]) == [5.0, 5.0, 5.0]
    assert list(m.df["x"]) == [1.0, 2.0, 3.0]
    assert "scale" not in m.data_modification

# TEST THAT SCALE WITH SINGLE ROW RAISES ERROR
def test_m1_single_row_raises():
    m = _m(pd.DataFrame({"y": [5.0]}))
    with pytest.raises(ValueError, match="zero or undefined standard deviation"):
        m.scale(["y"])

# TEST THAT SCALE WITH VARIABLE COLUMN STILL SCALES
def test_m1_variable_column_still_scales():
    m = _m()
    m.scale(["y"])
    assert abs(m.df["y"].mean()) < 1e-12
    assert abs(m.df["y"].std() - 1) < 1e-12  # pandas std is ddof=1

# TEST THAT PANDAS AND JAX STD ARE EQUAL
# ---------------------------------------------------------------- M2
def test_m2_pandas_and_jax_std_equal():
    y = [1.0, 2.0, 3.0, 4.0]
    m = _m(pd.DataFrame({"y": y}))
    m.scale(["y"])
    scaled_pandas = np.asarray(m.df["y"])
    scaled_jax = np.asarray(m.z_score(np.asarray(y)))
    np.testing.assert_allclose(scaled_pandas, scaled_jax, rtol=1e-6)
    # the shared convention is the sample standard deviation (ddof=1):
    # pandas Series.std() of the scaled column is exactly 1
    assert abs(m.df["y"].std() - 1) < 1e-9


# ---------------------------------------------------------------- M3
# TEST THAT PD_TO_JAX RUNS WITH X64 DISABLED AND DOES NOT RAISE TRUNCATION WARNING
def test_m3_pd_to_jax_x64_off_no_truncation_warning():
    # x64 state is process-global (an earlier bf() may have turned it on):
    # force it off for this test and restore it after.
    m = _m(pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [1, 2, 3]}))

    def model(y, x):
        pass

    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = m.pd_to_jax(model)
    finally:
        jax.config.update("jax_enable_x64", prev)
    assert not [w for w in caught if "truncat" in str(w.message).lower()]
    assert str(out["y"].dtype) == "float32"
    assert str(out["x"].dtype) == "int32"


# ---------------------------------------------------------------- M4
# TEST THAT OHE ENCODES CATEGORICAL DATA TYPE
def test_m4_ohe_encodes_category_dtype():
    m = _m(pd.DataFrame({"y": [1.0, 2.0, 3.0], "f": pd.Categorical(["a", "b", "a"])}))
    new = m.OHE()
    assert sorted(new.columns) == ["f_a", "f_b"]
    assert list(m.df.columns) == ["y", "f", "f_a", "f_b"]
    assert list(m.df["f_a"]) == [1, 0, 1]


# ---------------------------------------------------------------- M5
# Check that no name collisions occur after normalization.
def test_m5_names_are_normalized_once_at_load_and_original_names_still_work(capsys):
    # index() used to rename every column as a side effect, so a later call
    # with the original name raised KeyError. Names are now normalized once,
    # by data(), and the original spelling stays accepted.
    m = _m(pd.DataFrame({"my.col": [1.0, 2.0, 3.0], "g": ["a", "b", "a"]}))
    out = capsys.readouterr().out
    assert "renamed columns" in out and "'my.col' -> 'my_col'" in out
    assert list(m.df.columns) == ["my_col", "g"]
    assert m.column_aliases == {"my.col": "my_col"}
    m.index(["g"])
    assert list(m.df.columns) == ["my_col", "g", "index_g"]
    m.scale(["my.col"])  # original name
    assert abs(m.df["my_col"].mean()) < 1e-12
    m.scale(["my_col"])  # normalized name, usable as a model argument
    m.data_to_model(["my.col", "index_g"])
    assert sorted(m.data_on_model) == ["index_g", "my_col"]

def test_m5_no_rename_message_when_names_are_clean(capsys):
    _m(pd.DataFrame({"y": [1.0, 2.0], "g": ["a", "b"]}))
    assert "renamed columns" not in capsys.readouterr().out

# Ensure error is raised when names collide after normalization.
def test_m5_names_colliding_after_normalization_raise():
    with pytest.raises(ValueError, match="collide"):
        _m(pd.DataFrame({"a.b": [1.0, 2.0], "a_b": [3.0, 4.0]}))

# Ensure only normalized column is indexed.
def test_m5_index_sanitizes_only_added_column():
    m = _m(pd.DataFrame({"my g": ["a", "b", "a"], "y": [1.0, 2.0, 3.0]}))
    out = m.index(["my g"])  # original name accepted
    assert list(out.columns) == ["my_g", "y", "index_my_g"]
    assert m.index_map["my_g"] == {0: "a", 1: "b"}


# ---------------------------------------------------------------- M6
# Ensure datetime dtype raises error.
def test_m6_datetime_dtype_raises():
    m = _m(pd.DataFrame({"y": [1.0, 2.0, 3.0],
                         "t": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])}))

    def model(y, t):
        pass

    with pytest.raises(TypeError, match=r"pd_to_jax: column 't' has dtype datetime64.*cannot be converted"):
        m.pd_to_jax(model)

# Ensure error is raised when complex dtype is passed to pd_to_jax.
def test_m6_complex_dtype_raises():
    m = _m(pd.DataFrame({"y": [1.0, 2.0, 3.0],
                         "c": np.array([1 + 2j, 3 + 4j, 5 + 6j])}))

    def model(y, c):
        pass

    with pytest.raises(TypeError, match=r"pd_to_jax: column 'c' has dtype complex128"):
        m.pd_to_jax(model)

# Ensure mapped dtypes correctly convert.
def test_m6_mapped_dtypes_still_convert():
    m = _m()

    def model(y, x):
        pass

    out = m.pd_to_jax(model)
    assert str(out["y"].dtype) in ("float32", "float64")
    assert str(out["x"].dtype) in ("int32", "int64")

# Ensure bool and small int columns convert correctly.
def test_m6_bool_and_small_int_columns_convert():
    # bool / int8 / uint8 columns (binary outcomes, small codes) must keep
    # converting: only genuinely unconvertible dtypes raise.
    m = _m(pd.DataFrame({"y": np.array([True, False, True]),
                         "g": np.array([0, 1, 2], dtype=np.int8),
                         "u": np.array([3, 4, 5], dtype=np.uint8)}))

    def model(y, g, u):
        pass

    out = m.pd_to_jax(model)
    assert np.asarray(out["y"]).dtype == np.bool_
    assert np.asarray(out["g"]).tolist() == [0, 1, 2]
    assert np.asarray(out["u"]).tolist() == [3, 4, 5]


# ---------------------------------------------------------------- C8 option
# Ensure None get their own index when missing="category"
def test_index_missing_category_gets_own_code():
    m = _m(pd.DataFrame({"g": ["b", None, "a", "b"], "y": [1.0, 2.0, 3.0, 4.0]}))
    m.index(["g"], missing="category")
    assert list(m.df["index_g"]) == [1, 2, 0, 1]
    assert m.index_map["g"] == {0: "a", 1: "b", 2: "missing"}

# Ensure error is raised when "missing" is used as a category name.
def test_index_missing_category_on_r_factor():
    df = pd.DataFrame({"g": pd.Categorical(["z", None, "y"], categories=["z", "y"]), "y": [1.0, 2.0, 3.0]})
    m = _m(df)
    m.index("all", missing="category")
    assert list(m.df["index_g"]) == [0, 2, 1]
    assert m.index_map["g"] == {0: "z", 1: "y", 2: "missing"}

# Ensure "missing" category is not added when there are no missing values.
def test_index_missing_category_without_nan_unchanged():
    m = _m(pd.DataFrame({"g": ["b", "a"], "y": [1.0, 2.0]}))
    m.index(["g"], missing="category")
    assert m.index_map["g"] == {0: "a", 1: "b"}

# Ensure error is raised when "missing" is used as a category name.
def test_index_missing_category_level_collision_raises():
    m = _m(pd.DataFrame({"g": ["missing", None, "a"], "y": [1.0, 2.0, 3.0]}))
    with pytest.raises(ValueError, match="already has a level 'missing'"):
        m.index(["g"], missing="category")

# Ensure error is raised when "drop" is used as missing value strategy.
def test_index_missing_bad_value_raises():
    m = _m(pd.DataFrame({"g": ["a", "b"], "y": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="missing must be"):
        m.index(["g"], missing="drop")
