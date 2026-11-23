"""Regression tests for RELEASE_AUDIT.md, module Data, minors m1-m12."""

import ast
import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import pytest

from BayesForge.Data.manip import manip

_SRC = Path(inspect.getsourcefile(manip)).read_text()
_DATA_DIR = Path(inspect.getsourcefile(manip)).parent

# BF object with a default DataFrame
def _m(df=None):
    m = manip()
    if df is None:
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0], "x": [1, 2, 3, 5], "sex": ["f", "m", "f", "m"]})
    m.data(df)
    return m

# Dummy model
def _model(y):
    return y

# m1: index() must not reset index_map; entries of earlier calls are kept.
def test_m1_index_map_survives_later_calls():
    # m1: index() must not reset index_map; entries of earlier calls are kept.
    assert manip().index_map == {}  # created in __init__
    m = _m(pd.DataFrame({"g": ["a", "b", "a"], "h": ["x", "y", "x"], "z": [1.0, 2.0, 3.0]}))
    m.index(["g"])
    g_map = dict(m.index_map["g"])
    m.index(["h"])
    assert set(m.index_map) == {"g", "h"}
    assert m.index_map["g"] == g_map == {0: "a", 1: "b"}
    assert m.index_map["h"] == {0: "x", 1: "y"}

# Ensure columns treated are saved in data_modification
def test_m2_data_modification_stores_resolved_columns():
    # m2: with 'all', the log must hold the resolved column list, not 'all'.
    m = _m()
    m.index()
    assert m.data_modification["index"] == ["sex"]
    m2 = _m()
    m2.OHE()
    assert m2.data_modification["OHE"] == ["sex"]
    # explicit lists are still stored as lists
    m3 = _m(pd.DataFrame({"g": ["a", "b"], "h": ["x", "y"], "z": [1.0, 2.0]}))
    m3.index(["g", "h"])
    assert m3.data_modification["index"] == ["g", "h"]

# Ensure to_int returns m.df and chaining works
def test_m3_to_int_returns_self_df_like_to_float():
    # m3: to_int returns m.df (already fixed in the majors pass) -> chaining works.
    m = _m()
    out = m.to_int(["x"])
    assert isinstance(out, pd.DataFrame)
    assert out is m.df
    assert str(out["x"].dtype) == "int32"
    assert list(m.to_int(["x"]).head(2)["x"]) == [1, 2]  # chainable like to_float

# Ensure regex=False in str.replace() calls
def test_m4_str_replace_uses_regex_false():
    # m4: source check — every pandas .str.replace() call pins regex=False.
    lines = [ln for ln in _SRC.splitlines() if ".str.replace(" in ln]
    assert lines, "expected pandas .str.replace() calls in manip.py"
    for ln in lines:
        assert "regex=False" in ln, ln
    # functional: '.' in a category label is replaced literally, not as a wildcard
    m = _m(pd.DataFrame({"cat": ["a.b", "c d", "a.b"], "y": [1.0, 2.0, 3.0]}))
    OHE = m.OHE(["cat"])
    assert "cat_a_b" in OHE.columns and "cat_c_d" in OHE.columns

# Ensure bit parameter docstring and robust values
def test_m5_bit_param_docstring_and_robust_values():
    # m5: docstring no longer claims default 32; '32'/'64'/32/64 accepted, others raise.
    doc = inspect.getdoc(manip.pd_to_jax)
    assert "default: 32" not in doc
    assert "jax_enable_x64" in doc
    m = _m(pd.DataFrame({"y": [1.0, 2.0, 3.0]}))
    x64 = jax.config.jax_enable_x64
    for bit in ("32", 32):
        out = m.pd_to_jax(_model, bit=bit)
        assert out["y"].dtype == jnp.float32
    for bit in ("64", 64, None):
        out = m.pd_to_jax(_model, bit=bit)
        # '64' is honoured only when x64 is on (documented M3 behaviour);
        # crucially it must not silently produce a wrong dtype via 'float6464'.
        assert out["y"].dtype == (jnp.float64 if x64 else jnp.float32)
    for bad in ("float64", "16", 0, "abc"):
        with pytest.raises(ValueError):
            m.pd_to_jax(_model, bit=bad)

# Evaluate duplicated data columns error and int64 type
def test_m6_data_to_model_duplicate_and_int64():
    # m6: duplicated column name -> ValueError naming it (no silent 2-D array).
    m = _m(pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"]))
    with pytest.raises(ValueError, match="duplicated"):
        m.data_to_model(["x"])
    x64 = jax.config.jax_enable_x64
    # int64 cast explicitly to the dtype matching x64; no warning for small values
    m2 = _m(pd.DataFrame({"n": np.array([1, 2, 3], dtype="int64")}))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m2.data_to_model(["n"])
    assert m2.data_on_model["n"].dtype == (jnp.int64 if x64 else jnp.int32)
    assert not [w for w in rec if "truncat" in str(w.message)]
    # overflow warning only when int32 would actually truncate the values
    m3 = _m(pd.DataFrame({"n": np.array([2 ** 40], dtype="int64")}))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m3.data_to_model(["n"])
    overflow = [w for w in rec if "truncat" in str(w.message)]
    if x64:
        assert not overflow
        assert m3.data_on_model["n"].dtype == jnp.int64
    else:
        assert overflow and overflow[0].category is UserWarning

# Ensure NaN values in scale_var/z_score warn (UserWarning) and use nanmean/nanstd
def test_m7_nan_warns_and_scales_non_nan_rows():
    # m7: scale_var/z_score warn (UserWarning) and use nanmean/nanstd.
    m = _m()
    x = np.array([1.0, 2.0, np.nan, 4.0])
    with pytest.warns(UserWarning, match="'q'"):
        out = np.asarray(m.scale_var(x, name="q"))
    assert np.isnan(out[2])
    assert np.all(np.isfinite(np.delete(out, 2)))
    assert abs(np.nanmean(out)) < 1e-5  # scaled on the non-NaN entries

    X = jnp.asarray([[1.0, 10.0], [2.0, 20.0], [np.nan, 30.0], [4.0, 40.0]])
    with pytest.warns(UserWarning, match=r"column\(s\) \[0\]"):
        Z = np.asarray(m.z_score(X))
    assert np.isnan(Z[2, 0])                      # NaN entry stays NaN
    assert np.all(np.isfinite(Z[:, 1]))           # clean column fully scaled
    assert abs(np.nanmean(Z[:, 1])) < 1e-5
    # no NaN -> no NaN warning
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        m.z_score(jnp.asarray([[1.0], [2.0], [3.0]]))
        m.scale_var(jnp.asarray([1.0, 2.0, 3.0]))
    assert not [w for w in rec if "NaN" in str(w.message)]

# Ensure OHE with NaN documented, all-zero dummies, and kept original column
def test_m8_ohe_nan_documented_all_zero_dummies():
    # m8: docstring documents dummy_na=False behaviour and the kept original column.
    doc = inspect.getdoc(manip.OHE)
    assert "dummy_na" in doc and "all-zero" in doc and "kept" in doc
    m = _m(pd.DataFrame({"cat": ["a", "b", np.nan], "y": [1.0, 2.0, 3.0]}))
    OHE = m.OHE(["cat"])
    assert list(OHE.columns) == ["cat_a", "cat_b"]  # no _<na> column
    assert (OHE.iloc[2] == 0).all()                 # NaN row -> all-zero dummies
    assert "cat" in m.df.columns                    # original column kept

def test_m8_ohe_missing_category_adds_nan_column():
    m = _m(pd.DataFrame({"cat": ["a", "b", np.nan], "d": ["x", "y", "x"], "y": [1.0, 2.0, 3.0]}))
    OHE = m.OHE(["cat", "d"], missing="category")
    assert "cat_nan" in OHE.columns and "d_nan" not in OHE.columns
    assert OHE["cat_nan"].tolist() == [0, 0, 1]
    assert OHE.loc[2, ["cat_a", "cat_b"]].sum() == 0
    with pytest.raises(ValueError):
        _m().OHE(missing="bogus")

# Ensure type kwarg public and commented
def test_m9_type_kwarg_public_and_commented():
    # m9: `type` stays the public kwarg (no rename); the shadowing is commented.
    assert "type" in inspect.signature(manip.to_float).parameters
    assert "type" in inspect.signature(manip.to_int).parameters
    assert "m9" in _SRC and "builtin" in _SRC
    m = _m()
    m.to_float(cols=["y"], type="float64")  # keyword contract still works
    assert str(m.df["y"].dtype) == "float64"

# Ensure __init__.py is a docstring only
def test_m10_data_package_has_docstring_only_init():
    # m10: Data/ is a regular package; __init__.py is a docstring, no imports.
    init = _DATA_DIR / "__init__.py"
    assert init.is_file()
    tree = ast.parse(init.read_text())
    assert ast.get_docstring(tree), "__init__.py must carry a module docstring"
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)), \
        "__init__.py must not import anything"

# Ensure no jax.jit on preprocessing functions
def test_m11_no_jax_jit_on_preprocessing():
    # m11: one-off preprocessing must not be jitted (already fixed; pinned here).
    assert "jax.jit" not in _SRC

# Ensure nonstring column labels are handled properly
def test_m12_nonstring_column_labels():
    # m12: labels are looked up as given; str() only as a fallback match.
    m = _m(pd.DataFrame({0: ["a", "b", "a"], 1: [1.0, 2.0, 3.0]}))
    m.index([0])
    assert "index_0" in m.df.columns
    assert m.index_map[0] == {0: "a", 1: "b"}
    assert m.data_modification["index"] == [0]  # original int label, not '0'
    # str fallback when the label is not found as-is
    m_b = _m(pd.DataFrame({0: ["a", "b", "a"], 1: [1.0, 2.0, 3.0]}))
    m_b.index(["0"])
    assert "index_0" in m_b.df.columns and 0 in m_b.index_map
    # OHE with the original (integer) label
    m_c = _m(pd.DataFrame({0: ["a", "b"], 1: [1.0, 2.0]}))
    OHE = m_c.OHE([0])
    assert "0_a" in OHE.columns and m_c.data_modification["OHE"] == [0]
