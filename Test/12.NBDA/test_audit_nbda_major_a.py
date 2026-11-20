"""Regression tests for RELEASE_AUDIT.md, module NBDA, majors M1-M6 and M11-M13 (covariates)."""

import numpy as np
import pytest

from BayesForge.NBDA.NBDA import NBDA


def _nbda(n=4, t=3):
    rng = np.random.default_rng(0)
    net = np.asarray(rng.integers(0, 2, size=(n, n)).astype(float))
    status = np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=float)[:n, :t]
    return NBDA(net, status)


def test_m1_import_covnf_asocial_on_fresh_object():
    # where='asocial' forces get_cov_j=False, so self.covNF_j stays None on a
    # fresh object; covNF_get_j must be extended from the imported block only.
    nb = _nbda()
    nb.import_covNF(np.arange(4.0).reshape(4, 1), where='asocial')
    assert nb.covNF_j is None
    assert nb.covNF_get_j == [False]
    assert len(nb.names['covNF']) == 1


def test_m2_auto_names_one_per_column_and_unique():
    nb = _nbda()
    for k in range(12):
        nb.import_covNF(np.arange(4.0).reshape(4, 1) + k)
    names = nb.names['covNF']
    assert len(names) == 12
    assert len(set(names)) == 12


def test_m3_locations_stored_per_column():
    nb = _nbda()
    nb.import_covNF(np.array([[1.0], [2.0], [3.0], [4.0]]), names=['a'], where='social')
    nb.import_covNF(np.array([[4.0], [3.0], [2.0], [1.0]]), names=['b'], where='asocial')
    nb.get_cov()
    # Asocial design: intercept + b only (a must NOT leak into it).
    assert nb.T_asocial.shape[-1] == 2
    assert nb.T_asocial_names == ['asocial_learning', 'covNF_i_b']
    # Social design: intercept + a (learner, covNF_i) + a (demonstrator, covNF_j).
    # The first import used the default get_cov_j=True, so covNF_j holds column
    # 'a' and get_cov appends the whole covNF_j block to D_social, naming the
    # columns kept by covNF_get_j == [True, False] -> only 'a'. Expected: 3.
    assert nb.T_social.shape[-1] == 3
    assert nb.T_social_names == ['social_learning', 'covNF_i_a', 'covNF_j_a']


def test_m4_import_covdf_scale_false_keeps_binary_values():
    nb = _nbda()
    mat = np.array([[0, 1, 1, 0],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [0, 1, 1, 0]], dtype=float)
    nb.import_covDF(mat, names=['kin'], scale=False)
    vals = set(np.unique(np.asarray(nb.covDF)).tolist())
    assert vals <= {0.0, 1.0}


def test_m5_constant_column_scaled_to_zero_and_nan_input_rejected():
    nb = _nbda()
    nb.import_covNF(np.ones((4, 1)))
    stored = np.asarray(nb.covNF_i)
    assert not np.isnan(stored).any()
    np.testing.assert_allclose(stored, 0.0)

    nb2 = _nbda()
    bad = np.ones((4, 1))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match='import_covNF'):
        nb2.import_covNF(bad)


def test_m6_scale_along_time_standardizes_each_series_separately():
    nb = _nbda()
    rng = np.random.default_rng(1)
    covV = np.stack([rng.normal(0.0, 1.0, (4, 3)),
                     rng.normal(50.0, 1000.0, (4, 3))])
    nb.import_covNV(covV, names=['x', 'y'])
    stored = np.asarray(nb.covNV_i)  # (n, n, t, var), broadcast over axis 1
    series = stored[0, :, :, :]      # (n, t, var): one series per (variable, node)
    for v in range(2):
        for node in range(4):
            s = series[node, :, v]
            assert abs(s.mean()) < 1e-4
            assert abs(s.std() - 1.0) < 1e-4


def test_m11_covnv_asocial_names_match_columns():
    nb = _nbda()
    rng = np.random.default_rng(2)
    covV = rng.normal(size=(1, 4, 3))
    nb.import_covNV(covV, names=['exp'], where='asocial')
    nb.get_cov()
    assert len(nb.T_asocial_names) == nb.T_asocial.shape[-1]
    assert len(nb.T_social_names) == nb.T_social.shape[-1]
    assert nb.T_asocial_names == ['asocial_learning', 'covNV_i_exp']


def test_m12_invalid_where_raises():
    nb = _nbda()
    with pytest.raises(ValueError, match="where must be"):
        nb.import_covNF(np.arange(4.0).reshape(4, 1), where='social ')


def test_m13a_import_covnf_accepts_1d_input():
    nb = _nbda()
    nb.import_covNF(np.arange(4.0))
    assert len(nb.names['covNF']) == 1
    assert nb.covNF_i.shape[3] == 1
