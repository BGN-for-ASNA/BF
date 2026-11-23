"""Regression tests for RELEASE_AUDIT.md, module NBDA, minors m1-m15."""

import ast
import inspect
import textwrap

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest

import BayesForge.NBDA.NBDA as nbda_module
from BayesForge.NBDA.NBDA import NBDA


def _net(n=4):
    rng = np.random.default_rng(0)
    return np.asarray(rng.integers(0, 2, size=(n, n)).astype(float))


def _status(n=4, t=3):
    return np.array([[0, 0, 1], [0, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=float)[:n, :t]


def _nbda(n=4, t=3):
    return NBDA(_net(n), _status(n, t))


def _trace(nb, CES=False):
    """Trace nb.model the way the critical tests run it (seed + trace handler, no fit)."""
    return numpyro.handlers.trace(numpyro.handlers.seed(nb.model, rng_seed=0)).get_trace(CES=CES)


_SRC = inspect.getsource(nbda_module)


# ------------------------------- m1: `is` with a literal -------------------------------

def test_m1_no_identity_comparison_with_literal():
    # `is`/`is not` may only compare against None; no `is <literal>` and no
    # `is  None` double spacing left in the module (the old give_name had it).
    tree = ast.parse(_SRC)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            for comp in node.comparators:
                assert not (isinstance(comp, ast.Constant) and comp.value is not None), (
                    f"`is` compared with a literal at line {node.lineno}"
                )
    assert 'is  None' not in _SRC
    assert 'is not  None' not in _SRC


# ------------------------------- m2: dead give_name -------------------------------

def test_m2_give_name_removed():
    # give_name was never called anywhere in the repo (only commented-out
    # call sites); it is removed rather than fixed.
    assert not hasattr(NBDA, 'give_name')
    assert 'give_name' not in _SRC


# ------------------------------- m3: network dimension error message -------------------------------

def test_m3_network_dim_error_message_lists_accepted_dims():
    # The code accepts 2-D (n, n), 3-D (n, n, t) and 4-D (n, n, t, 1);
    # anything else must raise with a message listing exactly those.
    for bad_net in (np.ones(4), np.ones((4, 4, 3, 1, 2))):  # 1-D and 5-D
        with pytest.raises(ValueError, match=r"2 \(n, n\), 3 \(n, n, t\) or 4"):
            NBDA(bad_net, _status())
    # the accepted dimensions still build
    assert NBDA(_net(), _status()).network.shape == (4, 4, 3, 1)


# ------------------------------- m4: empty constructor -------------------------------

def test_m4_empty_constructor_raises_value_error():
    nb = NBDA()
    assert nb.network is None and nb.status is None
    assert nb.names is None and nb.n is None and nb.t is None
    for call in (nb.get_cov, nb.stack_cov, nb.model, nb.print_model):
        with pytest.raises(ValueError, match="NBDA requires network and status"):
            call()
    with pytest.raises(ValueError, match="NBDA requires network and status"):
        nb.import_covNF(np.ones((4, 1)))
    with pytest.raises(ValueError, match="NBDA requires network and status"):
        nb.import_covNV(np.ones((1, 4, 3)))
    with pytest.raises(ValueError, match="NBDA requires network and status"):
        nb.import_covDF(np.ones((4, 4)))
    with pytest.raises(ValueError, match="NBDA requires network and status"):
        nb.import_covDV(np.ones((4, 4, 3)))


# ------------------------------- m5: status stored as (n, t, 1) -------------------------------

def test_m5_status_stored_as_n_t_1_and_documented():
    nb = _nbda()
    assert nb.status.shape == (4, 3, 1)  # shape unchanged by the audit fixes
    assert np.allclose(np.asarray(nb.status)[:, :, 0], _status())
    doc = inspect.getdoc(NBDA) or ''
    assert '(n, t, 1)' in doc


# ------------------------------- m6: status_i/status_j removed -------------------------------

def test_m6_status_i_status_j_removed():
    nb = _nbda()
    assert not hasattr(nb, 'status_i') and not hasattr(nb, 'status_j')
    assert not hasattr(nb, 'arrays_status_i') and not hasattr(nb, 'arrays_status_j')
    assert 'status_i' not in inspect.getsource(NBDA.stack_cov)
    assert 'status_j' not in inspect.getsource(NBDA.stack_cov)
    assert 'status_i' not in inspect.getsource(NBDA.get_cov)
    # the model still traces without them
    nb.get_cov()
    tr = _trace(nb)
    assert np.asarray(tr["y"]["value"]).shape == (4, 3)


# ------------------------------- m7: memory — broadcast instead of loop copies -------------------------------

def test_m7_covdf_3d_broadcast_identical_values_and_covdv_cost_documented():
    rng = np.random.default_rng(1)
    mats = rng.normal(size=(4, 4, 2))
    nb = _nbda()
    nb.import_covDF(mats, names=['a', 'b'], scale=False)
    got = np.asarray(nb.covDF)
    assert got.shape == (4, 4, 3, 2)
    for v in range(2):
        assert np.allclose(got[:, :, :, v], mats[:, :, v][:, :, None])
    # scaled path: same per-matrix standardization as the old loop
    nb2 = _nbda()
    nb2.import_covDF(mats, names=['a', 'b'], scale=True)
    got2 = np.asarray(nb2.covDF)
    for v in range(2):
        m = mats[:, :, v]
        exp = (m - m.mean()) / m.std()
        assert np.allclose(got2[:, :, :, v], exp[:, :, None], atol=1e-5)
    # convert_status no longer builds (n, n, t) copies in a Python loop
    assert 'for i in range(self.t)' not in inspect.getsource(NBDA.convert_status)
    # the 4-D covDV path keeps its loop (code unchanged, m12) but the cost is documented
    doc = inspect.getdoc(NBDA.import_covDV) or ''
    assert 'O(num_variables * n^2 * t)' in doc


# ------------------------------- m8: ces jit cache not tied to the instance -------------------------------

def test_m8_ces_is_jitted_staticmethod_same_values():
    assert isinstance(inspect.getattr_static(NBDA, 'ces'), staticmethod)
    assert hasattr(NBDA.ces, 'lower')  # jax.jit wrapper
    # the compiled function is shared across instances (not an instance-bound jit)
    assert NBDA().ces is NBDA().ces
    K = jnp.arange(6.0).reshape(2, 3)
    L = jnp.arange(6.0, 12.0).reshape(2, 3)
    out = _nbda().ces(K, L, 0.3)
    assert np.allclose(np.asarray(out), 0.3 * np.asarray(K) + 0.7 * np.asarray(L))
    # the CES model path still traces with the staticmethod
    nb = _nbda()
    nb.get_cov()
    tr = _trace(nb, CES=True)
    assert 'y' in tr


# ------------------------------- m9: warnings.warn instead of print, typos -------------------------------

def test_m9_import_covnv_scale_false_warns_and_no_print(capsys):
    nb = _nbda()
    covV = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    with pytest.warns(UserWarning, match='Not scaling covariates'):
        nb.import_covNV(covV, scale=False)
    assert capsys.readouterr().out == ''
    # the misspellings must be gone from the code (comments may name them)
    code_only = '\n'.join(line.split('#')[0] for line in _SRC.splitlines())
    assert 'coeffcients' not in code_only
    assert 'affet' not in code_only
    assert 'coefficients' in code_only and 'affect' in code_only


# ------------------------------- m10: print_model LaTeX -------------------------------

def test_m10_print_model_scale_and_replacements(monkeypatch):
    captured = {}

    def _capture(*a, **k):
        obj = a[0] if a else ''
        captured['md'] = str(getattr(obj, 'data', None) or obj)

    monkeypatch.setattr(nbda_module, 'display', _capture)
    nb = _nbda()
    nb.import_covNF(np.arange(4.0).reshape(4, 1), names=['age'])
    nb.get_cov()
    nb.print_model()
    md = captured['md']
    # priors() uses dist.normal(0, 2, ...) -> scale 2 everywhere
    assert 'Normal(0,2)' in md
    assert 'Normal(0,4)' not in md
    assert 'Normal(0,1)' not in md
    # the covariate replacements actually fire now (they were silent no-ops)
    assert 'X =' in md and 'Z =' in md
    # without covariates nothing is replaced (no dangling "X =")
    captured.clear()
    _nbda().print_model()
    md0 = captured['md']
    assert 'X =' not in md0 and 'Z =' not in md0
    assert 'Normal(0,2)' in md0


# ------------------------------- m11: dead `if k is not None` in get_cov -------------------------------

def test_m11_dead_key_test_removed_and_get_cov_unchanged():
    # AST check: get_cov contains no `k is ...` identity comparison anymore
    # (the loop key was always a non-None string).
    tree = ast.parse(textwrap.dedent(inspect.getsource(NBDA.get_cov)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.Is, ast.IsNot)):
            assert not (isinstance(node.left, ast.Name) and node.left.id == 'k'), (
                f"dead identity test on the loop key at line {node.lineno}"
            )
    nb = _nbda()
    nb.import_covNF(np.arange(4.0).reshape(4, 1), names=['age'])
    nb.get_cov()
    # unchanged behaviour: intercept + covariate on learner i + on demonstrator j
    assert nb.T_social.shape[3] == 3
    assert nb.T_asocial.shape[2] == 2
    assert nb.T_social_names == ['social_learning', 'covNF_i_age', 'covNF_j_age']


# ------------------------------- m12: import_covDV 4-D docstring axis order -------------------------------

def test_m12_import_covdv_docstring_matches_code_axis_order():
    doc = inspect.getdoc(NBDA.import_covDV) or ''
    # the code iterates covDV[i, :, :] over shape[0] -> first axis is the variable axis
    assert '(num_variables, n, n, t)' in doc
    assert '(n, n, t, num_dyadics_effects)' not in doc
    src = inspect.getsource(NBDA.import_covDV)
    assert 'for i in range(covDV.shape[0])' in src  # code unchanged


# ------------------------------- m13: "CES" is a linear mixture -------------------------------

def test_m13_ces_documented_as_linear_mixture():
    doc = inspect.getdoc(inspect.getattr_static(NBDA, 'ces').__func__) or ''
    assert 'linear mixture' in doc
    assert 'perfect substitutes' in doc
    assert 'rho' in doc  # explicitly says it is not a CES with curvature rho
    mdoc = inspect.getdoc(NBDA.model) or ''
    assert 'perfect substitutes' in mdoc
    # no rename: the method and the model keyword are unchanged
    assert hasattr(NBDA, 'ces')
    assert 'CES' in inspect.signature(NBDA.model).parameters


# ------------------------------- m14: NaN sentinel documented -------------------------------

def test_m14_nan_sentinel_documented():
    # a concrete-input check would have to touch the traced likelihood, so the
    # silent masking of accidental NaNs is documented in model() instead.
    doc = inspect.getdoc(NBDA.model) or ''
    assert 'NaN' in doc
    assert 'already informed' in doc
    assert 'silently' in doc


# ------------------------------- m15: numpyro-only backend documented -------------------------------

def test_m15_module_docstring_documents_numpyro_backend():
    doc = nbda_module.__doc__ or ''
    assert 'numpyro' in doc
    assert 'backend' in doc
    # the direct numpyro usage the docstring refers to is still there (no refactor)
    assert 'numpyro.handlers.mask' in _SRC
    assert 'numpyro.distributions.Binomial' in _SRC
