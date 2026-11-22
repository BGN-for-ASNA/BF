# Run with simulated multi-device support for the pmap test:
#   XLA_FLAGS='--xla_force_host_platform_device_count=4' pytest -v "Test/Jax extension/test_sampleddata.py"
#
# Purpose
# -------
# `SampledData` is the object a distribution returns when called with
# `sample=True, to_jax=False`. It wraps a jax array and is meant to be a drop-in
# extension of it: every jax operation should work on the wrapper directly, and
# `isinstance(sd, jnp.ndarray)` should be True (achieved by spoofing __class__).
#
# This file has two parts:
#   1. "Jax extension" suite — exercises a SampledData object *directly* (without
#      unwrapping ._data) through the jax operations that matter (jnp/ufunc
#      interop, jit/vmap/grad/scan/pmap, pytree, pickle, use as a distribution
#      parameter, and the __class__ spoof).
#   2. "Display / repr" suite — pins the fix for the bug where a SampledData
#      printed as "<jax.Array at 0x...>" (a bare id) in IPython/Jupyter instead
#      of its values. Root cause: IPython's text/plain pretty-printer dispatches
#      on the spoofed __class__ (jnp.ndarray) and fell back to a default id repr.

from BayesForge.Utils.SampledData import SampledData, _register_ipython_pprinter

import copy
import pickle

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, random
from jax import enable_x64 as _jax_x64
def enable_x64(): return _jax_x64(True)
def disable_x64(): return _jax_x64(False)
import numpyro.distributions as npd

try:
    import cloudpickle
except ImportError:  # cloudpickle is a BayesForge dependency; skip gracefully if absent
    cloudpickle = None


# ================== Test Fixtures ==================

@pytest.fixture
def data_1d():
    """Provides a 1D SampledData object for testing."""
    return SampledData(jnp.arange(10))

@pytest.fixture
def data_2d():
    """Provides a 2D SampledData object for testing."""
    return SampledData(jnp.arange(12).reshape(4, 3))


# ================== Core Delegation Tests ==================

def test_property_delegation(data_2d):
    """Properties like .shape, .ndim, and .T are correctly delegated."""
    assert data_2d.shape == (4, 3)
    assert data_2d.ndim == 2

    transposed_data = data_2d.T
    assert isinstance(transposed_data, SampledData)  # should return a new wrapper
    assert transposed_data.shape == (3, 4)
    np.testing.assert_array_equal(transposed_data._data, jnp.arange(12).reshape(4, 3).T)

def test_method_delegation_no_args(data_1d):
    """Reductions like .mean() return a bare scalar, not a wrapper."""
    mean_val = data_1d.mean()
    assert not isinstance(mean_val, SampledData)
    assert mean_val == pytest.approx(4.5)

def test_method_delegation_with_args(data_2d):
    """Methods with arguments like .reshape() work and stay wrapped."""
    reshaped_data = data_2d.reshape(6, 2)
    assert isinstance(reshaped_data, SampledData)
    assert reshaped_data.shape == (6, 2)

def test_method_chaining(data_2d):
    """Method chaining works, which relies on _wrap_result at each step."""
    result = data_2d.T.reshape(6, 2).mean(axis=0)
    assert isinstance(result, SampledData)
    assert result.shape == (2,)
    np.testing.assert_array_almost_equal(result._data, jnp.array([4.0, 7.0]))

def test_dtype_and_astype(data_1d):
    """dtype delegation and astype conversion behave like a jax array."""
    assert data_1d.dtype == jnp.arange(10).dtype
    as_float = data_1d.astype(jnp.float32)
    assert isinstance(as_float, SampledData)
    assert as_float.dtype == jnp.float32


# ================== Arithmetic and Operator Tests ==================

def test_arithmetic_operators(data_1d):
    """Dunder methods like __add__, __mul__, __sub__ stay wrapped."""
    result_add = data_1d + 10
    assert isinstance(result_add, SampledData)
    np.testing.assert_array_equal(result_add._data, jnp.arange(10) + 10)

    result_mul = data_1d * 2
    np.testing.assert_array_equal(result_mul._data, jnp.arange(10) * 2)

    other_data = SampledData(jnp.ones(10))
    result_sub = data_1d - other_data
    assert isinstance(result_sub, SampledData)
    np.testing.assert_array_equal(result_sub._data, jnp.arange(10) - 1)

def test_reflected_operators(data_1d):
    """Reflected operators (scalar op SampledData) are handled."""
    result_radd = 10 + data_1d
    assert isinstance(result_radd, SampledData)
    np.testing.assert_array_equal(result_radd._data, 10 + jnp.arange(10))

    result_rsub = 100 - data_1d
    np.testing.assert_array_equal(result_rsub._data, 100 - jnp.arange(10))

def test_matmul_operator():
    """The @ operator works both wrapped@wrapped and wrapped@raw."""
    a = SampledData(jnp.arange(6.).reshape(2, 3))
    b = SampledData(jnp.arange(6.).reshape(3, 2))
    result = a @ b
    assert isinstance(result, SampledData)
    np.testing.assert_allclose(result._data, np.arange(6.).reshape(2, 3) @ np.arange(6.).reshape(3, 2))

    result_raw = a @ jnp.arange(6.).reshape(3, 2)
    np.testing.assert_allclose(result_raw._data, result._data)

def test_comparison_operators_return_bool_arrays(data_1d):
    """Comparisons return plain boolean arrays usable as masks."""
    mask = data_1d > 5
    assert not isinstance(mask, SampledData)
    np.testing.assert_array_equal(np.asarray(mask), np.arange(10) > 5)

def test_unary_operators(data_1d):
    """Negation / abs stay wrapped."""
    neg = -data_1d
    assert isinstance(neg, SampledData)
    np.testing.assert_array_equal(neg._data, -jnp.arange(10))
    assert isinstance(abs(neg), SampledData)
    np.testing.assert_array_equal(abs(neg)._data, jnp.arange(10))


# ================== Indexing and Slicing Tests ==================

def test_getitem_slicing(data_2d):
    """Slicing returns a new SampledData object."""
    sliced_data = data_2d[:, 0]
    assert isinstance(sliced_data, SampledData)
    assert sliced_data.shape == (4,)
    np.testing.assert_array_equal(sliced_data._data, jnp.array([0, 3, 6, 9]))

def test_getitem_single_element(data_1d):
    """Getting a single element returns a scalar, not a wrapper."""
    element = data_1d[5]
    assert not isinstance(element, SampledData)
    assert element == 5

def test_boolean_mask_indexing(data_1d):
    """A boolean mask produced from the object indexes it correctly."""
    mask = data_1d > 5
    selected = data_1d[mask]
    assert isinstance(selected, SampledData)
    np.testing.assert_array_equal(selected._data, jnp.arange(10)[np.arange(10) > 5])

def test_fancy_indexing(data_1d):
    """Integer-array (fancy) indexing is delegated."""
    idx = jnp.array([0, 2, 4])
    selected = data_1d[idx]
    assert isinstance(selected, SampledData)
    np.testing.assert_array_equal(selected._data, jnp.array([0, 2, 4]))

def test_setitem_raises_like_jax(data_1d):
    """__setitem__ is rejected: the wrapper is immutable, same as a jax array."""
    with pytest.raises(TypeError):
        data_1d[0] = 99
    assert data_1d[0] == 0  # unchanged

def test_at_set_returns_new_wrapper(data_1d):
    """.at[].set produces an updated SampledData, leaving the original intact."""
    updated = data_1d.at[0].set(100)
    assert isinstance(updated, SampledData)
    assert updated[0] == 100
    assert data_1d[0] == 0  # original unchanged (jax immutability)

def test_at_add_returns_new_wrapper(data_1d):
    """.at[].add also round-trips through the wrapper."""
    updated = data_1d.at[jnp.array([0, 1])].add(10)
    assert isinstance(updated, SampledData)
    np.testing.assert_array_equal(np.asarray(updated)[:2], np.array([10, 11]))
    np.testing.assert_array_equal(np.asarray(data_1d)[:2], np.array([0, 1]))


# ================== Native jnp / ufunc interop ==================
# These are the crux: can a SampledData be handed straight to jnp.* / np.*
# functions the way a raw array can? This is what `to_jax=True` sidesteps.

def test_jnp_reductions_on_object(data_2d):
    """jnp.sum / jnp.mean accept the object directly via __jax_array__."""
    assert jnp.sum(data_2d) == jnp.sum(jnp.arange(12))
    np.testing.assert_allclose(jnp.mean(data_2d, axis=0), jnp.arange(12).reshape(4, 3).mean(axis=0))

def test_jnp_elementwise_on_object(data_1d):
    """Elementwise jnp ufuncs accept the object directly (no jnp.asarray first).

    Passing the object itself is the whole point: an earlier version of this
    test called jnp.exp(jnp.asarray(data_1d, ...)) and therefore only exercised
    a raw array.
    """
    floats = SampledData(jnp.arange(10, dtype=jnp.float32))
    expected = jnp.exp(jnp.arange(10, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(jnp.exp(floats)), expected, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(jnp.sqrt(floats)),
                               jnp.sqrt(jnp.arange(10, dtype=jnp.float32)), rtol=1e-6)

def test_numpy_ufunc_on_object(data_1d):
    """A raw numpy ufunc accepts the object via __array_ufunc__, either side."""
    floats = SampledData(jnp.arange(3, dtype=jnp.float32))
    np.testing.assert_allclose(np.asarray(np.exp(floats)), np.exp(np.arange(3.)), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(np.add(np.arange(3.), floats)), np.arange(3.) * 2)

def test_jnp_dot_on_object():
    """jnp.dot works with SampledData arguments (no manual unwrap)."""
    v = SampledData(jnp.array([1., 2., 3.]))
    assert jnp.dot(v, v) == pytest.approx(14.0)

def test_jnp_where_with_object(data_1d):
    """jnp.where accepts a SampledData as a branch value."""
    out = jnp.where(jnp.arange(10) % 2 == 0, data_1d, -1)
    expected = jnp.where(jnp.arange(10) % 2 == 0, jnp.arange(10), -1)
    np.testing.assert_array_equal(out, expected)

def test_jnp_concatenate_and_stack():
    """SampledData mixes with raw arrays in concatenate / stack."""
    a = SampledData(jnp.array([1., 2., 3.]))
    cat = jnp.concatenate([a, jnp.array([4., 5.])])
    np.testing.assert_array_equal(cat, jnp.array([1., 2., 3., 4., 5.]))
    stacked = jnp.stack([a, a])
    assert stacked.shape == (2, 3)

def test_broadcasting_with_raw_array(data_2d):
    """A raw array broadcasts against a SampledData (and vice versa)."""
    col = jnp.array([10, 20, 30])
    out = data_2d + col
    assert isinstance(out, SampledData)
    np.testing.assert_array_equal(out._data, jnp.arange(12).reshape(4, 3) + col)

def test_numpy_conversion(data_2d):
    """np.asarray / __array__ produces a real numpy array."""
    arr = np.asarray(data_2d)
    assert isinstance(arr, np.ndarray)
    np.testing.assert_array_equal(arr, np.arange(12).reshape(4, 3))

def test_jnp_asarray_and_to_jax(data_2d):
    """jnp.asarray and .to_jax() both yield an unwrapped jax array."""
    as_jax = jnp.asarray(data_2d)
    assert isinstance(as_jax, jnp.ndarray) and not isinstance(as_jax, SampledData)
    np.testing.assert_array_equal(as_jax, jnp.arange(12).reshape(4, 3))

    to_jax = data_2d.to_jax()
    assert isinstance(to_jax, jnp.ndarray) and not isinstance(to_jax, SampledData)
    np.testing.assert_array_equal(to_jax, jnp.arange(12).reshape(4, 3))


# ================== Pytree registration ==================

def test_tree_flatten_unflatten_roundtrip(data_2d):
    """SampledData round-trips through jax pytree flatten/unflatten."""
    leaves, treedef = jax.tree_util.tree_flatten(data_2d)
    assert len(leaves) == 1
    np.testing.assert_array_equal(leaves[0], jnp.arange(12).reshape(4, 3))
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, SampledData)
    np.testing.assert_array_equal(rebuilt._data, data_2d._data)

def test_tree_map_preserves_wrapper(data_1d):
    """jax.tree_map over the object applies to the leaf and re-wraps."""
    doubled = jax.tree_util.tree_map(lambda x: x * 2, data_1d)
    assert isinstance(doubled, SampledData)
    np.testing.assert_array_equal(doubled._data, jnp.arange(10) * 2)


# ================== Transformations on the wrapped object ==================
# These pass the SampledData object itself into the transformation rather than
# `obj._data`.

@jax.jit
def _jitted_math(x):
    """Operations jax must trace through the SampledData object."""
    y = x.T * 2 + 5
    z = y.sum() / y.size
    return z

def test_jit_takes_object_directly():
    """A jitted function accepts the SampledData object and traces it."""
    data = SampledData(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
    result = _jitted_math(data)
    # x.T*2+5 -> [[5,11],[7,13],[9,15]]; sum=60; size=6; -> 10.0
    np.testing.assert_allclose(result, 10.0)
    assert not isinstance(result, SampledData)

@jax.jit
def _jitted_returns_wrapper(x):
    """A jitted function that returns a SampledData (exercises unflatten under jit)."""
    return x * 2 + 1

def test_jit_returns_wrapper():
    """A jitted function can return a SampledData via pytree reconstruction."""
    data = SampledData(jnp.arange(4, dtype=jnp.float32))
    out = _jitted_returns_wrapper(data)
    assert isinstance(out, SampledData)
    np.testing.assert_allclose(out._data, jnp.arange(4) * 2 + 1)

def test_grad_through_object():
    """jax.grad differentiates a function that consumes the object."""
    def square_and_sum(sd_object):
        return jnp.sum(jnp.square(jnp.asarray(sd_object)))

    grad_func = jax.grad(square_and_sum)
    data = SampledData(jnp.array([1.0, 2.0, 3.0]))
    gradient = grad_func(data)
    # d/dx sum(x^2) = 2x; grad of a pytree input is the same pytree type.
    assert isinstance(gradient, SampledData)
    np.testing.assert_allclose(gradient._data, jnp.array([2.0, 4.0, 6.0]))

def test_vmap_over_object():
    """vmap maps over the wrapped object's leading axis, re-wrapping each slice."""
    def row_dot(row):  # row arrives as a SampledData slice
        return jnp.dot(row, row)

    data = SampledData(jnp.array([[1., 2.], [3., 4.], [5., 6.]]))
    result = jax.vmap(row_dot)(data)
    assert not isinstance(result, SampledData)
    np.testing.assert_allclose(result, jnp.array([5., 25., 61.]))

def test_lax_scan_with_wrapper_carry():
    """SampledData works as the carry state of lax.scan.

    Carry and xs are built at the SAME dtype deliberately. Hardcoding float32
    for the carry while letting xs take the default made this test
    order-dependent: constructing a bf() anywhere earlier in the session
    enables jax x64 process-wide (float_precision=64 is BF's default), xs then
    comes out float64, the carry gets promoted, and lax.scan rejects it with
    "carry input and carry output must have equal types". This test is about
    pytree plumbing through scan, not about dtype promotion.
    """
    def accumulate(sd_carry, x):
        new = SampledData(sd_carry._data + x)
        return new, jnp.sum(new._data)

    dtype = jnp.ones(1).dtype                 # float32 or float64, per x64 mode
    initial = SampledData(jnp.zeros(3, dtype=dtype))
    xs = jnp.ones((5, 3), dtype=dtype)
    final_carry, outputs = lax.scan(accumulate, initial, xs)
    assert isinstance(final_carry, SampledData)
    np.testing.assert_allclose(final_carry._data, jnp.array([5., 5., 5.]))
    np.testing.assert_allclose(outputs, jnp.array([3., 6., 9., 12., 15.]))

def test_lax_cond_with_wrapper():
    """SampledData flows through both branches of lax.cond as operand/result."""
    def run(pred, sd):
        return lax.cond(pred, lambda s: s * 2, lambda s: s + 100, sd)

    data = SampledData(jnp.arange(3, dtype=jnp.float32))
    np.testing.assert_allclose(run(True, data)._data, jnp.arange(3) * 2)
    np.testing.assert_allclose(run(False, data)._data, jnp.arange(3) + 100)

def test_pmap_over_object():
    """SampledData works with jax.pmap when multiple devices are available."""
    if jax.device_count() < 2:
        pytest.skip("pmap test requires multiple devices "
                    "(run with XLA_FLAGS='--xla_force_host_platform_device_count=4').")
    n = jax.device_count()
    shape = (n, 3)
    data = SampledData(jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape))
    result = jax.pmap(lambda r: r * 2)(data)
    assert result.shape == shape
    np.testing.assert_allclose(np.asarray(result),
                               np.arange(np.prod(shape)).reshape(shape) * 2)


# ================== type identity (no __class__ spoof) ==================
# SampledData does NOT spoof __class__: type() is honest and
# isinstance(sd, jnp.ndarray) is False. jax interop comes from __jax_array__
# (coercion), the pytree registration, and __array__ (numpy). These pin that
# contract plus self-identity, pytree behaviour, and serialization.

def test_not_an_instance_of_jax_ndarray(data_1d):
    """No __class__ lie: a SampledData is not a jnp.ndarray, but coerces to one."""
    assert not isinstance(data_1d, jnp.ndarray)
    assert isinstance(jnp.asarray(data_1d), jnp.ndarray)   # __jax_array__ coercion
    assert data_1d.__jax_array__() is data_1d._data

def test_isinstance_self_identity_preserved(data_1d):
    """isinstance(sd, SampledData) and type() are the plain truth."""
    assert isinstance(data_1d, SampledData)
    assert type(data_1d) is SampledData

def test_type_keyed_pytree_still_works(data_2d):
    """type()-keyed pytree registration flattens to the bare leaf."""
    leaves, _ = jax.tree_util.tree_flatten(data_2d)
    assert len(leaves) == 1
    assert not isinstance(leaves[0], SampledData)

def test_pickle_roundtrip(data_2d):
    """__reduce__ keeps stdlib pickle round-tripping to the real class."""
    restored = pickle.loads(pickle.dumps(data_2d))
    assert isinstance(restored, SampledData)
    np.testing.assert_array_equal(restored._data, data_2d._data)

@pytest.mark.skipif(cloudpickle is None, reason="cloudpickle not installed")
def test_cloudpickle_roundtrip(data_2d):
    """BayesForge uses cloudpickle for model save/load; it must round-trip."""
    restored = cloudpickle.loads(cloudpickle.dumps(data_2d))
    assert isinstance(restored, SampledData)
    np.testing.assert_array_equal(restored._data, data_2d._data)


# ================== As a distribution parameter ==================
# The downstream reason a sampled value gets reused: feeding it back into
# another distribution. This is exactly the path `to_jax=True` was protecting.

def test_sampled_data_as_distribution_loc():
    """A SampledData can parameterize a numpyro distribution and be sampled."""
    loc = SampledData(jnp.array([0.0, 5.0, 10.0]))
    d = npd.Normal(loc, 1.0)
    s = d.sample(random.PRNGKey(0))
    assert s.shape == (3,)
    # Samples should sit near their respective loc values.
    assert bool(jnp.all(jnp.abs(s - jnp.array([0., 5., 10.])) < 6))

def test_sampled_data_as_distribution_log_prob():
    """log_prob accepts a SampledData observation and matches the raw-array result."""
    obs = SampledData(jnp.array([0.1, 4.9, 9.8]))
    d = npd.Normal(jnp.array([0.0, 5.0, 10.0]), 1.0)
    lp_wrapped = d.log_prob(obs)
    lp_raw = d.log_prob(jnp.asarray(obs))
    np.testing.assert_allclose(np.asarray(lp_wrapped), np.asarray(lp_raw))

def test_chained_sampling_with_wrapper():
    """A wrapped sample feeds a second distribution (hierarchical-style chaining).

    Characterization, not a bug: because numpyro samples via `loc + scale * eps`,
    a `SampledData` passed as `loc` makes the *returned* sample a `SampledData`
    too (the wrapper rides along through `__radd__`/`__rmul__`); whether it does
    depends on internal op order (wrapping only `scale` returns a raw array).
    Either way the result stays fully array-compatible. Callers that need a
    guaranteed raw array should finish with `.to_jax()` / `jnp.asarray(...)`.
    """
    mu = SampledData(npd.Normal(0.0, 1.0).sample(random.PRNGKey(0), sample_shape=(4,)))
    y = npd.Normal(mu, 0.5).sample(random.PRNGKey(1))
    assert y.shape == (4,)
    # Regardless of whether the wrapper propagated, the value is array-usable.
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(y))))
    assert isinstance(jnp.asarray(y), jnp.ndarray) and not isinstance(jnp.asarray(y), SampledData)


# ================== Display / repr (print & IPython/Jupyter) ==================
# Regression tests for the bug where `sample=True, to_jax=False` values printed
# as "<jax.Array at 0x...>" (a bare memory id) instead of showing the array.
# The spoofed __class__ (jnp.ndarray) makes IPython's text/plain pretty-printer
# dispatch on the wrong class; plain print()/repr() use the real type and were
# always fine, but IPython display fell back to the id form.

class _CapturePrinter:
    """Minimal stand-in for IPython's PrettyPrinter, capturing .text() output."""
    def __init__(self):
        self.buf = []
    def text(self, s):
        self.buf.append(s)
    def getvalue(self):
        return "".join(self.buf)


def test_repr_shows_values_not_pointer(data_1d):
    """repr() shows the array contents, never an id like '... at 0x...'."""
    r = repr(data_1d)
    assert "SampledData(" in r
    assert "at 0x" not in r
    for i in range(10):
        assert str(i) in r

def test_str_and_print_show_values(data_1d, capsys):
    """print(sd) (i.e. str) shows values, not a memory pointer."""
    s = str(data_1d)
    assert "at 0x" not in s
    print(data_1d)
    captured = capsys.readouterr().out
    assert "at 0x" not in captured
    assert "SampledData(" in captured

def test_repr_reports_dtype(data_1d):
    """repr() carries the jax dtype (int/float + precision 8/16/32/64)."""
    # Integer array keeps its integer dtype label.
    assert f"dtype={data_1d.dtype}" in repr(data_1d)
    assert "int" in repr(data_1d)

    # Precision levels and kinds are surfaced exactly as jax labels them.
    cases = {
        jnp.int8:    "int8",
        jnp.int16:   "int16",
        jnp.int32:   "int32",
        jnp.float16: "float16",
        jnp.float32: "float32",
    }
    for dt, label in cases.items():
        sd = SampledData(jnp.array([1, 2, 3], dtype=dt))
        r = repr(sd)
        assert r.startswith("SampledData(")
        assert f"dtype={label}" in r
        assert "at 0x" not in r

def test_repr_html_shows_values(data_1d):
    """_repr_html_ (used by HTML-capable frontends) renders the values."""
    html = data_1d._repr_html_()
    assert "at 0x" not in html
    assert "SampledData(" in html
    assert html.startswith("<pre>") and html.endswith("</pre>")

def test_repr_pretty_shows_values(data_1d):
    """_repr_pretty_ writes the repr into the printer, not an id."""
    p = _CapturePrinter()
    data_1d._repr_pretty_(p, cycle=False)
    out = p.getvalue()
    assert out == repr(data_1d)
    assert "at 0x" not in out


# --- IPython text/plain integration (the actual bug surface) ---

@pytest.fixture(scope="module")
def ipython_text_formatter():
    """A running IPython's text/plain formatter, with the fix registered.

    `start_ipython()` may only be called once per process, so share it across
    tests. Registration is a no-op at import time when no shell is running
    (pytest), so invoke it here once a shell exists.
    """
    pytest.importorskip("IPython")
    from IPython.testing.globalipapp import start_ipython
    from IPython import get_ipython
    ip = start_ipython() or get_ipython()
    _register_ipython_pprinter()
    return ip.display_formatter.formatters["text/plain"]


def test_ipython_text_plain_shows_values(data_1d, ipython_text_formatter):
    """Inside a running IPython, text/plain formatting shows values, not id.

    This is the path that produced '<jax.Array at 0x...>'. The fix registers a
    printer for jnp.ndarray that delegates to repr(), so the formatter now shows
    the values instead of a bare id.
    """
    out = ipython_text_formatter(data_1d)
    assert "at 0x" not in out
    assert "SampledData(" in out

def test_ipython_real_jax_array_unaffected(ipython_text_formatter):
    """The registration must not change how a genuine jax array displays."""
    arr = jnp.arange(3)
    out = ipython_text_formatter(arr)
    # Real arrays keep their native repr (Array([...], dtype=...)), and repr of
    # the array equals the formatter output (that's exactly what we delegate to).
    assert out == repr(arr)
    assert "at 0x" not in out


# ================== End-to-end: sample=True from a distribution ==================
# The user-facing entry point: `m.dist.<dist>(..., sample=True)` returns a raw
# jax array (to_jax=True, the default now); `to_jax=False` opts into the
# SampledData wrapper, which must print its values.

def test_distribution_sample_true_returns_printable_sampleddata():
    """to_jax=False yields a SampledData that prints values; default is raw jax."""
    from BayesForge import bf
    m = bf("cpu")

    arr = m.dist.normal(0, 1, shape=(3,), sample=True, to_jax=False)
    assert isinstance(arr, SampledData)
    assert "at 0x" not in repr(arr)
    assert "SampledData(" in repr(arr)

    raw = m.dist.normal(0, 1, shape=(3,), sample=True)          # to_jax=True default
    assert not isinstance(raw, SampledData)                     # raw jax array path
    assert isinstance(raw, jnp.ndarray)


# ================== Operators with no coverage above ==================
# Every dunder SampledData defines but the suite never exercised. They all work
# today; these pin them so a refactor of _wrap_result / _extract_data can't
# silently drop one.

@pytest.mark.parametrize("op, expected", [
    (lambda x: x / 2,       lambda a: a / 2),
    (lambda x: 8 / x,       lambda a: 8 / a),
    (lambda x: x // 2,      lambda a: a // 2),
    (lambda x: 8 // x,      lambda a: 8 // a),
    (lambda x: x % 3,       lambda a: a % 3),
    (lambda x: 10 % x,      lambda a: 10 % a),
    (lambda x: x ** 2,      lambda a: a ** 2),
    (lambda x: 2 ** x,      lambda a: 2 ** a),
    (lambda x: +x,          lambda a: +a),
    (lambda x: x << 1,      lambda a: a << 1),
    (lambda x: x >> 1,      lambda a: a >> 1),
])
def test_uncovered_arithmetic_operators_stay_wrapped(op, expected):
    """div / floordiv / mod / pow / pos / shifts wrap and match the raw result."""
    raw = jnp.arange(1, 5)
    got = op(SampledData(raw))
    assert isinstance(got, SampledData)
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected(raw)))


@pytest.mark.parametrize("op", [
    lambda x, y: x & y,
    lambda x, y: x | y,
    lambda x, y: x ^ y,
])
def test_bitwise_binary_operators(op):
    """&, |, ^ work wrapped-vs-wrapped and match the raw arrays."""
    b = jnp.array([True, True, False, False])
    c = jnp.array([True, False, True, False])
    got = op(SampledData(b), SampledData(c))
    assert isinstance(got, SampledData)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(op(b, c)))

def test_invert_operator():
    """~ on a boolean wrapper."""
    b = SampledData(jnp.array([True, False]))
    assert isinstance(~b, SampledData)
    np.testing.assert_array_equal(np.asarray(~b), np.array([False, True]))

def test_reflected_bitwise_operators():
    """Reflected bitwise ops (raw op wrapper) route through __rand__ etc."""
    b = jnp.array([True, True, False, False])
    c = SampledData(jnp.array([True, False, True, False]))
    np.testing.assert_array_equal(np.asarray(b & c), np.asarray(b & np.asarray(c)))
    np.testing.assert_array_equal(np.asarray(b | c), np.asarray(b | np.asarray(c)))

def test_remaining_comparison_operators(data_1d):
    """!=, <=, >= return plain boolean arrays like ==, <, > already tested."""
    for got, want in [(data_1d != 3, np.arange(10) != 3),
                      (data_1d <= 3, np.arange(10) <= 3),
                      (data_1d >= 3, np.arange(10) >= 3)]:
        assert not isinstance(got, SampledData)
        np.testing.assert_array_equal(np.asarray(got), want)

def test_len_iter_contains(data_2d):
    """__len__, __iter__ and __contains__ delegate to the wrapped array."""
    assert len(data_2d) == 4
    rows = list(data_2d)
    assert len(rows) == 4
    np.testing.assert_array_equal(np.asarray(rows[0]), jnp.arange(12).reshape(4, 3)[0])
    assert 5 in SampledData(jnp.arange(10))
    assert 99 not in SampledData(jnp.arange(10))

def test_scalar_casts():
    """int() / float() / complex() on a 0-d wrapper."""
    s = SampledData(jnp.array(2.0))
    assert float(s) == 2.0
    assert int(s) == 2
    assert complex(s) == complex(2.0)

def test_bool_on_multi_element_raises_like_jax(data_1d):
    """bool() on a >1-element wrapper raises, exactly as a jax array does."""
    with pytest.raises(Exception):
        bool(data_1d)


# ================== Edge shapes: 0-d and empty ==================

def test_zero_dim_wrapper():
    """A 0-d SampledData reprs, casts and takes part in jnp ops."""
    s = SampledData(jnp.array(5.0))
    assert s.shape == ()
    assert s.ndim == 0
    assert "at 0x" not in repr(s)
    assert float(jnp.sum(s)) == 5.0
    assert float(jnp.asarray(s + 1)) == 6.0

def test_empty_wrapper():
    """A zero-length SampledData behaves like an empty jax array."""
    s = SampledData(jnp.array([]))
    assert len(s) == 0
    assert s.shape == (0,)
    assert "at 0x" not in repr(s)

def test_nested_construction_does_not_double_wrap():
    """SampledData(SampledData(x)) keeps a bare array in _data."""
    s = SampledData(SampledData(jnp.arange(3)))
    assert not isinstance(s._data, SampledData)
    np.testing.assert_array_equal(np.asarray(s), np.arange(3))


# ================== Device / array-protocol surface ==================

def test_item_and_tolist(data_1d):
    """.item() and .tolist() delegate and return plain python objects."""
    assert SampledData(jnp.array(3.0)).item() == 3.0
    assert data_1d.tolist() == list(range(10))

def test_block_until_ready(data_1d):
    """.block_until_ready() (used before timing / IO) delegates."""
    r = data_1d.block_until_ready()
    np.testing.assert_array_equal(np.asarray(r), np.arange(10))

def test_device_attributes(data_1d):
    """.devices() / .sharding / .device are reachable through delegation."""
    assert len(data_1d.devices()) >= 1
    assert data_1d.sharding is not None
    assert data_1d.device is not None

def test_device_put(data_1d):
    """jax.device_put accepts the object (pytree path) and keeps the values."""
    r = jax.device_put(data_1d, jax.devices()[0])
    np.testing.assert_array_equal(np.asarray(r), np.arange(10))

def test_internal_jax_attributes(data_2d):
    """.aval / .weak_type are what jax internals reach for; delegation covers them."""
    assert data_2d.aval.shape == (4, 3)
    assert isinstance(data_2d.weak_type, bool)

def test_np_save_load_roundtrip(data_2d, tmp_path):
    """np.save writes through __array__ and reloads to the same values."""
    p = tmp_path / "sd.npy"
    np.save(p, data_2d)
    np.testing.assert_array_equal(np.load(p), np.arange(12).reshape(4, 3))


# ================== Transformations not covered above ==================

def test_value_and_grad_through_object():
    """value_and_grad returns the value plus a SampledData-shaped gradient."""
    f = lambda x: jnp.sum(jnp.asarray(x) ** 2)
    v, g = jax.value_and_grad(f)(SampledData(jnp.array([1., 2.])))
    assert float(v) == pytest.approx(5.0)
    assert isinstance(g, SampledData)
    np.testing.assert_allclose(np.asarray(g), np.array([2., 4.]))

def test_jacobian_and_hessian_through_object():
    """Higher-order autodiff traces the wrapper without unwrapping."""
    f = lambda x: jnp.sum(jnp.asarray(x) ** 3)
    x = SampledData(jnp.array([1., 2.]))
    np.testing.assert_allclose(np.asarray(jax.jacobian(f)(x)), np.array([3., 12.]))
    np.testing.assert_allclose(np.asarray(jax.hessian(f)(x)), np.diag([6., 12.]))

def test_grad_of_wrapper_nested_in_a_pytree():
    """A SampledData inside a dict differentiates and comes back wrapped."""
    f = lambda d: jnp.sum(jnp.asarray(d["x"]) ** 2)
    g = jax.grad(f)({"x": SampledData(jnp.array([1., 2.]))})
    assert isinstance(g["x"], SampledData)
    np.testing.assert_allclose(np.asarray(g["x"]), np.array([2., 4.]))

def test_eval_shape_with_wrapper(data_2d):
    """jax.eval_shape works without materializing the array."""
    out = jax.eval_shape(lambda x: x * 2, data_2d)
    assert jax.tree_util.tree_leaves(out)[0].shape == (4, 3)

def test_aot_lower_and_compile():
    """Ahead-of-time lowering/compilation accepts a SampledData argument."""
    compiled = jax.jit(lambda x: x * 2).lower(SampledData(jnp.arange(3.))).compile()
    out = compiled(SampledData(jnp.arange(3.)))
    np.testing.assert_allclose(np.asarray(out), np.arange(3.) * 2)

def test_jit_cache_is_reused_for_same_shape():
    """The treedef hashes stably, so a second call with the same shape doesn't retrace."""
    traces = []

    @jax.jit
    def f(x):
        traces.append(1)
        return x * 2

    f(SampledData(jnp.arange(12.).reshape(4, 3)))
    f(SampledData(jnp.arange(12.).reshape(4, 3)))
    assert len(traces) == 1

def test_fori_loop_with_wrapper_carry():
    """SampledData as a lax.fori_loop carry."""
    out = lax.fori_loop(0, 3, lambda i, c: c + 1, SampledData(jnp.zeros(3)))
    assert isinstance(out, SampledData)
    np.testing.assert_allclose(np.asarray(out), np.full(3, 3.0))

def test_while_loop_with_wrapper_carry():
    """SampledData as a lax.while_loop carry."""
    out = lax.while_loop(lambda c: jnp.sum(jnp.asarray(c)) < 9,
                         lambda c: c + 1,
                         SampledData(jnp.zeros(3)))
    assert isinstance(out, SampledData)
    np.testing.assert_allclose(np.asarray(out), np.full(3, 3.0))

def test_vmap_with_mixed_in_axes():
    """vmap over a wrapper with a broadcast (in_axes=None) partner argument."""
    out = jax.vmap(lambda a, b: jnp.dot(a, b), in_axes=(0, None))(
        SampledData(jnp.arange(6.).reshape(3, 2)), jnp.ones(2))
    np.testing.assert_allclose(np.asarray(out), np.arange(6.).reshape(3, 2).sum(axis=1))

def test_tree_map_over_two_wrappers(data_1d):
    """tree_map with two SampledData arguments (treedefs must compare equal)."""
    r = jax.tree_util.tree_map(lambda x, y: x + y, data_1d, SampledData(jnp.arange(10)))
    assert isinstance(r, SampledData)
    np.testing.assert_array_equal(np.asarray(r), np.arange(10) * 2)

def test_tree_map_wrapper_and_raw_array_mismatch(data_1d):
    """A wrapper and a raw array are DIFFERENT treedefs; mixing them must error."""
    with pytest.raises(ValueError):
        jax.tree_util.tree_map(lambda x, y: x + y, data_1d, jnp.arange(10))


# ================== copy / serialization ==================

def test_copy_and_deepcopy(data_2d):
    """copy / deepcopy go through __reduce__ and return real SampledData objects."""
    for c in (copy.copy(data_2d), copy.deepcopy(data_2d)):
        assert isinstance(c, SampledData)
        assert type(c) is SampledData
        np.testing.assert_array_equal(np.asarray(c), np.asarray(data_2d))

def test_pickle_inside_a_container(data_2d):
    """A SampledData nested in a dict/list survives pickling (model save/load path)."""
    restored = pickle.loads(pickle.dumps({"a": [data_2d]}))
    assert isinstance(restored["a"][0], SampledData)
    np.testing.assert_array_equal(np.asarray(restored["a"][0]), np.asarray(data_2d))

@pytest.mark.parametrize("dtype", [jnp.int8, jnp.int16, jnp.int32, jnp.float16, jnp.float32])
def test_pickle_preserves_dtype(dtype):
    """Round-tripping must not silently promote the dtype."""
    s = SampledData(jnp.array([1, 2, 3], dtype=dtype))
    assert pickle.loads(pickle.dumps(s)).dtype == dtype

def test_hash_raises_like_a_jax_array(data_1d):
    """Defining __eq__ leaves the object unhashable — same as a real jax array."""
    with pytest.raises(TypeError):
        hash(data_1d)
    with pytest.raises(TypeError):
        hash(jnp.arange(10))


# ================== dtype / x64 mode ==================
# BayesForge's default is float_precision=64, which flips jax's x64 flag
# process-wide. The rest of the suite runs in whatever mode the import order
# happens to leave behind; these pin both modes explicitly.

@pytest.mark.parametrize("x64", [False, True])
def test_dtype_preserved_in_both_precision_modes(x64):
    """The wrapper never changes the dtype the raw array would have had."""
    ctx = enable_x64() if x64 else disable_x64()
    with ctx:
        raw = jnp.arange(3.)
        sd = SampledData(raw)
        assert sd.dtype == raw.dtype
        assert jnp.asarray(sd + 1).dtype == (raw + 1).dtype
        assert jnp.asarray(sd * 2.0).dtype == (raw * 2.0).dtype

@pytest.mark.parametrize("dtype", [jnp.float32, jnp.int32])
def test_explicit_dtype_survives_x64(dtype):
    """An explicitly-typed array keeps its dtype even with x64 enabled."""
    with enable_x64():
        sd = SampledData(jnp.arange(3, dtype=dtype))
        assert sd.dtype == dtype
        assert jnp.asarray(sd).dtype == dtype

def test_weak_type_promotion_matches_raw():
    """sd + python-float promotes exactly like array + python-float."""
    raw = jnp.arange(3, dtype=jnp.int32)
    assert jnp.asarray(SampledData(raw) + 1.0).dtype == (raw + 1.0).dtype


# ================== Numeric reproducibility (the actual claim) ==================
# `to_jax=False` is only safe if the wrapper is numerically invisible: the same
# key and the same inputs must give BITWISE identical results whether the value
# is wrapped or raw. Nothing above asserted that.

def test_wrapped_path_is_bitwise_identical_to_raw_path():
    """Sampling with a wrapped loc equals sampling with the raw array, bit for bit."""
    key = random.PRNGKey(42)
    loc = jnp.zeros(5)
    wrapped = np.asarray(npd.Normal(SampledData(loc), 1.0).sample(key))
    raw = np.asarray(npd.Normal(loc, 1.0).sample(key))
    np.testing.assert_array_equal(wrapped, raw)

def test_same_key_gives_identical_samples_through_the_wrapper():
    """Determinism is not lost by the wrapper: same key, same draw."""
    key = random.PRNGKey(0)
    a = npd.Normal(SampledData(jnp.zeros(5)), 1.0).sample(key)
    b = npd.Normal(SampledData(jnp.zeros(5)), 1.0).sample(key)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

@pytest.mark.parametrize("fn", [
    lambda x: jnp.sum(x),
    lambda x: jnp.mean(x, axis=0),
    lambda x: jnp.exp(x),
    lambda x: jnp.linalg.norm(x),
    lambda x: jnp.einsum("ij,kj->ik", x, x),
    lambda x: jax.jit(lambda y: y * 2 + 1)(x),
])
def test_computations_bitwise_identical_wrapped_vs_raw(fn):
    """Any jnp computation gives the identical result on the wrapper and the array."""
    raw = jnp.arange(12., dtype=jnp.float32).reshape(4, 3) / 7.0
    np.testing.assert_array_equal(np.asarray(fn(SampledData(raw))), np.asarray(fn(raw)))


# ================== numpyro downstream ==================

def test_sampled_data_as_scale_parameter():
    """A SampledData works as `scale`, not only as `loc`."""
    d = npd.Normal(0.0, SampledData(jnp.array([1.0, 2.0])))
    s = d.sample(random.PRNGKey(0))
    assert s.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(jnp.asarray(s))))

def test_sampled_data_as_multivariate_covariance():
    """A wrapped covariance matrix parameterizes MultivariateNormal."""
    d = npd.MultivariateNormal(jnp.zeros(2), SampledData(jnp.eye(2)))
    s = d.sample(random.PRNGKey(0))
    assert s.shape == (2,)

def test_sampled_data_as_mcmc_observation():
    """A SampledData passed as `obs=` runs through a full NUTS fit.

    This is the end of the road for a sampled value: feeding it back in as data.
    """
    import numpyro
    from numpyro.infer import MCMC, NUTS

    obs = SampledData(npd.Normal(3.0, 1.0).sample(random.PRNGKey(0), (50,)))

    def model(y):
        mu = numpyro.sample("mu", npd.Normal(0.0, 10.0))
        numpyro.sample("y", npd.Normal(mu, 1.0), obs=y)

    mcmc = MCMC(NUTS(model), num_warmup=50, num_samples=50, progress_bar=False)
    mcmc.run(random.PRNGKey(1), obs)
    mu = np.asarray(mcmc.get_samples()["mu"])
    assert np.isfinite(mu).all()
    assert abs(mu.mean() - 3.0) < 1.0

def test_mcmc_result_identical_wrapped_vs_raw_obs():
    """Wrapping the observations must not change the posterior draws at all."""
    import numpyro
    from numpyro.infer import MCMC, NUTS

    raw_obs = npd.Normal(3.0, 1.0).sample(random.PRNGKey(0), (50,))

    def model(y):
        mu = numpyro.sample("mu", npd.Normal(0.0, 10.0))
        numpyro.sample("y", npd.Normal(mu, 1.0), obs=y)

    def run(obs):
        mcmc = MCMC(NUTS(model), num_warmup=50, num_samples=50, progress_bar=False)
        mcmc.run(random.PRNGKey(1), obs)
        return np.asarray(mcmc.get_samples()["mu"])

    np.testing.assert_array_equal(run(SampledData(raw_obs)), run(raw_obs))


# ================== Known divergences from jax array semantics ==================
# SampledData is NOT a perfect stand-in. These pin the places where it differs,
# so the difference is a decision on record rather than a surprise. The xfails
# are strict: they turn red the moment the gap is closed, prompting the flip to
# a positive assertion.

def test_setitem_matches_jax_immutability(data_1d):
    """Both a raw jax array and a SampledData reject in-place item assignment."""
    with pytest.raises(TypeError):
        jnp.arange(10)[0] = 99
    with pytest.raises(TypeError):
        data_1d[0] = 99
    assert int(np.asarray(data_1d)[0]) == 0   # untouched

def test_at_update_does_not_touch_aliases():
    """Value semantics: `.at[].set` returns a new wrapper; other names unaffected."""
    a = SampledData(jnp.arange(3))
    b = a
    a2 = a.at[0].set(99)
    assert int(np.asarray(a2)[0]) == 99
    assert int(np.asarray(a)[0]) == 0
    assert int(np.asarray(b)[0]) == 0

def test_at_set_promotes_dtype_like_jax():
    """`.at[].set` follows jax dtype rules; the int wrapper stays int."""
    s = SampledData(jnp.arange(3))
    out = s.at[0].set(1)
    assert out.dtype == jnp.arange(3).dtype
    assert int(np.asarray(out)[0]) == 1

def test_at_update_is_invisible_to_a_cached_jit():
    """`.at[].set` leaves the original buffer intact, so a cached jit is stable."""
    @jax.jit
    def f(x):
        return x * 2

    s = SampledData(jnp.arange(3))
    before = np.asarray(f(s))
    s = s.at[0].set(100)               # new wrapper, original buffer untouched
    after = np.asarray(f(s))
    np.testing.assert_array_equal(before, np.array([0, 2, 4]))
    np.testing.assert_array_equal(after, np.array([200, 2, 4]))

def test_numpy_array_on_the_left_keeps_the_wrapper():
    """`np_array + sd` returns a SampledData backed by a jax array.

    A raw jax array wins the numpy binop and yields a jax Array. SampledData
    does the same through `__array_priority__` and `__array_ufunc__`, so the
    wrapper is kept and the data stays a jax array.
    """
    raw = jnp.arange(3.)
    assert isinstance(np.ones(3) + raw, jnp.ndarray)

    out = np.ones(3) + SampledData(raw)
    assert isinstance(out, SampledData)
    assert isinstance(jnp.asarray(out), jnp.ndarray)
    np.testing.assert_allclose(np.asarray(out), np.ones(3) + np.arange(3.))

def test_divmod_matches_jax():
    """divmod() works on a jax array and on a SampledData (via __divmod__)."""
    q, r = divmod(SampledData(jnp.arange(1, 5)), 2)
    np.testing.assert_array_equal(np.asarray(q), np.arange(1, 5) // 2)
    np.testing.assert_array_equal(np.asarray(r), np.arange(1, 5) % 2)

def test_format_spec_matches_jax():
    """f'{x:.2f}' works on a 0-d jax array and on a wrapper (via __format__)."""
    assert f"{SampledData(jnp.array(3.0)):.2f}" == f"{jnp.array(3.0):.2f}"


def test_array_protocol_accepts_copy_keyword():
    """np.asarray(x, copy=False) works on a jax array; on a wrapper it warns."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        np.asarray(SampledData(jnp.arange(3.)), copy=False)


def test_no_class_spoof_contract():
    """Pins the post-spoof contract: honest type, jax interop via __jax_array__.

    If a future change re-introduces a __class__ lie, the first assert fails.
    """
    sd = SampledData(jnp.arange(3))
    assert type(sd) is SampledData
    assert sd.__class__ is SampledData
    assert not isinstance(sd, jnp.ndarray)
    # interop still holds through the coercion hook / pytree / numpy protocol
    assert isinstance(jnp.asarray(sd), jnp.ndarray)
    assert isinstance(np.asarray(sd), np.ndarray)
    np.testing.assert_array_equal(jnp.asarray(sd), np.arange(3))


# ================== Plotting methods and hdi ==================
# The plotting/summary half of SampledData had no tests at all. These are smoke
# tests: they assert the methods run on correctly-shaped input and raise a clear
# ValueError on wrongly-shaped input. show() is stubbed so nothing opens a
# browser or a GUI window.

@pytest.fixture
def headless(monkeypatch):
    """Stub out plotly's and matplotlib's show() so plots never open a window."""
    import plotly.graph_objects as go
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")

@pytest.fixture
def samples_1d():
    return SampledData(jnp.asarray(np.random.default_rng(0).normal(size=200)))

@pytest.fixture
def samples_2d():
    return SampledData(jnp.asarray(np.random.default_rng(0).normal(size=(200, 3))))


def test_hdi_returns_narrowest_interval():
    """hdi() returns an interval of the requested width covering cred_mass.

    On uniformly spaced samples every candidate interval has the same width, so
    which one argmin picks is not pinned down (floating-point noise decides);
    the width and the coverage are.
    """
    s = SampledData(jnp.linspace(0.0, 1.0, 101))
    lo, hi = s.hdi(cred_mass=0.95)
    assert float(hi) - float(lo) == pytest.approx(0.95, abs=1e-6)
    covered = np.mean((np.linspace(0, 1, 101) >= float(lo)) &
                      (np.linspace(0, 1, 101) <= float(hi)))
    assert covered >= 0.95

def test_hdi_is_tighter_than_the_full_range():
    """A 50% HDI of normal draws is strictly inside the 95% one."""
    s = SampledData(jnp.asarray(np.random.default_rng(0).normal(size=2000)))
    lo50, hi50 = s.hdi(cred_mass=0.5)
    lo95, hi95 = s.hdi(cred_mass=0.95)
    assert float(lo95) < float(lo50) < float(hi50) < float(hi95)

def test_hdi_on_a_normal_sample_brackets_the_mean(samples_1d):
    """A 95% HDI of standard-normal draws sits near +/- 2."""
    lo, hi = samples_1d.hdi()
    assert float(lo) < 0.0 < float(hi)
    assert 3.0 < float(hi) - float(lo) < 5.0

@pytest.mark.parametrize("method", ["hist", "autocorr", "density"])
def test_plot_methods_accept_1d(method, samples_1d, headless):
    """1D-capable plotting methods run without error."""
    getattr(samples_1d, method)()

@pytest.mark.parametrize("method", [
    "hist", "corr_heatmap", "boxplot", "violinplot", "pairplot",
    "timeseries", "scatter3d", "traceplot", "autocorr", "density",
])
def test_plot_methods_accept_2d(method, samples_2d, headless):
    """2D-capable plotting methods run without error (interactive path)."""
    getattr(samples_2d, method)()

@pytest.mark.parametrize("method", [
    "hist", "corr_heatmap", "boxplot", "violinplot", "pairplot",
    "timeseries", "scatter3d", "traceplot", "autocorr", "ridgeline",
])
def test_plot_methods_static_backend(method, samples_2d, headless):
    """The interactive=False (seaborn/matplotlib) branch also runs."""
    getattr(samples_2d, method)(interactive=False)

@pytest.mark.parametrize("method", [
    "corr_heatmap", "boxplot", "violinplot", "pairplot",
    "timeseries", "scatter3d", "traceplot", "ridgeline", "surface_3d",
])
def test_plot_methods_reject_1d_with_a_clear_error(method, samples_1d, headless):
    """Methods that need 2D input raise ValueError, not an obscure downstream error."""
    with pytest.raises(ValueError):
        getattr(samples_1d, method)()

def test_surface_3d_accepts_2d(samples_2d, headless):
    """surface_3d takes no `interactive` flag; it runs on 2D data."""
    samples_2d.surface_3d()

def test_ppc_plot_runs(samples_2d, headless):
    """ppc_plot overlays replicates against an observed vector."""
    observed = np.random.default_rng(1).normal(size=3)
    samples_2d.ppc_plot(observed)

def test_ridgeline_interactive_runs(samples_2d, headless):
    """The interactive ridgeline path runs after the lazy-import alias fix."""
    samples_2d.ridgeline()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
