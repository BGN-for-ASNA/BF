"""Regression tests for RELEASE_AUDIT.md, module 13 (Build), critical T-1..T-3, N-1..N-4."""

import importlib, inspect, os, sys
import numpy as np
import pytest
# module importation
def _import(name):
    sys.modules.pop(name, None)
    return importlib.import_module(name)
# test that tfp module import do not create file
def test_t1_import_has_no_side_effect(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow_probability")
    monkeypatch.chdir(tmp_path)
    _import("BayesForge.Build.Build_TFP_light")
    assert list(tmp_path.iterdir()) == []
# test that tfp do not overwrite already existing files
def test_t1_refuses_production_name(tmp_path):
    pytest.importorskip("tensorflow_probability")
    mod = _import("BayesForge.Build.Build_TFP_light")
    with pytest.raises(ValueError):
        mod.generate(str(tmp_path / "tfp_dists.py"))
# test tfp compile and record the observations
def test_t2_t3_generated_tfp_file_compiles_and_conditions_obs(tmp_path):
    pytest.importorskip("tensorflow_probability")
    mod = _import("BayesForge.Build.Build_TFP_light")
    out = mod.generate(str(tmp_path / "gen.py"))
    src = open(out).read()
    compile(src, out, "exec")
    assert "kwargs=kwargs" not in src and "args=args" not in src
    assert "<class" not in src
    # obs handling now lives in the shared tail UnifiedDist._finish, which
    # the generator copies into the generated class (RELEASE_AUDIT tfp M2).
    assert "def _finish(" in src and "UnifiedDist._finish(dist" in src
    assert "def distribution(" not in src  # tfd.Distribution base class excluded
    # The generated module must run, not only compile: _finish is defined in it.
    import importlib.util
    import jax.numpy as jnp
    spec = importlib.util.spec_from_file_location("gen_tfp", out)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    d = gen.UnifiedDist.normal(0.0, 1.0, obs=jnp.zeros(3), wrap=False)
    assert d.log_prob(jnp.zeros(3)).shape == ()
# test signature extraction
def _sig(f):
    return inspect.signature(f).parameters
# test signature rules
def test_t2_n1_signature_rules():
    mod = _import("BayesForge.Build.Build_NP_light")
    def f(a, b=1, *args, mask=None, **kwargs): pass
    s = mod.build_signature(_sig(f), {"mask": "None", "seed": "None"})
    compile(f"def g({s}): pass", "x", "exec")
    assert s.count("mask") == 1
    assert s.endswith("**kwargs")
    assert mod.build_call_args(_sig(f)) == "a=a, b=b, *args, mask=mask, **kwargs"
    import jax.numpy as jnp
    def h(dtype=jnp.int32): pass
    s2 = mod.build_signature(_sig(h), {"seed": "None"})
    assert "jnp.int32" in s2
    compile(f"import jax.numpy as jnp\ndef g({s2}): pass", "x", "exec")
# test np do not import ollama
def test_n1_n4_np_import_has_no_side_effect_and_no_ollama(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "ollama", None)  # import ollama would fail
    mod = _import("BayesForge.Build.Build_NP_light")
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        mod.generate(str(tmp_path / "np_dists.py"), docs={})
# test np file compilation and LLM doc clean up
def test_n1_n2_generated_np_file_compiles(tmp_path):
    mod = _import("BayesForge.Build.Build_NP_light")
    classes = mod.collect_dist_classes()
    docs = {k: '```python\n"""' + k + ' doc with a """ inside\n```' for k in classes}
    out = str(tmp_path / "gen.py")
    mod.generate(out, docs=docs)
    src = open(out).read()
    compile(src, out, "exec")
    assert "kwargs=kwargs" not in src and "```" not in src
# test LLM doc clean up
def test_n2_clean_llm_docstring():
    mod = _import("BayesForge.Build.Build_NP_light")
    d = mod.clean_llm_docstring('```python\n"""HalfNormal\nbody\n```')
    compile(f"def f():\n    {d}\n    pass\n", "x", "exec")
    assert d.startswith('r"""') and "```" not in d
# test LLM doc formulas
def test_n3_half_normal_doc_formula():
    from BayesForge.Distributions.np_dists import UnifiedDist
    doc = UnifiedDist.half_normal.__doc__
    assert r"\sqrt{2}" in doc and r"\exp" in doc
    assert r"\frac{2}{\pi,\sigma}" not in doc
