"""Execution-based introspection of a BF/NumPyro model.

``to_latex`` and ``model_graph`` historically read the *source text* of a
model and guessed its structure. That cannot be exact: loops, helper
functions, ``jnp`` operations and BF building blocks (``m.effects``,
``m.net``) have no static meaning in the text, and models written in R or
Julia have no Python source at all.

This module runs the model instead:

1. ``numpyro.handlers.trace`` gives every sample/deterministic site, its
   distribution, shape, plates and whether it is observed -- including the
   sites created inside BF building blocks.
2. ``jax.make_jaxpr`` over a function mapping site values to distribution
   parameters gives the *exact* dependencies between sites, through loops,
   helpers and ``jnp`` calls. Every site value (latent and deterministic)
   is substituted as an input, so an edge ``a -> b`` means that a parameter
   of ``b`` is computed from the value of ``a``.
3. For Python models, each site is mapped to the source line that created it
   (by walking the call stack), so the source-based renderers can attach the
   sites that do not appear literally in the code.

Usage::

    from BayesForge.PostModel.model_trace import trace_model
    tr = trace_model(model, m.data_on_model)
    tr.sites["sigma"].dist     # 'Exponential'
    tr.parents["y"]            # {'mu', 'sigma'} if mu is a deterministic site
"""

import functools
import inspect
import re
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.primitives import Messenger

LATENT = "latent"
OBSERVED = "observed"
DETERMINISTIC = "deterministic"
DATA = "data"

# Distribution wrappers to look through for the user-facing distribution name.
_WRAPPERS = ("ExpandedDistribution", "Independent", "MaskedDistribution",
             "MaskedDistributionWrapper", "TransformedDistribution")


@dataclass
class Param:
    """One parameter of a site's distribution."""
    name: str
    value: object          # concrete array/scalar from the trace
    parents: set = field(default_factory=set)   # site / data names it depends on


@dataclass
class Site:
    name: str
    kind: str                  # LATENT, OBSERVED or DETERMINISTIC
    dist: str = None           # e.g. 'Normal' (None for deterministic sites)
    shape: tuple = ()
    plates: tuple = ()         # names of enclosing plates, outermost first
    line: int = None           # source line in the model function (Python only)
    params: list = field(default_factory=list)
    discrete: bool = False


class _LineRecorder(Messenger):
    """Record, for each site, the line of the model function that created it."""

    def __init__(self, fn, code):
        super().__init__(fn)
        self.code = code
        self.lines = {}

    def process_message(self, msg):
        if msg["type"] not in ("sample", "deterministic") or self.code is None:
            return
        frame = inspect.currentframe()
        while frame is not None:
            if frame.f_code is self.code:
                self.lines.setdefault(msg["name"], frame.f_lineno)
                break
            frame = frame.f_back


def _model_code(model):
    """Code object of the user's model function (through partial/wrappers)."""
    fn = model
    while isinstance(fn, functools.partial):
        fn = fn.func
    try:
        fn = inspect.unwrap(fn)
    except Exception:
        pass
    return getattr(fn, "__code__", None)


def _base_dist(d):
    while type(d).__name__ in _WRAPPERS and hasattr(d, "base_dist"):
        d = d.base_dist
    return d


def dist_label(d):
    """User-facing distribution name of a NumPyro distribution object."""
    return type(_base_dist(d)).__name__


def dist_snake_name(label):
    """Convert CamelCase distribution class name to BF snake_case label."""
    if not label:
        return None
    # e.g. LKJCholesky -> lkj_cholesky, MultivariateNormal -> multivariate_normal
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", label)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _param_leaves(d):
    """(names, leaves) of a distribution's parameters, wrappers included.

    The names come from the innermost distribution's ``arg_constraints``;
    leaves that do not map onto a named argument (wrapper state) are dropped
    from the names list but still traced for dependencies.
    """
    base = _base_dist(d)
    names, leaves = [], []
    for k in getattr(base, "arg_constraints", {}) or {}:
        v = getattr(base, k, None)
        if v is None:
            continue
        try:
            jnp.asarray(v)
        except Exception:
            continue
        names.append(k)
        leaves.append(v)
    if not leaves:
        # Fall back to the pytree leaves of the whole distribution.
        try:
            flat = jax.tree_util.tree_flatten(d)[0]
        except Exception:
            flat = []
        for i, v in enumerate(flat):
            names.append(f"arg{i}")
            leaves.append(v)
    return names, leaves


def _split_data(data):
    """Array-like data entries are traced inputs; the rest stay static."""
    arrays, static = {}, {}
    for k, v in (data or {}).items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            static[k] = v
            continue
        try:
            a = jnp.asarray(v)
        except Exception:
            static[k] = v
            continue
        if a.dtype == object:
            static[k] = v
        else:
            arrays[k] = a
    return arrays, static


def _is_literal(v):
    return type(v).__name__ == "Literal"


def _propagate_jaxpr(jaxpr, in_sets, const_sets=()):
    """Propagate input dependency sets through a Jaxpr to its outputs."""
    deps = {}
    for var, s in zip(getattr(jaxpr, "constvars", ()), const_sets):
        if not _is_literal(var):
            deps[var] = set(s)
    for var, s in zip(jaxpr.invars, in_sets):
        if not _is_literal(var):
            deps[var] = set(s)
    for eqn in jaxpr.eqns:
        in_deps = [set() if _is_literal(v) else deps.get(v, set())
                   for v in eqn.invars]
        sub = eqn.params.get("jaxpr") or eqn.params.get("call_jaxpr")
        sub_jaxpr = getattr(sub, "jaxpr", sub) if sub is not None else None
        if (sub_jaxpr is not None
                and hasattr(sub_jaxpr, "invars")
                and len(sub_jaxpr.invars) == len(eqn.invars)
                and len(sub_jaxpr.outvars) == len(eqn.outvars)
                and not getattr(sub_jaxpr, "constvars", ())):
            out_deps = _propagate_jaxpr(sub_jaxpr, in_deps)
            for v, od in zip(eqn.outvars, out_deps):
                if not _is_literal(v):
                    deps[v] = od
        else:
            union_s = set().union(*in_deps) if in_deps else set()
            for v in eqn.outvars:
                if not _is_literal(v):
                    deps[v] = union_s
    return [set() if _is_literal(v) else deps.get(v, set())
            for v in jaxpr.outvars]


def _jaxpr_dependencies(jaxpr, input_names):
    """Forward dataflow: for each jaxpr output, the set of input names it uses.

    Equations with sub-jaxprs (pjit, scan, while, cond, custom_jvp...) are
    propagated per-output when the sub-jaxpr signature matches, and treated
    conservatively (union of inputs) otherwise.
    """
    in_sets = [{name} for name in input_names]
    return _propagate_jaxpr(jaxpr, in_sets)


@dataclass
class ModelTrace:
    sites: dict                    # name -> Site, in execution order
    parents: dict                  # name -> set of parent site/data names
    data_names: list               # array-like data entries used as inputs
    source_lines: bool             # True when site -> line mapping is available

    def children(self, name):
        return {s for s, ps in self.parents.items() if name in ps}


def trace_model(model, data=None, seed=0):
    """Run ``model`` once and return its exact structure as a :class:`ModelTrace`.

    Args:
        model: BF/NumPyro model function (Python, or an R/Julia model exposed
            to Python).
        data: dict of model arguments (typically ``m.data_on_model``).
        seed: seed for the single forward run.
    """
    data = dict(data or {})
    code = _model_code(model)

    # 1. Forward run: sites, distributions, shapes, plates, source lines,
    #    and concrete parameter values.
    rec = _LineRecorder(model, code)
    with handlers.seed(rng_seed=seed):
        tr = handlers.trace(rec).get_trace(**data)

    sites = {}
    values = {}           # site values substituted as inputs in step 2
    for name, s in tr.items():
        if s["type"] == "sample":
            observed = bool(s.get("is_observed"))
            if s["fn"] is None:
                continue
            d = s["fn"]
            pnames, leaves = _param_leaves(d)
            params = [Param(pn, np.asarray(leaf), set())
                      for pn, leaf in zip(pnames, leaves)]
            sites[name] = Site(
                name=name,
                kind=OBSERVED if observed else LATENT,
                dist=dist_label(d),
                shape=tuple(np.shape(s["value"])),
                plates=tuple(f.name for f in reversed(s.get("cond_indep_stack", []))),
                line=rec.lines.get(name),
                params=params,
                discrete=bool(getattr(_base_dist(d), "is_discrete", False)),
            )
            if not observed:
                values[name] = s["value"]
        elif s["type"] == "deterministic":
            sites[name] = Site(
                name=name, kind=DETERMINISTIC,
                shape=tuple(np.shape(s["value"])),
                plates=tuple(f.name for f in reversed(s.get("cond_indep_stack", []))),
                line=rec.lines.get(name),
            )
            values[name] = s["value"]

    # 2. Dependencies through jaxpr. Site values and array data are inputs;
    #    outputs are every site's distribution parameters (or, for a
    #    deterministic site, the value it was computed from).
    arrays, static = _split_data(data)
    out_index = []        # (site name, param index or -1) per flat output

    def mapping(site_values, array_data):
        out_index.clear()
        outs = []
        kwargs = {**static, **array_data}
        # Deterministic sites are substituted only AFTER their own value has
        # been recorded, so their inputs are traced too: record the raw value
        # passed to numpyro.deterministic through a messenger inside substitute.
        raw = {}

        class _RawDeterministic(Messenger):
            def process_message(self, msg):
                if msg["type"] == "deterministic":
                    raw[msg["name"]] = msg["value"]

        sub_values = {k: v for k, v in site_values.items()}
        with handlers.seed(rng_seed=seed):
            t = handlers.trace(
                handlers.substitute(_RawDeterministic(model), data=sub_values)
            ).get_trace(**kwargs)
        for name, s in t.items():
            if name not in sites:
                continue
            if s["type"] == "sample" and s["fn"] is not None:
                _, leaves = _param_leaves(s["fn"])
                for idx, leaf in enumerate(leaves):
                    out_index.append((name, idx))
                    outs.append(jnp.asarray(leaf))
            elif s["type"] == "deterministic" and name in raw:
                out_index.append((name, -1))
                outs.append(jnp.asarray(raw[name]))
        return outs

    in_names = []
    # tree_flatten sorts dict keys: rebuild the matching name list.
    for k in sorted(values):
        in_names.append(k)
    for k in sorted(arrays):
        in_names.append(k)

    parents = {name: set() for name in sites}
    try:
        closed = jax.make_jaxpr(mapping)(values, arrays)
        out_deps = _jaxpr_dependencies(closed.jaxpr, in_names)
        for (name, pidx), deps in zip(list(out_index), out_deps):
            deps = {d for d in deps if d != name}
            parents[name] |= deps
            if pidx >= 0 and pidx < len(sites[name].params):
                sites[name].params[pidx].parents |= deps
    except Exception as exc:  # pragma: no cover - reported, not fatal
        import warnings
        warnings.warn(f"model_trace: dependency tracing failed ({exc!r}); "
                      "the graph has nodes but no edges.", stacklevel=2)

    used = set().union(*parents.values()) if parents else set()
    data_names = [k for k in sorted(arrays) if k in used]
    return ModelTrace(sites=sites, parents=parents, data_names=data_names,
                      source_lines=code is not None and any(
                          s.line is not None for s in sites.values()))
