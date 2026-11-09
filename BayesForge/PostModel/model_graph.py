"""Probabilistic-model graph extraction & plotting for BayesForge.

Reconstructs the directed graph of a BF/NumPyro probabilistic model using a
hybrid of AST analysis and execution-based JAX tracing (``model_trace``):

    * stochastic (latent) sample sites        -- ellipse, white
    * observed sample sites (``obs=`` given)  -- ellipse, shaded
    * deterministic assignments               -- rounded box
    * data / model inputs (function arguments) -- plain box

When ``data`` is provided (automatic on ``m.graph()`` after ``m.fit()``),
``trace_model`` runs the model once through ``numpyro.handlers.trace`` and
``jax.make_jaxpr`` so that sites inside BF building blocks (``m.effects``,
``m.net``), ``for`` loops, helper functions, and R/Julia models (which have
no Python source) are captured with exact dependency edges. When called on an
unfitted Python model without ``data``, an enhanced AST analyzer handles
tuple unpacking, loop recurrences (``z.append(...)``), augmented assignments,
and static expansion of BF building blocks.

Usage from a BF instance::

    m.graph()              # graph of the last-fitted / current model
    m.graph(my_model)      # graph of an arbitrary model function
"""

import ast
import inspect
import textwrap

from BayesForge.PostModel.model_trace import (
    LATENT,
    OBSERVED,
    DETERMINISTIC,
    DATA,
    dist_snake_name,
    trace_model,
)


_WARNED = False


def _warn_experimental():
    global _WARNED
    if not _WARNED:
        print("[WARNING] Probabilistic model-graph plotting is an experimental "
              "feature. Use it with caution. [WARNING]")
        _WARNED = True


def _names_in(node):
    """Collect all bare ``Name`` identifiers referenced inside an AST node."""
    if node is None:
        return []
    callee_root = set()
    out = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            # Exclude the root object/function names of calls (e.g. `m`, `jnp`, `numpyro`)
            cur = sub.func
            while isinstance(cur, ast.Attribute):
                cur = cur.value
            if isinstance(cur, ast.Name):
                callee_root.add(id(cur))
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and id(sub) not in callee_root:
            out.append(sub.id)
    return out


def _callee_name(call):
    """Dotted name of a call's callee, or '' when it is not a plain name."""
    try:
        return ast.unparse(call.func)
    except Exception:
        return ""


# NumPyro's own primitives; their site name is the FIRST POSITIONAL argument.
_NUMPYRO_SITE_FNS = ("numpyro.sample", "sample", "numpyro.deterministic",
                     "deterministic", "numpyro.param", "param")


def _is_numpyro_site(node):
    return (isinstance(node, ast.Call)
            and _callee_name(node) in _NUMPYRO_SITE_FNS
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str))


def _is_numpyro_deterministic(node):
    return (isinstance(node, ast.Call)
            and _callee_name(node) in ("numpyro.deterministic", "deterministic")
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str))


def _is_dist_call(node):
    """True for a BF ``m.dist.xxx(...)`` call or a NumPyro sample/deterministic."""
    if not isinstance(node, ast.Call):
        return False
    if _is_numpyro_site(node):
        return True
    name = _callee_name(node)
    return any(part == "dist" for part in name.split("."))


def _dist_name(call):
    """The distribution name, e.g. ``m.dist.normal(...)`` -> 'normal'."""
    if _is_numpyro_deterministic(call):
        return None
    if _is_numpyro_site(call):
        for arg in call.args[1:]:
            if isinstance(arg, ast.Call):
                return _dist_name(arg)
        return _callee_name(call).split(".")[-1]
    func = call.func
    if isinstance(func, ast.Attribute):
        return dist_snake_name(func.attr)
    return dist_snake_name(getattr(func, "id", _callee_name(call)))


def _call_meta(call):
    """Return (site_name, obs_value_name, dependency_names) for a dist call."""
    name_kw, obs_name = None, None
    dep_nodes = list(call.args)

    if _is_numpyro_site(call):
        name_kw = call.args[0].value
        dep_nodes = list(call.args[1:])

    for kw in call.keywords:
        if kw.arg == "name" and isinstance(kw.value, ast.Constant):
            name_kw = kw.value.value
        elif kw.arg == "obs":
            obs_name = (kw.value.id if isinstance(kw.value, ast.Name)
                        else ast.unparse(kw.value))
        elif kw.arg in ("shape", "sample", "to_jax", "create_obj", "rng_key", "event"):
            continue  # structural, not a probabilistic dependency
        else:
            dep_nodes.append(kw.value)
    deps = []
    for n in dep_nodes:
        deps.extend(_names_in(n))
    return name_kw, obs_name, deps


def _kw_str(call, key, default):
    for kw in call.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return default


def _kw_bool(call, key, default=False):
    for kw in call.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
            return kw.value.value
    return default


def _expand_bf_block_static(call):
    """If ``call`` is a high-level BF building block (``m.effects.*``, ``m.net.*``,
    ``m.gaussian.*``), return ``(latent_sites, index_deps)`` where:
        * ``latent_sites`` is a list of ``(site_name, dist_name, parent_names)``
        * ``index_deps`` is a list of referenced AST variable names (e.g. ``group_id``)
          that feed the returned effect array.
    Returns ``None`` if ``call`` is not a known building block.
    """
    if not isinstance(call, ast.Call):
        return None
    cname = _callee_name(call)
    parts = cname.split(".")
    if len(parts) < 2:
        return None
    submod, fn = parts[-2], parts[-1]
    all_refs = _names_in(call)

    if submod == "effects":
        if fn == "varying_effects":
            gname = _kw_str(call, "group_name", "age")
            centered = _kw_bool(call, "centered", False)
            sites = [
                ("global_intercept", "normal", []),
                ("global_beta", "normal", []),
                (f"{gname}_sd_intercept", "exponential", []),
                (f"{gname}_sd_beta", "exponential", []),
            ]
            if centered:
                sites.append((f"corr_{gname}", "lkj", []))
                sites.append((f"{gname}_mvn", "multivariate_normal",
                              ["global_intercept", "global_beta",
                               f"{gname}_sd_intercept", f"{gname}_sd_beta",
                               f"corr_{gname}"]))
            else:
                sites.append((f"{gname}_L_corr", "lkj_cholesky", []))
                sites.append((f"{gname}_z", "normal", []))
            return sites, all_refs
        if fn in ("varying_intercept", "varying_intercept_non_centered"):
            gname = _kw_str(call, "group_name", "group")
            sites = [
                (f"global_intercept_{gname}", "normal", []),
                (f"sd_{gname}", "exponential", []),
            ]
            if fn == "varying_intercept":
                sites.append((f"intercept_{gname}", "normal",
                              [f"global_intercept_{gname}", f"sd_{gname}"]))
            else:
                sites.append((f"z_intercept_{gname}", "normal", []))
            return sites, all_refs
        if fn in ("varying_slope", "varying_slope_non_centered"):
            gname = _kw_str(call, "group_name", "group")
            sites = [
                (f"L_Rho_{gname}", "lkj_cholesky", []),
                (f"sigma_{gname}", "exponential", []),
                (f"z_{gname}", "normal", []),
            ]
            return sites, all_refs
        if fn in ("varying_intercept_slope", "varying_intercept_slope_non_centered"):
            gname = _kw_str(call, "group_name", "group")
            sites = [
                (f"{gname}_std" if fn == "varying_intercept_slope" else f"{gname}_sigma", "exponential", []),
                (f"L_corr_{gname}" if fn == "varying_intercept_slope" else f"L_Rho_{gname}", "lkj_cholesky", []),
                (f"{gname}_params" if fn == "varying_intercept_slope" else f"z_{gname}",
                 "multivariate_normal" if fn == "varying_intercept_slope" else "normal", []),
            ]
            return sites, all_refs
        if fn == "phylo_effect":
            sname = _kw_str(call, "name", "z_phylo")
            return [(sname, "normal", [])], all_refs
        if fn == "nested_varying_effects":
            sites = [
                ("global_intercept", "normal", []),
                ("global_beta", "normal", []),
            ]
            for kw in call.keywords:
                if kw.arg == "names" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    for elt in kw.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            nm = elt.value
                            sites.append((f"sigma_{nm}", "exponential", []))
                            sites.append((f"L_corr_{nm}", "lkj_cholesky", []))
                            sites.append((f"z_{nm}", "normal", []))
            return sites, all_refs

    if submod in ("net", "net2"):
        if fn == "sender_receiver":
            sites = [
                ("sender_effects", "normal", []),
                ("receiver_effects", "normal", []),
                ("sr_raw", "normal", []),
                ("sr_sigma", "truncated_normal", []),
                ("sr_L", "lkj_cholesky", []),
            ]
            return sites, all_refs
        if fn == "nodes_random_effects":
            return [
                ("sr_raw", "normal", []),
                ("sr_sigma", "truncated_normal", []),
                ("sr_L", "lkj_cholesky", []),
            ], all_refs
        if fn == "nodes_terms":
            return [
                ("sender_effects", "normal", []),
                ("receiver_effects", "normal", []),
            ], all_refs
        if fn in ("dyadic_effect", "dyadic_random_effects"):
            sites = [
                ("dr_raw", "normal", []),
                ("dr_sigma", "truncated_normal", []),
                ("dr_L", "lkj_cholesky", []),
            ]
            if fn == "dyadic_effect":
                sites.insert(0, ("dyad_effects", "normal", []))
            return sites, all_refs
        if fn == "block_model":
            bname = _kw_str(call, "name", "")
            return [(f"b_{bname}", "normal", [])], all_refs
        if fn == "edge_weights":
            ename = _kw_str(call, "name", "edge_weight")
            return [
                (f"{ename}_sigma", "half_normal", []),
                (ename, "normal", []),
            ], all_refs
        if fn == "nodal_fixed_effects":
            fname = _kw_str(call, "name", "beta_fixed")
            return [(fname, "normal", [])], all_refs
        if fn == "nodal_random_effects":
            rname = _kw_str(call, "name", "beta_random")
            return [
                (f"{rname}_group_mu", "normal", []),
                (f"{rname}_group_sigma", "half_normal", []),
                (rname, "normal", [f"{rname}_group_mu", f"{rname}_group_sigma"]),
            ], all_refs

    if submod == "gaussian" and fn == "gaussian_process":
        return [
            ("etasq", "exponential", []),
            ("rhosq", "exponential", []),
            ("kernel", "multivariate_normal", ["etasq", "rhosq"] + all_refs),
        ], all_refs

    return None


class ModelGraph:
    """A parsed probabilistic-model graph with a matplotlib renderer."""

    def __init__(self, nodes, edges, kinds, dist_of, shapes=None, plates=None):
        self.nodes = nodes          # ordered list of node names
        self.edges = edges          # list of (src, dst)
        self.kinds = kinds          # name -> kind
        self.dist_of = dist_of      # name -> distribution label (or None)
        self.shapes = dict(shapes or {})
        self.plates = dict(plates or {})

    # -- layout ---------------------------------------------------------------
    def _layers(self):
        parents = {n: set() for n in self.nodes}
        children = {n: set() for n in self.nodes}
        for a, b in self.edges:
            if a in parents and b in parents:
                parents[b].add(a)
                children[a].add(b)
        # longest-path layering over a topological order
        indeg = {n: len(parents[n]) for n in self.nodes}
        queue = [n for n in self.nodes if indeg[n] == 0]
        order, layer = [], {n: 0 for n in self.nodes}
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c in children[n]:
                layer[c] = max(layer[c], layer[n] + 1)
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) < len(self.nodes):
            import warnings
            stuck = [n for n in self.nodes if n not in set(order)]
            warnings.warn(
                f"model graph contains a cycle through {stuck}; those nodes are "
                "drawn in the first layer.", stacklevel=2)
        return layer

    def draw(self, figsize=(8, 6), title=None, show=True):
        """Render the model graph with matplotlib.

        Args:
            show: call plt.show(). Set False to compose the axes yourself or to
                render headless.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        layer = self._layers()
        by_layer = {}
        for n in self.nodes:
            by_layer.setdefault(layer[n], []).append(n)

        pos = {}
        for l, nodes in by_layer.items():
            k = len(nodes)
            for i, node in enumerate(nodes):
                pos[node] = (l * 2.2, (k - 1) / 2 - i)

        style = {
            LATENT:        dict(fc="white",   ec="#333333", box="ellipse"),
            OBSERVED:      dict(fc="#9ecae1", ec="#3182bd", box="ellipse"),
            DETERMINISTIC: dict(fc="#fff2cc", ec="#d6b656", box="round"),
            DATA:          dict(fc="#eeeeee", ec="#999999", box="square"),
        }

        fig, ax = plt.subplots(figsize=figsize)
        for a, b in self.edges:
            if a not in pos or b not in pos:
                continue
            x0, y0 = pos[a]
            x1, y1 = pos[b]
            ax.add_patch(FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
                shrinkA=20, shrinkB=20, color="#666666", lw=1.3, zorder=1))

        for node, (px, py) in pos.items():
            st = style[self.kinds[node]]
            boxstyle = {
                "ellipse": "circle,pad=0.45",
                "round": "round,pad=0.45",
                "square": "square,pad=0.4",
            }[st["box"]]
            label = node
            d = self.dist_of.get(node)
            if d:
                label = f"{node}\n~{d}"
            ax.text(px, py, label, ha="center", va="center", fontsize=9, zorder=2,
                    bbox=dict(boxstyle=boxstyle, fc=st["fc"], ec=st["ec"], lw=1.5))

        xs = [p[0] for p in pos.values()] or [0]
        ys = [p[1] for p in pos.values()] or [0]
        ax.set_xlim(min(xs) - 1.6, max(xs) + 1.6)
        ax.set_ylim(min(ys) - 1.6, max(ys) + 1.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title or "Probabilistic model graph")

        from matplotlib.patches import Patch
        legend = [
            Patch(fc="white", ec="#333333", label="latent"),
            Patch(fc="#9ecae1", ec="#3182bd", label="observed"),
            Patch(fc="#fff2cc", ec="#d6b656", label="deterministic"),
            Patch(fc="#eeeeee", ec="#999999", label="data/input"),
        ]
        ax.legend(handles=legend, loc="upper left", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        if show:
            plt.show()
        return ax

    def __repr__(self):
        e = ", ".join(f"{a}->{b}" for a, b in self.edges)
        return f"ModelGraph(nodes={len(self.nodes)}, edges=[{e}])"


def _graph_from_trace(tr):
    """Build a :class:`ModelGraph` purely from a :class:`ModelTrace`."""
    nodes = []
    kinds = {}
    dist_of = {}
    shapes = {}
    plates = {}

    for dname in tr.data_names:
        nodes.append(dname)
        kinds[dname] = DATA
        dist_of[dname] = None

    for name, site in tr.sites.items():
        if name not in kinds:
            nodes.append(name)
        kinds[name] = site.kind
        dist_of[name] = dist_snake_name(site.dist) if site.dist else None
        shapes[name] = site.shape
        plates[name] = site.plates

    edges = []
    known = set(nodes)
    for name in tr.sites:
        for p in tr.parents.get(name, set()):
            if p in known and p != name:
                edges.append((p, name))
    edges = list(dict.fromkeys(edges))
    return ModelGraph(nodes, edges, kinds, dist_of, shapes=shapes, plates=plates)


def _lhs_names(target):
    """Extract assigned variable names from an AST target (Name, Tuple, List, Subscript)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for elt in target.elts:
            out.extend(_lhs_names(elt))
        return out
    if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
        return [target.value.id]
    return []


def model_to_graph(model, data=None):
    """Parse a model function into a :class:`ModelGraph`.

    Args:
        model: A BF/NumPyro model function (Python, or an R/Julia model when
            ``data`` is provided).
        data: Optional dict of model inputs (e.g. ``m.data_on_model``). When
            supplied, runs ``trace_model(model, data)`` to capture exact JAX
            dependencies, dynamic sites inside loops/building blocks, and
            R/Julia models without Python source.
    """
    _warn_experimental()

    tr = None
    if data:
        try:
            tr = trace_model(model, data)
        except Exception:
            tr = None

    try:
        src = textwrap.dedent(inspect.getsource(model))
        tree = ast.parse(src)
        func = next((n for n in ast.walk(tree)
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    except Exception:
        func = None

    if func is None:
        if tr is not None:
            return _graph_from_trace(tr)
        raise ValueError("Could not find a function definition in the model source "
                         "and no model trace data was provided.")

    # Map relative AST line numbers to absolute code line numbers if trace is available.
    line_offset = 0
    if tr is not None and tr.source_lines:
        try:
            _, start_line = inspect.getsourcelines(model)
            line_offset = start_line - func.lineno
        except Exception:
            line_offset = 0

    a = func.args
    all_args = list(getattr(a, "posonlyargs", [])) + list(a.args) + list(a.kwonlyargs)
    if a.vararg:
        all_args.append(a.vararg)
    if a.kwarg:
        all_args.append(a.kwarg)
    inputs = [x.arg for x in all_args if x.arg != "self"]

    nodes = []
    kinds = {}
    dist_of = {}
    deps = {}          # node -> list of referenced names
    var_to_node = {}   # python LHS variable -> node name (often identical)
    shapes = {}
    plates = {}
    line_to_lhs = {}   # absolute line number -> list of LHS variable names

    _rank = {DATA: 0, DETERMINISTIC: 1, LATENT: 2, OBSERVED: 3}

    def add_node(name, kind, dist=None, depends=()):
        if name not in kinds:
            nodes.append(name)
            kinds[name] = kind
            dist_of[name] = dist
            deps[name] = list(depends)
            return
        if _rank.get(kind, 0) >= _rank.get(kinds[name], 0):
            kinds[name] = kind
            if dist is not None:
                dist_of[name] = dist
        deps[name] = list(dict.fromkeys(list(deps.get(name, [])) + list(depends)))

    for inp in inputs:
        add_node(inp, DATA)

    loop_vars = set()

    def _iter_stmts(body):
        for stmt in body:
            if isinstance(stmt, ast.For):
                for lv in _lhs_names(stmt.target):
                    loop_vars.add(lv)
            yield stmt
            for attr in ("body", "orelse", "finalbody"):
                inner = getattr(stmt, attr, None)
                if isinstance(inner, list):
                    yield from _iter_stmts(inner)
            for handler in getattr(stmt, "handlers", []) or []:
                yield from _iter_stmts(handler.body)

    for stmt in _iter_stmts(func.body):
        abs_line = getattr(stmt, "lineno", 0) + line_offset

        # 1. `lhs = <something>` (single Name, Tuple/List unpacking, or Subscript `z[t] = ...`)
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            lhs_list = _lhs_names(target)
            if not lhs_list:
                continue
            line_to_lhs[abs_line] = lhs_list

            if len(lhs_list) == 1 and isinstance(target, ast.Name) and _is_dist_call(stmt.value):
                lhs = lhs_list[0]
                name_kw, obs_name, d = _call_meta(stmt.value)
                node = name_kw or lhs
                if _is_numpyro_deterministic(stmt.value):
                    kind = DETERMINISTIC
                else:
                    kind = OBSERVED if obs_name else LATENT
                add_node(node, kind, _dist_name(stmt.value), d)
                var_to_node[lhs] = node
                if obs_name:
                    var_to_node.setdefault(obs_name, node)
                continue

            # Check if RHS is a BF high-level building block (e.g. m.effects.varying_effects)
            block_info = _expand_bf_block_static(stmt.value) if tr is None else None
            if block_info is not None:
                block_sites, index_refs = block_info
                block_site_names = []
                for sname, sdist, sparents in block_sites:
                    add_node(sname, LATENT, sdist, sparents)
                    block_site_names.append(sname)
                for lhs in lhs_list:
                    add_node(lhs, DETERMINISTIC, None, block_site_names + index_refs)
                    var_to_node[lhs] = lhs
                continue

            # Standard deterministic assignment (including tuple unpacking or subscript `z[t] = ...`)
            rhs_deps = _names_in(stmt.value)
            if isinstance(target, ast.Subscript):
                rhs_deps = rhs_deps + _names_in(target.slice)
            for lhs in lhs_list:
                add_node(lhs, DETERMINISTIC, None, rhs_deps)
                var_to_node.setdefault(lhs, lhs)

        # 2. `lhs += <something>` (AugAssign)
        elif isinstance(stmt, ast.AugAssign):
            lhs_list = _lhs_names(stmt.target)
            rhs_deps = _names_in(stmt.value)
            for lhs in lhs_list:
                add_node(lhs, DETERMINISTIC, None, rhs_deps)
                var_to_node.setdefault(lhs, lhs)

        # 3. Bare expression: `m.dist.xxx(..., obs=y)` or `z.append(expr)`
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if _is_dist_call(call):
                name_kw, obs_name, d = _call_meta(call)
                if _is_numpyro_deterministic(call):
                    node = name_kw or "deterministic"
                    add_node(node, DETERMINISTIC, None, d)
                else:
                    node = name_kw or obs_name or f"obs_{_dist_name(call)}"
                    add_node(node, OBSERVED if obs_name else LATENT,
                             _dist_name(call), d)
                if obs_name:
                    var_to_node.setdefault(obs_name, node)
            elif (isinstance(call.func, ast.Attribute)
                  and call.func.attr in ("append", "extend", "insert")
                  and isinstance(call.func.value, ast.Name)):
                # e.g. `z.append(z[t - 1] * 0.9)` inside a for loop
                target_var = call.func.value.id
                arg_deps = []
                for arg in call.args:
                    arg_deps.extend(_names_in(arg))
                add_node(target_var, DETERMINISTIC, None, arg_deps)
                var_to_node.setdefault(target_var, target_var)

    # Merge execution trace if available (adds sites inside m.effects / m.net / loops / helpers)
    if tr is not None:
        for name, site in tr.sites.items():
            sdist = dist_snake_name(site.dist) if site.dist else None
            add_node(name, site.kind, sdist, list(tr.parents.get(name, ())))
            shapes[name] = site.shape
            plates[name] = site.plates
            # If this site was created inside a helper / building block on line L
            # (e.g. `alpha, beta = m.effects.varying_effects(...)`), connect the
            # site to the LHS variables assigned on line L!
            if site.line in line_to_lhs:
                for lhs in line_to_lhs[site.line]:
                    mapped = var_to_node.get(lhs, lhs)
                    if mapped != name and mapped in kinds:
                        deps[mapped] = list(dict.fromkeys(list(deps.get(mapped, [])) + [name]))

    # Resolve dependency names to actual node names and build edges
    edges = []
    known = set(nodes)
    for node in nodes:
        for ref in deps.get(node, []):
            if ref in loop_vars and ref not in inputs:
                continue
            target = var_to_node.get(ref, ref)
            if target in known and target != node:
                edges.append((target, node))
    edges = list(dict.fromkeys(edges))  # dedupe, preserve order

    # If trace is available and both a direct trace edge (a -> y) and an AST
    # path through deterministic local variables (a -> mu -> y) exist, drop the
    # redundant bypass edge (a -> y) so intermediate deterministic nodes stay
    # cleanly in the middle of the graph.
    if tr is not None:
        det_nodes = {n for n, k in kinds.items() if k == DETERMINISTIC and n not in tr.sites}
        if det_nodes:
            adj = {n: set() for n in nodes}
            for u, v in edges:
                adj[u].add(v)

            def _reaches_via_det(src, dst):
                stack = [nxt for nxt in adj.get(src, ()) if nxt in det_nodes]
                seen = set(stack)
                while stack:
                    cur = stack.pop()
                    if dst in adj.get(cur, ()):
                        return True
                    for nxt in adj.get(cur, ()):
                        if nxt not in seen and nxt in det_nodes:
                            seen.add(nxt)
                            stack.append(nxt)
                return False

            edges = [(u, v) for (u, v) in edges if not _reaches_via_det(u, v)]

    # Drop data/input nodes (and unreferenced loop temporaries) that are never referenced
    referenced = {a for a, _ in edges} | {b for _, b in edges}
    pruned = [n for n in nodes
              if kinds[n] != DATA or n in referenced]
    edges = [(a, b) for a, b in edges if a in pruned and b in pruned]

    return ModelGraph(pruned, edges, kinds, dist_of, shapes=shapes, plates=plates)
