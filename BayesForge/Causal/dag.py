"""Causal DAG tooling for BayesForge.

Pure-python (no networkx/graphviz dependency) implementation of the core
graphical-causal-inference primitives that *Statistical Rethinking* users
expect: d-separation, the back-door criterion, minimal adjustment-set search,
collider/mediator warnings and interventional (do-operator) graph mutilation.

Example
-------
>>> g = DAG("z -> x, z -> y, x -> y")          # confounder z
>>> g.adjustment_sets("x", "y")
[{'z'}]
>>> g.d_separated("x", "y", given={"z"})
False
>>> g = DAG("x -> m, m -> y, x -> y")         # m is a mediator x ~> y
>>> g.check("x", "y", given={"m"})
["conditioning on ['m'] (descendant(s) of x) can bias the effect",
 "conditioning on ['m'] blocks a directed path x ~> y (mediator/over-control)"]
"""

import warnings
from collections import deque
from itertools import combinations


def _warn_experimental():
    # One-shot per process via warnings' default "once per location" filter;
    # a UserWarning (not print) so it is capturable, filterable and
    # thread-safe (same pattern as PostModel.model_graph, m27).
    warnings.warn("Causal DAG tooling is an experimental feature. "
                  "Use it with caution.", UserWarning, stacklevel=2)


def _check_name(node):
    """Reject empty / whitespace-only node names (an empty name silently
    corrupts specs like ``"a ->"`` and makes labels invisible in plots)."""
    if isinstance(node, str) and not node.strip():
        raise ValueError("Node names must be non-empty strings")


def _parse_spec(spec):
    """Parse a DAG specification string into a list of (parent, child) edges.

    Accepts ``->`` / ``<-`` edges separated by commas, semicolons or newlines,
    e.g. ``"z -> x, z -> y; x -> y"`` or ``"y <- x"``.
    """
    edges = []
    for chunk in spec.replace("\n", ",").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "<->" in chunk:
            raise ValueError(f"Bidirected edge {chunk!r} is not supported: "
                             "model the latent confounder explicitly ('u -> x, u -> y')")
        n_ops = chunk.count("->") + chunk.count("<-")
        if n_ops > 1:
            raise ValueError(f"Cannot parse edge: {chunk!r} has {n_ops} arrows; "
                             "write one edge per chunk ('x -> m, w -> m')")
        if "<-" in chunk:
            rhs, lhs = chunk.split("<-")
            a, b = rhs.strip(), lhs.strip()
            if not a or not b:
                raise ValueError(f"Cannot parse edge: {chunk!r} (empty node name)")
            edges.append((b, a))  # lhs is the parent
        elif "->" in chunk:
            a, b = (s.strip() for s in chunk.split("->"))
            if not a or not b:
                raise ValueError(f"Cannot parse edge: {chunk!r} (empty node name)")
            edges.append((a, b))
        else:
            raise ValueError(f"Cannot parse edge: {chunk!r} (use 'a -> b' or 'b <- a')")
    return edges


def _as_node_set(nodes):
    """Normalise a node argument into a set of node names.

    ``None`` -> empty set; a string (or any non-iterable hashable) -> the
    single-node set ``{nodes}`` (so ``given="age"`` means the node ``"age"``,
    not its characters); any other iterable -> ``set(nodes)``.
    """
    if nodes is None:
        return set()
    if isinstance(nodes, str):
        return {nodes}
    try:
        return set(nodes)
    except TypeError:
        return {nodes}


class DAG:
    """A directed acyclic graph for causal reasoning.

    Args:
        spec: Either a specification string (``"z -> x, x -> y"``) or an
            iterable whose elements are ``(parent, child)`` pairs or
            ``"a -> b"`` strings (``["z -> x", "x -> y"]``). Anything else
            raises ``TypeError``.
    """

    def __init__(self, spec=None):
        _warn_experimental()
        self._parents = {}   # node -> set of parents
        self._children = {}  # node -> set of children
        if spec is not None:
            if isinstance(spec, str):
                edges = _parse_spec(spec)
            else:
                edges = []
                for item in spec:
                    if isinstance(item, str):
                        edges.extend(_parse_spec(item))
                    elif isinstance(item, (tuple, list)) and len(item) == 2:
                        edges.append((item[0], item[1]))
                    else:
                        raise TypeError(
                            "DAG edges must be (parent, child) pairs or "
                            f"'a -> b' strings, got {item!r}")
            for a, b in edges:
                self.add_edge(a, b)

    # -- construction ---------------------------------------------------------
    def _ensure(self, node):
        self._parents.setdefault(node, set())
        self._children.setdefault(node, set())

    def add_edge(self, parent, child):
        """Add a directed edge ``parent -> child`` (raises if it creates a cycle).

        The node names and the cycle risk are validated *before* the graph is
        mutated: a refused edge (self-loop, cycle, empty name) leaves the
        graph exactly as it was — no orphan nodes are added on the error path.

        Complexity: the cycle check is one reachability DFS from ``child``,
        O(V + E) per call, so building a graph of E edges costs O(E·(V+E)).
        """
        _check_name(parent)
        _check_name(child)
        if child in self._children.get(parent, ()):
            return self  # edge already present: idempotent, nothing to check
        if parent == child or self._reaches(child, parent):
            raise ValueError(f"Edge {parent} -> {child} introduces a cycle (not a DAG)")
        self._ensure(parent)
        self._ensure(child)
        self._children[parent].add(child)
        self._parents[child].add(parent)
        return self

    def _reaches(self, start, target):
        """True iff ``target`` is reachable from ``start`` through the *existing*
        edges. Iterative DFS (no recursion limit, O(V + E))."""
        if start == target:
            return True
        seen, stack = {start}, [start]
        while stack:
            for c in self._children.get(stack.pop(), ()):
                if c == target:
                    return True
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return False

    @property
    def nodes(self):
        return set(self._parents)

    @property
    def edges(self):
        """Edges as a list of ``(parent, child)`` tuples, sorted by ``str`` of
        both endpoints (the underlying sets iterate in hash order, which
        varies between processes; sorting keeps output reproducible)."""
        return sorted(((p, c) for p, cs in self._children.items() for c in cs),
                      key=lambda e: (str(e[0]), str(e[1])))

    def parents(self, node):
        return set(self._parents.get(node, set()))

    def children(self, node):
        return set(self._children.get(node, set()))

    # -- ancestry -------------------------------------------------------------
    def ancestors(self, nodes):
        nodes = _as_node_set(nodes)
        seen, stack = set(), list(nodes)
        while stack:
            n = stack.pop()
            for p in self._parents.get(n, set()):
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return seen

    def descendants(self, nodes):
        nodes = _as_node_set(nodes)
        seen, stack = set(), list(nodes)
        while stack:
            n = stack.pop()
            for c in self._children.get(n, set()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return seen

    def _require(self, *nodes):
        """Raise ValueError for node names that are not in the graph."""
        unknown = []
        for n in nodes:
            items = [n] if isinstance(n, str) or not hasattr(n, "__iter__") else list(n)
            unknown.extend(i for i in items if i not in self._parents)
        if unknown:
            raise ValueError(f"Unknown node(s) {sorted(map(str, unknown))}; "
                             f"graph nodes are {sorted(map(str, self._parents))}")

    # -- paths ----------------------------------------------------------------
    def _neighbors(self, node):
        return self._parents[node] | self._children[node]

    def _all_paths(self, x, y, max_paths=100_000):
        """Enumerate simple undirected paths between x and y (as node lists).

        Limit: enumeration stops with a ``RuntimeError`` once more than
        ``max_paths`` (default 100 000) simple paths would be produced;
        the graph is then too large for exhaustive path enumeration.

        Complexity: the number of simple paths is exponential in the graph
        size; each extension copies the path prefix (``path + [nb]``), so
        memory is O(number of paths × path length).
        """
        paths, stack = [], [(x, [x])]
        while stack:
            node, path = stack.pop()
            for nb in self._neighbors(node):
                if nb in path:
                    continue
                if nb == y:
                    paths.append(path + [nb])
                    if len(paths) > max_paths:
                        raise RuntimeError(
                            f"_all_paths found more than {max_paths} simple "
                            f"paths between {x} and {y}; the graph is too "
                            "large for exhaustive path enumeration")
                else:
                    stack.append((nb, path + [nb]))
        return paths

    def _edge_dir(self, a, b):
        """Return '->' if a is a parent of b, else '<-'."""
        return "->" if b in self._children[a] else "<-"

    def _is_open(self, path, given):
        """Is an undirected path d-connected (open) given the set ``given``?

        Complexity: O(len(path) · (V + E)) — ``descendants(b)`` is recomputed
        for every collider on the path, which matters to callers that run
        this over all paths for many conditioning sets (``check``).
        """
        given = set(given)
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            left = self._edge_dir(a, b)    # direction of a-b at b
            right = self._edge_dir(b, c)   # direction of b-c at b
            is_collider = (left == "->" and right == "<-")
            if is_collider:
                # Collider blocks unless b or a descendant of b is conditioned on.
                if b not in given and not (self.descendants(b) & given):
                    return False
            else:
                # Chain or fork: blocked iff the middle node is conditioned on.
                if b in given:
                    return False
        return True

    def d_separated(self, x, y, given=None):
        """True iff x and y are d-separated (conditionally independent) given ``given``.

        ``given`` may be a set of node names or a single node name as a string.
        Raises ``ValueError`` when ``x == y``: a node is never d-separated
        from itself (no simple path x↔x is enumerated, so the empty disjunction
        would wrongly answer ``True``).
        """
        self._require(x, y, given or [])
        if x == y:
            raise ValueError(f"{x!r} is not d-separated from itself")
        given = _as_node_set(given)
        return not any(self._is_open(p, given) for p in self._all_paths(x, y))

    # -- back-door criterion --------------------------------------------------
    def backdoor_paths(self, x, y):
        """All paths from x to y that start with an arrow *into* x (confounding paths)."""
        self._require(x, y)
        return [p for p in self._all_paths(x, y)
                if len(p) > 1 and self._edge_dir(p[0], p[1]) == "<-"]

    def is_valid_adjustment_set(self, x, y, z):
        """Check the back-door criterion for adjustment set ``z`` w.r.t. effect x -> y.

        Deviation from Pearl's standard criterion: the standard criterion only
        forbids ``z`` to contain ``x`` or a descendant of ``x``; this
        implementation additionally forbids ``y`` in ``z``. Conditioning on the
        outcome opens colliders at ``y`` and is statistically indefensible for
        an effect estimate, so such sets are rejected here (``z = {y}`` returns
        ``False`` even when it satisfies the textbook criterion).
        """
        self._require(x, y, z)
        z = set(z)
        if z & ({x, y} | self.descendants(x)):
            return False  # must not contain X, Y or any descendant of X
        return all(not self._is_open(p, z) for p in self.backdoor_paths(x, y))

    def adjustment_sets(self, x, y, max_size=None, minimal=True):
        """Find adjustment sets satisfying the back-door criterion for x -> y.

        Args:
            max_size: Cap on the size of candidate sets (default: None, all
                candidates, guarded by the 2**16 limit below).
            minimal: If True, return only minimal (non-redundant) valid sets.

        Returns:
            List of sets (smallest first). ``[set()]`` means no adjustment is
            needed; ``[]`` means no valid set exists (effect not identifiable by
            back-door adjustment).

        Limits:
            With ``max_size=None`` every subset of the candidates is tested
            (2**len(candidates) sets); a ``ValueError`` is raised when that
            exceeds 2**16 (more than 16 candidates) — pass ``max_size=`` to
            cap the search. Path enumeration inside each check is capped at
            100 000 simple paths (see :meth:`_all_paths`). Candidates exclude
            ``x``, ``y`` and the descendants of ``x`` (see
            :meth:`is_valid_adjustment_set` for why ``y`` is excluded).
        """
        self._require(x, y)
        candidates = self.nodes - {x, y} - self.descendants(x)
        # key=str: node sets may mix types (int/str); sorting them directly
        # would raise TypeError late and obscurely.
        candidates = sorted(candidates, key=str)
        if max_size is None and len(candidates) > 16:
            raise ValueError(
                f"adjustment_sets would test {2 ** len(candidates)} candidate "
                f"subsets of {len(candidates)} nodes (limit 2**16 = 65536); "
                "pass max_size= to cap the size of the candidate sets")
        hi = len(candidates) if max_size is None else min(max_size, len(candidates))
        valid = []
        for k in range(hi + 1):
            for combo in combinations(candidates, k):
                s = set(combo)
                if self.is_valid_adjustment_set(x, y, s):
                    if minimal and any(v <= s for v in valid):
                        continue  # superset of an already-found valid set
                    valid.append(s)
        return valid

    # -- intervention ---------------------------------------------------------
    def do(self, x):
        """Return the mutilated graph for ``do(x)`` (all arrows into x removed).

        ``x`` may be a set of node names or a single node name as a string.
        """
        self._require(x)
        x = _as_node_set(x)
        g = DAG()
        for node in self.nodes:
            g._ensure(node)
        for p, c in self.edges:
            if c in x:
                continue  # cut incoming arrows to intervened nodes
            g.add_edge(p, c)
        return g

    # -- diagnostics ----------------------------------------------------------
    def check(self, x, y, given=None):
        """Warn about common back-door mistakes when conditioning on ``given``.

        Flags: conditioning on a collider (opens a spurious path), on a mediator
        (blocks part of the causal effect), or on a descendant of X.
        Returns a list of human-readable warnings (empty if the set looks safe).
        ``given`` may be a set of node names or a single node name as a string.

        Complexity: the collider scan re-enumerates **all** x↔y paths for each
        conditioned node and re-tests d-connectivity (see :meth:`_all_paths`,
        :meth:`_is_open`) — O(len(given) × number of paths × (V + E)).
        """
        self._require(x, y, given or [])
        given = _as_node_set(given)
        msgs = []  # not `warnings`: the module name must stay reachable
        descendants_x = self.descendants(x)

        bad = sorted(given & descendants_x, key=str)
        if bad:
            msgs.append(
                f"conditioning on {bad} (descendant(s) of {x}) can bias the effect")

        # Mediator: a node on a directed path x ~> y.
        on_causal = self.descendants(x) & self.ancestors(y)
        med = sorted(given & on_causal, key=str)
        if med:
            msgs.append(
                f"conditioning on {med} blocks a directed path {x} ~> {y} "
                f"(mediator/over-control)")

        # Collider: conditioning opens a path that was closed when unconditioned.
        for node in sorted(given, key=str):
            without = given - {node}
            opened = [p for p in self._all_paths(x, y)
                      if self._is_open(p, given) and not self._is_open(p, without)]
            if opened:
                msgs.append(
                    f"conditioning on '{node}' opens a spurious (collider) path "
                    f"between {x} and {y}")
        return msgs

    # -- visualisation --------------------------------------------------------
    def _layers(self):
        """Assign each node a layer = longest directed distance from a root."""
        layer = {}
        order = self._topo_order()
        for n in order:
            ps = self._parents[n]
            layer[n] = 0 if not ps else 1 + max(layer[p] for p in ps)
        return layer

    def _topo_order(self):
        """Kahn's algorithm, O(V + E) with a deque (a list + pop(0) was O(V²))."""
        indeg = {n: len(self._parents[n]) for n in self.nodes}
        queue = deque(sorted([n for n, d in indeg.items() if d == 0]))
        order = []
        while queue:
            n = queue.popleft()
            order.append(n)
            for c in sorted(self._children[n]):
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        return order

    def plot(self, x=None, y=None, adjust=None, figsize=(7, 5), title=None,
             show=False):
        """Draw the DAG with matplotlib.

        Args:
            x, y: Optionally highlight the exposure (blue) and outcome (green).
            adjust: Node name or iterable of nodes to mark as adjusted-for
                (grey, boxed).
            figsize: Figure size.
            title: Optional title.
            show: If True, call ``plt.show()``; by default the figure is only
                built and the Axes returned (no blocking side effect in
                scripts, notebooks or CI).

        An empty DAG returns an empty figure (no crash on the layout maths).
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch

        adjust = _as_node_set(adjust)

        if not self._parents:  # empty graph: nothing to lay out
            fig, ax = plt.subplots(figsize=figsize)
            if title:
                ax.set_title(title)
            ax.set_aspect("equal")
            ax.axis("off")
            plt.tight_layout()
            if show:
                plt.show()
            return ax

        layer = self._layers()
        by_layer = {}
        for n, l in layer.items():
            by_layer.setdefault(l, []).append(n)

        pos = {}
        for l, nodes in by_layer.items():
            nodes = sorted(nodes)
            n = len(nodes)
            for i, node in enumerate(nodes):
                ypos = (n - 1) / 2 - i  # center the column vertically
                pos[node] = (l * 2.0, ypos)

        fig, ax = plt.subplots(figsize=figsize)
        for p, c in self.edges:
            x0, y0 = pos[p]
            x1, y1 = pos[c]
            # curve edges that skip layers so they don't hide behind
            # intermediate nodes (e.g. confounder triangles). A real edge
            # always spans >= 1 layer, so no separate "same row" case exists.
            span = abs(layer[c] - layer[p])
            rad = 0.3 * span if span > 1 else 0.0
            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18,
                shrinkA=16, shrinkB=16, color="#555555", lw=1.4, zorder=1,
                connectionstyle=f"arc3,rad={rad}")
            ax.add_patch(arrow)

        for node, (px, py) in pos.items():
            if node == x:
                fc, ec = "#9ecae1", "#3182bd"
            elif node == y:
                fc, ec = "#a1d99b", "#31a354"
            elif node in adjust:
                fc, ec = "#d9d9d9", "#969696"
            else:
                fc, ec = "white", "#333333"
            boxstyle = "square,pad=0.4" if node in adjust else "circle,pad=0.4"
            ax.text(px, py, node, ha="center", va="center", fontsize=11, zorder=2,
                    bbox=dict(boxstyle=boxstyle, fc=fc, ec=ec, lw=1.6))

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
        ax.set_ylim(min(ys) - 1.5, max(ys) + 1.5)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title)
        plt.tight_layout()
        if show:
            plt.show()
        return ax

    def __repr__(self):
        # self.edges is sorted by str, so the repr is stable across processes
        # (iterating the raw sets would vary with hash randomization).
        e = ", ".join(f"{p} -> {c}" for p, c in self.edges)
        return f"DAG({e})"


class causal:
    """High-level entry point exposed as ``m.causal`` on a BF instance."""

    DAG = DAG

    def dag(self, spec=None):
        """Build a :class:`DAG` from a spec string or edge list."""
        return DAG(spec)
