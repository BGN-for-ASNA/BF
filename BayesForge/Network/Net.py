from BayesForge.Network.metrics import met
from BayesForge.Network.util import array_manip
from BayesForge.Network.model_effects import Neteffect
from BayesForge.Network.model_effects2 import NeteffectMatrix


class _LazyViz:
    """Descriptor: build the d3 ``NetExplorer`` front-end on first access only.

    ``NetExplorer.__init__`` raises ``FileNotFoundError`` when the bundled d3 /
    CSS assets are absent (e.g. a wheel built before the ``package-data`` entry
    shipped). ``net`` / ``net2`` are primarily metrics + model-effects classes,
    so that failure must not block construction — only an actual ``m.net.viz``
    call should surface it.
    """

    def __set_name__(self, owner, name):
        self._attr = f"_{name}_obj"

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        cached = getattr(obj, self._attr, None)
        if cached is None:
            from BayesForge.Network.explorer import NetExplorer

            cached = NetExplorer()
            setattr(obj, self._attr, cached)
        return cached

    def __set__(self, obj, value):
        setattr(obj, self._attr, value)


class net(met, Neteffect, array_manip):
    """The net class serves as a high-level interface for managing and utilizing various network metrics and effects within the BF framework.
    It encapsulates functionalities for computing clustering coefficients, eigenvector centrality, Dijkstra's algorithm for shortest paths, and other network metrics.
    Additionally, it extends the array_manip class to provide methods for handling network effects, including sender-receiver effects, dyadic effects, and block models.
    This class is designed to simplify the process of working with complex network structures, allowing users to easily compute metrics and model network interactions through a consistent API.
    """
    # d3.js network-visualisation front-end (R `NetExplorer` port), built lazily
    # on first access:  m.net.viz(df, adj, col_id="id", col_size="strength", ...)
    viz = _LazyViz()

    def __init__(self, *args, **kwargs):
        # Call super() without specifying the class name in a multiple inheritance context
        super().__init__(*args, **kwargs)


class net2(met, NeteffectMatrix, array_manip):
    """Matrix-form network interface — mirrors net but all effect outputs are (N, N).

    Use m.net2 for shard-compatible SRM models. Replaces edgelist scatter-gather
    with outer sums and matrix operations that shard cleanly along axis 0.
    See BayesForge.Network.model_effects2 for full documentation.
    """
    viz = _LazyViz()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


