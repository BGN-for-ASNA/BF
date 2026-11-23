import importlib

class LazyImporter:
    """
    Imports heavy plotting libraries lazily and synchronously.

    ``schedule_import()`` only records the ``alias -> module_name`` mapping;
    the actual ``importlib.import_module()`` happens on the first
    ``get_module()`` call and the module is cached. No thread pool is used:
    importing related packages concurrently can deadlock, and it turned
    ImportErrors into late surprises raised from the background thread.

    RELEASE_AUDIT (Utils, ImportManager) resolution notes:
    * ``get_module`` has no ``timeout`` on purpose — there is no future to
      wait on any more, the import runs in the caller's thread, so a frozen
      import freezes nothing that a plain ``import`` would not.
    * Deferred errors are gone with the executor: the ImportError is raised
      by the ``get_module`` call itself, with the missing package named and
      an install hint, chained (``__cause__``) to the original ImportError.
    * "Executor never closed / close it with atexit" no longer applies:
      there is no executor to close, so no ``atexit`` hook is registered
      either. ``shutdown()`` stays as a no-op for backward compatibility.
    """
    def __init__(self):
        self._scheduled = {}  # alias -> module_name
        self._modules = {}    # alias -> imported module (cache)

    def schedule_import(self, module_name, alias):
        """Records a module to import on first use (no thread, no import yet)."""
        self._scheduled.setdefault(alias, module_name)

    def get_module(self, alias):
        """
        Returns the module registered under ``alias``, importing it on first use.

        Raises ImportError with an install hint if the module is unavailable.
        """
        if alias in self._modules:
            return self._modules[alias]
        if alias not in self._scheduled:
            raise ImportError(f"Module for alias '{alias}' was not scheduled for import.")

        name = self._scheduled[alias]
        try:
            module = importlib.import_module(name)
        except ImportError as e:
            raise ImportError(
                f"{name} is needed for this plot (pip install {name.split('.')[0]})"
            ) from e
        self._modules[alias] = module
        return module

    def shutdown(self):
        """No-op kept for backward compatibility (there is no executor any more)."""
        pass
