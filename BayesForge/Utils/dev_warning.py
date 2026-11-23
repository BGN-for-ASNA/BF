import warnings

DEV_WARNING_MSG = "⚠️This function is still in development. Use it with caution. ⚠️"
_WARNED = set()


def warn_dev(key="default"):
    """Emit the standard BayesForge development warning at most once per key.

    The message is a ``RuntimeWarning``, so ``warnings.simplefilter("error", RuntimeWarning)``
    and ``pytest.warns`` work. The per-key anti-duplication set is kept;
    ``reset_dev_warnings()`` re-arms it (note that Python's own per-location
    warning registry can still deduplicate — tests should use
    ``warnings.catch_warnings()`` with ``simplefilter("always")`` or
    ``pytest.warns``).
    """
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(DEV_WARNING_MSG, RuntimeWarning, stacklevel=2)


def reset_dev_warnings():
    """Reset the set of emitted development warnings."""
    _WARNED.clear()



def dev_warning():
    """Display the standard development warning as a callout (for the docs).

    Used as ``dev_warning()`` in a hidden cell under a section, so every
    in-development function shows the identical message.
    """
    from IPython.display import Markdown, display
    display(Markdown("::: {.callout-warning}\n" + DEV_WARNING_MSG + "\n:::"))
