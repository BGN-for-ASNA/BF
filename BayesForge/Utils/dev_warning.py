DEV_WARNING_MSG = "⚠️This function is still in development. Use it with caution. ⚠️"
_WARNED = set()


def warn_dev(key="default"):
    """Print the standard BayesForge development warning at most once per key."""
    if key not in _WARNED:
        _WARNED.add(key)
        print(DEV_WARNING_MSG)


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
