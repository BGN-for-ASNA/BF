"""float_precision follow-up: switching the process-global x64 between bf() instances warns."""
import warnings

import jax
import pytest

import BayesForge.SetDevice.set as set_mod
from BayesForge.SetDevice.set import setup_device


@pytest.fixture
def restore_x64():
    before = bool(jax.config.jax_enable_x64)
    flag = set_mod._X64_CONFIGURED
    yield
    jax.config.update("jax_enable_x64", before)
    set_mod._X64_CONFIGURED = flag


def _x64_warnings(record):
    return [w for w in record if "process-global jax_enable_x64" in str(w.message)]


def test_switching_precision_warns(restore_x64):
    jax.config.update("jax_enable_x64", False)
    set_mod._X64_CONFIGURED = True  # an earlier bf() already configured x64
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        setup_device(platform="cpu", float_precision=64, print_devices_found=False)
    hits = _x64_warnings(rec)
    assert len(hits) == 1 and issubclass(hits[0].category, RuntimeWarning)
    assert jax.config.jax_enable_x64


def test_same_precision_and_first_call_are_quiet(restore_x64):
    jax.config.update("jax_enable_x64", True)
    set_mod._X64_CONFIGURED = True
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        setup_device(platform="cpu", float_precision=64, print_devices_found=False)
    assert not _x64_warnings(rec)
    # First configuration of the process: no warning even if the value changes.
    set_mod._X64_CONFIGURED = False
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        setup_device(platform="cpu", float_precision=32, print_devices_found=False)
    assert not _x64_warnings(rec)
    assert not jax.config.jax_enable_x64
