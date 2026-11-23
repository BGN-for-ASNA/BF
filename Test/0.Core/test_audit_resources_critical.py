"""Regression tests for RELEASE_AUDIT.md, module 15 (Resources), critical C-1..C-3."""

import os
import pytest

from BayesForge.Resources.datasets import load


def test_c1_no_loader_points_to_a_missing_file():
    ld = load()
    missing = []
    for name in dir(ld):
        if name.startswith("_") or not callable(getattr(ld, name)):
            continue
        fn = getattr(ld, name)
        try:
            p = fn(only_path=True)
        except TypeError:
            continue  # not a dataset loader
        if isinstance(p, (str, os.PathLike)) and not os.path.exists(p):
            missing.append(name)
    assert missing == []
    assert not hasattr(ld, "trolley")


@pytest.mark.parametrize("loader,cols,nrows", [
    ("WaffleDivorce", {"Location", "Divorce", "WaffleHouses"}, 50),
    ("elephants", {"AGE", "MATINGS"}, 41),
])
def test_c2_c3_default_frame_has_columns(loader, cols, nrows):
    df = getattr(load(), loader)()
    assert cols <= set(df.columns)
    assert df.shape[0] == nrows


@pytest.mark.parametrize("loader", ["WaffleDivorce", "elephants"])
def test_c2_c3_frame_and_bunch_share_schema(loader):
    ld = load()
    df = getattr(ld, loader)()
    b = getattr(ld, loader)(frame=False)
    assert list(df.columns) == list(b.feature_names)
    assert isinstance(getattr(ld, loader)(only_path=True), str)
