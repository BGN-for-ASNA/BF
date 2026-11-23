"""RELEASE_AUDIT X13: m.data() accepts the documented frame=False Bunch."""
import numpy as np
import pytest

from BayesForge import bf
from BayesForge.Resources.datasets import Bunch


def test_data_accepts_a_frame_false_bunch():
    m = bf(platform="cpu", print_devices_found=False)
    bunch = m.load.howell1(frame=False)
    df = m.data(bunch)
    ref = m.load.howell1()
    assert list(df.columns) == list(ref.columns)
    np.testing.assert_allclose(df.values.astype(float), ref.values.astype(float))


def test_bunch_without_frame_gives_a_clear_error():
    m = bf(platform="cpu", print_devices_found=False)
    with pytest.raises(ValueError, match="frame=True"):
        m.data(Bunch(data=np.zeros((4, 2)), feature_names=["a", "b"]))
