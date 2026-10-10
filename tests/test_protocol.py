"""The evaluation harness must treat every detector identically."""

import numpy as np

from prdbench.data import build_dataset
from prdbench.detectors import build


def _small():
    return build_dataset("stub", {"n_maps": 12, "n_range": 32, "n_doppler": 32,
                                  "snr_db": [8.0, 14.0]}, seed=0)


def test_every_detector_returns_a_per_cell_score():
    ds = _small()
    for kind in ("ca_cfar", "os_cfar", "go_cfar"):
        s = build(kind).score(ds.maps)
        assert s.shape == ds.maps.shape
        assert np.isfinite(s).all()
