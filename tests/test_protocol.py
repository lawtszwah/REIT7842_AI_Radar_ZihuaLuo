"""The evaluation harness must treat every detector identically."""

import numpy as np

from prdbench.data import build_dataset
from prdbench.detectors import build
from prdbench.evaluation import metrics


def _small():
    return build_dataset("stub", {"n_maps": 12, "n_range": 32, "n_doppler": 32,
                                  "snr_db": [8.0, 14.0]}, seed=0)


def test_every_detector_returns_a_per_cell_score():
    ds = _small()
    for kind in ("ca_cfar", "os_cfar", "go_cfar"):
        s = build(kind).score(ds.maps)
        assert s.shape == ds.maps.shape
        assert np.isfinite(s).all()


def test_pd_is_monotone_in_pfa():
    ds = _small()
    tgt, bg, _ = metrics.split_scores(build("ca_cfar").score(ds.maps), ds.targets)
    pds = [metrics.pd_at_pfa(tgt, bg, p) for p in (1e-4, 1e-3, 1e-2, 1e-1)]
    # strict=False is deliberate: successive pairs, so the operands differ in length.
    assert all(a <= b + 1e-9 for a, b in zip(pds, pds[1:], strict=False))


def test_strong_targets_are_detectable():
    ds = build_dataset("stub", {"n_maps": 24, "n_range": 32, "n_doppler": 32,
                                "snr_db": [12.0, 16.0], "clutter_patches": [0, 0]}, seed=1)
    tgt, bg, _ = metrics.split_scores(build("ca_cfar").score(ds.maps), ds.targets)
    assert metrics.pd_at_pfa(tgt, bg, 1e-2) > 0.5
