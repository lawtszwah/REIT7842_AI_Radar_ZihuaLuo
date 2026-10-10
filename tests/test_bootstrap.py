"""Map resampling preserves radar eligibility and unequal map denominators."""

import numpy as np
import pytest

from prdbench.evaluation import bootstrap, metrics
from prdbench.simulation.interface import Target


@pytest.mark.parametrize("tol, exclude, guard", [(1.0, 3.0, 2), (0.5, 2.0, 1)])
def test_per_map_counts_reproduce_pooled_counts(tol, exclude, guard):
    scores = np.random.default_rng(123).integers(0, 5, size=(4, 12, 16)).astype(float)
    targets = [
        [Target(4, 13, 10), Target(6, 8, 8)],  # guarded target contributes no exclusion
        [Target(0.25, 0.25, 6), Target(1.5, 1.5, 8)],  # clipped, overlapping windows
        [],
        [Target(5, 8 + guard + tol, 9), Target(6.25, 14.25, 12)],  # guard boundary
    ]
    thresholds = {"0.001": 2.0, "0.01": 3.0, "0.1": 4.0}
    kwargs = {"tol": tol, "exclude": exclude, "zero_doppler_guard": guard}
    tgt, bg, _ = metrics.split_scores(scores, targets, **kwargs)
    counts = metrics.per_map_counts(scores, targets, thresholds, **kwargs)
    assert counts["n_targets"].tolist() == [1, 2, 0, 1]
    assert counts["n_background_cells"][2] == 12 * (16 - (2 * guard + 1))
    assert counts["n_targets"].sum() == len(tgt)
    assert counts["n_background_cells"].sum() == len(bg)
    for p, tau in thresholds.items():
        assert counts["detected"][p].sum() == np.count_nonzero(tgt > tau)
        assert counts["alarms"][p].sum() == np.count_nonzero(bg > tau)
        assert counts["detected"][p].sum() / len(tgt) == metrics.pd_at_threshold(tgt, tau)
        assert counts["alarms"][p].sum() / len(bg) == metrics.pfa_at_threshold(bg, tau)
    assert counts["detected"]["0.1"].sum() == counts["alarms"]["0.1"].sum() == 0


def test_identical_maps_collapse_intervals_to_the_point():
    one_map = np.ones((12, 16))
    one_map[3, 13] = 3.0
    scores = np.repeat(one_map[None], 5, axis=0)
    targets = [[Target(3, 13, 10), Target(8, 2, 6)] for _ in range(5)]
    thresholds = {"0.01": 2.0}
    counts = metrics.per_map_counts(scores, targets, thresholds)
    indices = bootstrap.map_indices(123, 5, 64)
    result = bootstrap.intervals(counts, indices, [2.5, 97.5])
    tgt, bg, _ = metrics.split_scores(scores, targets)
    for field, point in (("pd_interval", metrics.pd_at_threshold(tgt, 2.0)),
                         ("pfa_interval", metrics.pfa_at_threshold(bg, 2.0))):
        assert result[field]["0.01"] == [point, point]
    assert np.array_equal(indices, bootstrap.map_indices(123, 5, 64))


def test_resamples_sum_counts_before_dividing_and_handle_empty_targets():
    counts = {"n_targets": np.array([1, 9]), "n_background_cells": np.array([10, 30]),
              "detected": {"p": np.array([1, 0])}, "alarms": {"p": np.array([2, 0])}}
    indices = np.array([[0, 1], [0, 0], [1, 1]], dtype=np.int32)
    result = bootstrap.intervals(counts, indices, [0, 100])
    assert result == {"pd_interval": {"p": [0, 1]}, "pfa_interval": {"p": [0, 0.2]}}
    single = bootstrap.intervals(counts, indices[:1], [2.5, 97.5])
    assert single["pd_interval"]["p"] == [0.1, 0.1]
    assert single["pfa_interval"]["p"] == [0.05, 0.05]
    counts["n_targets"] = np.array([0, 9])
    counts["detected"]["p"] = np.array([0, 1])
    result = bootstrap.intervals(counts, indices, [0, 100])
    assert result["pd_interval"]["p"] == [1 / 9, 1 / 9]
    counts["n_targets"][:] = 0
    assert np.isnan(bootstrap.intervals(counts, indices, [2.5, 97.5])["pd_interval"]["p"]).all()
