"""Percentile intervals from map counts, with bounded temporary memory."""

from __future__ import annotations

import numpy as np


def map_indices(seed: int, n_maps: int, resamples: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(
        0, n_maps, size=(resamples, n_maps), dtype=np.int32,
    )


def _rates(numerator: np.ndarray, denominator: np.ndarray, indices: np.ndarray) -> np.ndarray:
    rates = np.full(len(indices), np.nan)
    # Work on map counts in batches; never construct per-cell bootstrap arrays.
    for start in range(0, len(indices), 256):
        selected = indices[start:start + 256]
        totals = denominator[selected].sum(axis=1)
        hits = numerator[selected].sum(axis=1)
        np.divide(hits, totals, out=rates[start:start + len(selected)], where=totals > 0)
    return rates


def intervals(counts: dict, indices: np.ndarray, percentiles) -> dict:
    """Undefined zero-denominator resamples are omitted; all undefined gives NaNs."""
    result = {}
    for metric, numerator, denominator in (
        ("pd_interval", "detected", "n_targets"),
        ("pfa_interval", "alarms", "n_background_cells"),
    ):
        result[metric] = {}
        for p, values in counts[numerator].items():
            rates = _rates(values, counts[denominator], indices)
            finite = rates[np.isfinite(rates)]
            result[metric][p] = (
                np.percentile(finite, percentiles).tolist() if finite.size
                else [float("nan"), float("nan")]
            )
    return result
