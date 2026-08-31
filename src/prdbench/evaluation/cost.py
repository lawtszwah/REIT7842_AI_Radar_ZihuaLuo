"""Inference cost, measured the same way for every detector."""

from __future__ import annotations

import time

import numpy as np


def measure_latency(detector, maps: np.ndarray, repeats: int = 5, warmup: int = 1) -> dict:
    """Median wall-clock seconds per map, single process, batch of `len(maps)`."""
    for _ in range(warmup):
        detector.score(maps[: min(4, len(maps))])
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        detector.score(maps)
        times.append((time.perf_counter() - t0) / len(maps))
    return {
        "latency_s_per_map_median": float(np.median(times)),
        "latency_s_per_map_iqr": float(np.subtract(*np.percentile(times, [75, 25]))),
        "n_repeats": repeats,
        "n_maps": int(len(maps)),
    }
