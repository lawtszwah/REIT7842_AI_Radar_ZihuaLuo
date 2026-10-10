"""Detection metrics on radar terms, not classification terms.

A target counts as detected if any cell within `tol` bins of its true position
exceeds the threshold.  False alarms are counted over background cells only,
with a guard region around every true target excluded so that a near-miss is
neither a hit nor a false alarm.  Pd is therefore per target and Pfa is per
cell, which is what a CFAR design curve means and what makes "Pd at a fixed
Pfa" comparable across detectors.
"""

from __future__ import annotations

import numpy as np

from ..simulation.interface import Target


def _window(scores: np.ndarray, r: float, d: float, tol: float) -> float:
    nr, nd = scores.shape
    r0, r1 = max(0, int(np.floor(r - tol))), min(nr, int(np.ceil(r + tol)) + 1)
    d0, d1 = max(0, int(np.floor(d - tol))), min(nd, int(np.ceil(d + tol)) + 1)
    if r0 >= r1 or d0 >= d1:
        return -np.inf
    return float(scores[r0:r1, d0:d1].max())


def _map_scores(scores, targets, tol, exclude, zero_doppler_guard):
    """One geometry implementation shared by pooled scores and map counts."""
    nr, nd = scores.shape[1:]
    zero_dop = nd // 2
    rr, dd = np.meshgrid(np.arange(nr), np.arange(nd), indexing="ij")
    in_guard = np.abs(dd - zero_dop) <= zero_doppler_guard
    for i, tl in enumerate(targets):
        tgt_scores, tgt_snr = [], []
        mask = ~in_guard.copy()
        for t in tl:
            if abs(t.doppler_bin - zero_dop) <= zero_doppler_guard + tol:
                continue
            tgt_scores.append(_window(scores[i], t.range_bin, t.doppler_bin, tol))
            tgt_snr.append(t.snr_db)
            near = (np.abs(rr - t.range_bin) <= exclude) & (np.abs(dd - t.doppler_bin) <= exclude)
            mask &= ~near
        yield (np.asarray(tgt_scores, dtype=np.float64),
               scores[i][mask].astype(np.float64), np.asarray(tgt_snr, dtype=np.float64))


def split_scores(scores: np.ndarray, targets: list[list[Target]],
                 tol: float = 1.0, exclude: float = 3.0, zero_doppler_guard: int = 2):
    """Return (per-target peaks, background cells, per-target SNR).

    Zero-Doppler guard columns are excluded from background. Targets whose
    tolerance window touches the guard are omitted from both target counts and
    target exclusions, exactly as in the calibrated evaluation protocol.
    """
    maps = list(_map_scores(scores, targets, tol, exclude, zero_doppler_guard))
    return tuple(np.concatenate([m[j] for m in maps]) for j in range(3))


def per_map_counts(scores: np.ndarray, targets: list[list[Target]], thresholds: dict,
                   tol: float = 1.0, exclude: float = 3.0,
                   zero_doppler_guard: int = 2) -> dict:
    """Eligible denominators and strict-threshold numerators, one count per map."""
    n_maps = len(targets)
    counts = {
        "n_targets": np.zeros(n_maps, dtype=np.int64),
        "n_background_cells": np.zeros(n_maps, dtype=np.int64),
        "detected": {p: np.zeros(n_maps, dtype=np.int64) for p in thresholds},
        "alarms": {p: np.zeros(n_maps, dtype=np.int64) for p in thresholds},
    }
    for i, (tgt, bg, _) in enumerate(
        _map_scores(scores, targets, tol, exclude, zero_doppler_guard),
    ):
        counts["n_targets"][i] = len(tgt)
        counts["n_background_cells"][i] = len(bg)
        for p, tau in thresholds.items():
            counts["detected"][p][i] = np.count_nonzero(tgt > tau)
            counts["alarms"][p][i] = np.count_nonzero(bg > tau)
    return counts


def roc(tgt_scores: np.ndarray, bg_scores: np.ndarray, n_points: int = 200):
    """Sweep the threshold over background quantiles -> exact (Pfa, Pd) curve."""
    pfa_grid = np.logspace(-6, -0.3, n_points)
    thresholds = np.quantile(bg_scores, 1.0 - np.clip(pfa_grid, 0, 1))
    bg_sorted = np.sort(bg_scores)
    tg_sorted = np.sort(tgt_scores)
    pfa = 1.0 - np.searchsorted(bg_sorted, thresholds, side="left") / len(bg_sorted)
    pd = 1.0 - np.searchsorted(tg_sorted, thresholds, side="left") / len(tg_sorted)
    return pfa, pd, thresholds


def pd_at_threshold(tgt_scores: np.ndarray, tau: float) -> float:
    """Per-target Pd at an externally selected threshold, using strict >."""
    return float((tgt_scores > tau).mean()) if tgt_scores.size else float("nan")


def pfa_at_threshold(bg_scores: np.ndarray, tau: float) -> float:
    """Per-cell Pfa at an externally selected threshold, using strict >."""
    return float((bg_scores > tau).mean()) if bg_scores.size else float("nan")


def pd_by_snr_at_threshold(tgt_scores: np.ndarray, tgt_snr: np.ndarray,
                           tau: float, edges: np.ndarray) -> list[dict]:
    """Stratify Pd using the same externally calibrated threshold in every bin."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        selected = (tgt_snr >= lo) & (tgt_snr < hi)
        out.append({
            "snr_lo": float(lo), "snr_hi": float(hi), "n": int(selected.sum()),
            "pd": pd_at_threshold(tgt_scores[selected], tau),
        })
    return out
