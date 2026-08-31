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


def split_scores(scores: np.ndarray, targets: list[list[Target]],
                 tol: float = 1.0, exclude: float = 3.0, zero_doppler_guard: int = 2):
    """Return (per-target peak scores, background cell scores, per-target SNR).

    `zero_doppler_guard` removes the columns either side of zero Doppler from
    both counts.  Direct-path leakage lives there and is removed by clutter
    cancellation before any detector runs in a real passive radar chain; leaving
    it in would let leakage, not detector quality, set the false-alarm threshold.
    Targets falling inside the guard are excluded rather than scored as misses.
    """
    tgt_scores, tgt_snr, bg = [], [], []
    nr, nd = scores.shape[1:]
    zero_dop = nd // 2
    rr, dd = np.meshgrid(np.arange(nr), np.arange(nd), indexing="ij")
    in_guard = np.abs(dd - zero_dop) <= zero_doppler_guard
    for i, tl in enumerate(targets):
        mask = ~in_guard.copy()
        for t in tl:
            if abs(t.doppler_bin - zero_dop) <= zero_doppler_guard + tol:
                continue
            tgt_scores.append(_window(scores[i], t.range_bin, t.doppler_bin, tol))
            tgt_snr.append(t.snr_db)
            near = (np.abs(rr - t.range_bin) <= exclude) & (np.abs(dd - t.doppler_bin) <= exclude)
            mask &= ~near
        bg.append(scores[i][mask])
    return (np.asarray(tgt_scores, dtype=np.float64),
            np.concatenate(bg).astype(np.float64),
            np.asarray(tgt_snr, dtype=np.float64))


def roc(tgt_scores: np.ndarray, bg_scores: np.ndarray, n_points: int = 200):
    """Sweep the threshold over background quantiles -> exact (Pfa, Pd) curve."""
    pfa_grid = np.logspace(-6, -0.3, n_points)
    thresholds = np.quantile(bg_scores, 1.0 - np.clip(pfa_grid, 0, 1))
    bg_sorted = np.sort(bg_scores)
    tg_sorted = np.sort(tgt_scores)
    pfa = 1.0 - np.searchsorted(bg_sorted, thresholds, side="left") / len(bg_sorted)
    pd = 1.0 - np.searchsorted(tg_sorted, thresholds, side="left") / len(tg_sorted)
    return pfa, pd, thresholds


def pd_at_pfa(tgt_scores: np.ndarray, bg_scores: np.ndarray, pfa: float) -> float:
    """Probability of detection at a fixed false-alarm rate."""
    thr = np.quantile(bg_scores, 1.0 - pfa)
    return float((tgt_scores >= thr).mean())


def pd_by_snr(tgt_scores: np.ndarray, tgt_snr: np.ndarray, bg_scores: np.ndarray,
              pfa: float, edges: np.ndarray) -> list[dict]:
    """Where do gains occur?  Pd at fixed Pfa, stratified by target SNR."""
    thr = np.quantile(bg_scores, 1.0 - pfa)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (tgt_snr >= lo) & (tgt_snr < hi)
        out.append({
            "snr_lo": float(lo), "snr_hi": float(hi), "n": int(sel.sum()),
            "pd": float((tgt_scores[sel] >= thr).mean()) if sel.any() else float("nan"),
        })
    return out
