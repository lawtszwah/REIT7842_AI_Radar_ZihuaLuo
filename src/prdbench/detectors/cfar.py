"""CFAR baselines, properly tuned rather than left at default settings.

The score returned is the ratio of the cell under test to the local noise
estimate.  Sweeping a threshold on that ratio traces the detector's own ROC,
which is exactly the CFAR design curve -- so the baseline is evaluated on the
same footing as the learned detectors instead of at one arbitrary alpha.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter

from .base import Detector

_EPS = 1e-12


def _window_sums(maps: np.ndarray, train: int, guard: int) -> np.ndarray:
    """Sum over the training annulus (outer square minus guard square)."""
    outer = 2 * (train + guard) + 1
    inner = 2 * guard + 1
    tot = uniform_filter(maps, size=(1, outer, outer), mode="nearest") * outer * outer
    gap = uniform_filter(maps, size=(1, inner, inner), mode="nearest") * inner * inner
    n_cells = outer * outer - inner * inner
    return (tot - gap) / n_cells


class CACFAR(Detector):
    """Cell-averaging CFAR."""

    family = "cfar"

    def __init__(self, train: int = 4, guard: int = 2):
        super().__init__(train=train, guard=guard)
        self.name = f"CA-CFAR(t={train},g={guard})"
        self.train, self.guard = train, guard

    def score(self, maps: np.ndarray) -> np.ndarray:
        maps = np.asarray(maps, dtype=np.float32)
        noise = _window_sums(maps, self.train, self.guard)
        return maps / (noise + _EPS)


class OSCFAR(Detector):
    """Ordered-statistic CFAR: robust where the training window is contaminated."""

    family = "cfar"

    def __init__(self, train: int = 4, guard: int = 2, rank_pct: float = 75.0):
        super().__init__(train=train, guard=guard, rank_pct=rank_pct)
        self.name = f"OS-CFAR(t={train},g={guard},p={rank_pct:g})"
        self.train, self.guard, self.rank_pct = train, guard, rank_pct

    def _footprint(self) -> np.ndarray:
        outer = 2 * (self.train + self.guard) + 1
        fp = np.ones((outer, outer), dtype=bool)
        c, g = self.train + self.guard, self.guard
        fp[c - g : c + g + 1, c - g : c + g + 1] = False
        return fp

    def score(self, maps: np.ndarray) -> np.ndarray:
        maps = np.asarray(maps, dtype=np.float32)
        fp = self._footprint()
        out = np.empty_like(maps)
        for i, m in enumerate(maps):
            noise = percentile_filter(m, self.rank_pct, footprint=fp, mode="nearest")
            out[i] = m / (noise + _EPS)
        return out


class GOCFAR(Detector):
    """Greatest-of CFAR: takes the larger of the two leading/lagging half-windows."""

    family = "cfar"

    def __init__(self, train: int = 4, guard: int = 2):
        super().__init__(train=train, guard=guard)
        self.name = f"GO-CFAR(t={train},g={guard})"
        self.train, self.guard = train, guard

    def score(self, maps: np.ndarray) -> np.ndarray:
        maps = np.asarray(maps, dtype=np.float32)
        t, g = self.train, self.guard
        lead, lag = [], []
        for m in maps:
            padded = np.pad(m, ((t + g, t + g), (0, 0)), mode="edge")
            cs = np.cumsum(padded, axis=0)
            cs = np.vstack([np.zeros((1, m.shape[1]), np.float32), cs])
            n = m.shape[0]
            idx = np.arange(n) + t + g
            up = (cs[idx - g] - cs[idx - g - t]) / t
            dn = (cs[idx + g + t + 1] - cs[idx + g + 1]) / t
            lead.append(up)
            lag.append(dn)
        noise = np.maximum(np.stack(lead), np.stack(lag))
        return maps / (noise + _EPS)
