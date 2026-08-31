"""Classical feature-based detector: hand-designed local statistics + a classifier.

Sits between CFAR and the deep models.  It sees the same local neighbourhood a
CFAR detector does, but learns how to combine several scales instead of using a
fixed cell-averaged threshold, which isolates how much of any gain comes from
learning rather than from a larger receptive field.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter
from sklearn.ensemble import HistGradientBoostingClassifier

from .base import Detector
from .cfar import _window_sums

_EPS = 1e-12


def cell_features(maps: np.ndarray) -> np.ndarray:
    """(N, R, D) -> (N, R, D, F).  Cheap, local, CFAR-like statistics."""
    maps = np.asarray(maps, dtype=np.float32)
    n, nr, nd = maps.shape
    log_m = np.log10(maps + _EPS)
    feats = [log_m]

    for train, guard in ((2, 1), (4, 2), (8, 3)):
        noise = _window_sums(maps, train, guard)
        feats.append(np.log10(maps / (noise + _EPS) + _EPS))
        feats.append(np.log10(noise + _EPS))

    for size in (3, 7):
        mu = uniform_filter(log_m, size=(1, size, size), mode="nearest")
        var = uniform_filter(log_m ** 2, size=(1, size, size), mode="nearest") - mu ** 2
        feats.append(mu)
        feats.append(np.sqrt(np.clip(var, 0, None)))

    feats.append(maximum_filter(log_m, size=(1, 3, 3), mode="nearest") - log_m)
    feats.append(np.broadcast_to(log_m.mean(axis=2, keepdims=True), (n, nr, nd)))
    feats.append(np.broadcast_to(log_m.mean(axis=1, keepdims=True), (n, nr, nd)))

    # Geometry: distance from the zero-Doppler column, where direct-path
    # leakage lives.  Lets the model learn the structure CFAR cannot see.
    d = np.abs(np.arange(nd, dtype=np.float32) - nd // 2)
    feats.append(np.broadcast_to(d[None, None, :], (n, nr, nd)))

    return np.stack([np.broadcast_to(f, (n, nr, nd)) for f in feats], axis=-1).astype(np.float32)


class FeatureGBDT(Detector):
    family = "feature"
    trainable = True

    def __init__(self, max_iter: int = 120, learning_rate: float = 0.1,
                 max_depth: int | None = 6, neg_per_map: int = 400, seed: int = 0):
        super().__init__(max_iter=max_iter, learning_rate=learning_rate,
                         max_depth=max_depth, neg_per_map=neg_per_map, seed=seed)
        self.name = f"GBDT(iters={max_iter},depth={max_depth})"
        self.neg_per_map = neg_per_map
        self.seed = seed
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter, learning_rate=learning_rate,
            max_depth=max_depth, random_state=seed,
        )

    def fit(self, maps: np.ndarray, labels: np.ndarray, **kw) -> FeatureGBDT:
        rng = np.random.default_rng(self.seed)
        feats = cell_features(maps)
        n, nr, nd, nf = feats.shape
        X, y = [], []
        for i in range(n):
            pos = np.argwhere(labels[i] > 0)
            neg = np.argwhere(labels[i] == 0)
            take = rng.choice(len(neg), size=min(self.neg_per_map, len(neg)), replace=False)
            neg = neg[take]
            for arr, lab in ((pos, 1), (neg, 0)):
                if len(arr):
                    X.append(feats[i, arr[:, 0], arr[:, 1]])
                    y.append(np.full(len(arr), lab, dtype=np.int8))
        self.clf.fit(np.concatenate(X), np.concatenate(y))
        return self

    def score(self, maps: np.ndarray) -> np.ndarray:
        feats = cell_features(maps)
        n, nr, nd, nf = feats.shape
        flat = feats.reshape(-1, nf)
        p = self.clf.predict_proba(flat)[:, 1]
        return p.reshape(n, nr, nd).astype(np.float32)

    def n_parameters(self) -> int:
        try:
            return int(sum(len(t[0].nodes) for t in self.clf._predictors))
        except Exception:
            return -1
