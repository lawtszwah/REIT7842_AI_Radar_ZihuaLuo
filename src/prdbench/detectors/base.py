"""The single interface every detector in the benchmark implements.

Design decision (ADR-0001): a detector returns a *continuous per-cell score*,
never a binary decision.  The threshold is owned by the evaluation harness, not
by the detector.  That is what makes a tuned CFAR variant and a neural network
comparable at a fixed false-alarm rate instead of at whatever operating point
each happens to default to.
"""

from __future__ import annotations

import abc

import numpy as np


class Detector(abc.ABC):
    name: str = "detector"
    family: str = "unknown"  # cfar | feature | deep
    trainable: bool = False

    def __init__(self, **params):
        self.params = params

    def fit(self, maps: np.ndarray, labels: np.ndarray | None = None, **kw) -> Detector:
        """Fit on training maps.  No-op for the CFAR family."""
        return self

    @abc.abstractmethod
    def score(self, maps: np.ndarray) -> np.ndarray:
        """(N, R, D) power maps -> (N, R, D) scores, higher = more target-like."""

    def n_parameters(self) -> int:
        """Free parameters, for the accuracy-vs-cost axis of the benchmark."""
        return 0

    def describe(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "trainable": self.trainable,
            "n_parameters": self.n_parameters(),
            "params": self.params,
        }
