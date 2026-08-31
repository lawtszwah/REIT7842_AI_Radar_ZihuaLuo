"""Attention-based detector -- the third model family in the research question.

STATUS: interface fixed, implementation scheduled for Milestone M3
(see README, Plan of work).  Implementing it after the CFAR and CNN arms are
locked keeps the evaluation protocol frozen before the most flexible model is
introduced, which is the failure mode the project is criticising in the
literature.
"""

from __future__ import annotations


from .base import Detector


class PatchAttention(Detector):
    family = "deep"
    trainable = True

    def __init__(self, patch: int = 8, dim: int = 64, heads: int = 4,
                 layers: int = 3, epochs: int = 20, seed: int = 0):
        super().__init__(patch=patch, dim=dim, heads=heads, layers=layers,
                         epochs=epochs, seed=seed)
        self.name = f"Attn(p={patch},d={dim},L={layers})"

    def fit(self, maps, labels, **kw):  # pragma: no cover
        raise NotImplementedError("Milestone M3; see README plan of work.")

    def score(self, maps):  # pragma: no cover
        raise NotImplementedError("Milestone M3; see README plan of work.")
