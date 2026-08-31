"""Equal-budget hyperparameter search.

Every detector -- including the CFAR baselines -- gets the same number of
configuration trials, scored by Pd at the reference Pfa on a validation split
that is disjoint from both training and test.  Tuning CFAR under the same
budget as the networks is what stops the baseline from being a straw man.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from ..data.generate import build_dataset
from ..detectors import build as build_detector
from ..evaluation import metrics


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*(space[k] for k in keys))]


def search(simulator: str, data_cfg: dict, kind: str, space: dict, budget: int,
           ref_pfa: float = 1e-3, seed: int = 0, out: Path | None = None) -> dict:
    rng = np.random.default_rng(seed)
    trials = grid(space)
    if len(trials) > budget:
        trials = [trials[i] for i in rng.choice(len(trials), budget, replace=False)]

    val = build_dataset(simulator, data_cfg, seed=5000 + seed)
    train = build_dataset(simulator, data_cfg, seed=seed)

    scored = []
    for params in trials:
        det = build_detector(kind, **params)
        if det.trainable:
            det.fit(train.maps, train.labels)
        tgt, bg, _ = metrics.split_scores(det.score(val.maps), val.targets)
        scored.append({"params": params, "pd": metrics.pd_at_pfa(tgt, bg, ref_pfa)})

    scored.sort(key=lambda r: -r["pd"])
    result = {"detector": kind, "ref_pfa": ref_pfa, "budget": budget,
              "n_trials": len(trials), "best": scored[0], "all": scored}
    if out:
        Path(out).write_text(json.dumps(result, indent=2))
    return result
