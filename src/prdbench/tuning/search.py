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
from ..detectors import REGISTRY
from ..detectors import build as build_detector
from ..evaluation import metrics
from ..evaluation.protocol import ProtocolConfig, _split_scores, _validate_score_rule
from ..evaluation.seeds import make_streams, stream_record


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, vals, strict=True))
            for vals in itertools.product(*(space[k] for k in keys))]


def search(spec: dict, kind: str, space: dict, budget: int,
           root_seed: int = 0, out: Path | None = None) -> dict:
    """Rank by validation Pd; ties retain trial order after subsampling.

    Calibration and test splits are never generated or consulted here. The
    calibration split remains reserved for thresholds after model selection.
    """
    if "validation" not in spec:
        raise ValueError("an explicit validation split is required for model selection")
    if budget < 1:
        raise ValueError("budget must be positive")
    proto = ProtocolConfig(**spec.get("protocol", {}))
    _validate_score_rule(proto)
    required = {"train_maps", "validation_maps"}
    trainable = REGISTRY[kind].trainable
    if trainable:
        required.add("model_initialization")
    trials = grid(space)
    if not trials:
        raise ValueError("search space must contain at least one trial")
    if len(trials) > budget:
        required.add("tuning_trial_selection")
    streams = make_streams(root_seed, proto.seed_streams, required)
    used = {
        name: stream_record(streams[name]) for name in ("train_maps", "validation_maps")
    }
    if trainable:
        used["model_initialization"] = stream_record(streams["model_initialization"], model=True)
    if len(trials) > budget:
        name = "tuning_trial_selection"
        used[name] = stream_record(streams[name])
        rng = np.random.default_rng(used[name]["materialised_seed"])
        trials = [trials[i] for i in rng.choice(len(trials), budget, replace=False)]

    val = build_dataset(spec["simulator"], spec["validation"],
                        seed=used["validation_maps"]["materialised_seed"])
    train = build_dataset(spec["simulator"], spec["train"],
                          seed=used["train_maps"]["materialised_seed"])

    scored = []
    for params in trials:
        params = dict(params)
        if trainable:
            params["seed"] = used["model_initialization"]["materialised_seed"]
        det = build_detector(kind, **params)
        if det.trainable:
            det.fit(train.maps, train.labels)
        tgt, bg, _ = _split_scores(det, val, proto, "validation")
        if not tgt.size:
            raise ValueError("validation must contain eligible targets for model selection")
        tau = float(np.quantile(bg, 1.0 - proto.selection_pfa, method=proto.quantile_method))
        scored.append({
            "params": params, "pd": metrics.pd_at_threshold(tgt, tau), "threshold": tau,
            "n_targets": len(tgt), "n_background_cells": len(bg),
        })

    # max returns the first tied entry; all retains the actual trial execution order.
    result = {"detector": kind, "root_seed": root_seed, "streams": used,
              "selection_pfa": proto.selection_pfa, "budget": budget,
              "n_trials": len(trials), "best": max(scored, key=lambda r: r["pd"]), "all": scored}
    if out:
        Path(out).write_text(json.dumps(result, indent=2))
    return result
