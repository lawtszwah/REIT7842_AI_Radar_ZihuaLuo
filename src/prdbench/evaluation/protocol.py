"""The single evaluation protocol every detector goes through.

One function, one code path.  Detectors differ only in the `score` they return;
data, thresholds, metrics, seeds and cost measurement are identical by
construction.  This is the design decision that the research question turns on.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.generate import build_dataset
from ..detectors import build as build_detector
from . import cost, metrics


@dataclass
class ProtocolConfig:
    pfa_points: tuple[float, ...] = (1e-4, 1e-3, 1e-2)
    tol_bins: float = 1.0
    exclude_bins: float = 3.0
    zero_doppler_guard: int = 2
    snr_edges: tuple[float, ...] = (-2, 2, 6, 10, 14)
    n_seeds: int = 3
    latency_repeats: int = 5


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_benchmark(spec: dict, out_dir: Path) -> dict:
    """spec: {'simulator', 'train', 'test', 'detectors', 'protocol'} -> results dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proto = ProtocolConfig(**spec.get("protocol", {}))

    results = {
        "run": {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "spec": spec,
        "protocol": asdict(proto),
        "detectors": [],
    }

    per_seed: dict[str, list[dict]] = {}
    for seed in range(proto.n_seeds):
        train = build_dataset(spec["simulator"], spec["train"], seed=seed)
        test = build_dataset(spec["simulator"], spec["test"], seed=1000 + seed)
        results["run"].setdefault("simulator_meta", train.meta)

        for entry in spec["detectors"]:
            kind, params = entry["kind"], dict(entry.get("params", {}))
            if "seed" in params or kind in {"feat_gbdt", "cnn", "attention"}:
                params.setdefault("seed", seed)
            det = build_detector(kind, **params)
            if det.trainable:
                det.fit(train.maps, train.labels)

            scores = det.score(test.maps)
            tgt, bg, snr = metrics.split_scores(
                scores, test.targets, tol=proto.tol_bins, exclude=proto.exclude_bins,
                zero_doppler_guard=proto.zero_doppler_guard,
            )
            pfa_c, pd_c, _ = metrics.roc(tgt, bg)
            rec = {
                "key": entry.get("key", kind),
                "seed": seed,
                "describe": det.describe(),
                "pd_at_pfa": {f"{p:g}": metrics.pd_at_pfa(tgt, bg, p) for p in proto.pfa_points},
                "pd_by_snr": metrics.pd_by_snr(
                    tgt, snr, bg, proto.pfa_points[1], np.asarray(proto.snr_edges)
                ),
                "roc": {"pfa": pfa_c.tolist(), "pd": pd_c.tolist()},
                "cost": cost.measure_latency(det, test.maps, repeats=proto.latency_repeats)
                | {"n_parameters": det.n_parameters()},
                "n_targets": len(tgt),
                "n_background_cells": len(bg),
            }
            per_seed.setdefault(rec["key"], []).append(rec)

    for key, runs in per_seed.items():
        agg = {"key": key, "describe": runs[0]["describe"], "seeds": len(runs)}
        for p in proto.pfa_points:
            vals = [r["pd_at_pfa"][f"{p:g}"] for r in runs]
            agg[f"pd@pfa={p:g}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        lat = [r["cost"]["latency_s_per_map_median"] for r in runs]
        agg["latency_s_per_map"] = {"mean": float(np.mean(lat)), "std": float(np.std(lat))}
        agg["n_parameters"] = runs[0]["cost"]["n_parameters"]
        agg["runs"] = runs
        results["detectors"].append(agg)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results
