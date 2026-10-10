"""The single evaluation protocol every detector goes through.

One function, one code path.  Detectors differ only in the `score` they return;
data, thresholds, metrics, seeds and cost measurement are identical by
construction.  This is the design decision that the research question turns on.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..data.generate import build_dataset
from ..detectors import REGISTRY
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
    seed_streams: tuple[str, ...] = ()
    min_eligible_background_cells: int = 10_000_000
    calibration_method: str = "empirical_background_quantile"
    quantile_method: str = "higher"
    threshold_comparison: str = "score_gt_threshold"
    # Recorded protocol settings; bootstrap analysis is reserved for step 6.
    bootstrap_resamples: int = 10_000
    gate3_permitted_repeats: int = 1


SPLIT_STREAMS = {
    "train": "train_maps",
    "calibration": "calibration_maps",
    "test_in_distribution": "test_in_distribution_maps",
    "extended_test": "extended_test_maps",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _validate_spec(spec: dict, proto: ProtocolConfig, test_splits: list[str]) -> None:
    if "calibration" not in spec:
        raise ValueError("calibration split is required")
    if tuple(spec["calibration"].get("n_targets", (1, 1))) != (0, 0):
        raise ValueError("calibration must be target-free: set n_targets to [0, 0]")
    if not test_splits:
        raise ValueError("at least one of test_in_distribution or extended_test is required")
    if not proto.seed_streams or not isinstance(proto.seed_streams, (list, tuple)):
        raise ValueError("protocol.seed_streams is required as an ordered list of names")
    if any(not isinstance(name, str) for name in proto.seed_streams):
        raise ValueError("protocol.seed_streams must contain string names")
    if len(set(proto.seed_streams)) != len(proto.seed_streams):
        raise ValueError("protocol.seed_streams must contain unique names")
    required = {SPLIT_STREAMS[name] for name in ("train", "calibration", *test_splits)}
    if any(REGISTRY[entry["kind"]].trainable for entry in spec["detectors"]):
        required.add("model_initialization")
    missing = required - set(proto.seed_streams)
    if missing:
        raise ValueError(f"protocol.seed_streams missing required names: {sorted(missing)}")
    if proto.quantile_method != "higher":
        raise ValueError("protocol.quantile_method must be 'higher'")
    if proto.threshold_comparison != "score_gt_threshold":
        raise ValueError("protocol.threshold_comparison must be 'score_gt_threshold'")
    if proto.calibration_method != "empirical_background_quantile":
        raise ValueError("protocol.calibration_method must be 'empirical_background_quantile'")
    if proto.n_seeds < 1 or proto.min_eligible_background_cells < 1:
        raise ValueError("n_seeds and min_eligible_background_cells must be positive")
    if not proto.pfa_points or any(not 0 < p < 1 for p in proto.pfa_points):
        raise ValueError("protocol.pfa_points must contain probabilities strictly between 0 and 1")


def _stream_record(stream: np.random.SeedSequence, *, model: bool = False) -> dict:
    # Model initialization uses a native uint32 draw: sklearn's integer random_state
    # is limited to 32 bits, and this same integer also works with numpy and torch.
    dtype = np.uint32 if model else np.uint64
    return {
        "spawn_key": list(stream.spawn_key),
        "materialised_seed": int(stream.generate_state(1, dtype=dtype)[0]),
    }


def _calibrate_thresholds(cal_bg: np.ndarray, proto: ProtocolConfig) -> dict[str, float]:
    """Select thresholds using only eligible calibration background scores."""
    return {
        f"{p:g}": float(np.quantile(cal_bg, 1.0 - p, method=proto.quantile_method))
        for p in proto.pfa_points
    }


def _split_scores(detector, dataset, proto: ProtocolConfig, split_name: str):
    tgt, bg, snr = metrics.split_scores(
        detector.score(dataset.maps), dataset.targets,
        tol=proto.tol_bins, exclude=proto.exclude_bins,
        zero_doppler_guard=proto.zero_doppler_guard,
    )
    if len(bg) < proto.min_eligible_background_cells:
        raise ValueError(
            f"{split_name}: {len(bg)} eligible background cells; "
            f"minimum required is {proto.min_eligible_background_cells}"
        )
    return tgt, bg, snr


def _evaluate_split(tgt, bg, snr, thresholds: dict, proto: ProtocolConfig) -> dict:
    # ROC is a descriptive test-score sweep; it never selects an operating threshold.
    pfa_curve, pd_curve, _ = metrics.roc(tgt, bg)
    return {
        "pd": {p: metrics.pd_at_threshold(tgt, tau) for p, tau in thresholds.items()},
        "pfa_achieved": {p: metrics.pfa_at_threshold(bg, tau) for p, tau in thresholds.items()},
        "pd_by_snr": {
            p: metrics.pd_by_snr_at_threshold(tgt, snr, tau, np.asarray(proto.snr_edges))
            for p, tau in thresholds.items()
        },
        "roc": {"pfa": pfa_curve.tolist(), "pd": pd_curve.tolist()},
        "n_targets": len(tgt),
        "n_background_cells": len(bg),
    }


def run_benchmark(spec: dict, out_dir: Path) -> dict:
    """Fit on train, calibrate on null maps, and evaluate named test splits."""
    proto = ProtocolConfig(**spec.get("protocol", {}))
    test_splits = [name for name in ("test_in_distribution", "extended_test") if name in spec]
    _validate_spec(spec, proto, test_splits)
    reporting_split = "test_in_distribution" if "test_in_distribution" in spec else test_splits[0]
    reference_pfa = proto.pfa_points[min(1, len(proto.pfa_points) - 1)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "run": {
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "config_sha256": hashlib.sha256(
                json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "root_seeds": list(range(proto.n_seeds)),
            "reporting_split": reporting_split,
        },
        "spec": spec,
        "protocol": asdict(proto),
        "detectors": [],
    }

    per_seed: dict[str, list[dict]] = {}
    for root in results["run"]["root_seeds"]:
        streams = dict(zip(
            proto.seed_streams, np.random.SeedSequence(root).spawn(len(proto.seed_streams)),
            strict=True,
        ))
        map_streams = {
            SPLIT_STREAMS[name]: _stream_record(streams[SPLIT_STREAMS[name]])
            for name in ("train", "calibration", *test_splits)
        }
        datasets = {
            name: build_dataset(
                spec["simulator"], spec[name],
                seed=map_streams[SPLIT_STREAMS[name]]["materialised_seed"],
            )
            for name in ("train", "calibration", *test_splits)
        }
        train = datasets["train"]
        results["run"].setdefault("simulator_meta", train.meta)
        results["run"].setdefault("simulator", train.meta["simulator"])
        results["run"].setdefault("simulator_version", train.meta["simulator_version"])

        for entry in spec["detectors"]:
            kind, params = entry["kind"], dict(entry.get("params", {}))
            used_streams = dict(map_streams)
            if REGISTRY[kind].trainable:
                model_record = _stream_record(streams["model_initialization"], model=True)
                params["seed"] = model_record["materialised_seed"]
                used_streams["model_initialization"] = model_record
            det = build_detector(kind, **params)
            if det.trainable:
                det.fit(train.maps, train.labels)

            _, cal_bg, _ = _split_scores(det, datasets["calibration"], proto, "calibration")
            thresholds = _calibrate_thresholds(cal_bg, proto)
            splits = {
                name: _evaluate_split(
                    *_split_scores(det, datasets[name], proto, name), thresholds, proto,
                )
                for name in test_splits
            }
            report = splits[reporting_split]
            rec = {
                "key": entry.get("key", kind),
                "seed": root,
                "root_seed": root,
                "streams": used_streams,
                "describe": det.describe(),
                "thresholds": thresholds,
                "calibration": {
                    "pfa_achieved": {
                        p: metrics.pfa_at_threshold(cal_bg, tau) for p, tau in thresholds.items()
                    },
                    "n_background_cells": len(cal_bg),
                },
                "splits": splits,
                # Compatibility fields use the same reporting split as detector aggregates.
                "pd_at_pfa": report["pd"],
                "pd_by_snr": report["pd_by_snr"][f"{reference_pfa:g}"],
                "roc": report["roc"],
                "cost": cost.measure_latency(
                    det, datasets[reporting_split].maps, repeats=proto.latency_repeats,
                )
                | {"n_parameters": det.n_parameters()},
                "n_targets": report["n_targets"],
                "n_background_cells": report["n_background_cells"],
            }
            per_seed.setdefault(rec["key"], []).append(rec)

    for key, runs in per_seed.items():
        agg = {"key": key, "describe": runs[0]["describe"], "seeds": len(runs)}
        # Aggregate Pd uses calibrated thresholds on test_in_distribution when present,
        # otherwise the first test split. Reporting keeps its existing aggregate keys.
        for p in proto.pfa_points:
            vals = [r["pd_at_pfa"][f"{p:g}"] for r in runs]
            agg[f"pd@pfa={p:g}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            achieved = [r["splits"][reporting_split]["pfa_achieved"][f"{p:g}"] for r in runs]
            agg[f"pfa_achieved@pfa={p:g}"] = {
                "mean": float(np.mean(achieved)), "std": float(np.std(achieved)),
            }
        lat = [r["cost"]["latency_s_per_map_median"] for r in runs]
        agg["latency_s_per_map"] = {"mean": float(np.mean(lat)), "std": float(np.std(lat))}
        agg["n_parameters"] = runs[0]["cost"]["n_parameters"]
        agg["runs"] = runs
        results["detectors"].append(agg)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    return results
