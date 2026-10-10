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
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from ..data.generate import build_dataset
from ..detectors import REGISTRY
from ..detectors import build as build_detector
from . import bootstrap, cost, metrics
from .seeds import make_streams, stream_record


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
    selection_pfa: float = 1e-3
    # Trials per detector in model selection. No default: equal budget across arms
    # (RSK-03) only holds if the number is fixed in the config, not chosen per call.
    tuning_budget: int | None = None
    bootstrap_resamples: int = 10_000
    bootstrap_interval_percentiles: tuple[float, float] = (2.5, 97.5)
    gate3_permitted_repeats: int = 1


SPLIT_STREAMS = {
    "train": "train_maps",
    "calibration": "calibration_maps",
    "test_in_distribution": "test_in_distribution_maps",
    "extended_test": "extended_test_maps",
    "joint_shift": "joint_shift_maps",
}
TEST_SPLITS = ("test_in_distribution", "extended_test", "joint_shift")


def merge_joint_shift(spec: dict, repo_root: Path | None = None) -> dict:
    """Resolve joint_shift_config relative to the repo root, before hashing.

    The CLI handles external config reads here; run_benchmark only sees the
    merged spec. An explicit repo_root also permits standalone tests.
    """
    merged = deepcopy(spec)
    if "joint_shift_config" not in spec:
        return merged
    path = Path(spec["joint_shift_config"])
    if not path.is_absolute():
        path = (repo_root or Path(__file__).resolve().parents[3]) / path
    raw = path.read_bytes()
    shift = yaml.safe_load(raw)
    if not isinstance(shift, dict) or shift.get("simulator") != spec["simulator"]:
        raise ValueError("joint_shift_config simulator must equal the protocol simulator")
    if not isinstance(shift.get("test"), dict):
        raise ValueError("joint_shift_config must contain a test block")
    merged["joint_shift"] = shift["test"]
    merged["joint_shift_source"] = {
        "path": str(spec["joint_shift_config"]),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return merged


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
    if not any(name in test_splits for name in TEST_SPLITS[:2]):
        raise ValueError("at least one of test_in_distribution or extended_test is required")
    required = {SPLIT_STREAMS[name] for name in ("train", "calibration", *test_splits)}
    required.add("analysis_bootstrap")
    if any(REGISTRY[entry["kind"]].trainable for entry in spec["detectors"]):
        required.add("model_initialization")
    make_streams(0, proto.seed_streams, required)
    _validate_score_rule(proto)
    percentiles = proto.bootstrap_interval_percentiles
    if proto.bootstrap_resamples < 1:
        raise ValueError("protocol.bootstrap_resamples must be positive")
    if (len(percentiles) != 2 or not all(np.isfinite(percentiles))
            or not 0 <= percentiles[0] < percentiles[1] <= 100):
        raise ValueError("protocol.bootstrap_interval_percentiles must be two ordered percentiles")
    if proto.n_seeds < 1:
        raise ValueError("n_seeds must be positive")
    if not proto.pfa_points or any(not 0 < p < 1 for p in proto.pfa_points):
        raise ValueError("protocol.pfa_points must contain probabilities strictly between 0 and 1")


def _validate_score_rule(proto: ProtocolConfig) -> None:
    """Evaluation and selection share quantiles, comparisons and eligibility."""
    if proto.quantile_method != "higher":
        raise ValueError("protocol.quantile_method must be 'higher'")
    if proto.threshold_comparison != "score_gt_threshold":
        raise ValueError("protocol.threshold_comparison must be 'score_gt_threshold'")
    if proto.calibration_method != "empirical_background_quantile":
        raise ValueError("protocol.calibration_method must be 'empirical_background_quantile'")
    if proto.min_eligible_background_cells < 1:
        raise ValueError("min_eligible_background_cells must be positive")
    if not 0 < proto.selection_pfa < 1:
        raise ValueError("protocol.selection_pfa must be strictly between 0 and 1")


def _calibrate_thresholds(cal_bg: np.ndarray, proto: ProtocolConfig) -> dict[str, float]:
    """Select thresholds using only eligible calibration background scores."""
    return {
        f"{p:g}": float(np.quantile(cal_bg, 1.0 - p, method=proto.quantile_method))
        for p in proto.pfa_points
    }


def _split_scores(detector, dataset, proto: ProtocolConfig, split_name: str, *, scores=None):
    tgt, bg, snr = metrics.split_scores(
        detector.score(dataset.maps) if scores is None else scores, dataset.targets,
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
    test_splits = [name for name in TEST_SPLITS if name in spec]
    if "joint_shift_config" in spec and "joint_shift_source" not in spec:
        raise ValueError("joint_shift_config must be merged before run_benchmark")
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
    if "joint_shift_source" in spec:
        results["run"]["joint_shift_source"] = deepcopy(spec["joint_shift_source"])

    per_seed: dict[str, list[dict]] = {}
    for root in results["run"]["root_seeds"]:
        streams = make_streams(root, proto.seed_streams)
        map_streams = {
            SPLIT_STREAMS[name]: stream_record(streams[SPLIT_STREAMS[name]])
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
        # Configs are recorded for root 0; later roots replace only config.seed
        # with the split's recorded map seed. Per-map metadata is reproducible.
        results["run"].setdefault("simulator_meta", {
            name: {key: dataset.meta[key] for key in ("simulator", "simulator_version", "config")}
            for name, dataset in datasets.items()
        })
        results["run"].setdefault("simulator", train.meta["simulator"])
        results["run"].setdefault("simulator_version", train.meta["simulator_version"])

        # Spawn all three children even when some splits are absent: positions are fixed.
        children = streams["analysis_bootstrap"].spawn(len(TEST_SPLITS))
        bootstrap_records, indices = {}, {}
        for name in test_splits:
            record = stream_record(children[TEST_SPLITS.index(name)])
            bootstrap_records[f"analysis_bootstrap/{name}"] = record
            # All detectors share this one int32 matrix for valid paired comparisons.
            indices[name] = bootstrap.map_indices(
                record["materialised_seed"], len(datasets[name].maps), proto.bootstrap_resamples,
            )

        for entry in spec["detectors"]:
            kind, params = entry["kind"], dict(entry.get("params", {}))
            used_streams = map_streams | bootstrap_records
            if REGISTRY[kind].trainable:
                model_record = stream_record(streams["model_initialization"], model=True)
                params["seed"] = model_record["materialised_seed"]
                used_streams["model_initialization"] = model_record
            det = build_detector(kind, **params)
            if det.trainable:
                det.fit(train.maps, train.labels)

            _, cal_bg, _ = _split_scores(det, datasets["calibration"], proto, "calibration")
            thresholds = _calibrate_thresholds(cal_bg, proto)
            splits = {}
            for name in test_splits:
                dataset = datasets[name]
                scores = det.score(dataset.maps)
                splits[name] = _evaluate_split(
                    *_split_scores(det, dataset, proto, name, scores=scores), thresholds, proto,
                )
                counts = metrics.per_map_counts(
                    scores, dataset.targets, thresholds, tol=proto.tol_bins,
                    exclude=proto.exclude_bins, zero_doppler_guard=proto.zero_doppler_guard,
                )
                splits[name].update(bootstrap.intervals(
                    counts, indices[name], proto.bootstrap_interval_percentiles,
                ))
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
