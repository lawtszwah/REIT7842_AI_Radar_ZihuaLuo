"""The harness calibrates independently and records the spawned seed protocol."""

import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml

from prdbench.data.generate import build_dataset
from prdbench.detectors import build
from prdbench.evaluation import metrics
from prdbench.evaluation.protocol import run_benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL = yaml.safe_load((REPO_ROOT / "configs/protocol/full.yaml").read_text())
MAP_STREAMS = {
    "train_maps", "calibration_maps", "test_in_distribution_maps", "extended_test_maps",
}


def _spec():
    grid = {
        "n_maps": 4, "n_range": 24, "n_doppler": 24, "n_targets": [1, 1],
        "range_bin": [8.0, 8.0], "doppler_bin": [18.0, 18.0], "snr_db": [8.0, 12.0],
        "clutter_patches": [0, 0], "clutter_ridge_db": [24.0, 24.0],
    }
    return {
        "simulator": "stub",
        "train": copy.deepcopy(grid),
        "calibration": grid | {"n_targets": [0, 0]},
        "test_in_distribution": copy.deepcopy(grid),
        "extended_test": grid | {"snr_db": [2.0, 6.0]},
        "protocol": {
            "pfa_points": [1e-3, 1e-2, 0.1], "snr_edges": [0, 6, 12, 18],
            "n_seeds": 2, "latency_repeats": 1, "min_eligible_background_cells": 1000,
            "quantile_method": "higher", "threshold_comparison": "score_gt_threshold",
            "seed_streams": list(FULL["protocol"]["seed_streams"]),
        },
        "detectors": [{"kind": "ca_cfar", "params": {"train": 2, "guard": 1}}],
    }


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return run_benchmark(_spec(), tmp_path_factory.mktemp("harness"))


def _threshold_bytes(results):
    return [[{p: struct.pack("!d", tau) for p, tau in run["thresholds"].items()}
             for run in detector["runs"]] for detector in results["detectors"]]


def _without_timing(value):
    if isinstance(value, dict):
        return {key: _without_timing(item) for key, item in value.items()
                if key != "started" and not key.startswith("latency")}
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def test_map_stream_seeds_are_spawned_distinct_and_deterministic(baseline):
    names = _spec()["protocol"]["seed_streams"]
    observed = []
    for root, run in enumerate(baseline["detectors"][0]["runs"]):
        assert run["root_seed"] == root
        assert set(run["streams"]) == MAP_STREAMS
        for name, record in run["streams"].items():
            child = np.random.SeedSequence(root).spawn(len(names))[names.index(name)]
            assert record["spawn_key"] == list(child.spawn_key)
            assert record["materialised_seed"] == int(child.generate_state(1, dtype=np.uint64)[0])
            observed.append(record["materialised_seed"])
    assert len(set(observed)) == len(observed)


def test_model_seed_is_native_uint32_and_shared_by_trainable_detectors(tmp_path):
    spec = _spec()
    spec["detectors"] = [
        {"kind": "feat_gbdt", "params": {"max_iter": 2, "max_depth": 2,
                                         "neg_per_map": 16, "seed": 999}},
        {"kind": "cnn", "params": {"width": 2, "depth": 1, "epochs": 1,
                                   "batch_size": 4, "seed": 999}},
    ]
    results = run_benchmark(spec, tmp_path)
    names = spec["protocol"]["seed_streams"]
    model_seeds = []
    for root in range(2):
        child = np.random.SeedSequence(root).spawn(len(names))[names.index("model_initialization")]
        expected = int(child.generate_state(1, dtype=np.uint32)[0])
        model_seeds.append(expected)
        for detector in results["detectors"]:
            run = detector["runs"][root]
            record = run["streams"]["model_initialization"]
            assert record == {"spawn_key": list(child.spawn_key), "materialised_seed": expected}
            assert 0 <= record["materialised_seed"] < 2**32
            assert run["describe"]["params"]["seed"] == expected
            assert set(run["streams"]) == MAP_STREAMS | {"model_initialization"}
    assert model_seeds[0] != model_seeds[1]


def test_thresholds_depend_only_on_calibration_grid(baseline, tmp_path):
    altered_test = _spec()
    altered_test["test_in_distribution"]["snr_db"] = [40.0, 40.0]
    altered_test["extended_test"]["snr_db"] = [-10.0, -10.0]
    tested = run_benchmark(altered_test, tmp_path / "test_changed")
    assert _threshold_bytes(tested) == _threshold_bytes(baseline)

    altered_calibration = _spec()
    altered_calibration["calibration"]["clutter_patches"] = [2, 2]
    calibrated = run_benchmark(altered_calibration, tmp_path / "calibration_changed")
    assert _threshold_bytes(calibrated) != _threshold_bytes(baseline)


def test_calibration_achieves_at_most_each_design_pfa(baseline):
    for detector in baseline["detectors"]:
        for run in detector["runs"]:
            for p, achieved in run["calibration"]["pfa_achieved"].items():
                assert achieved <= float(p)


def test_threshold_comparison_is_strict():
    scores = np.array([1.0, 2.0, 2.0, 3.0])
    assert metrics.pfa_at_threshold(scores, 2.0) == 0.25
    assert metrics.pd_at_threshold(scores, 2.0) == 0.25
    assert metrics.pfa_at_threshold(np.array([2.0, 2.0]), 2.0) == 0.0
    bins = metrics.pd_by_snr_at_threshold(scores, np.full(4, 8.0), 2.0, np.array([0, 10]))
    assert bins == [{"snr_lo": 0.0, "snr_hi": 10.0, "n": 4, "pd": 0.25}]


def test_each_split_uses_the_calibrated_thresholds(baseline):
    spec = _spec()
    detector = build("ca_cfar", train=2, guard=1)
    for run in baseline["detectors"][0]["runs"]:
        for name, record in run["splits"].items():
            stream = f"{name}_maps"
            data = build_dataset(
                "stub", spec[name], seed=run["streams"][stream]["materialised_seed"],
            )
            tgt, bg, snr = metrics.split_scores(detector.score(data.maps), data.targets)
            assert record["n_targets"] == len(tgt)
            assert record["n_background_cells"] == len(bg)
            for p, tau in run["thresholds"].items():
                assert record["pd"][p] == float((tgt > tau).mean())
                assert record["pfa_achieved"][p] == float((bg > tau).mean())
                bins = record["pd_by_snr"][p]
                for item in bins:
                    selected = (snr >= item["snr_lo"]) & (snr < item["snr_hi"])
                    assert item["n"] == int(selected.sum())
                    if selected.any():
                        assert item["pd"] == float((tgt[selected] > tau).mean())


@pytest.mark.parametrize("split", ["calibration", "test_in_distribution", "extended_test"])
def test_background_floor_is_enforced_on_every_split(split, tmp_path):
    spec = _spec()
    spec[split]["n_maps"] = 1
    # 24 * 19 after the Doppler mask; target exclusions remove another 49 cells.
    count = 456 if split == "calibration" else 407
    with pytest.raises(ValueError, match=rf"{split}: {count} .*1000"):
        run_benchmark(spec, tmp_path)


def test_background_floor_defaults_to_ten_million(tmp_path):
    spec = _spec()
    del spec["protocol"]["min_eligible_background_cells"]
    with pytest.raises(ValueError, match=r"calibration: 1824 .*10000000"):
        run_benchmark(spec, tmp_path)


@pytest.mark.parametrize("targets", [[0, 1], [1, 1], None])
def test_calibration_must_be_explicitly_target_free(targets, tmp_path):
    spec = _spec()
    if targets is None:
        del spec["calibration"]["n_targets"]
    else:
        spec["calibration"]["n_targets"] = targets
    with pytest.raises(ValueError, match="calibration must be target-free"):
        run_benchmark(spec, tmp_path)


def test_manifest_hash_counts_and_reproducibility(baseline, tmp_path):
    spec = _spec()
    repeated = run_benchmark(spec, tmp_path / "same")
    expected_hash = hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert baseline["run"]["config_sha256"] == repeated["run"]["config_sha256"] == expected_hash
    assert baseline["run"]["root_seeds"] == [0, 1]
    assert baseline["run"]["simulator"] == "stub"
    assert baseline["run"]["simulator_version"]
    # Compare JSON strings so NaN in empty SNR bins does not break equality.
    assert json.dumps(_without_timing(baseline), sort_keys=True) == json.dumps(
        _without_timing(repeated), sort_keys=True,
    )
    assert spec == _spec(), "the harness must not mutate its input"
    stored = json.loads((tmp_path / "same" / "results.json").read_text())
    assert stored["run"]["config_sha256"] == expected_hash
    for run in stored["detectors"][0]["runs"]:
        assert run["streams"] and run["thresholds"]
        assert run["calibration"]["n_background_cells"] >= 1000
        assert set(run["splits"]) == {"test_in_distribution", "extended_test"}
        assert all(item["n_background_cells"] >= 1000 for item in run["splits"].values())
    changed = _spec()
    changed["extended_test"]["snr_db"][0] = 1.0
    assert run_benchmark(changed, tmp_path / "changed")["run"]["config_sha256"] != expected_hash


def test_reporting_prefers_in_distribution_and_supports_extended_only(baseline, tmp_path):
    spec = _spec()
    del spec["test_in_distribution"]
    extended_only = run_benchmark(spec, tmp_path)
    for results, split in ((baseline, "test_in_distribution"), (extended_only, "extended_test")):
        assert results["run"]["reporting_split"] == split
        detector = results["detectors"][0]
        for run in detector["runs"]:
            report = run["splits"][split]
            assert run["pd_at_pfa"] == report["pd"]
            assert run["roc"] == report["roc"]
            assert run["n_targets"] == report["n_targets"]
        for p in results["protocol"]["pfa_points"]:
            values = [run["splits"][split]["pd"][f"{p:g}"] for run in detector["runs"]]
            achieved = [run["splits"][split]["pfa_achieved"][f"{p:g}"] for run in detector["runs"]]
            assert detector[f"pd@pfa={p:g}"] == {"mean": np.mean(values), "std": np.std(values)}
            assert detector[f"pfa_achieved@pfa={p:g}"] == {
                "mean": np.mean(achieved), "std": np.std(achieved),
            }


@pytest.mark.parametrize("field, value", [
    ("quantile_method", "linear"), ("threshold_comparison", "score_ge_threshold"),
    ("seed_streams", []), ("seed_streams", ["train_maps"]),
])
def test_invalid_protocol_settings_fail_clearly(field, value, tmp_path):
    spec = _spec()
    spec["protocol"][field] = value
    with pytest.raises(ValueError, match=field):
        run_benchmark(spec, tmp_path)


@pytest.mark.parametrize("missing", [
    "calibration", "tests", "seed_streams", "model_initialization",
])
def test_required_splits_and_streams_are_checked(missing, tmp_path):
    spec = _spec()
    if missing == "calibration":
        del spec["calibration"]
    elif missing == "tests":
        del spec["test_in_distribution"]
        del spec["extended_test"]
    elif missing == "seed_streams":
        del spec["protocol"]["seed_streams"]
    else:
        spec["protocol"]["seed_streams"].remove("model_initialization")
        spec["detectors"] = [{"kind": "feat_gbdt"}]
    with pytest.raises(ValueError, match="required|missing"):
        run_benchmark(spec, tmp_path)
