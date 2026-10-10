"""The harness calibrates independently and records the spawned seed protocol."""

import copy
import hashlib
import json
import struct
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
import yaml

from prdbench.data.generate import build_dataset
from prdbench.detectors import build
from prdbench.evaluation import metrics, protocol
from prdbench.evaluation.protocol import merge_joint_shift, run_benchmark
from prdbench.evaluation.seeds import make_streams, stream_record
from prdbench.tuning.search import search

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL = yaml.safe_load((REPO_ROOT / "configs/protocol/full.yaml").read_text())
MAP_STREAMS = {
    "train_maps", "calibration_maps", "test_in_distribution_maps", "extended_test_maps",
}
BOOTSTRAP_STREAMS = {
    "analysis_bootstrap/test_in_distribution", "analysis_bootstrap/extended_test",
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
        "validation": copy.deepcopy(grid),
        "calibration": grid | {"n_targets": [0, 0]},
        "test_in_distribution": copy.deepcopy(grid),
        "extended_test": grid | {"snr_db": [2.0, 6.0]},
        "protocol": {
            "pfa_points": [1e-3, 1e-2, 0.1], "snr_edges": [0, 6, 12, 18],
            "n_seeds": 2, "latency_repeats": 1, "min_eligible_background_cells": 1000,
            "quantile_method": "higher", "threshold_comparison": "score_gt_threshold",
            "seed_streams": list(FULL["protocol"]["seed_streams"]),
            "selection_pfa": 1e-3, "bootstrap_resamples": 64,
            "bootstrap_interval_percentiles": [2.5, 97.5],
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
        assert set(run["streams"]) == MAP_STREAMS | BOOTSTRAP_STREAMS
        for name, record in run["streams"].items():
            parent, _, split = name.partition("/")
            child = np.random.SeedSequence(root).spawn(len(names))[names.index(parent)]
            if split:
                child = child.spawn(3)[protocol.TEST_SPLITS.index(split)]
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
            assert set(run["streams"]) == MAP_STREAMS | BOOTSTRAP_STREAMS | {
                "model_initialization",
            }
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


@pytest.mark.parametrize("root, expected, model_seed", [
    (0, [8668861027912758289, 4881901421217228719, 16452687389592421897,
         13238389300853459902, 16732062099500631801, 14153166575370824094,
         675977917276771736, 16247558764769518856], 23012616),
    (1, [8431846347943309920, 4042681867674859579, 1275975541612323131,
         10440292027562320097, 11724249445994358946, 5892472470897923492,
         1745718741773654819, 15460960693284752821], 3112422837),
])
def test_appending_tuning_stream_preserves_frozen_seeds(root, expected, model_seed):
    # Captured from the eight-stream protocol before appending the tuning stream.
    original = ["train_maps", "test_in_distribution_maps", "extended_test_maps",
                "joint_shift_maps", "validation_maps", "calibration_maps",
                "analysis_bootstrap", "model_initialization"]
    for config in (FULL, yaml.safe_load((REPO_ROOT / "configs/protocol/smoke.yaml").read_text())):
        names = config["protocol"]["seed_streams"]
        assert names == [*original, "tuning_trial_selection"]
        streams = make_streams(root, names)
        assert [stream_record(streams[name])["materialised_seed"]
                for name in original] == expected
        assert stream_record(streams["model_initialization"], model=True)[
            "materialised_seed"
        ] == model_seed
        assert config["protocol"]["selection_pfa"] == 1e-3
        assert config["protocol"]["bootstrap_interval_percentiles"] == [2.5, 97.5]


def test_selection_uses_only_validation_and_records_each_trial(tmp_path):
    spec = _spec()
    spec["validation"]["snr_db"] = [40.0, 40.0]
    space = {"train": [2, 3, 4], "guard": [1, 2]}
    result = search(spec, "ca_cfar", space, budget=3, root_seed=17, out=tmp_path / "tune.json")
    changed = copy.deepcopy(spec)
    changed["calibration"]["n_targets"] = [99, 99]
    changed["test_in_distribution"]["snr_db"] = [-40.0, -40.0]
    changed["extended_test"]["snr_db"] = [-40.0, -40.0]
    assert search(changed, "ca_cfar", space, budget=3, root_seed=17) == result
    changed["validation"]["snr_db"] = [-40.0, -40.0]
    weaker = search(changed, "ca_cfar", space, budget=3, root_seed=17)
    assert [r["pd"] for r in weaker["all"]] != [r["pd"] for r in result["all"]]
    assert result["root_seed"] == 17
    assert result["selection_pfa"] == 1e-3
    assert result["n_trials"] == result["budget"] == 3
    assert set(result["streams"]) == {"train_maps", "validation_maps", "tuning_trial_selection"}
    assert result["best"] == result["all"][0], "ties use trial order after subsampling"
    val = build_dataset("stub", spec["validation"],
                        seed=result["streams"]["validation_maps"]["materialised_seed"])
    for trial in result["all"]:
        tgt, bg, _ = metrics.split_scores(build("ca_cfar", **trial["params"]).score(val.maps),
                                          val.targets)
        assert trial["threshold"] == float(np.quantile(bg, 0.999, method="higher"))
        assert trial["pd"] == float((tgt > trial["threshold"]).mean())
        assert trial["n_targets"] == len(tgt)
        assert trial["n_background_cells"] == len(bg)
    assert json.loads((tmp_path / "tune.json").read_text()) == result


def test_search_and_benchmark_share_map_and_native_model_seeds(tmp_path):
    spec = _spec()
    params = {"max_iter": 2, "max_depth": 2, "neg_per_map": 16, "seed": 999}
    spec["detectors"] = [{"kind": "feat_gbdt", "params": params}]
    run = run_benchmark(spec, tmp_path)["detectors"][0]["runs"][1]
    tuned = search(spec, "feat_gbdt", {key: [v] for key, v in params.items()}, budget=1,
                   root_seed=1)
    for name in ("train_maps", "model_initialization"):
        assert tuned["streams"][name] == run["streams"][name]
    assert tuned["best"]["params"]["seed"] == run["describe"]["params"]["seed"] < 2**32
    assert "tuning_trial_selection" not in tuned["streams"], "no unused stream is recorded"


@pytest.mark.parametrize("problem", ["missing", "floor", "no_targets"])
def test_selection_requires_eligible_validation_data(problem):
    spec = _spec()
    if problem == "missing":
        del spec["validation"]
        message = "explicit validation"
    elif problem == "floor":
        spec["validation"]["n_maps"] = 1
        message = "validation: 407 .*1000"
    else:
        spec["validation"]["n_targets"] = [0, 0]
        message = "eligible targets"
    with pytest.raises(ValueError, match=message):
        search(spec, "ca_cfar", {"train": [2], "guard": [1]}, budget=1)


def test_selection_uses_strict_comparison_with_tied_scores(monkeypatch):
    monkeypatch.setattr(import_module("prdbench.tuning.search"), "_split_scores",
                        lambda *args: (np.array([2.0, 3.0]), np.full(1000, 2.0), None))
    tuned = search(_spec(), "ca_cfar", {"train": [2], "guard": [1]}, budget=1)
    assert tuned["best"]["threshold"] == 2.0
    assert tuned["best"]["pd"] == 0.5


def test_joint_shift_merge_hash_thresholds_and_stream(tmp_path):
    spec = _spec()
    spec["joint_shift_config"] = "shift.yaml"
    shift = {"simulator": "stub", "test": spec["extended_test"] | {"snr_db": [-6.0, -1.0]}}
    file = tmp_path / "shift.yaml"
    file.write_text(yaml.safe_dump(shift))
    merged = merge_joint_shift(spec, repo_root=tmp_path)
    assert "joint_shift" not in spec
    assert merged["joint_shift"] == shift["test"]
    assert merged["joint_shift_source"] == {
        "path": "shift.yaml", "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
    }
    res = run_benchmark(merged, tmp_path / "joint")
    assert res["run"]["joint_shift_source"] == merged["joint_shift_source"]
    assert res["run"]["reporting_split"] == "test_in_distribution"
    for run in res["detectors"][0]["runs"]:
        streams = make_streams(run["root_seed"], spec["protocol"]["seed_streams"])
        assert run["streams"]["joint_shift_maps"] == stream_record(streams["joint_shift_maps"])
        data = build_dataset("stub", shift["test"],
                             seed=run["streams"]["joint_shift_maps"]["materialised_seed"])
        tgt, bg, _ = metrics.split_scores(build("ca_cfar", train=2, guard=1).score(data.maps),
                                          data.targets)
        for p, tau in run["thresholds"].items():
            assert run["splits"]["joint_shift"]["pd"][p] == float((tgt > tau).mean())
            assert run["splits"]["joint_shift"]["pfa_achieved"][p] == float((bg > tau).mean())
        assert run["pd_at_pfa"] == run["splits"]["test_in_distribution"]["pd"]
    original = run_benchmark(_spec(), tmp_path / "original")
    assert _threshold_bytes(res) == _threshold_bytes(original)
    # Even a byte-only edit must be covered by provenance and the merged spec hash.
    file.write_text(file.read_text() + "\n# revised source\n")
    amended = merge_joint_shift(spec, repo_root=tmp_path)
    assert amended["joint_shift"] == merged["joint_shift"]
    assert run_benchmark(amended, tmp_path / "changed")["run"]["config_sha256"] != res[
        "run"
    ]["config_sha256"]
    shift["test"]["snr_db"] = [-12.0, -10.0]
    file.write_text(yaml.safe_dump(shift))
    changed = run_benchmark(merge_joint_shift(spec, repo_root=tmp_path), tmp_path / "grid_changed")
    assert changed["run"]["config_sha256"] != res["run"]["config_sha256"]
    assert _threshold_bytes(changed) == _threshold_bytes(res)
    shift["simulator"] = "prdsim"
    file.write_text(yaml.safe_dump(shift))
    with pytest.raises(ValueError, match="simulator"):
        merge_joint_shift(spec, repo_root=tmp_path)
    with pytest.raises(ValueError, match="merged"):
        run_benchmark(spec, tmp_path / "unmerged")


def test_joint_shift_floor_and_reporting_with_extended_only(tmp_path):
    spec = _spec()
    spec["joint_shift"] = copy.deepcopy(spec["extended_test"])
    spec["joint_shift"]["n_maps"] = 1
    with pytest.raises(ValueError, match="joint_shift: 407 .*1000"):
        run_benchmark(spec, tmp_path / "floor")
    spec["joint_shift"]["n_maps"] = 4
    del spec["test_in_distribution"]
    results = run_benchmark(spec, tmp_path / "extended")
    assert results["run"]["reporting_split"] == "extended_test"
    del spec["extended_test"]
    with pytest.raises(ValueError, match="at least one"):
        run_benchmark(spec, tmp_path / "joint_only")


def test_joint_shift_path_is_relative_to_repo_root_not_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    merged = merge_joint_shift(FULL)
    authoritative = REPO_ROOT / "configs/data/shift_test.yaml"
    assert merged["joint_shift"] == yaml.safe_load(authoritative.read_text())["test"]
    assert merged["joint_shift_source"]["sha256"] == hashlib.sha256(
        authoritative.read_bytes(),
    ).hexdigest()


def test_joint_shift_requires_test_block(tmp_path):
    (tmp_path / "shift.yaml").write_text("simulator: stub\n")
    with pytest.raises(ValueError, match="test block"):
        merge_joint_shift(_spec() | {"joint_shift_config": "shift.yaml"}, repo_root=tmp_path)


def test_bootstrap_indices_are_paired_and_stable_when_splits_are_absent(monkeypatch, tmp_path):
    original = protocol.bootstrap.intervals
    seen = []

    def capture(counts, indices, percentiles):
        seen.append(indices)
        return original(counts, indices, percentiles)

    monkeypatch.setattr(protocol.bootstrap, "intervals", capture)
    spec = _spec()
    spec["detectors"] = [spec["detectors"][0] | {"key": "first"},
                         spec["detectors"][0] | {"key": "second"}]
    res = run_benchmark(spec, tmp_path / "paired")
    for offset in (0, 4):
        assert seen[offset] is seen[offset + 2]
        assert seen[offset + 1] is seen[offset + 3]
        assert seen[offset].dtype == np.int32
        assert seen[offset].shape == (64, 4)
    assert not np.array_equal(seen[0], seen[4])
    for first, second in zip(res["detectors"][0]["runs"], res["detectors"][1]["runs"], strict=True):
        assert json.dumps(first["splits"], sort_keys=True) == json.dumps(
            second["splits"], sort_keys=True,
        )
        for record in first["splits"].values():
            for field in ("pd_interval", "pfa_interval"):
                for lo, hi in record[field].values():
                    assert 0 <= lo <= hi <= 1
    full_indices = seen.copy()
    seen.clear()
    spec["joint_shift"] = copy.deepcopy(spec["extended_test"])
    del spec["test_in_distribution"]
    shifted = run_benchmark(spec, tmp_path / "absent")
    for root in range(2):
        assert np.array_equal(full_indices[root * 4 + 1], seen[root * 4])
        name = "analysis_bootstrap/extended_test"
        assert shifted["detectors"][0]["runs"][root]["streams"][name] == res[
            "detectors"
        ][0]["runs"][root]["streams"][name]
        assert shifted["detectors"][0]["runs"][root]["streams"][
            "analysis_bootstrap/joint_shift"
        ]["spawn_key"] == [6, 2]


def test_simulator_manifest_whitelists_only_split_provenance(monkeypatch, tmp_path):
    original = protocol.build_dataset

    def with_per_map_metadata(*args, **kwargs):
        dataset = original(*args, **kwargs)
        dataset.meta["derived_seeds"] = [{"seed": 123}] * len(dataset.maps)
        dataset.meta["clutter_records"] = [{"patch": "large metadata"}] * len(dataset.maps)
        return dataset

    monkeypatch.setattr(protocol, "build_dataset", with_per_map_metadata)
    spec = _spec()
    res = run_benchmark(spec, tmp_path)
    meta = res["run"]["simulator_meta"]
    assert set(meta) == {"train", "calibration", "test_in_distribution", "extended_test"}
    for split, record in meta.items():
        assert set(record) == {"simulator", "simulator_version", "config"}
        assert record["simulator"] == "stub"
        assert record["config"]["n_maps"] == spec[split]["n_maps"]
        stream = f"{split}_maps" if split != "train" else "train_maps"
        assert record["config"]["seed"] == res["detectors"][0]["runs"][0]["streams"][
            stream
        ]["materialised_seed"]


@pytest.mark.parametrize("field, value", [
    ("bootstrap_resamples", 0), ("bootstrap_interval_percentiles", [97.5, 2.5]),
    ("bootstrap_interval_percentiles", [-1, 97.5]),
    ("bootstrap_interval_percentiles", [float("nan"), 97.5]), ("selection_pfa", 0),
])
def test_invalid_analysis_settings_fail_clearly(field, value, tmp_path):
    spec = _spec()
    spec["protocol"][field] = value
    with pytest.raises(ValueError, match=field):
        run_benchmark(spec, tmp_path)


def test_tuning_budget_comes_from_the_config():
    """RSK-03: equal budget only holds if the frozen config fixes it."""
    spec = _spec()
    space = {"train": [2, 3], "guard": [1, 2]}
    spec["protocol"]["tuning_budget"] = 2
    assert search(spec, "ca_cfar", space)["budget"] == 2
    assert search(spec, "ca_cfar", space, budget=2)["n_trials"] == 2
    with pytest.raises(ValueError, match="tuning_budget is 2"):
        search(spec, "ca_cfar", space, budget=3)


def test_tuning_budget_must_be_set_somewhere():
    with pytest.raises(ValueError, match="tuning_budget"):
        search(_spec(), "ca_cfar", {"train": [2], "guard": [1]})


@pytest.mark.parametrize("value", [0, -1, 1.5, True])
def test_tuning_budget_must_be_a_positive_integer(value):
    spec = _spec()
    spec["protocol"]["tuning_budget"] = value
    with pytest.raises(ValueError, match="positive integer"):
        search(spec, "ca_cfar", {"train": [2], "guard": [1]})


def test_full_protocol_freezes_a_tuning_budget():
    assert isinstance(FULL["protocol"]["tuning_budget"], int)
    assert FULL["protocol"]["tuning_budget"] >= 1
