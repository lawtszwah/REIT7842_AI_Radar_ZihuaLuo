"""Reporting merges arms run separately, shows every root, and keeps GPU latency apart."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from prdbench.reporting import (
    ReportError,
    load,
    markdown,
    over_pfa,
    pfa_text,
    render,
    summarise,
    tied_at_threshold,
)

PFA = [1e-4, 1e-3]
SPLITS = ["test_in_distribution", "joint_shift"]
N_CAL = 10_533_600


def _streams(root: int, trained: bool) -> dict:
    streams = {"test_in_distribution_maps": {"spawn_key": [1], "materialised_seed": 100 + root},
               "calibration_maps": {"spawn_key": [5], "materialised_seed": 200 + root}}
    if trained:
        streams["model_initialization"] = {"spawn_key": [7], "materialised_seed": 300 + root}
    return streams


def arm(key: str, family: str, pd: list[float], *, device: str | None = None,
        pfa_achieved: float | None = None, cal_zero_root: int | None = None) -> dict:
    """One detector record with one run per entry of `pd` (the Pd at every point)."""
    runs = []
    for root, value in enumerate(pd):
        split = {"pd": {f"{p:g}": value for p in PFA},
                 "pfa_achieved": {f"{p:g}": p if pfa_achieved is None else pfa_achieved
                                  for p in PFA},
                 "roc": {"pfa": [1e-5, 1e-4, 1e-3], "pd": [0.1, value, 0.6]},
                 "pd_by_snr": {f"{p:g}": [{"snr_lo": 0.0, "snr_hi": 4.0, "n": 5, "pd": value},
                                          {"snr_lo": 4.0, "snr_hi": 8.0, "n": 0,
                                           "pd": float("nan")}] for p in PFA}}
        cal = {f"{p:g}": 0.0 if root == cal_zero_root else p for p in PFA}
        runs.append({"key": key, "seed": root, "root_seed": root,
                     "streams": _streams(root, family != "cfar"),
                     "thresholds": {f"{p:g}": 1.5 for p in PFA},
                     "calibration": {"pfa_achieved": cal, "n_background_cells": N_CAL},
                     "splits": {s: split for s in SPLITS},
                     "cost": {"latency_s_per_map_median": 0.002 + root * 1e-4}})
    describe = {"name": key.upper(), "family": family, "params": {}}
    if device:
        describe["device"] = device
    return {"key": key, "describe": describe, "n_parameters": 0 if family == "cfar" else 99,
            "runs": runs}


def results(*arms: dict, commit: str = "a" * 40, config: str = "c" * 64) -> dict:
    return {"run": {"git_commit": commit, "git_dirty": False, "config_sha256": config,
                    "simulator": "groupgen", "simulator_version": "1.0.0+gen.x",
                    "root_seeds": [0, 1], "reporting_split": "test_in_distribution"},
            "protocol": {"pfa_points": PFA, "selection_pfa": 1e-3},
            "spec": {"detectors": [{"key": "ca_cfar"}, {"key": "gbdt"}, {"key": "cnn"}]},
            "detectors": list(arms)}


def write(tmp_path, name: str, res: dict):
    path = tmp_path / name / "results.json"
    path.parent.mkdir()
    path.write_text(json.dumps(res))
    return path


@pytest.fixture
def pair(tmp_path):
    gpu = write(tmp_path, "gpu", results(arm("cnn", "deep", [0.30, 0.0], device="cuda",
                                             cal_zero_root=1)))
    cpu = write(tmp_path, "cpu", results(arm("ca_cfar", "cfar", [0.28, 0.29]),
                                         arm("gbdt", "feature", [0.31, 0.30],
                                             pfa_achieved=0.05)))
    return gpu, cpu


def test_files_merge_in_config_order_with_their_provenance(pair):
    merged = load(list(pair))
    assert [d["key"] for d in merged["detectors"]] == ["ca_cfar", "gbdt", "cnn"]
    assert [s["detectors"] for s in merged["sources"]] == [["cnn"], ["ca_cfar", "gbdt"]]
    assert all(len(s["sha256"]) == 64 for s in merged["sources"])


@pytest.mark.parametrize("field, value", [("config_sha256", "d" * 64),
                                          ("simulator_version", "2.0.0"),
                                          ("root_seeds", [0, 1, 2])])
def test_files_from_different_experiments_are_refused(tmp_path, field, value):
    a = write(tmp_path, "a", results(arm("ca_cfar", "cfar", [0.3, 0.3])))
    other = results(arm("cnn", "deep", [0.3, 0.3]))
    other["run"][field] = value
    b = write(tmp_path, "b", other)
    with pytest.raises(ReportError, match=field):
        load([a, b])


def test_unpaired_data_streams_are_refused(tmp_path):
    a = write(tmp_path, "a", results(arm("ca_cfar", "cfar", [0.3, 0.3])))
    shifted = arm("cnn", "deep", [0.3, 0.3])
    shifted["runs"][1]["streams"]["calibration_maps"]["materialised_seed"] = 999
    b = write(tmp_path, "b", results(shifted))
    with pytest.raises(ReportError, match="calibration_maps"):
        load([a, b])


def test_untrained_arms_pair_without_a_model_stream(pair):
    """CFAR draws no model_initialization stream; the streams in common must agree."""
    load(list(pair))


def test_a_detector_in_two_files_is_refused(tmp_path):
    a = write(tmp_path, "a", results(arm("cnn", "deep", [0.3, 0.3])))
    b = write(tmp_path, "b", results(arm("cnn", "deep", [0.3, 0.3])))
    with pytest.raises(ReportError, match="more than one file"):
        load([a, b])


def test_summary_keeps_every_root_and_uses_the_population_sd(pair):
    summary = summarise(load(list(pair)))
    assert summary["headline_pfa"] == 1e-4
    cnn = next(d for d in summary["detectors"] if d["detector"] == "cnn")
    s = cnn["splits"]["test_in_distribution"]["0.0001"]["pd"]
    assert s["per_root"] == [0.30, 0.0]
    assert s["mean"] == pytest.approx(0.15) and s["sd"] == pytest.approx(np.std([0.3, 0.0]))
    assert cnn["latency_device"] == "cuda"
    assert cnn["latency_ms"]["per_root"] == pytest.approx([2.0, 2.1])


def test_unattained_operating_points_are_named_with_their_ties(pair):
    summary = summarise(load(list(pair)))
    cnn = next(d for d in summary["detectors"] if d["detector"] == "cnn")
    assert {(u["root_seed"], u["pfa"]) for u in cnn["unattained"]} == {(1, 1e-4), (1, 1e-3)}
    tied = next(u for u in cnn["unattained"] if u["pfa"] == 1e-4)
    assert tied["min_cells_tied_at_threshold"] == 1054
    assert all(not d["unattained"] for d in summary["detectors"] if d["detector"] != "cnn")


def test_tie_count_matches_the_higher_quantile():
    """At least that many cells sit at or above numpy's "higher" quantile."""
    for n, p in [(N_CAL, 1e-4), (N_CAL, 1e-5), (12345, 1e-2)]:
        scores = np.arange(n, dtype=float)
        threshold = np.quantile(scores, 1 - p, method="higher")
        assert tied_at_threshold(n, p) == int(np.sum(scores >= threshold))
    assert tied_at_threshold(N_CAL, 1e-4) == N_CAL - math.ceil((1 - 1e-4) * (N_CAL - 1))


def test_pd_beside_a_broken_pfa_is_flagged(pair):
    summary = summarise(load(list(pair)))
    assert over_pfa(summary, "joint_shift", 1e-4) == ["gbdt"]
    text = markdown(summary)
    assert "more than 2× the design Pfa for gbdt" in text
    assert "## Pd at design Pfa = 1e-4 (headline operating point)" in text
    assert "## Pd at design Pfa = 1e-3 (selection operating point)" in text
    assert "| cnn | 1 | 1e-4 | 1.5 | 1,054 of 10,533,600 |" in text
    assert "| cnn | 99 | CUDA |" in text


def test_render_writes_every_output(pair, tmp_path):
    out = tmp_path / "figures"
    written = render(list(pair), out)
    assert sorted(p.name for p in written) == sorted(
        ["pd_by_root.png", "roc.png", "pd_by_snr.png", "pd_vs_cost.png", "summary.json",
         "summary.md"])
    assert all(p.stat().st_size > 0 for p in written)
    assert json.loads((out / "summary.json").read_text())["detectors"][2]["detector"] == "cnn"


def test_pfa_text():
    assert pfa_text(1e-4) == "1e-4"
    assert pfa_text(1e-2) == "1e-2"
    assert pfa_text(5e-3) == "5e-3"
