"""A run takes its hyperparameters from checked tuning records, never from the config."""

import hashlib
import json
from importlib import import_module

import pytest
import yaml
from test_protocol_harness import _spec

from prdbench import cli
from prdbench.cli import main
from prdbench.evaluation.protocol import config_sha256, run_benchmark
from prdbench.tuning.search import search
from prdbench.tuning.selection import load_selections, record_path

# The package re-exports the function `search`, which shadows the module name.
search_module = import_module("prdbench.tuning.search")

CFAR_SPACE = {"train": [2, 4], "guard": [1]}
GBDT_SPACE = {"max_iter": [2], "max_depth": [2], "neg_per_map": [16]}


def _tuned_spec():
    spec = _spec()
    spec["protocol"]["tuning_budget"] = 2
    spec["detectors"] = [{"key": "ca_cfar", "kind": "ca_cfar"},
                         {"key": "feat_gbdt", "kind": "feat_gbdt"}]
    spec["search_spaces"] = {"ca_cfar": CFAR_SPACE, "feat_gbdt": GBDT_SPACE}
    return spec


@pytest.fixture(autouse=True)
def clean_tree(monkeypatch):
    # Records made while this repo has uncommitted edits would otherwise be refused.
    monkeypatch.setattr(search_module, "_git_dirty", lambda: False)


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tune")
    spec = _tuned_spec()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(search_module, "_git_dirty", lambda: False)
        for key, space in spec["search_spaces"].items():
            search(spec, key, space, root_seed=0, out=record_path(directory, key))
    return directory


def _edited(records, tmp_path, key, edit):
    record = json.loads(record_path(records, key).read_text())
    edit(record)
    for other in ("ca_cfar", "feat_gbdt"):
        record_path(tmp_path, other).write_text(record_path(records, other).read_text())
    record_path(tmp_path, key).write_text(json.dumps(record))
    return tmp_path


def test_run_uses_selected_params_and_records_where_they_came_from(records, tmp_path):
    spec = _tuned_spec()
    tuned = load_selections(spec, records, ["ca_cfar", "feat_gbdt"])
    result = run_benchmark(spec, tmp_path, tuned=tuned)

    assert result["run"]["config_sha256"] == config_sha256(_tuned_spec()), "spec untouched"
    assert result["spec"] == _tuned_spec()
    assert result["run"]["detectors_run"] == ["ca_cfar", "feat_gbdt"]
    for detector in result["detectors"]:
        key = detector["key"]
        record = json.loads(record_path(records, key).read_text())
        source = result["run"]["tuned"][key]["source"]
        raw = record_path(records, key).read_bytes()
        assert source["sha256"] == hashlib.sha256(raw).hexdigest()
        assert source["validation_pd"] == record["best"]["pd"]
        selected = {k: v for k, v in record["best"]["params"].items() if k != "seed"}
        for run in detector["runs"]:
            used = run["describe"]["params"]
            assert {k: used[k] for k in selected} == selected


def test_model_seed_still_comes_from_each_root(records, tmp_path):
    spec = _tuned_spec()
    tuned = load_selections(spec, records, ["feat_gbdt"])
    assert "seed" not in tuned["feat_gbdt"]["params"]
    runs = run_benchmark(spec, tmp_path, tuned=tuned, only=["feat_gbdt"])["detectors"][0]["runs"]
    record = json.loads(record_path(records, "feat_gbdt").read_text())
    assert runs[0]["describe"]["params"]["seed"] == record["best"]["params"]["seed"]
    assert runs[1]["describe"]["params"]["seed"] != runs[0]["describe"]["params"]["seed"]


def test_only_runs_the_named_detectors_with_the_same_data(records, tmp_path):
    spec = _tuned_spec()
    tuned = load_selections(spec, records, ["ca_cfar", "feat_gbdt"])
    both = run_benchmark(spec, tmp_path / "both", tuned=tuned)
    one = run_benchmark(spec, tmp_path / "one", tuned=tuned, only=["ca_cfar"])
    assert [d["key"] for d in one["detectors"]] == ["ca_cfar"]
    assert one["run"]["detectors_run"] == ["ca_cfar"]
    assert list(one["run"]["tuned"]) == ["ca_cfar"]
    assert one["run"]["config_sha256"] == both["run"]["config_sha256"]
    for a, b in zip(one["detectors"][0]["runs"], both["detectors"][0]["runs"], strict=True):
        assert a["streams"] == b["streams"] and a["thresholds"] == b["thresholds"]


def test_every_detector_that_runs_must_be_tuned(records, tmp_path):
    spec = _tuned_spec()
    tuned = load_selections(spec, records, ["ca_cfar"])
    with pytest.raises(ValueError, match="no tuned hyperparameters for \\['feat_gbdt'\\]"):
        run_benchmark(spec, tmp_path, tuned=tuned)


@pytest.mark.parametrize("only", [["nope"], []])
def test_only_rejects_unknown_or_empty_keys(only, tmp_path):
    with pytest.raises(ValueError, match="unknown detector keys"):
        run_benchmark(_tuned_spec(), tmp_path, only=only)


def test_config_params_may_not_contradict_the_selection(records, tmp_path):
    spec = _tuned_spec()
    tuned = load_selections(spec, records, ["ca_cfar"])
    spec["detectors"][0]["params"] = {"guard": 9}
    with pytest.raises(ValueError, match="contradict"):
        run_benchmark(spec, tmp_path, tuned=tuned, only=["ca_cfar"])


def _set(path, value):
    def edit(record):
        target = record
        for step in path[:-1]:
            target = target[step]
        target[path[-1]] = value
    return edit


@pytest.mark.parametrize("key, edit, message", [
    ("ca_cfar", _set(["detector"], "go_cfar"), "is for 'go_cfar'"),
    ("ca_cfar", _set(["root_seed"], 1), "root seed 1"),
    ("ca_cfar", _set(["run", "config_sha256"], "0" * 64), "was made under config"),
    ("ca_cfar", _set(["run", "git_dirty"], True), "uncommitted changes"),
    ("ca_cfar", _set(["selection_pfa"], 0.01), "selected at Pfa"),
    ("ca_cfar", _set(["budget"], 36), "budget of 36"),
    ("ca_cfar", _set(["n_trials"], 1), "trials"),
    ("ca_cfar", lambda r: r["all"].pop(), "trials"),
    ("ca_cfar", lambda r: r["best"].update(pd=r["best"]["pd"] + 1), "not the first highest"),
    ("ca_cfar", lambda r: [t["params"].update(train=99) for t in r["all"]]
     + [r["best"]["params"].update(train=99)], "outside"),
    ("ca_cfar", lambda r: [t["params"].update(extra=1) for t in r["all"]]
     + [r["best"]["params"].update(extra=1)], "selects parameters"),
    ("feat_gbdt", lambda r: [t["params"].update(seed=1) for t in r["all"]]
     + [r["best"]["params"].update(seed=1)], "model seed"),
])
def test_records_the_frozen_protocol_could_not_have_made_are_refused(
        records, tmp_path, key, edit, message):
    directory = _edited(records, tmp_path, key, edit)
    with pytest.raises(ValueError, match=message):
        load_selections(_tuned_spec(), directory, [key])


def test_a_missing_record_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no tuning record"):
        load_selections(_tuned_spec(), tmp_path, ["ca_cfar"])


def test_tuned_on_another_simulator_version_is_refused(records, tmp_path):
    spec = _tuned_spec()
    tuned = load_selections(spec, records, ["ca_cfar"])
    tuned["ca_cfar"]["source"]["simulator_version"] = "other"
    with pytest.raises(ValueError, match="tuned on simulator other"):
        run_benchmark(spec, tmp_path, tuned=tuned, only=["ca_cfar"])


def test_cli_checks_records_before_dispatch(records, tmp_path, monkeypatch):
    config = tmp_path / "protocol.yaml"
    config.write_text(yaml.safe_dump(_tuned_spec()))
    seen = []
    monkeypatch.setattr(cli, "run_benchmark", lambda spec, out, **kw: (
        seen.append(kw) or {"protocol": {"pfa_points": [1e-3]}, "detectors": []}))
    assert main(["run", "--config", str(config), "--out", str(tmp_path / "r"),
                 "--tuned", str(records), "--detectors", "ca_cfar"]) == 0
    assert seen[0]["only"] == ["ca_cfar"]
    assert list(seen[0]["tuned"]) == ["ca_cfar"]
    assert seen[0]["tuned"]["ca_cfar"]["params"] in ({"train": 2, "guard": 1},
                                                     {"train": 4, "guard": 1})
    with pytest.raises(SystemExit, match="unknown detector keys"):
        main(["run", "--config", str(config), "--detectors", "nope"])
    with pytest.raises(ValueError, match="no tuning record"):
        main(["run", "--config", str(config), "--tuned", str(tmp_path / "empty")])
    assert len(seen) == 1
