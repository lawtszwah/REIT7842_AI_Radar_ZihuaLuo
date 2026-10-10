"""Every declared subcommand must actually dispatch.

`prdbench report` shipped broken: it imported a module that did not exist, so
the command crashed the moment anyone ran it. Nothing caught that, because no
test ever invoked the CLI. These tests do.
"""

import json

import pytest
import yaml

from prdbench import cli
from prdbench.cli import main


def _results_fixture() -> dict:
    """The smallest results.json the reporting code accepts."""
    pfa = [1e-4, 1e-3, 1e-2]
    roc = {"pfa": [1e-3, 1e-2], "pd": [0.4, 0.7]}
    snr = [{"snr_lo": 0.0, "snr_hi": 8.0, "n": 10, "pd": 0.5}]

    def arm(key, family, params):
        run = {"key": key, "seed": 0, "roc": roc, "pd_by_snr": snr}
        rec = {
            "key": key,
            "describe": {"name": key, "family": family, "trainable": False,
                         "n_parameters": params, "params": {}},
            "seeds": 1,
            "latency_s_per_map": {"mean": 0.001, "std": 0.0},
            "n_parameters": params,
            "runs": [run],
        }
        for p in pfa:
            rec[f"pd@pfa={p:g}"] = {"mean": 0.5, "std": 0.01}
        return rec

    return {
        "run": {"git_commit": "0" * 40},
        "protocol": {"pfa_points": pfa},
        "detectors": [arm("ca_cfar", "cfar", 0), arm("cnn", "deep", 1234)],
    }


def test_report_renders_every_figure(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_results_fixture()))

    assert main(["report", "--results", str(results)]) == 0

    figures = tmp_path / "figures"
    for name in ("roc.png", "pd_by_snr.png", "pd_vs_cost.png", "summary.json"):
        assert (figures / name).exists(), f"{name} was not produced"


def test_report_honours_explicit_out_dir(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_results_fixture()))
    out = tmp_path / "elsewhere"

    assert main(["report", "--results", str(results), "--out", str(out)]) == 0
    assert (out / "roc.png").exists()


@pytest.mark.parametrize("cmd", ["run", "tune", "report"])
def test_subcommand_help_does_not_crash(cmd):
    """Catches an import error in a subcommand's module before a user hits it."""
    with pytest.raises(SystemExit) as exc:
        main([cmd, "--help"])
    assert exc.value.code == 0


def test_run_help_lists_current_simulator_choices(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["run", "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "{stub,prdsim}" in help_text
    assert "group" not in help_text


def test_run_rejects_removed_group_simulator():
    with pytest.raises(SystemExit) as exc:
        main(["run", "--config", "unused.yaml", "--simulator", "group"])
    assert exc.value.code == 2


def test_run_merges_joint_shift_before_dispatch(tmp_path, monkeypatch):
    shift = tmp_path / "shift.yaml"
    shift.write_text(yaml.safe_dump({"simulator": "stub", "test": {"n_maps": 4}}))
    spec = {"simulator": "stub", "joint_shift_config": str(shift), "detectors": []}
    config = tmp_path / "protocol.yaml"
    config.write_text(yaml.safe_dump(spec))
    seen = []

    def capture(merged, out):
        seen.append((merged, out))
        return {"protocol": {"pfa_points": [1e-3, 1e-2]}, "detectors": []}

    monkeypatch.setattr(cli, "run_benchmark", capture)
    out = tmp_path / "results"
    assert main(["run", "--config", str(config), "--out", str(out)]) == 0
    assert seen[0][0]["joint_shift"] == {"n_maps": 4}
    assert seen[0][0]["joint_shift_source"]["path"] == str(shift)
    assert seen[0][1] == out
    # An override is checked against the external file's simulator before dispatch.
    with pytest.raises(ValueError, match="simulator"):
        main(["run", "--config", str(config), "--simulator", "prdsim"])
    assert len(seen) == 1


def test_tune_passes_whole_spec_and_preserves_flags(tmp_path, monkeypatch, capsys):
    spec = {"simulator": "stub", "train": {"n_maps": 4}, "validation": {"n_maps": 3},
            "protocol": {"selection_pfa": 1e-3}, "search_spaces": {"ca_cfar": {"train": [2]}}}
    config = tmp_path / "protocol.yaml"
    config.write_text(yaml.safe_dump(spec))
    seen = []

    def capture(spec, kind, space, budget, out):
        seen.append((spec, kind, space, budget, out))
        return {"best": {"params": {"train": 2}, "pd": 0.5}}

    monkeypatch.setattr(cli, "search", capture)
    out = str(tmp_path / "tuning.json")
    assert main(["tune", "--config", str(config), "--detector", "ca_cfar",
                 "--budget", "3", "--out", out]) == 0
    assert seen == [(spec, "ca_cfar", {"train": [2]}, 3, out)]
    assert json.loads(capsys.readouterr().out) == {"params": {"train": 2}, "pd": 0.5}


def test_tune_budget_flag_defaults_to_the_config(tmp_path, monkeypatch):
    spec = {"simulator": "stub", "train": {"n_maps": 4}, "validation": {"n_maps": 3},
            "protocol": {"tuning_budget": 5}, "search_spaces": {"ca_cfar": {"train": [2]}}}
    config = tmp_path / "protocol.yaml"
    config.write_text(yaml.safe_dump(spec))
    seen = []
    monkeypatch.setattr(cli, "search", lambda spec, kind, space, budget, out: (
        seen.append(budget) or {"best": {}}))
    assert main(["tune", "--config", str(config), "--detector", "ca_cfar"]) == 0
    assert seen == [None], "the CLI must not invent a budget; search reads the config"
