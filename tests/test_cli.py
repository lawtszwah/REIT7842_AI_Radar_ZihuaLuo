"""Every declared subcommand must actually dispatch.

`prdbench report` shipped broken: it imported a module that did not exist, so
the command crashed the moment anyone ran it. Nothing caught that, because no
test ever invoked the CLI. These tests do.
"""

import json

import pytest
import yaml
from test_reporting import arm, results

from prdbench import cli
from prdbench.cli import main


def _results_fixture() -> dict:
    """A small results.json of the shape `prdbench run` writes."""
    return results(arm("ca_cfar", "cfar", [0.5, 0.5]), arm("cnn", "deep", [0.5, 0.4]))


def test_report_renders_every_figure(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_results_fixture()))

    assert main(["report", "--results", str(results)]) == 0

    figures = tmp_path / "figures"
    for name in ("pd_by_root.png", "roc.png", "pd_by_snr.png", "pd_vs_cost.png",
                 "summary.json", "summary.md"):
        assert (figures / name).exists(), f"{name} was not produced"


def test_report_honours_explicit_out_dir(tmp_path):
    results = tmp_path / "results.json"
    results.write_text(json.dumps(_results_fixture()))
    out = tmp_path / "elsewhere"

    assert main(["report", "--results", str(results), "--out", str(out)]) == 0
    assert (out / "roc.png").exists()


def test_report_merges_several_results_files(tmp_path):
    gpu, cpu = tmp_path / "gpu.json", tmp_path / "cpu.json"
    gpu.write_text(json.dumps(results(arm("cnn", "deep", [0.5, 0.4], device="cuda"))))
    cpu.write_text(json.dumps(results(arm("ca_cfar", "cfar", [0.5, 0.5]))))
    out = tmp_path / "merged"

    assert main(["report", "--results", str(gpu), str(cpu), "--out", str(out)]) == 0
    assert "| cnn | 99 | CUDA |" in (out / "summary.md").read_text(encoding="utf-8")


def test_report_of_several_files_needs_an_out_dir(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    with pytest.raises(SystemExit, match="--out"):
        main(["report", "--results", str(a), str(b)])


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

    def capture(merged, out, **kw):
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
