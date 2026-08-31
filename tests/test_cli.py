"""Every declared subcommand must actually dispatch.

`prdbench report` shipped broken: it imported a module that did not exist, so
the command crashed the moment anyone ran it. Nothing caught that, because no
test ever invoked the CLI. These tests do.
"""

import json

import pytest

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
