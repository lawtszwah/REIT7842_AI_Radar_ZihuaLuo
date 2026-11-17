"""The torch device is an execution setting: chosen at run time, recorded, never frozen."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from prdbench import cli
from prdbench.cli import main
from prdbench.detectors.cnn import DEVICE_ENV, DilatedCNN, resolve_device


@pytest.fixture(autouse=True)
def _no_device_env(monkeypatch):
    monkeypatch.delenv(DEVICE_ENV, raising=False)


def test_default_is_cpu():
    assert resolve_device() == torch.device("cpu")
    assert DilatedCNN(width=2, depth=1).device == torch.device("cpu")


def test_argument_wins_over_environment(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV, "auto")
    assert resolve_device("cpu") == torch.device("cpu")


def test_environment_is_used_without_an_argument(monkeypatch):
    monkeypatch.setenv(DEVICE_ENV, " CPU ")
    assert resolve_device() == torch.device("cpu")


def test_auto_picks_an_available_device():
    dev = resolve_device("auto")
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    elif torch.backends.mps.is_available():
        assert dev.type == "mps"
    else:
        assert dev.type == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is present here")
def test_unavailable_cuda_raises_instead_of_falling_back():
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device("cuda")


def test_unsupported_device_is_rejected():
    with pytest.raises(ValueError, match="unsupported device"):
        resolve_device("meta")


def test_device_is_recorded_but_not_a_hyperparameter():
    det = DilatedCNN(width=2, depth=1, device="cpu")
    described = det.describe()
    assert described["device"] == "cpu"
    assert "device" not in described["params"], "trial records must not depend on the device"


def test_cnn_trains_and_scores_on_the_resolved_device():
    rng = np.random.default_rng(0)
    maps = rng.exponential(size=(4, 8, 8)).astype(np.float32)
    labels = np.zeros_like(maps)
    labels[:, 4, 4] = 1
    det = DilatedCNN(width=2, depth=1, epochs=1, batch_size=2, device="auto")
    scores = det.fit(maps, labels).score(maps)
    assert scores.shape == maps.shape and scores.dtype == np.float32


def test_cli_device_sets_the_environment_for_the_run(tmp_path, monkeypatch):
    config = tmp_path / "protocol.yaml"
    config.write_text(yaml.safe_dump({"simulator": "stub", "detectors": []}))
    seen = []

    def capture(spec, out):
        import os
        seen.append(os.environ.get(DEVICE_ENV))
        return {"protocol": {"pfa_points": [1e-3, 1e-2]}, "detectors": []}

    monkeypatch.setattr(cli, "run_benchmark", capture)
    assert main(["run", "--config", str(config), "--out", str(tmp_path / "r"),
                 "--device", "cpu"]) == 0
    assert seen == ["cpu"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is present here")
def test_cli_rejects_an_unavailable_device_before_running(tmp_path, monkeypatch):
    config = tmp_path / "protocol.yaml"
    config.write_text(yaml.safe_dump({"simulator": "stub", "detectors": []}))
    monkeypatch.setattr(cli, "run_benchmark", lambda *a: pytest.fail("must not run"))
    with pytest.raises(RuntimeError, match="CUDA"):
        main(["run", "--config", str(config), "--device", "cuda"])
