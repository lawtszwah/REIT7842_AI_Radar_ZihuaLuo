"""The adapter must wait for MATLAB to exit, not for every process holding its output.

MATLAB can start a background service that inherits its stdout/stderr and outlives it.
On 2026-11-17 that left the freeze run waiting on a pipe after MATLAB had finished.
These tests stand in a fake `matlab` script, so they need neither MATLAB nor the generator.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from prdbench.simulation.groupgen import GroupGenerator

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake matlab is a shell script")


def _fake_matlab(tmp_path, body: str):
    script = tmp_path / "matlab"
    script.write_text("#!/bin/sh\n" + body + "\n")
    os.chmod(script, 0o755)
    return GroupGenerator(generator_dir=tmp_path, matlab=str(script))


def test_a_lingering_background_child_does_not_block(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    # The background sleep keeps MATLAB's stdout and stderr open after MATLAB exits.
    sim = _fake_matlab(tmp_path, "echo generating; sleep 30 & exit 0")
    start = time.monotonic()
    sim._run_matlab([], out)
    assert time.monotonic() - start < 10


def test_matlab_failure_reports_its_output(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    sim = _fake_matlab(tmp_path, "echo 'Undefined function prd_groupgen' >&2; exit 3")
    with pytest.raises(RuntimeError, match=r"exit 3[\s\S]*Undefined function prd_groupgen"):
        sim._run_matlab([], out)
