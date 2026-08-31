"""Adapter for the research group's parameterised simulation function.

STATUS: not yet received (see docs/risk_register.md, RSK-01).

When it arrives, implement `GroupSimulator.generate` so that it
  1. maps `SimConfig` fields onto the group function's own arguments,
  2. converts its output to linear-power `float32` maps of shape (N, R, D),
  3. converts its ground truth to `Target(range_bin, doppler_bin, snr_db)` in
     the same bin convention as the maps (0-indexed, zero-Doppler at nd // 2),
  4. records the function's version / commit in `meta`.

Nothing else in the benchmark should need to change.  `tests/test_simulator_
contract.py` runs the same conformance checks against any simulator, so the
adapter can be validated the day the function is handed over.
"""

from __future__ import annotations

from .interface import SimBatch, SimConfig


class GroupSimulator:
    name = "group"
    version = "unreleased"

    def __init__(self, fn=None, **fixed_kwargs):
        self._fn = fn
        self._fixed = fixed_kwargs

    def generate(self, config: SimConfig) -> SimBatch:  # pragma: no cover
        raise NotImplementedError(
            "Waiting on the research group's simulation function. "
            "Use --simulator stub until then; see docs/risk_register.md RSK-01."
        )
