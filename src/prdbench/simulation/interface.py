"""Contract between the benchmark and whatever produces range-Doppler maps.

The parameterised simulation function is supplied by the research group and is
not yet in hand.  Everything downstream of this module is written against the
`RDMapSimulator` protocol below, so the group's function can be dropped in via
`adapter.py` without touching the detectors or the evaluation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Target:
    """Ground truth for one target in one range-Doppler map."""

    range_bin: float
    doppler_bin: float
    snr_db: float


@dataclass
class SimConfig:
    """Parameters swept by the benchmark.

    Names mirror the four quantities named in the topic definition: target
    range, Doppler shift, signal-to-noise ratio and noise level.  Ranges are
    inclusive `[lo, hi]` and sampled uniformly unless a scalar is given.
    """

    n_maps: int = 256
    n_range: int = 64
    n_doppler: int = 64
    n_targets: tuple[int, int] = (1, 1)
    # `None` means "anywhere except a 10% margin at each edge", so a config is
    # valid at any map size; give an explicit pair to pin a range/Doppler regime.
    range_bin: tuple[float, float] | None = None
    doppler_bin: tuple[float, float] | None = None
    snr_db: tuple[float, float] = (-2.0, 14.0)
    noise_power: tuple[float, float] = (1.0, 1.0)
    clutter_ridge_db: tuple[float, float] = (18.0, 30.0)
    clutter_patches: tuple[int, int] = (0, 3)
    seed: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class SimBatch:
    """A materialised batch of maps plus exact ground truth.

    maps    : (N, R, D) float32, linear power, already normalised by nothing.
    targets : list of length N, each a list of `Target`.
    meta    : provenance -- simulator name/version, config, seed.
    """

    maps: np.ndarray
    targets: list[list[Target]]
    meta: dict

    def __post_init__(self) -> None:
        if self.maps.ndim != 3:
            raise ValueError(f"maps must be (N, R, D), got {self.maps.shape}")
        if len(self.targets) != self.maps.shape[0]:
            raise ValueError("one target list per map is required")


@runtime_checkable
class RDMapSimulator(Protocol):
    """Anything that can turn a `SimConfig` into a `SimBatch`."""

    name: str
    version: str

    def generate(self, config: SimConfig) -> SimBatch: ...
