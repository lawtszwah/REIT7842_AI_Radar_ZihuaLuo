"""Receiver noise and square-law detection: docs/simulator_model.md, section 3."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .interface import SimBatch, SimConfig

# Stable component IDs: keep these fixed as additional components are implemented.
NOISE_PARAMETERS_ID = 0
NOISE_FIELD_ID = 1
TARGETS_ID = 2
RIDGE_ID = 3
CLUTTER_ID = 4


class PrdSim:
    """Noise-only first step of the reportable simulator."""

    name = "prdsim"
    version = "0.1.0"

    def generate(self, config: SimConfig) -> SimBatch:
        if max(config.n_targets) > 0:
            raise NotImplementedError("prdsim 0.1.0: targets not implemented yet")
        if config.ridge_enabled:
            raise NotImplementedError("prdsim 0.1.0: ridge not implemented yet")
        if config.clutter_enabled:
            raise NotImplementedError("prdsim 0.1.0: clutter not implemented yet")
        if min(config.n_targets) < 0:
            raise ValueError("n_targets must be non-negative")

        lo, hi = (
            (config.noise_power, config.noise_power)
            if np.isscalar(config.noise_power)
            else config.noise_power
        )
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 <= lo <= hi):
            raise ValueError("noise_power must have finite, non-negative, ordered bounds")

        maps = np.empty((config.n_maps, config.n_range, config.n_doppler), dtype=np.float32)
        derived_seeds = []
        for map_index in range(config.n_maps):
            parameter_key = (map_index, NOISE_PARAMETERS_ID)
            field_key = (map_index, NOISE_FIELD_ID)
            parameter_rng = np.random.default_rng(
                np.random.SeedSequence(config.seed, spawn_key=parameter_key)
            )
            field_rng = np.random.default_rng(
                np.random.SeedSequence(config.seed, spawn_key=field_key)
            )
            sigma2 = float(lo) if lo == hi else float(parameter_rng.uniform(lo, hi))
            sd = np.sqrt(sigma2 / 2.0)
            shape = (config.n_range, config.n_doppler)
            field = field_rng.normal(0.0, sd, shape) + 1j * field_rng.normal(0.0, sd, shape)
            maps[map_index] = np.abs(field) ** 2
            derived_seeds.append(
                {
                    "noise_parameters": list(parameter_key),
                    "noise_field": list(field_key),
                }
            )

        meta = {
            "simulator": self.name,
            "simulator_version": self.version,
            "config": json.loads(json.dumps(asdict(config))),
            "seed": config.seed,
            "derived_seeds": derived_seeds,
        }
        return SimBatch(maps=maps, targets=[[] for _ in range(config.n_maps)], meta=meta)
