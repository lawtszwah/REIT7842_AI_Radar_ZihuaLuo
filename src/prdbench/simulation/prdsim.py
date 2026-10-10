"""Noise, point targets and residual ridge: docs/simulator_model.md, sections 3, 4 and 5."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .interface import SimBatch, SimConfig, Target

# Stable component IDs: keep these fixed as additional components are implemented.
NOISE_PARAMETERS_ID = 0
NOISE_FIELD_ID = 1
TARGETS_ID = 2
RIDGE_ID = 3
CLUTTER_ID = 4

# Placement geometry matches evaluation/metrics.py and simulator_model.md section 2.
ZERO_DOPPLER_TARGET_GUARD = 3  # +/-2 guard columns plus +/-1 matching tolerance.
TARGET_EXCLUSION_SEPARATION = 6  # Twice the +/-3 truth-exclusion square.
MAX_PLACEMENT_ATTEMPTS = 1000


def _bounds(
    value: float | tuple[float, float] | list[float], name: str,
) -> tuple[float, float]:
    """Normalise a scalar or pair without drawing or perturbing pinned values."""
    try:
        lo, hi = (value, value) if np.isscalar(value) else value
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a scalar or a pair of finite bounds") from exc
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise ValueError(f"{name} must have finite, ordered bounds")
    return lo, hi


def _draw(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return lo if lo == hi else float(rng.uniform(lo, hi))


def _position(
    rng: np.random.Generator,
    range_bounds: tuple[float, float],
    doppler_bounds: tuple[float, float],
    zero_doppler: int,
    placed: list[Target],
    pinned: bool,
) -> tuple[float, float]:
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        r = _draw(rng, range_bounds)
        d = _draw(rng, doppler_bounds)
        if pinned:
            return r, d
        if abs(d - zero_doppler) <= ZERO_DOPPLER_TARGET_GUARD:
            continue
        if any(
            abs(r - target.range_bin) <= TARGET_EXCLUSION_SEPARATION
            and abs(d - target.doppler_bin) <= TARGET_EXCLUSION_SEPARATION
            for target in placed
        ):
            continue
        return r, d
    raise ValueError(
        f"could not place target {len(placed) + 1} after {MAX_PLACEMENT_ATTEMPTS} "
        "attempts: zero-Doppler guard or target exclusion overlap prevents placement"
    )


class PrdSim:
    """Complex receiver noise with coherently summed targets and residual ridge."""

    name = "prdsim"
    version = "0.3.0"

    def generate(self, config: SimConfig) -> SimBatch:
        if config.clutter_enabled:
            raise NotImplementedError("prdsim 0.3.0: clutter not implemented yet")
        for name in ("ridge_width_bins", "ridge_range_decay"):
            value = getattr(config, name)
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and greater than zero")
        target_lo, target_hi = config.n_targets
        if not 0 <= target_lo <= target_hi:
            raise ValueError("n_targets must have non-negative, ordered bounds")

        range_bounds = _bounds(
            config.range_bin if config.range_bin is not None
            else (0.1 * config.n_range, 0.9 * config.n_range), "range_bin",
        )
        doppler_bounds = _bounds(
            config.doppler_bin if config.doppler_bin is not None
            else (0.1 * config.n_doppler, 0.9 * config.n_doppler), "doppler_bin",
        )
        snr_bounds = _bounds(config.snr_db, "snr_db")
        lo, hi = _bounds(config.noise_power, "noise_power")
        ridge_bounds = _bounds(config.clutter_ridge_db, "clutter_ridge_db")
        if lo < 0:
            raise ValueError("noise_power must be non-negative")
        pinned = (
            config.range_bin is not None and config.doppler_bin is not None
            and range_bounds[0] == range_bounds[1] and doppler_bounds[0] == doppler_bounds[1]
        )
        if pinned and target_hi > 1:
            raise ValueError("a pinned position cannot request more than one target")

        maps = np.empty((config.n_maps, config.n_range, config.n_doppler), dtype=np.float32)
        range_axis = np.arange(config.n_range, dtype=np.float64)
        doppler_axis = np.arange(config.n_doppler, dtype=np.float64)
        if config.ridge_enabled:
            range_profile = np.exp(-range_axis / (config.ridge_range_decay * config.n_range))
            doppler_profile = np.exp(
                -0.5 * ((doppler_axis - config.n_doppler // 2) / config.ridge_width_bins) ** 2
            )
            ridge_profile = range_profile[:, None] * doppler_profile[None, :]
        all_targets: list[list[Target]] = []
        derived_seeds = []
        for map_index in range(config.n_maps):
            parameter_key = (map_index, NOISE_PARAMETERS_ID)
            field_key = (map_index, NOISE_FIELD_ID)
            target_key = (map_index, TARGETS_ID)
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

            target_rng = np.random.default_rng(
                np.random.SeedSequence(config.seed, spawn_key=target_key)
            )
            # Fixed draw order: count, then position pairs (including redraws), SNR, phase.
            count = int(target_rng.integers(target_lo, target_hi + 1))
            targets: list[Target] = []
            for _ in range(count):
                r, d = _position(
                    target_rng, range_bounds, doppler_bounds, config.n_doppler // 2,
                    targets, pinned,
                )
                snr_db = _draw(target_rng, snr_bounds)
                phase = target_rng.uniform(0.0, 2.0 * np.pi)
                amplitude = np.sqrt(sigma2 * 10.0 ** (snr_db / 10.0))
                response = (
                    np.sinc(range_axis - r)[:, None] * np.sinc(doppler_axis - d)[None, :]
                )
                field += amplitude * response * np.exp(1j * phase)
                peak_response = abs(np.sinc(r - round(r)) * np.sinc(d - round(d)))
                peak_snr_db = float(snr_db + 20.0 * np.log10(peak_response))
                targets.append(Target(r, d, snr_db, peak_snr_db))

            map_seeds = {
                "noise_parameters": list(parameter_key),
                "noise_field": list(field_key),
                "targets": list(target_key),
            }
            if config.ridge_enabled:
                ridge_key = (map_index, RIDGE_ID)
                ridge_rng = np.random.default_rng(
                    np.random.SeedSequence(config.seed, spawn_key=ridge_key)
                )
                # Fixed draw order: ridge level, then one phase shared by the whole map.
                level_db = _draw(ridge_rng, ridge_bounds)
                ridge_phase = ridge_rng.uniform(0.0, 2.0 * np.pi)
                ridge_amplitude = np.sqrt(sigma2 * 10.0 ** (level_db / 10.0))
                field += ridge_amplitude * ridge_profile * np.exp(1j * ridge_phase)
                map_seeds["ridge"] = list(ridge_key)

            maps[map_index] = np.abs(field) ** 2
            all_targets.append(targets)
            derived_seeds.append(map_seeds)

        meta = {
            "simulator": self.name,
            "simulator_version": self.version,
            "config": json.loads(json.dumps(asdict(config))),
            "seed": config.seed,
            "derived_seeds": derived_seeds,
        }
        return SimBatch(maps=maps, targets=all_targets, meta=meta)
