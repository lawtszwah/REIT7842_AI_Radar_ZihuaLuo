"""Characterise the group's generator on the adapter's grid, before the data grid is set.

No detector runs here. Two measurements, both on the default 10 Hz x 150 m grid:

1. Target-free background: per-cell power statistics outside the zero-Doppler guard, the
   reach of the uncancelled zero-Doppler ridge, and correlation between neighbouring cells.
2. The SNR mapping: how the generator's own `SNR_dB` translates into the effective SNR at
   the recorded target cell, mean(target cell) / mean(background) - 1, which already
   includes straddle loss on this grid.

It then evaluates the ideal single-cell detection probability for a non-fluctuating target
in exponential noise, Pd = Q1(sqrt(2 snr), sqrt(2 ln(1/Pfa))), which tells which effective
SNRs separate detectors at all: below the transition every detector misses, above it every
detector hits.

    python3 scripts/characterise_groupgen.py      # about 6 minutes of MATLAB the first time
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import ncx2

from prdbench.simulation import SimConfig, get_simulator
from prdbench.simulation.groupgen import DEFAULTS, _grids

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "validation" / "groupgen_characterisation.json"
FMAP, GRIDX = _grids(DEFAULTS)
ZERO = len(FMAP) // 2
GUARD = 2  # protocol.zero_doppler_guard
SNR_LEVELS = [-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0]
SCENES_PER_LEVEL = 8
FRAMES = DEFAULTS["range_size"]


def config(n_scenes: int, seed: int, snr, n_targets=(1, 1)) -> SimConfig:
    return SimConfig(n_maps=n_scenes * FRAMES, n_range=len(GRIDX), n_doppler=len(FMAP),
                     n_targets=n_targets, snr_db=snr, seed=seed)


def background(sim) -> dict:
    batch = sim.generate(config(20, seed=9001, snr=(-25.0, 5.0), n_targets=(0, 0)))
    p = batch.maps.astype(np.float64)                    # maps x range x Doppler
    eligible = np.abs(np.arange(len(FMAP)) - ZERO) > GUARD
    bg = p[:, :, eligible]
    floor = float(np.median(bg) / np.log(2))             # exponential mean from the median
    # ridge reach: mean power per Doppler offset at short and long range, relative to floor
    by_offset = {int(k): float(p[:, :, ZERO + k].mean() / floor) for k in range(0, 8)}
    by_range = [float(p[:, r, ZERO].mean() / floor) for r in range(len(GRIDX))]
    reach = next((r for r in range(len(GRIDX)) if by_range[r] < 2.0), len(GRIDX))
    far = p[:, 40:, :][:, :, eligible]                   # well past the clutter extent
    x = far.ravel()

    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    return {
        "maps": int(p.shape[0]),
        "eligible_cells_per_map": int(bg.shape[1] * bg.shape[2]),
        "noise_floor_linear": floor,
        "far_background_cv": float(x.std() / x.mean()),
        "far_background_mean_over_floor": float(x.mean() / floor),
        "zero_doppler_power_over_floor_by_doppler_offset": by_offset,
        "ridge_reach_range_bins_at_3dB": int(reach),
        "corr_doppler_lag1": corr(far[:, :, :-1], far[:, :, 1:]),
        "corr_range_lag1": corr(far[:, :-1, :], far[:, 1:, :]),
    }


def snr_mapping(sim, floor: float) -> list[dict]:
    rows = []
    for k, snr in enumerate(SNR_LEVELS):
        batch = sim.generate(config(SCENES_PER_LEVEL, seed=9100 + k, snr=(snr, snr)))
        cell = []
        for power, (t,) in zip(batch.maps, batch.targets, strict=True):
            cell.append(power[int(round(t.range_bin)), int(round(t.doppler_bin))])
        eff = np.mean(cell) / floor - 1.0
        rows.append({"snr_db_generator": snr, "frames": len(cell),
                     "effective_snr_db": float(10 * np.log10(eff)) if eff > 0 else None})
    return rows


def ideal_pd(snr_db: np.ndarray, pfa: float) -> np.ndarray:
    snr = 10 ** (snr_db / 10)
    t = np.log(1 / pfa)
    return ncx2.sf(2 * t, df=2, nc=2 * snr)


def main() -> dict:
    sim = get_simulator("groupgen")
    bg = background(sim)
    mapping = snr_mapping(sim, bg["noise_floor_linear"])
    grid = np.arange(-5.0, 25.01, 0.25)
    theory = {}
    for pfa in (1e-5, 1e-4, 1e-3, 1e-2):
        pd = ideal_pd(grid, pfa)
        theory[f"{pfa:g}"] = {
            f"pd_{q}": float(grid[np.searchsorted(pd, q)]) for q in (0.05, 0.5, 0.95, 0.99)
        }
    report = {"simulator_version": sim.version, "grid": {"fmap": DEFAULTS["fmap"],
              "gridx": DEFAULTS["gridx"]}, "background": bg, "snr_mapping": mapping,
              "ideal_detector_effective_snr_db": theory}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
