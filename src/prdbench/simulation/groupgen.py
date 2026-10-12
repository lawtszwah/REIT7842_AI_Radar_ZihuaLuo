"""Adapter for the research group's passive-radar generator (Dr Lu Zhang, MATLAB).

The generator was supplied on 2026-09-10 as student interface scripts around protected
P-code cores; it is not part of this repository. `matlab/prd_groupgen.m` calls its three
cores for a batch of scenes, and this module turns the result into a `SimBatch`.

One *scene* is one target trajectory: the generator simulates `range_size` consecutive
positions and the range-Doppler processing produces one frame per position. Frames of one
scene are strongly dependent, so a scene never straddles two splits: each split draws its
own scenes from its own seed.

What the adapter controls, per scene, and where each value comes from:

  target start position, speed, heading   drawn here, then passed to the trajectory core
  SNR_dB                                   `SimConfig.snr_db`, in the generator's own definition
  CSR_dB, DNR_dB, speed, heading           `SimConfig.extra["groupgen"]`, see `DEFAULTS`
  MATLAB seed                              rng(seed, 'twister') immediately before the cores

Ground truth is computed from the positions the trajectory core actually returns, with the
grid conventions measured on 2026-10-12 (see `doppler_hz` and `delay_samples`).

`SimConfig` fields that describe the in-project reference simulator (noise power, ridge,
clutter patches and their shapes) have no counterpart in this generator and are not read.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import scipy.io as sio

from .interface import SimBatch, SimConfig, Target

REPO_ROOT = Path(__file__).resolve().parents[3]
MATLAB_DIR = REPO_ROOT / "matlab"

# Defaults are the values in the group's interface scripts, except the Doppler grid, see
# `DEFAULTS["fmap"]`. Every value can be overridden in SimConfig.extra["groupgen"].
DEFAULTS: dict = {
    "c": 3e8,
    "fs": 2e6,
    "fc": 220e6,
    "tx": [-1e3, 1000.0],
    "rx": [0.0, 0.0],
    "time_duration": 1.0,
    "frame_duration": 0.1,
    "frame_step": 0.1,
    "num_points": 50,
    "update_interval": 0.1,
    "range_size": 10,
    "target_start_idx": 1,
    "clutter_bins": 50,
    "decay_factor": 0.15,
    "min_delay": 1,
    "use_random_phase": True,
    # [start, step, stop] in Hz. The interface script uses -250:1:400; a 1 Hz step
    # oversamples the 10 Hz resolution of a 0.1 s frame tenfold, so neighbouring cells are
    # strongly correlated, and it is 15x slower. The grid here is symmetric about 0 Hz so
    # that zero Doppler sits at index n_doppler // 2, the convention the metrics assume.
    "fmap": [-400.0, 10.0, 400.0],
    "gridx": [2, 100],
    # Drawn per scene, uniformly over [lo, hi].
    "csr_db": [25.0, 25.0],
    "dnr_db": [45.0, 45.0],
    "speed_kmh": [200.0, 600.0],
    "heading_deg": [0.0, 360.0],
    # Placement: every frame's target must sit this far inside the range-Doppler window,
    # and this far from zero Doppler, in cells.
    "edge_margin_cells": 3,
    "zero_doppler_margin_cells": 4,
    "position_box_m": 16000.0,
}

TRAJECTORY_STREAM, MATLAB_STREAM = 0, 1
MAX_PLACEMENT_ATTEMPTS = 10000


def _bounds(value) -> tuple[float, float]:
    lo, hi = (value, value) if np.isscalar(value) else value
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo <= hi):
        raise ValueError(f"invalid bounds {value!r}")
    return lo, hi


def _draw(rng: np.random.Generator, value) -> float:
    lo, hi = _bounds(value)
    return lo if lo == hi else float(rng.uniform(lo, hi))


def delay_samples(pos: np.ndarray, p: dict) -> np.ndarray:
    """Bistatic delay relative to the direct path, in samples: (|T-Tx| + |T-Rx| - L) fs / c.

    Measured 2026-10-12: the generator's `cut` is columns 2..100 of the full delay axis, so
    gridx value g holds delay g - 1 samples, and the peak sits at gridx = delay + 1.
    """
    tx, rx = np.asarray(p["tx"], float), np.asarray(p["rx"], float)
    baseline = np.linalg.norm(tx - rx)
    path = np.linalg.norm(pos - tx, axis=-1) + np.linalg.norm(pos - rx, axis=-1)
    return (path - baseline) * p["fs"] / p["c"]


def doppler_hz(pos: np.ndarray, velocity: np.ndarray, p: dict) -> np.ndarray:
    """Doppler of the target peak in the generator's maps: 2 v . u_rx / lambda.

    Measured 2026-10-12 on six strong-target scenes in different geometries (60 frames):
    the peak follows the *monostatic* form about the receiver to within 0.01 +/- 0.11 Hz,
    while the bistatic form v . (u_tx + u_rx) / lambda is off by up to 13 Hz. The delay, by
    contrast, is bistatic. The sign is positive for a receding target. This is the
    generator's behaviour, not a modelling choice made here; it has been raised with the
    supervisors.
    """
    rx = np.asarray(p["rx"], float)
    u_rx = (pos - rx) / np.linalg.norm(pos - rx, axis=-1, keepdims=True)
    lam = p["c"] / p["fc"]
    return 2.0 * (u_rx @ np.asarray(velocity, float)) / lam


def _grids(p: dict) -> tuple[np.ndarray, np.ndarray]:
    start, step, stop = (float(x) for x in p["fmap"])
    n = int(round((stop - start) / step)) + 1
    fmap = start + step * np.arange(n)
    gridx = np.arange(int(p["gridx"][0]), int(p["gridx"][1]) + 1)
    if not np.isclose(fmap[n // 2], 0.0):
        raise ValueError("fmap must be symmetric about 0 Hz with an odd number of bins")
    return fmap, gridx


def _frame_positions(start: np.ndarray, velocity: np.ndarray, p: dict) -> np.ndarray:
    """Positions the trajectory core produces: straight line, one per update interval."""
    t = (int(p["target_start_idx"]) - 1 + np.arange(int(p["range_size"]))) * p["update_interval"]
    return start + t[:, None] * velocity


def _place(rng: np.random.Generator, p: dict, fmap, gridx) -> dict:
    """Draw one scene whose target stays inside the window and off zero Doppler throughout."""
    margin, zmargin = p["edge_margin_cells"], p["zero_doppler_margin_cells"]
    f_step = fmap[1] - fmap[0]
    d_lo, d_hi = gridx[0] - 1 + margin, gridx[-1] - 1 - margin
    f_lim = fmap[-1] - margin * f_step
    box = p["position_box_m"]
    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        start = rng.uniform(-box, box, size=2)
        speed = _draw(rng, p["speed_kmh"])
        heading = _draw(rng, p["heading_deg"])
        rad = np.radians(heading)
        velocity = speed / 3.6 * np.array([np.cos(rad), np.sin(rad)])
        pos = _frame_positions(start, velocity, p)
        delay, fd = delay_samples(pos, p), doppler_hz(pos, velocity, p)
        if (delay.min() >= d_lo and delay.max() <= d_hi
                and np.abs(fd).max() <= f_lim and np.abs(fd).min() >= zmargin * f_step):
            return {"initial_position": start.tolist(), "speed_kmh": speed,
                    "heading_deg": heading}
    raise ValueError("could not place a target inside the range-Doppler window")


def generator_fingerprint(generator_dir: Path) -> str:
    """SHA-256 over the generator's files, so every dataset names the exact code it came from."""
    h = hashlib.sha256()
    for f in sorted(generator_dir.glob("*")):
        if f.suffix in {".m", ".p"}:
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


class GroupGenerator:
    """The research group's generator behind the `RDMapSimulator` protocol."""

    name = "groupgen"
    adapter_version = "0.1.0"

    def __init__(self, generator_dir: str | Path | None = None, matlab: str | None = None):
        self.generator_dir = Path(
            generator_dir or os.environ.get("PRDBENCH_GENERATOR_DIR")
            or REPO_ROOT.parent / "generator"
        )
        self.matlab = matlab or os.environ.get(
            "PRDBENCH_MATLAB", "/Applications/MATLAB_R2024a.app/bin/matlab"
        )

    @property
    def version(self) -> str:
        if not self.generator_dir.exists():
            return f"{self.adapter_version}+gen.absent"
        fp = generator_fingerprint(self.generator_dir)[:12]
        return f"{self.adapter_version}+gen.{fp}"

    def available(self) -> bool:
        return self.generator_dir.is_dir() and Path(self.matlab).exists()

    def _params(self, config: SimConfig) -> dict:
        p = dict(DEFAULTS)
        p.update(config.extra.get("groupgen", {}))
        return p

    def plan(self, config: SimConfig) -> list[dict]:
        """The scenes a config asks for, fully specified, without running MATLAB."""
        p = self._params(config)
        fmap, gridx = _grids(p)
        if (config.n_range, config.n_doppler) != (len(gridx), len(fmap)):
            raise ValueError(
                f"n_range x n_doppler must match the generator grid: "
                f"{len(gridx)} x {len(fmap)} for gridx {p['gridx']} and fmap {p['fmap']}"
            )
        if tuple(config.n_targets) not in {(0, 0), (1, 1)}:
            raise ValueError(
                "the group's generator simulates exactly one target per scene; "
                "n_targets must be (1, 1), or (0, 0) for target-free maps"
            )
        target_free = tuple(config.n_targets) == (0, 0)
        frames = int(p["range_size"])
        if config.n_maps % frames:
            raise ValueError(f"n_maps must be a multiple of {frames} frames per scene")
        scenes = []
        for i in range(config.n_maps // frames):
            rng = np.random.default_rng(
                np.random.SeedSequence(config.seed, spawn_key=(i, TRAJECTORY_STREAM)))
            scene = _place(rng, p, fmap, gridx)
            scene.update({
                "id": i,
                # Target-free maps keep a trajectory (the cores need one) but no target power.
                "target_free": target_free,
                "snr_db": None if target_free else _draw(rng, config.snr_db),
                # The generator sets clutter relative to the target (CNR = SNR + CSR), so a
                # target-free map draws the SNR a target would have had, from the same range,
                # to keep the clutter level distributed as in maps with targets.
                "reference_snr_db": _draw(rng, config.snr_db) if target_free else None,
                "csr_db": _draw(rng, p["csr_db"]),
                "dnr_db": _draw(rng, p["dnr_db"]),
                "seed": int(np.random.SeedSequence(config.seed, spawn_key=(i, MATLAB_STREAM))
                            .generate_state(1, dtype=np.uint32)[0]),
            })
            for key in ("c", "fs", "fc", "tx", "rx", "time_duration", "frame_duration",
                        "frame_step", "num_points", "update_interval", "range_size",
                        "target_start_idx", "clutter_bins", "decay_factor", "min_delay",
                        "use_random_phase"):
                scene[key] = p[key]
            scene["fmap_start"], scene["fmap_step"], scene["fmap_stop"] = (
                float(x) for x in p["fmap"])
            scene["gridx_start"], scene["gridx_stop"] = (int(x) for x in p["gridx"])
            scenes.append(scene)
        return scenes

    def _run_matlab(self, scenes: list[dict], out_dir: Path) -> None:
        job = {"generator_dir": str(self.generator_dir), "out_dir": str(out_dir),
               "scenes": scenes}
        job_path = out_dir / "job.json"
        job_path.write_text(json.dumps(job))
        cmd = (f"addpath('{MATLAB_DIR}'); prd_groupgen('{job_path}');")
        result = subprocess.run([self.matlab, "-batch", cmd], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"MATLAB failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")

    def cache_dir(self) -> Path:
        return Path(os.environ.get("PRDBENCH_CACHE_DIR", REPO_ROOT / "data" / "groupgen_cache"))

    def scene_key(self, scene: dict) -> str:
        """Cache key: the complete scene specification, the generator files, the adapter."""
        blob = json.dumps({"scene": {k: v for k, v in scene.items() if k != "id"},
                           "generator": generator_fingerprint(self.generator_dir),
                           "adapter": self.adapter_version}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _ensure_cached(self, scenes: list[dict]) -> list[Path]:
        """Generate any scene not yet cached, in one MATLAB batch; return each scene's file.

        A scene is a pure function of its specification and seed, so a cached scene is the
        scene the generator would produce again; the key includes the generator's files.
        """
        cache = self.cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        paths = [cache / f"{self.scene_key(s)}.npz" for s in scenes]
        missing = [(s, path) for s, path in zip(scenes, paths, strict=True) if not path.exists()]
        if missing:
            if not self.available():
                raise RuntimeError(
                    f"group generator unavailable: need {self.generator_dir} and MATLAB at "
                    f"{self.matlab} (set PRDBENCH_GENERATOR_DIR / PRDBENCH_MATLAB)"
                )
            tmp = Path(tempfile.mkdtemp(prefix="groupgen_"))
            try:
                self._run_matlab([s for s, _ in missing], tmp)
                for s, path in missing:
                    d = sio.loadmat(tmp / f"scene_{s['id']:06d}.mat")
                    partial = path.with_suffix(".partial.npz")
                    np.savez(partial, power=d["power"].astype(np.float32),
                             target_position=d["target_position"])
                    partial.replace(path)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        return paths

    def generate(self, config: SimConfig) -> SimBatch:
        p = self._params(config)
        fmap, gridx = _grids(p)
        scenes = self.plan(config)
        maps, targets = [], []
        for scene, path in zip(scenes, self._ensure_cached(scenes), strict=True):
            with np.load(path) as d:
                power = d["power"]                           # frames x Doppler x range
                pos = d["target_position"]
            maps.append(np.transpose(power, (0, 2, 1)).astype(np.float32))
            if scene["target_free"]:
                targets.extend([] for _ in range(power.shape[0]))
                continue
            rad = np.radians(scene["heading_deg"])
            velocity = scene["speed_kmh"] / 3.6 * np.array([np.cos(rad), np.sin(rad)])
            rng_bins = delay_samples(pos, p) - (gridx[0] - 1)
            dop_bins = (doppler_hz(pos, velocity, p) - fmap[0]) / (fmap[1] - fmap[0])
            targets.extend(
                [Target(range_bin=float(r), doppler_bin=float(f), snr_db=scene["snr_db"])]
                for r, f in zip(rng_bins, dop_bins, strict=True)
            )
        meta = {
            "simulator": self.name,
            "simulator_version": self.version,
            "config": json.loads(json.dumps(asdict(config))),
            "seed": config.seed,
            "generator_params": json.loads(json.dumps(p)),
            "scenes": scenes,
            "fmap_hz": fmap.tolist(),
            "gridx": gridx.tolist(),
        }
        return SimBatch(maps=np.concatenate(maps), targets=targets, meta=meta)
