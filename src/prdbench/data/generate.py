"""Turn a config into a dataset: maps, per-cell labels, exact ground truth.

Datasets are never committed.  They are regenerated from (config, seed), which
is what makes the benchmark reproducible without storing gigabytes -- see
docs/reproducibility.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..simulation import SimConfig, get_simulator
from ..simulation.interface import Target


@dataclass
class Dataset:
    maps: np.ndarray            # (N, R, D) float32 linear power
    labels: np.ndarray          # (N, R, D) uint8 per-cell target mask
    targets: list[list[Target]]
    meta: dict


def _label_maps(shape, targets: list[list[Target]], radius: float = 1.0) -> np.ndarray:
    n, nr, nd = shape
    labels = np.zeros((n, nr, nd), dtype=np.uint8)
    rr, dd = np.meshgrid(np.arange(nr), np.arange(nd), indexing="ij")
    for i, tl in enumerate(targets):
        for t in tl:
            m = ((rr - t.range_bin) ** 2 + (dd - t.doppler_bin) ** 2) <= radius ** 2
            labels[i][m] = 1
    return labels


def build_dataset(simulator: str, cfg: dict, seed: int | None = None,
                  label_radius: float = 1.0) -> Dataset:
    cfg = dict(cfg)
    if seed is not None:
        cfg["seed"] = int(seed)
    sim_cfg = SimConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items()})
    batch = get_simulator(simulator).generate(sim_cfg)
    labels = _label_maps(batch.maps.shape, batch.targets, radius=label_radius)
    return Dataset(maps=batch.maps, labels=labels, targets=batch.targets,
                   meta=batch.meta | {"label_radius": label_radius})
