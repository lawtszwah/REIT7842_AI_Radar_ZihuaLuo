"""Stand-in simulator used until the group's function is available.

It is deliberately simple and deliberately NOT a research contribution: its
only job is to let the whole pipeline -- generation, tuning, detection,
evaluation, reporting -- run end to end today, so that swapping in the real
simulator is a one-file change rather than a rewrite.  Results produced with
this simulator are marked `simulator="stub"` in every manifest and must never
be reported as project findings.
"""

from __future__ import annotations

import numpy as np

from .interface import RDMapSimulator, SimBatch, SimConfig, Target


def _draw(rng: np.random.Generator, bounds) -> float:
    lo, hi = (bounds, bounds) if np.isscalar(bounds) else bounds
    return float(lo) if lo == hi else float(rng.uniform(lo, hi))


class StubSimulator:
    """Complex-Gaussian noise + zero-Doppler direct-path ridge + clutter + targets."""

    name = "stub"
    version = "0.2.0"

    def generate(self, config: SimConfig) -> SimBatch:
        rng = np.random.default_rng(config.seed)
        n, nr, nd = config.n_maps, config.n_range, config.n_doppler
        maps = np.empty((n, nr, nd), dtype=np.float32)
        all_targets: list[list[Target]] = []

        r_ax = np.arange(nr)[:, None]
        d_ax = np.arange(nd)[None, :]
        r_bounds = config.range_bin or (0.1 * nr, 0.9 * nr)
        d_bounds = config.doppler_bin or (0.1 * nd, 0.9 * nd)

        for i in range(n):
            noise_power = _draw(rng, config.noise_power)
            re = rng.normal(0.0, np.sqrt(noise_power / 2), (nr, nd))
            im = rng.normal(0.0, np.sqrt(noise_power / 2), (nr, nd))
            field = re + 1j * im

            # Direct-path leakage: a strong ridge along the zero-Doppler column.
            ridge_db = _draw(rng, config.clutter_ridge_db)
            ridge_amp = np.sqrt(noise_power * 10 ** (ridge_db / 10))
            zero_dop = nd // 2
            spread = np.exp(-0.5 * ((d_ax - zero_dop) / 1.1) ** 2)
            decay = np.exp(-r_ax / (0.35 * nr))
            field += ridge_amp * spread * decay * np.exp(1j * rng.uniform(0, 2 * np.pi))

            # Clutter patches: correlated bright regions off the ridge.
            n_patch = int(rng.integers(config.clutter_patches[0], config.clutter_patches[1] + 1))
            for _ in range(n_patch):
                cr, cd = rng.uniform(0, nr), rng.uniform(0, nd)
                amp = np.sqrt(noise_power * 10 ** (rng.uniform(4, 11) / 10))
                sr, sd = rng.uniform(1.5, 4.0), rng.uniform(1.5, 4.0)
                env = np.exp(-0.5 * (((r_ax - cr) / sr) ** 2 + ((d_ax - cd) / sd) ** 2))
                # Mostly coherent (a bright region) with a little speckle, rather
                # than i.i.d. complex noise, whose Rayleigh tail would otherwise
                # dominate the false-alarm statistics at every threshold.
                speck = 0.35 * (rng.normal(size=(nr, nd)) + 1j * rng.normal(size=(nr, nd)))
                field += amp * env * (np.exp(1j * rng.uniform(0, 2 * np.pi)) + speck)

            # Targets: separable sinc response, SNR defined against the noise floor.
            k = int(rng.integers(config.n_targets[0], config.n_targets[1] + 1))
            targets: list[Target] = []
            for _ in range(k):
                tr = _draw(rng, r_bounds)
                td = _draw(rng, d_bounds)
                snr_db = _draw(rng, config.snr_db)
                amp = np.sqrt(noise_power * 10 ** (snr_db / 10))
                resp = np.sinc(r_ax - tr) * np.sinc(d_ax - td)
                field += amp * resp * np.exp(1j * rng.uniform(0, 2 * np.pi))
                targets.append(Target(range_bin=tr, doppler_bin=td, snr_db=snr_db))

            maps[i] = np.abs(field).astype(np.float32) ** 2
            all_targets.append(targets)

        meta = {
            "simulator": self.name,
            "simulator_version": self.version,
            "seed": config.seed,
            "config": config.__dict__ | {"extra": dict(config.extra)},
            "warning": "synthetic stand-in, not the research group's simulator",
        }
        return SimBatch(maps=maps, targets=all_targets, meta=meta)


assert isinstance(StubSimulator(), RDMapSimulator)
