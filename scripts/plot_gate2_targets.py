"""Gate 2 figure: a map with targets, and measured peak loss against straddle-loss theory.

(a) One 128 x 128 map with three targets, ridge and clutter off, with each target's
    recorded ground-truth position circled.  This is the visual check that
    `Target(range_bin, doppler_bin)` lands where the response is drawn.
(b) The Gate 2 measurement (docs/simulator_model.md section 8), regenerated with the
    frozen validation seeds: mean power at the target cell over 4096 maps, converted
    to the loss relative to the nominal 10 dB SNR, against 20 log10|chi(dr, dd)|.
    Shown along the two cuts that bracket the 5 x 5 offset grid: offset in range only,
    and equal offset in both axes.  Error bars are the Gate 2 acceptance band (z = 4).

    python3 scripts/plot_gate2_targets.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from prdbench.simulation import SimConfig, get_simulator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
ONE_AXIS = "#2a78d6"  # categorical slot 1
BOTH_AXES = "#eb6834"  # categorical slot 2

# Panel (a) is an illustration, not a validation case: a fixed seed, outside the
# frozen validation seed tree, chosen once and never re-picked.
ILLUSTRATION_SEED = 20261011


def load_spec() -> dict:
    with (REPO_ROOT / "configs/protocol/full.yaml").open() as f:
        return yaml.safe_load(f)["simulator_validation"]


def example_map(sim, size: int):
    cfg = SimConfig(
        n_maps=1,
        n_range=size,
        n_doppler=size,
        n_targets=(3, 3),
        snr_db=(8.0, 16.0),
        noise_power=(1.0, 1.0),
        ridge_enabled=False,
        clutter_enabled=False,
        seed=ILLUSTRATION_SEED,
    )
    batch = sim.generate(cfg)
    return batch.maps[0], batch.targets[0], batch.meta


def gate2_mean_peak(sim, spec: dict, dr: float, dd: float) -> float:
    """Mean power at the Gate 2 cell, seeded exactly as the Gate 2 test seeds it."""
    g2 = spec["gate2"]
    offsets = [float(x) for x in g2["offsets"]]
    offset_index = offsets.index(dr) * len(offsets) + offsets.index(dd)
    stream = spec["seed_streams"].index("gate2_targets")
    root = int(spec["root_seeds"][0])
    size, cell = int(spec["map_size"]), int(spec["map_size"]) // 2
    n_maps, chunk = int(g2["n_maps"]), int(g2["chunk_size"])

    total, done, chunk_index = 0.0, 0, 0
    while done < n_maps:
        k = min(chunk, n_maps - done)
        seq = np.random.SeedSequence(root, spawn_key=(stream, offset_index, chunk_index))
        cfg = SimConfig(
            n_maps=k,
            n_range=size,
            n_doppler=size,
            n_targets=(1, 1),
            range_bin=(cell + dr, cell + dr),
            doppler_bin=(cell + dd, cell + dd),
            snr_db=(float(g2["snr_db"]),) * 2,
            noise_power=(float(g2["noise_power"]),) * 2,
            ridge_enabled=False,
            clutter_enabled=False,
            seed=int(seq.generate_state(1, dtype=np.uint64)[0]),
        )
        total += float(sim.generate(cfg).maps[:, cell, cell].astype(np.float64).sum())
        done += k
        chunk_index += 1
    return total / n_maps


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def main(out: Path) -> list[Path]:
    spec = load_spec()
    sim = get_simulator("prdsim")
    size = int(spec["map_size"])
    g2 = spec["gate2"]
    z = float(spec["acceptance_z"])
    sigma2, snr_db, n_maps = float(g2["noise_power"]), float(g2["snr_db"]), int(g2["n_maps"])
    snr_lin = 10 ** (snr_db / 10)
    offsets = [float(x) for x in g2["offsets"]]

    fig, (ax_map, ax_loss) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.35]}
    )

    # (a) example map with ground truth circled.
    power, targets, meta = example_map(sim, size)
    half = size // 2
    img = ax_map.imshow(
        10 * np.log10(power),
        origin="lower",
        aspect="equal",
        cmap="Greys_r",
        # Floor at the noise mean so the noise reads dark and target peaks stand out;
        # exponential noise exceeds +10 dB in only e^-10 = 4.5e-5 of cells.
        vmin=0,
        vmax=16,
        extent=(-half - 0.5, size - half - 0.5, -0.5, size - 0.5),
    )
    for t in sorted(targets, key=lambda t: t.snr_db):
        x, y = t.doppler_bin - half, t.range_bin
        ax_map.add_patch(
            plt.Circle((x, y), 7, fill=False, ec=BOTH_AXES, lw=1.4)
        )
        ax_map.annotate(
            f"{t.snr_db:.1f} dB",
            (x, y),
            xytext=(12, 12),
            textcoords="offset points",
            color=INK,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
        )
    ax_map.set_xlabel("Doppler bin (0 = zero Doppler)")
    ax_map.set_ylabel("Range bin")
    ax_map.set_title(
        "(a) Three targets, recorded positions circled", loc="left", color=INK, fontsize=11
    )
    style(ax_map)
    cbar = fig.colorbar(img, ax=ax_map, fraction=0.046, pad=0.03)
    cbar.set_label("Cell power / σ²  (dB)", color=INK)
    cbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED)
    cbar.outline.set_visible(False)

    # (b) measured vs theoretical straddle loss along two cuts.
    fine = np.linspace(0.0, 0.5, 200)
    cuts = [
        ("Offset in range only", ONE_AXIS, "o", lambda d: (d, 0.0),
         lambda x: 20 * np.log10(np.abs(np.sinc(x)))),
        ("Equal offset in both axes", BOTH_AXES, "s", lambda d: (d, d),
         lambda x: 40 * np.log10(np.abs(np.sinc(x)))),
    ]
    for label, colour, marker, pair, theory in cuts:
        ax_loss.plot(fine, theory(fine), color=colour, lw=2, ls="--")
        measured, lo_err, hi_err = [], [], []
        for d in offsets:
            dr, dd = pair(d)
            mean_power = gate2_mean_peak(sim, spec, dr, dd)
            chi2 = (np.sinc(dr) * np.sinc(dd)) ** 2
            sd = sigma2 * math.sqrt(1 + 2 * snr_lin * chi2)
            band = z * sd / math.sqrt(n_maps)

            def to_loss(p: float) -> float:
                return 10 * math.log10((p - sigma2) / sigma2) - snr_db

            m = to_loss(mean_power)
            measured.append(m)
            lo_err.append(m - to_loss(mean_power - band))
            hi_err.append(to_loss(mean_power + band) - m)
        ax_loss.errorbar(
            offsets, measured, yerr=[lo_err, hi_err], fmt=marker, color=colour,
            ms=8, mec="white", mew=1.5, capsize=3, lw=1.2, label=label,
        )
    ax_loss.plot([], [], color=INK_MUTED, lw=2, ls="--", label="Theory: 20 log₁₀|χ(δr, δd)|")
    for value, text in ((-3.92, "−3.92 dB"), (-7.84, "−7.84 dB")):
        ax_loss.annotate(text, (0.5, value), xytext=(-48, -14), textcoords="offset points",
                         color=INK_MUTED, fontsize=8.5)
    ax_loss.set_xlim(-0.02, 0.54)
    ax_loss.set_ylim(-9, 0.8)
    ax_loss.set_xlabel("Fractional offset from the cell centre (bins)")
    ax_loss.set_ylabel("Peak SNR − nominal SNR (dB)")
    ax_loss.set_title(
        f"(b) Straddle loss, {n_maps:,} maps per point", loc="left", color=INK, fontsize=11
    )
    ax_loss.grid(True, axis="y", color=GRID, lw=0.8)
    ax_loss.set_axisbelow(True)
    style(ax_loss)
    leg = ax_loss.legend(frameon=False, loc="lower left")
    for t in leg.get_texts():
        t.set_color(INK)
    ax_loss.text(
        0.98, 0.97,
        f"nominal SNR {snr_db:g} dB, σ² = {sigma2:g}\n"
        f"{meta['simulator']} {meta['simulator_version']}, error bars = Gate 2 band (z = {z:g})",
        transform=ax_loss.transAxes, ha="right", va="top", color=INK_MUTED, fontsize=8.5,
    )

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "gate2_targets.png", out / "gate2_targets.pdf"]
    fig.savefig(paths[0], dpi=200, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "results/validation"))
    for p in main(Path(ap.parse_args().out)):
        print(p)
