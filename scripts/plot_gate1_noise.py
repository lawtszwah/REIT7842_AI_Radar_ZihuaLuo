"""Gate 1 figure: one noise-only prdsim map, and its power distribution against theory.

Uses exactly the maps Gate 1 tests (docs/simulator_model.md section 8): the frozen
validation root seed, the `gate1_noise` stream, 64 maps of 128 x 128 at the chosen
noise power, ridge and clutter off, no targets.  Panel (b) overlays the analytical
exponential density, which on a log y-axis is a straight line of slope -1.

    python3 scripts/plot_gate1_noise.py                # sigma^2 = 1.0
    python3 scripts/plot_gate1_noise.py --noise-power 4.0 --out results/validation
"""

from __future__ import annotations

import argparse
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
SAMPLE = "#2a78d6"  # categorical slot 1
THEORY = "#eb6834"  # categorical slot 2


def gate1_maps(noise_power: float) -> tuple[np.ndarray, dict]:
    """Regenerate the Gate 1 maps for one noise power, seeded as the test seeds them."""
    with (REPO_ROOT / "configs/protocol/full.yaml").open() as f:
        spec = yaml.safe_load(f)["simulator_validation"]
    powers = [float(x) for x in spec["gate1"]["noise_powers"]]
    if noise_power not in powers:
        raise SystemExit(f"--noise-power must be one of the Gate 1 values {powers}")
    root = int(spec["root_seeds"][0])
    seq = np.random.SeedSequence(
        root,
        spawn_key=(spec["seed_streams"].index("gate1_noise"), powers.index(noise_power)),
    )
    seed = int(seq.generate_state(1, dtype=np.uint64)[0])
    size = int(spec["map_size"])
    cfg = SimConfig(
        n_maps=int(spec["gate1"]["n_maps"]),
        n_range=size,
        n_doppler=size,
        n_targets=(0, 0),
        noise_power=(noise_power, noise_power),
        ridge_enabled=False,
        clutter_enabled=False,
        seed=seed,
    )
    batch = get_simulator("prdsim").generate(cfg)
    return batch.maps, batch.meta


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot(maps: np.ndarray, meta: dict, sigma2: float, out: Path) -> list[Path]:
    x = maps.reshape(-1).astype(np.float64)
    m = x.size
    mean, cv = x.mean(), x.std(ddof=1) / x.mean()

    fig, (ax_map, ax_pdf) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1, 1.35]}
    )

    # (a) one map, in dB relative to sigma^2, zero Doppler centred.
    n_r, n_d = maps.shape[1:]
    img = ax_map.imshow(
        10 * np.log10(maps[0] / sigma2),
        origin="lower",
        aspect="equal",
        cmap="Greys_r",
        vmin=-20,
        vmax=10,
        extent=(-n_d // 2 - 0.5, n_d - n_d // 2 - 0.5, -0.5, n_r - 0.5),
    )
    ax_map.set_xlabel("Doppler bin (0 = zero Doppler)")
    ax_map.set_ylabel("Range bin")
    ax_map.set_title("(a) One noise-only map", loc="left", color=INK, fontsize=11)
    style(ax_map)
    cbar = fig.colorbar(img, ax=ax_map, fraction=0.046, pad=0.03)
    cbar.set_label("Cell power / σ²  (dB)", color=INK)
    cbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED)
    cbar.outline.set_visible(False)

    # (b) normalised power density, all cells of all maps, against Exp(mean 1).
    u = x / sigma2
    edges = np.linspace(0.0, 16.0, 65)
    density, _ = np.histogram(u, bins=edges, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    keep = density > 0
    ax_pdf.step(
        centres[keep], density[keep], where="mid", color=SAMPLE, lw=2,
        label=f"prdsim, {m:,} cells",
    )
    grid = np.linspace(0.0, 16.0, 400)
    ax_pdf.plot(grid, np.exp(-grid), color=THEORY, lw=2, ls="--",
                label="Theory: exponential, mean σ²")
    ax_pdf.set_yscale("log")
    ax_pdf.set_xlim(0, 16)
    ax_pdf.set_ylim(1e-7, 2)
    ax_pdf.set_xlabel("Cell power / σ²")
    ax_pdf.set_ylabel("Probability density (log scale)")
    ax_pdf.set_title("(b) Power distribution vs theory", loc="left", color=INK, fontsize=11)
    ax_pdf.grid(True, axis="y", color=GRID, lw=0.8)
    ax_pdf.set_axisbelow(True)
    style(ax_pdf)
    leg = ax_pdf.legend(frameon=False, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    ax_pdf.text(
        0.02, 0.04,
        f"σ² = {sigma2:g}   sample mean / σ² = {mean / sigma2:.4f}   CV = {cv:.4f}\n"
        f"{meta['simulator']} {meta['simulator_version']}, seed {meta['seed']}",
        transform=ax_pdf.transAxes, color=INK_MUTED, fontsize=8.5,
    )

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    stem = f"gate1_noise_sigma2_{sigma2:g}"
    paths = [out / f"{stem}.png", out / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=200, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--noise-power", type=float, default=1.0)
    ap.add_argument("--out", default=str(REPO_ROOT / "results/validation"))
    a = ap.parse_args()
    maps, meta = gate1_maps(a.noise_power)
    for p in plot(maps, meta, a.noise_power, Path(a.out)):
        print(p)
