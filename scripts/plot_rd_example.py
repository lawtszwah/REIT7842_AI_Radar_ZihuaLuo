"""Example range-Doppler maps from prdsim: the signal model built up one component at a time.

Every component draws from its own seed stream, so switching components on one by one
with the same seed adds exactly one term of z = n + g + sum c_k + sum t_q per panel
while every earlier draw stays where it was.  That makes the build-up a faithful
decomposition rather than four unrelated maps.

    rd_example_buildup.{png,pdf}   noise -> + ridge -> + clutter -> + targets
    rd_example_full.{png,pdf}      the complete map, with ground truth annotated

The seed is a fixed illustration seed, outside the validation seed tree, chosen once.
Parameters are the training regime of docs/simulator_model.md section 7, except that
three targets at 8-16 dB are requested so the panel has something to point at.

    python3 scripts/plot_rd_example.py
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

from prdbench.simulation import SimConfig, get_simulator  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
ILLUSTRATION_SEED = 20261012

INK = "#0b0b0b"
INK_MUTED = "#52514e"
TARGET = "#eb6834"   # categorical slot 2
CLUTTER = "#2a78d6"  # categorical slot 1
VMIN, VMAX = 0.0, 25.0


def full_config(size: int = 128) -> SimConfig:
    return SimConfig(
        n_maps=1,
        n_range=size,
        n_doppler=size,
        n_targets=(3, 3),
        snr_db=(8.0, 16.0),
        noise_power=(1.0, 1.0),
        clutter_patches=(4, 4),
        ridge_enabled=True,
        clutter_enabled=True,
        seed=ILLUSTRATION_SEED,
    )


def show(ax, power: np.ndarray, sigma2: float):
    n_r, n_d = power.shape
    half = n_d // 2
    img = ax.imshow(
        10 * np.log10(power / sigma2), origin="lower", aspect="equal", cmap="Greys_r",
        vmin=VMIN, vmax=VMAX, extent=(-half - 0.5, n_d - half - 0.5, -0.5, n_r - 0.5),
    )
    ax.set_xlim(-half - 0.5, n_d - half - 0.5)
    ax.set_ylim(-0.5, n_r - 0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED, labelsize=8)
    return img


def annotate(ax, batch, n_d: int, labels: bool) -> None:
    half = n_d // 2
    for patch in batch.meta["clutter"][0]:
        ax.add_patch(Ellipse(
            (patch["centre_doppler_bin"] - half, patch["centre_range_bin"]),
            width=4 * patch["width_doppler_bins"], height=4 * patch["width_range_bins"],
            fill=False, ec=CLUTTER, lw=1.3, ls="--",
        ))
    for t in batch.targets[0]:
        x, y = t.doppler_bin - half, t.range_bin
        ax.add_patch(plt.Circle((x, y), 5, fill=False, ec=TARGET, lw=1.6))
        if labels:
            # Labels for targets in the right third go to the left so they stay inside.
            right = x > n_d / 6
            ax.annotate(
                f"target {t.snr_db:.1f} dB", (x, y), xytext=(-9 if right else 9, 9),
                textcoords="offset points", ha="right" if right else "left",
                color=INK, fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85),
            )


def colorbar(fig, img, ax) -> None:
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Cell power / σ²  (dB)", color=INK)
    cbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED)
    cbar.outline.set_visible(False)


def save(fig, out: Path, stem: str) -> list[Path]:
    paths = [out / f"{stem}.png", out / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=200, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


def main(out: Path) -> list[Path]:
    sim = get_simulator("prdsim")
    full = full_config()
    steps = [
        ("Receiver noise  n", replace(full, n_targets=(0, 0), ridge_enabled=False,
                                      clutter_enabled=False)),
        ("+ zero-Doppler ridge  g", replace(full, n_targets=(0, 0), clutter_enabled=False)),
        ("+ clutter patches  c", replace(full, n_targets=(0, 0))),
        ("+ targets  t  (complete map)", full),
    ]
    batches = [sim.generate(cfg) for _, cfg in steps]
    sigma2 = 1.0
    n_d = full.n_doppler
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    # Build-up: four panels, shared colour scale.
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for i, (ax, (title, _), batch) in enumerate(zip(axes, steps, batches, strict=True)):
        img = show(ax, batch.maps[0], sigma2)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("Doppler bin", color=INK, fontsize=9)
        if i >= 2:
            annotate(ax, batch, n_d, labels=False)
    axes[0].set_ylabel("Range bin", color=INK)
    colorbar(fig, img, axes)
    fig.text(0.01, 0.01,
             f"Same seed in every panel; each component has its own random stream, so each "
             f"panel adds one term.  {batches[-1].meta['simulator']} "
             f"{batches[-1].meta['simulator_version']}, seed {ILLUSTRATION_SEED}.",
             color=INK_MUTED, fontsize=8.5)
    paths += save(fig, out, "rd_example_buildup")

    # Complete map, annotated.
    batch = batches[-1]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    img = show(ax, batch.maps[0], sigma2)
    annotate(ax, batch, n_d, labels=True)
    ax.annotate("residual zero-Doppler ridge", (1.5, 6), xytext=(18, 18),
                textcoords="offset points", color=INK, fontsize=8.5,
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=1),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    ax.plot([], [], ls="none", marker="o", mfc="none", mec=TARGET, ms=10, mew=1.6,
            label="Target (recorded truth)")
    ax.plot([], [], ls="--", color=CLUTTER, lw=1.3, label="Clutter patch (±2 widths)")
    leg = ax.legend(frameon=True, loc="upper right", fontsize=8.5, facecolor="white",
                    edgecolor="none", framealpha=0.9)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.set_xlabel("Doppler bin (0 = zero Doppler)", color=INK)
    ax.set_ylabel("Range bin", color=INK)
    ax.set_title("One simulated range-Doppler map", loc="left", color=INK, fontsize=11)
    colorbar(fig, img, ax)
    fig.tight_layout()
    paths += save(fig, out, "rd_example_full")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "results/figures"))
    for p in main(Path(ap.parse_args().out)):
        print(p)
