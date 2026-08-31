"""Draw the three figures the thesis will report -- as specifications, before any data exists.

Fixing the axes, the operating point and the stratification *before* running anything is a
control, not a formality: it is what stops the figures being chosen after the fact to suit
whatever the numbers turned out to be. These panels carry no data and are not predictions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MUTED, INK, ACCENT = "#64748b", "#0f172a", "#1d4ed8"


def _frame(ax, title):
    ax.set_title(title, fontsize=10, color=INK)
    ax.grid(alpha=0.3)
    for s in ax.spines.values():
        s.set_color("#cbd5e1")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_fontsize(8.5)
    ax.yaxis.label.set_fontsize(8.5)


def _note(ax, lines):
    ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center",
            fontsize=17, color="#e2e8f0", zorder=0, weight="bold")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, ha="left", va="top",
            fontsize=7.4, color=MUTED, linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f8fafc", ec="#cbd5e1", lw=0.8))


def build(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # 1 -- ROC
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e-1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Probability of false alarm (per cell)")
    ax.set_ylabel("Probability of detection (per target)")
    ax.axvline(1e-3, color="#94a3b8", ls=":", lw=1.1)
    ax.text(1.15e-3, 0.03, "reference operating point", fontsize=7, color=MUTED, rotation=90)
    _frame(ax, "Figure 1 — ROC under the shared protocol   [RQ1]")
    _note(ax, ["One curve per detector arm, six arms.",
               "Solid = CFAR, dashed = feature-based, dot-dash = deep.",
               "Shaded band = ±1 sd over 5 seeds.",
               "Headline number is read at Pfa = 10⁻³."])
    fig.tight_layout()
    p = out_dir / "spec_roc.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    written.append(p)

    # 2 -- Pd by SNR
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.set_xlim(-4, 16)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Target SNR (dB)")
    ax.set_ylabel("Pd at Pfa = 10⁻³")
    ax.set_xticks([-4, 0, 4, 8, 12, 16])
    _frame(ax, "Figure 2 — where detections are won or lost   [RQ2]")
    _note(ax, ["One line per arm, binned in 4 dB steps.",
               "Error bars = sd over 5 seeds.",
               "The SNR at which the curves separate is the answer:",
               "a gain concentrated at low SNR means something",
               "different from a gain spread evenly.",
               "Companion panels: by range, Doppler, clutter density."])
    fig.tight_layout()
    p = out_dir / "spec_pd_by_snr.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    written.append(p)

    # 3 -- Pd vs cost
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Inference latency (ms per map, CPU)")
    ax.set_ylabel("Pd at Pfa = 10⁻³")
    _frame(ax, "Figure 3 — performance against inference cost   [RQ3]")
    _note(ax, ["One point per arm, annotated with parameter count.",
               "Error bars = sd over 5 seeds.",
               "The Pareto frontier is the deliverable: which arm",
               "is best once a latency budget is imposed.",
               "Same machine, same batch size, median of 10 runs."])
    fig.tight_layout()
    p = out_dir / "spec_pd_vs_cost.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    written.append(p)
    return written


if __name__ == "__main__":
    for f in build(Path("docs/design/figure_specs")):
        print(f)
