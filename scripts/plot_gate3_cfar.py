"""Gate 3 figure: CA-CFAR measured false-alarm probability against its design value.

Reads the validation record that the Gate 3 test writes, so the figure shows the
recorded run rather than a re-computation:

    python3 -m pytest -q tests/test_prdsim_validation.py -k gate3 \
        --prdsim-validation-report results/validation/gate3_report.json
    python3 scripts/plot_gate3_cfar.py

(a) measured against design Pfa over three decades, with y = x;
(b) measured / design with the 99% map-clustered bootstrap interval, which is the
    Gate 3 acceptance test: the design value must lie inside the interval.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
MEASURED = "#2a78d6"  # categorical slot 1


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED, which="both")
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def plot(report: dict, out: Path) -> list[Path]:
    run = report["runs"][-1]
    pts = run["points"]
    design = np.array([p["design_pfa"] for p in pts])
    measured = np.array([p["measured_pfa"] for p in pts])
    low = np.array([p["interval_low"] for p in pts])
    high = np.array([p["interval_high"] for p in pts])
    alarms = [p["alarm_count"] for p in pts]
    eligible = pts[0]["eligible_cells"]

    fig, (ax_ll, ax_ratio) = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) log-log agreement.
    span = np.array([design.min() / 3, design.max() * 3])
    ax_ll.plot(span, span, color=INK_MUTED, lw=1.2, ls="--", label="Measured = design")
    ax_ll.plot(design, measured, "o", color=MEASURED, ms=9, mec="white", mew=1.5,
               label="CA-CFAR on prdsim noise")
    ax_ll.set_xscale("log")
    ax_ll.set_yscale("log")
    ax_ll.set_xlim(*span)
    ax_ll.set_ylim(*span)
    ax_ll.set_aspect("equal")
    ax_ll.set_xlabel("Design Pfa")
    ax_ll.set_ylabel("Measured Pfa")
    ax_ll.set_title("(a) Measured vs design, three decades", loc="left", color=INK,
                    fontsize=11)
    ax_ll.grid(True, which="major", color=GRID, lw=0.8)
    ax_ll.set_axisbelow(True)
    style(ax_ll)
    leg = ax_ll.legend(frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)

    # (b) ratio with the acceptance interval.
    ax_ratio.axhline(1.0, color=INK_MUTED, lw=1.2, ls="--")
    ax_ratio.errorbar(
        design, measured / design,
        yerr=[(measured - low) / design, (high - measured) / design],
        fmt="o", color=MEASURED, ms=9, mec="white", mew=1.5, capsize=4, lw=1.4,
    )
    for x, hi, n in zip(design, high / design, alarms, strict=True):
        ax_ratio.annotate(f"{n:,} alarms", (x, hi), xytext=(0, 6),
                          textcoords="offset points", ha="center", color=INK_MUTED,
                          fontsize=8.5)
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlim(design.min() / 3, design.max() * 3)
    ax_ratio.set_ylim(0.6, 1.45)
    ax_ratio.set_xlabel("Design Pfa")
    ax_ratio.set_ylabel("Measured / design Pfa")
    ax_ratio.set_title("(b) Ratio with 99% bootstrap interval", loc="left", color=INK,
                       fontsize=11)
    ax_ratio.grid(True, axis="y", color=GRID, lw=0.8)
    ax_ratio.set_axisbelow(True)
    style(ax_ratio)
    verdict = "all four design points inside" if report["passed"] else "FAILED"
    ax_ratio.text(
        0.02, 0.04,
        f"{eligible:,} eligible cells, {run['n_maps']} noise-only maps, "
        f"window {run['train']} train / {run['guard']} guard\n"
        f"{run['simulator']} {run['simulator_version']}, root seed {run['root_seed']}: "
        f"{verdict}",
        transform=ax_ratio.transAxes, color=INK_MUTED, fontsize=8.5,
    )

    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / "gate3_cfar.png", out / "gate3_cfar.pdf"]
    fig.savefig(paths[0], dpi=200, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(REPO_ROOT / "results/validation/gate3_report.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "results/validation"))
    a = ap.parse_args()
    report_path = Path(a.report)
    if not report_path.exists():
        raise SystemExit(f"{report_path} not found; run the Gate 3 test first (see --help)")
    for p in plot(json.loads(report_path.read_text()), Path(a.out)):
        print(p)
