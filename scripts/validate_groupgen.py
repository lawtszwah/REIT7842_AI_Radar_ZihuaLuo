"""The three validation gates, applied to the research group's generator, with figures.

The gates were written for the reference simulator (docs/simulator_model.md section 8);
here the same questions are asked of the data source the thesis actually uses. No learned
detector runs; Gate 3 runs CA-CFAR on target-free maps to measure its false-alarm rate.

  build-up  one scene, same seed: direct path + noise, + clutter, + target
  gate 1    target-free background against the exponential law (CV, quantiles, correlation)
  gate 2    loss of the target-cell peak against the fractional offset of the true position;
            in Doppler, against sinc^2 of a rectangular 0.1 s window
  gate 3    CA-CFAR measured Pfa on target-free maps against its design Pfa, >= 10^7 cells

All maps use the protocol's background: DNR 45 dB, CSR 20-30 dB, the 10 Hz x 150 m grid.
Output: results/validation/groupgen_gate{1,2,3}.{png,pdf,json}, groupgen_buildup.{png,pdf}.

    python3 scripts/validate_groupgen.py        # about 30 minutes of MATLAB the first time
    python3 scripts/validate_groupgen.py gate1  # one step; others keep their last record
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from prdbench.detectors.cfar import CACFAR  # noqa: E402
from prdbench.simulation import SimConfig, get_simulator  # noqa: E402
from prdbench.simulation.groupgen import DEFAULTS, _grids  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "validation"
FMAP, GRIDX = _grids(DEFAULTS)
ZERO = len(FMAP) // 2
FRAMES = DEFAULTS["range_size"]
CELL_KM = DEFAULTS["c"] / DEFAULTS["fs"] / 1e3
Z = 4.0
BACKGROUND = {"csr_db": [20.0, 30.0], "dnr_db": [45.0, 45.0]}

INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def cfg(n_scenes, seed, snr, n_targets=(1, 1), **overrides) -> SimConfig:
    extra = dict(BACKGROUND)
    extra.update(overrides)
    return SimConfig(n_maps=n_scenes * FRAMES, n_range=len(GRIDX), n_doppler=len(FMAP),
                     n_targets=n_targets, snr_db=snr, seed=seed, extra={"groupgen": extra})


def style(ax, ygrid=True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if ygrid:
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def save(fig, stem) -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [OUT / f"{stem}.png", OUT / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=200, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


def eligible_columns(reach: int = 2) -> np.ndarray:
    return np.abs(np.arange(len(FMAP)) - ZERO) > reach


def show_map(ax, power, ref, vmax=20):
    step = FMAP[1] - FMAP[0]
    km = (GRIDX - 1) * CELL_KM
    return ax.imshow(10 * np.log10(np.maximum(power / ref, 1e-12)), origin="lower",
                     aspect="auto", cmap="Greys_r", vmin=0, vmax=vmax,
                     extent=(FMAP[0] - step / 2, FMAP[-1] + step / 2,
                             km[0] - CELL_KM / 2, km[-1] + CELL_KM / 2))


# -- build-up -------------------------------------------------------------------------


def buildup(sim) -> list[Path]:
    seed, snr = 20261014, (5.0, 5.0)
    steps = [
        ("Direct path + noise", cfg(1, seed, snr, n_targets=(0, 0), csr_db=[-300.0, -300.0])),
        ("+ clutter", cfg(1, seed, snr, n_targets=(0, 0))),
        ("+ target (complete frame)", cfg(1, seed, snr)),
    ]
    batches = [sim.generate(c) for _, c in steps]
    ref = np.median(batches[0].maps[0][:, eligible_columns()]) / np.log(2)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, (title, _), batch in zip(axes, steps, batches, strict=True):
        img = show_map(ax, batch.maps[0].astype(np.float64), ref, vmax=25)
        ax.set_title(title, loc="left", color=INK, fontsize=10.5)
        ax.set_xlabel("Doppler (Hz)")
        style(ax, ygrid=False)
    (t,) = batches[-1].targets[0]
    step = FMAP[1] - FMAP[0]
    axes[-1].add_patch(matplotlib.patches.Ellipse(
        (FMAP[0] + t.doppler_bin * step, (t.range_bin + GRIDX[0] - 1) * CELL_KM),
        70, 1.5, fill=False, ec=ORANGE, lw=1.8))
    axes[0].set_ylabel("Bistatic range (km)")
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.2, top=0.88, wspace=0.08)
    cbar = fig.colorbar(img, cax=fig.add_axes([0.905, 0.2, 0.012, 0.68]))
    cbar.set_label("Power / background (dB)", color=INK)
    cbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED)
    cbar.outline.set_visible(False)
    fig.text(0.01, 0.01, f"groupgen {sim.version}, seed {seed}; one 0.1 s frame; "
             "same seed in every panel", color=INK_MUTED, fontsize=8.5)
    return save(fig, "groupgen_buildup")


# -- gate 1 ---------------------------------------------------------------------------


def _exponential_checks(u: np.ndarray, cube: np.ndarray) -> dict:
    """CV, tail quantiles and lag-1 correlation of power already normalised to mean 1."""
    m = u.size
    cv = float(u.std(ddof=1) / u.mean())
    quantiles = {}
    for q in (0.5, 0.9, 0.99, 0.999):
        expected = -math.log(1 - q)
        measured = float(np.quantile(u, q))
        tol = Z * math.sqrt(q / (m * (1 - q)))
        quantiles[str(q)] = {"measured": measured, "expected": expected, "tolerance": tol,
                             "passed": abs(measured - expected) <= tol}

    def lag1(axis):
        n = cube.shape[axis] // 2 * 2
        a = np.take(cube, np.arange(0, n, 2), axis=axis).ravel()
        b = np.take(cube, np.arange(1, n, 2), axis=axis).ravel()
        return float(np.corrcoef(a, b)[0, 1]), Z / math.sqrt(a.size)

    (r, r_tol), (d, d_tol) = lag1(1), lag1(2)
    out = {"cells": int(m),
           "cv": {"measured": cv, "tolerance": Z / math.sqrt(m),
                  "passed": abs(cv - 1) <= Z / math.sqrt(m)},
           "quantiles": quantiles,
           "lag1_corr": {"range": r, "range_tolerance": r_tol, "range_passed": abs(r) <= r_tol,
                         "doppler": d, "doppler_tolerance": d_tol,
                         "doppler_passed": abs(d) <= d_tol}}
    out["passed"] = bool(out["cv"]["passed"] and all(v["passed"] for v in quantiles.values())
                         and out["lag1_corr"]["range_passed"]
                         and out["lag1_corr"]["doppler_passed"])
    return out


def gate1(sim) -> dict:
    """Target-free background, pooled and with each map normalised by its own mean.

    Measured 2026-10-12: pooled over maps the background is *not* one exponential law,
    because each map's level rises with its clutter level (CNR = reference SNR + CSR).
    Within each map it is exponential. Both results are recorded.
    """
    batch = sim.generate(cfg(20, 9301, (-10.0, 10.0), n_targets=(0, 0)))
    cube = batch.maps.astype(np.float64)[:, :, eligible_columns()]
    level = cube.mean(axis=(1, 2), keepdims=True)
    pooled = _exponential_checks((cube / cube.mean()).ravel(), cube / cube.mean())
    per_map = _exponential_checks((cube / level).ravel(), cube / level)
    cnr = np.repeat([s["reference_snr_db"] + s["csr_db"] for s in batch.meta["scenes"]],
                    FRAMES)
    level_db = 10 * np.log10(level.ravel() / np.median(level))
    report = {"maps": int(cube.shape[0]), "pooled": pooled, "per_map_normalised": per_map,
              "map_level_db_range": [float(level_db.min()), float(level_db.max())],
              "corr_map_level_vs_cnr": float(np.corrcoef(cnr, level_db)[0, 1])}

    fig, (ax_map, ax_pdf, ax_lvl) = plt.subplots(
        1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1, 1.2, 0.9]})
    img = show_map(ax_map, batch.maps[0].astype(np.float64), level.ravel()[0], vmax=12)
    ax_map.set_xlabel("Doppler (Hz)")
    ax_map.set_ylabel("Bistatic range (km)")
    ax_map.set_title("(a) One target-free frame", loc="left", color=INK, fontsize=11)
    style(ax_map, ygrid=False)
    cbar = fig.colorbar(img, ax=ax_map, fraction=0.046, pad=0.03)
    cbar.set_label("Power / map mean (dB)", color=INK)
    cbar.ax.tick_params(colors=INK_MUTED, labelcolor=INK_MUTED)
    cbar.outline.set_visible(False)

    edges = np.linspace(0, 16, 65)
    c = 0.5 * (edges[:-1] + edges[1:])
    for u, colour, label in (((cube / cube.mean()).ravel(), INK_MUTED, "Pooled over maps"),
                             ((cube / level).ravel(), BLUE, "Each map / its own mean")):
        dens, _ = np.histogram(u, bins=edges, density=True)
        keep = dens > 0
        ax_pdf.step(c[keep], dens[keep], where="mid", color=colour, lw=2, label=label)
    g = np.linspace(0, 16, 400)
    ax_pdf.plot(g, np.exp(-g), color=ORANGE, lw=2, ls="--", label="Theory: exponential")
    ax_pdf.set_yscale("log")
    ax_pdf.set_xlim(0, 16)
    ax_pdf.set_ylim(1e-7, 2)
    ax_pdf.set_xlabel("Cell power / mean")
    ax_pdf.set_ylabel("Probability density (log scale)")
    ax_pdf.set_title("(b) Exponential within each map", loc="left", color=INK, fontsize=11)
    style(ax_pdf)
    leg = ax_pdf.legend(frameon=False, loc="upper right", fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    ax_pdf.text(0.02, 0.04, f"per map: CV = {per_map['cv']['measured']:.4f}\n"
                f"pooled:  CV = {pooled['cv']['measured']:.4f}",
                transform=ax_pdf.transAxes, color=INK_MUTED, fontsize=8.5)

    ax_lvl.plot(cnr, level_db, "o", color=BLUE, ms=4, alpha=0.6, mec="none")
    ax_lvl.set_xlabel("Clutter-to-noise ratio, CNR (dB)")
    ax_lvl.set_ylabel("Map background level (dB re median)")
    ax_lvl.set_title("(c) ...but its level follows clutter", loc="left", color=INK,
                     fontsize=11)
    style(ax_lvl)
    ax_lvl.text(0.02, 0.92, f"r = {report['corr_map_level_vs_cnr']:.2f}",
                transform=ax_lvl.transAxes, color=INK_MUTED, fontsize=9)
    fig.text(0.01, 0.01, f"groupgen {sim.version}; {cube.shape[0]} target-free frames, "
             f"{cube[0].size:,} cells each; DNR 45 dB, CSR 20–30 dB; zero-Doppler guard "
             "excluded", color=INK_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    report["figure"] = [str(x) for x in save(fig, "groupgen_gate1")]
    return report


# -- gate 2 ---------------------------------------------------------------------------


def gate2(sim) -> dict:
    batch = sim.generate(cfg(40, 9401, (10.0, 10.0)))
    p = batch.maps.astype(np.float64)
    floor = np.median(p[:, :, eligible_columns()]) / np.log(2)
    rows = []
    for power, (t,) in zip(p, batch.targets, strict=True):
        r, d = int(round(t.range_bin)), int(round(t.doppler_bin))
        rows.append((t.range_bin - r, t.doppler_bin - d, power[r, d] / floor - 1.0))
    a = np.array(rows)
    centre = a[(np.abs(a[:, 0]) < 0.15) & (np.abs(a[:, 1]) < 0.15), 2].mean()
    bins = np.linspace(0, 0.5, 6)

    def binned(offsets, other):
        out = []
        for lo, hi in zip(bins[:-1], bins[1:], strict=True):
            sel = (np.abs(offsets) >= lo) & (np.abs(offsets) < hi) & (np.abs(other) < 0.25)
            if sel.sum() >= 5:
                v = a[sel, 2] / centre
                out.append(((lo + hi) / 2, 10 * np.log10(v.mean()),
                            10 * np.log10(1 + Z * v.std(ddof=1) / math.sqrt(sel.sum())
                                          / v.mean()), int(sel.sum())))
        return out

    dop, rng_ = binned(a[:, 1], a[:, 0]), binned(a[:, 0], a[:, 1])
    fine = np.linspace(0, 0.5, 100)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, data, label, theory in (
        (axes[0], dop, "Doppler", True), (axes[1], rng_, "range", False),
    ):
        ax.plot(fine, 20 * np.log10(np.sinc(fine)), color=INK_MUTED, lw=1.5, ls="--",
                label="sinc² of a 0.1 s rectangular window" if theory else "sinc² (reference only)")
        x, y, e, n = (np.array(col) for col in zip(*data, strict=True))
        ax.errorbar(x, y, yerr=e, fmt="o", color=BLUE if theory else ORANGE, ms=8,
                    mec="white", mew=1.5, capsize=3, label="Generator, binned frames")
        ax.set_xlabel(f"Fractional {label} offset of the true position (cells)")
        ax.set_title(f"({'a' if theory else 'b'}) Peak loss vs {label} offset", loc="left",
                     color=INK, fontsize=11)
        style(ax)
        leg = ax.legend(frameon=False, loc="lower left")
        for t in leg.get_texts():
            t.set_color(INK)
    axes[0].set_ylabel("Target-cell SNR re centred target (dB)")
    axes[0].set_ylim(-9, 2)
    fig.text(0.01, 0.01, f"groupgen {sim.version}; SNR_dB 10, DNR 45 dB; {len(a)} frames; "
             "error bars: z = 4 standard errors", color=INK_MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    figure = save(fig, "groupgen_gate2")
    sinc_err = [abs(y - 20 * math.log10(np.sinc(x))) <= e for x, y, e, _ in dop]
    return {"frames": int(len(a)), "doppler": dop, "range": rng_,
            "doppler_within_band_of_sinc2": bool(all(sinc_err)), "figure": [str(f) for f in figure]}


# -- gate 3 ---------------------------------------------------------------------------


def gate3(sim) -> dict:
    train, guard = 4, 2
    border = train + guard
    reach = 2 + border            # keep every CFAR window clear of the zero-Doppler guard
    cols = np.zeros(len(FMAP), bool)
    cols[border:len(FMAP) - border] = True
    cols &= eligible_columns(reach)
    per_map = (len(GRIDX) - 2 * border) * int(cols.sum())
    n_scenes = math.ceil(1e7 / per_map / FRAMES)
    batch = sim.generate(cfg(n_scenes, 9501, (-10.0, 10.0), n_targets=(0, 0)))
    scores = CACFAR(train=train, guard=guard).score(batch.maps)[:, border:-border, :][:, :, cols]
    n_tr = (2 * border + 1) ** 2 - (2 * guard + 1) ** 2
    eligible = scores[0].size * scores.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence(9501, spawn_key=(1,)))
    idx = rng.integers(0, scores.shape[0], size=(10000, scores.shape[0]), dtype=np.int32)
    points = []
    for pfa in (1e-5, 1e-4, 1e-3, 1e-2):
        alpha = n_tr * (pfa ** (-1 / n_tr) - 1)
        counts = (scores > alpha).sum(axis=(1, 2))
        rates = counts[idx].sum(axis=1) / eligible
        lo, hi = np.percentile(rates, [0.5, 99.5])
        points.append({"design_pfa": pfa, "measured_pfa": float(counts.sum() / eligible),
                       "alarms": int(counts.sum()), "interval_low": float(lo),
                       "interval_high": float(hi), "passed": bool(lo <= pfa <= hi)})
    report = {"maps": int(scores.shape[0]), "eligible_cells": int(eligible), "points": points,
              "passed": all(pt["passed"] for pt in points)}

    d = np.array([pt["design_pfa"] for pt in points])
    meas = np.array([pt["measured_pfa"] for pt in points])
    lo = np.array([pt["interval_low"] for pt in points])
    hi = np.array([pt["interval_high"] for pt in points])
    fig, (ax_ll, ax_r) = plt.subplots(1, 2, figsize=(11, 4.4))
    span = np.array([d.min() / 3, d.max() * 3])
    ax_ll.plot(span, span, color=INK_MUTED, lw=1.2, ls="--", label="Measured = design")
    ax_ll.plot(d, meas, "o", color=BLUE, ms=9, mec="white", mew=1.5,
               label="CA-CFAR on generator background")
    ax_ll.set_xscale("log")
    ax_ll.set_yscale("log")
    ax_ll.set_xlim(*span)
    ax_ll.set_ylim(*span)
    ax_ll.set_aspect("equal")
    ax_ll.set_xlabel("Design Pfa")
    ax_ll.set_ylabel("Measured Pfa")
    ax_ll.set_title("(a) Measured vs design", loc="left", color=INK, fontsize=11)
    style(ax_ll)
    leg = ax_ll.legend(frameon=False, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    ax_r.axhline(1, color=INK_MUTED, lw=1.2, ls="--")
    ax_r.errorbar(d, meas / d, yerr=[(meas - lo) / d, (hi - meas) / d], fmt="o", color=BLUE,
                  ms=9, mec="white", mew=1.5, capsize=4)
    for x, h, pt in zip(d, hi / d, points, strict=True):
        ax_r.annotate(f"{pt['alarms']:,} alarms", (x, h), xytext=(0, 6),
                      textcoords="offset points", ha="center", color=INK_MUTED, fontsize=8.5)
    ax_r.set_xscale("log")
    ax_r.set_xlim(d.min() / 3, d.max() * 3)
    ax_r.set_ylim(0.55, 1.35)
    ax_r.set_xlabel("Design Pfa")
    ax_r.set_ylabel("Measured / design Pfa")
    ax_r.set_title("(b) Ratio with 99% bootstrap interval", loc="left", color=INK, fontsize=11)
    style(ax_r)
    ax_r.text(0.02, 0.04, f"{eligible:,} cells, {scores.shape[0]} target-free frames; "
              f"window {train} train / {guard} guard\ngroupgen {sim.version}: "
              f"{'all four design points inside' if report['passed'] else 'NOT all inside'}",
              transform=ax_r.transAxes, color=INK_MUTED, fontsize=8.5)
    fig.tight_layout()
    report["figure"] = [str(x) for x in save(fig, "groupgen_gate3")]
    return report


def main() -> dict:
    sim = get_simulator("groupgen")
    only = set(sys.argv[1:]) or {"buildup", "gate1", "gate2", "gate3"}
    path = OUT / "groupgen_gates.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update({"simulator_version": sim.version, "background": BACKGROUND})
    steps = {"buildup": lambda: [str(p) for p in buildup(sim)], "gate1": lambda: gate1(sim),
             "gate2": lambda: gate2(sim), "gate3": lambda: gate3(sim)}
    for name in ("buildup", "gate1", "gate2", "gate3"):
        if name in only:
            report[name] = steps[name]()
    path.write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: (v.get("passed") if isinstance(v, dict) else v) for k, v in r.items()},
                     indent=2))
