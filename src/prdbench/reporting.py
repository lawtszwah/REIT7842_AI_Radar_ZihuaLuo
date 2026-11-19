"""Render the figures and summary tables the thesis reports from results.json files.

Lives in the package rather than in scripts/ so that `prdbench report` and
`scripts/make_figures.py` are the same code path: a figure in the write-up is
traceable to its results files and one commit, whichever way it was produced.

Several results files can be reported together, as when the GPU arm and the CPU
arms ran separately (`prdbench run --detectors`). They are merged only if they are
one experiment: same frozen config, same simulator, same root seeds, and the same
data streams at every root, so the arms are compared on paired data.

What is shown follows ADR-0005 and the risk register:

* The headline operating point is Pfa = 1e-4; the selection point 1e-3 is reported
  beside it, and every design point is in summary.json.
* Every root seed is shown, not only root 0, and the spread is the population SD
  over roots, as in results.json.
* The shift splits are reported separately from the in-distribution test.
* Achieved Pfa is reported with Pd. An operating point a detector could not reach
  on the calibration background is named, with the tie count that caused it.
* Latency measured on a GPU is reported apart from CPU latency and is never put on
  the CPU cost axis (RSK-06).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HEADLINE_PFA = 1e-4  # ADR-0005: "the headline figure is 10^-4"
SPLITS = ("test_in_distribution", "extended_test", "joint_shift", "grid_shift")
SPLIT_TITLES = {
    "test_in_distribution": "In-distribution test",
    "extended_test": "Extended test",
    "joint_shift": "Joint shift",
    "grid_shift": "Grid shift",
}
# Categorical slots in fixed order: a detector keeps its colour in every figure.
COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6",
           "#e34948"]
MARKERS = {"cfar": "o", "feature": "s", "deep": "^"}
LINES = {"cfar": "-", "feature": "--", "deep": "-."}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
# The fields that make two results files one experiment.
SAME_RUN = ("config_sha256", "simulator", "simulator_version", "root_seeds",
            "reporting_split")


class ReportError(ValueError):
    """Results files that cannot be reported together, or are incomplete."""


# Pd is read against achieved Pfa: past this factor over design, a Pd is flagged.
PFA_EXCESS = 2.0


def pfa_text(p: float) -> str:
    """1e-4 rather than 0.0001."""
    mantissa, exponent = f"{p:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def pfa_math(p: float) -> str:
    mantissa, exponent = f"{p:.0e}".split("e")
    head = "" if mantissa == "1" else rf"{mantissa}	imes"
    return rf"${head}10^{{{int(exponent)}}}$"


def over_pfa(summary: dict, split: str, p: float) -> list[str]:
    """Detectors whose mean achieved test Pfa exceeds the design Pfa by PFA_EXCESS."""
    return [d["detector"] for d in summary["detectors"]
            if d["splits"][split][f"{p:g}"]["pfa_achieved"]["mean"] > PFA_EXCESS * p]


def headline_pfa(points: list[float]) -> float:
    return HEADLINE_PFA if HEADLINE_PFA in points else points[min(1, len(points) - 1)]


def device_of(det: dict) -> str:
    """Where latency was measured. Only the torch arms can run elsewhere than the CPU."""
    return det["describe"].get("device", "cpu")


def load(paths: list[Path]) -> dict:
    """Read one or more results files and merge them into one experiment."""
    if not paths:
        raise ReportError("no results files given")
    files = []
    for path in paths:
        raw = Path(path).read_bytes()
        files.append((Path(path), hashlib.sha256(raw).hexdigest(), json.loads(raw)))

    first = files[0][2]
    for path, _, res in files[1:]:
        for field in SAME_RUN:
            if res["run"].get(field) != first["run"].get(field):
                raise ReportError(f"{path}: run.{field} differs from {files[0][0]}; "
                                  "these are not one experiment")
        if res["protocol"] != first["protocol"]:
            raise ReportError(f"{path}: protocol differs from {files[0][0]}")

    detectors, streams = [], {}
    for path, _, res in files:
        for det in res["detectors"]:
            if any(d["key"] == det["key"] for d in detectors):
                raise ReportError(f"detector {det['key']!r} appears in more than one file")
            for run in det["runs"]:
                if "streams" not in run:
                    continue
                # Untrained arms draw no model_initialization stream, so compare
                # the streams each pair of arms has in common.
                seen = streams.setdefault(run["root_seed"], {})
                for name, stream in run["streams"].items():
                    if seen.setdefault(name, stream) != stream:
                        raise ReportError(f"{path}: {det['key']} root {run['root_seed']} used "
                                          f"a different {name} stream; the arms are not "
                                          "paired")
            detectors.append(det)

    order = [d["key"] for d in first.get("spec", {}).get("detectors", [])]
    detectors.sort(key=lambda d: order.index(d["key"]) if d["key"] in order else len(order))
    sources = [{"path": path.as_posix(), "sha256": sha,
                "git_commit": res["run"].get("git_commit"),
                "git_dirty": res["run"].get("git_dirty"),
                "detectors": [d["key"] for d in res["detectors"]]}
               for path, sha, res in files]
    return {"run": {f: first["run"].get(f) for f in SAME_RUN},
            "protocol": first["protocol"], "detectors": detectors, "sources": sources}


def _splits(det: dict) -> list[str]:
    present = det["runs"][0].get("splits", {})
    return [s for s in SPLITS if s in present] + [s for s in present if s not in SPLITS]


def tied_at_threshold(n_cells: int, pfa: float) -> int:
    """Calibration cells that must share the threshold score when none exceeds it.

    The threshold is the "higher" empirical quantile at 1 - pfa, so every cell from
    that order statistic up scores at least the threshold. If the achieved Pfa is
    zero, none scores above it, so all of them score exactly the threshold.
    """
    return n_cells - math.ceil((1.0 - pfa) * (n_cells - 1))


def _stats(values: list[float]) -> dict:
    return {"per_root": values, "mean": float(np.mean(values)), "sd": float(np.std(values))}


def summarise(merged: dict) -> dict:
    points = merged["protocol"]["pfa_points"]
    rows = []
    for det in merged["detectors"]:
        runs = sorted(det["runs"], key=lambda r: r["root_seed"])
        row = {
            "detector": det["key"], "name": det["describe"]["name"],
            "family": det["describe"]["family"], "params": det["describe"]["params"],
            "n_parameters": det["n_parameters"], "latency_device": device_of(det),
            "roots": [r["root_seed"] for r in runs],
            "latency_ms": _stats([r["cost"]["latency_s_per_map_median"] * 1e3 for r in runs]),
            "splits": {},
            "unattained": [],
        }
        for split in _splits(det):
            row["splits"][split] = {
                f"{p:g}": {"pd": _stats([r["splits"][split]["pd"][f"{p:g}"] for r in runs]),
                           "pfa_achieved": _stats([r["splits"][split]["pfa_achieved"][f"{p:g}"]
                                                   for r in runs])}
                for p in points}
        for r in runs:
            n_cells = r["calibration"]["n_background_cells"]
            for p in points:
                if r["calibration"]["pfa_achieved"][f"{p:g}"] == 0:
                    row["unattained"].append({
                        "root_seed": r["root_seed"], "pfa": p,
                        "threshold": r["thresholds"][f"{p:g}"],
                        "calibration_cells": n_cells,
                        "min_cells_tied_at_threshold": tied_at_threshold(n_cells, p)})
        rows.append(row)
    return {"sources": merged["sources"], "run": merged["run"],
            "headline_pfa": headline_pfa(points),
            "selection_pfa": merged["protocol"].get("selection_pfa"),
            "pfa_points": points, "detectors": rows}


def _fmt(s: dict, digits: int = 3) -> str:
    return f"{s['mean']:.{digits}f} ± {s['sd']:.{digits}f}"


def _table(head: list[str], body: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head),
            *("| " + " | ".join(r) + " |" for r in body), ""]


def markdown(summary: dict) -> str:
    dets = summary["detectors"]
    splits = [s for s in SPLITS if all(s in d["splits"] for d in dets)]
    head_p, sel_p = summary["headline_pfa"], summary["selection_pfa"]
    run = summary["run"]
    out = ["# Results summary", "",
           f"Config `{run['config_sha256']}`, simulator `{run['simulator_version']}`, "
           f"root seeds {run['root_seeds']}. Values are mean ± population SD over the root "
           "seeds; per-root values are in summary.json.", ""]
    out += _table(["Results file", "SHA-256", "Commit", "Dirty", "Detectors"],
                  [[s["path"], f"`{s['sha256'][:16]}`", f"`{(s['git_commit'] or '')[:7]}`",
                    str(s["git_dirty"]), ", ".join(s["detectors"])]
                   for s in summary["sources"]])

    operating = [(head_p, "headline operating point")]
    if sel_p is not None and sel_p != head_p and sel_p in summary["pfa_points"]:
        operating.append((sel_p, "selection operating point"))
    for p, what in operating:
        out += [f"## Pd at design Pfa = {pfa_text(p)} ({what})", ""]
        out += _table(["Detector", *(SPLIT_TITLES.get(s, s) for s in splits)],
                      [[d["detector"], *(_fmt(d["splits"][s][f"{p:g}"]["pd"]) for s in splits)]
                       for d in dets])
        out += [f"Achieved test Pfa at design Pfa = {pfa_text(p)}:", ""]
        out += _table(["Detector", *(SPLIT_TITLES.get(s, s) for s in splits)],
                      [[d["detector"],
                        *(f"{d['splits'][s][f'{p:g}']['pfa_achieved']['mean']:.2e}"
                          for s in splits)] for d in dets])
        flagged = [f"{k} on {SPLIT_TITLES.get(s, s).lower()}"
                   for s in splits for k in over_pfa(summary, s, p)]
        if flagged:
            out += [f"Achieved Pfa is more than {PFA_EXCESS:g}× the design Pfa for "
                    + "; ".join(flagged) + ". Those thresholds, set on in-distribution "
                    "calibration background, do not hold there, so the Pd beside them is "
                    "not at the stated Pfa.", ""]

    roots = dets[0]["roots"]
    first = splits[0]
    out += [f"## Pd per root seed, {SPLIT_TITLES.get(first, first)}, Pfa = {pfa_text(head_p)}",
            ""]
    out += _table(["Detector", *(f"root {r}" for r in roots)],
                  [[d["detector"],
                    *(f"{v:.3f}" for v in d["splits"][first][f"{head_p:g}"]["pd"]["per_root"])]
                   for d in dets])

    out += ["## Cost", "",
            "Latency is the median over repeated runs per root, then mean ± SD over roots. "
            "GPU and CPU latency are not comparable and are never plotted together "
            "(RSK-06).", ""]
    out += _table(["Detector", "Parameters", "Measured on", "Latency (ms per map)"],
                  [[d["detector"], str(d["n_parameters"]), d["latency_device"].upper(),
                    _fmt(d["latency_ms"], 3)] for d in dets])

    notes = [(d["detector"], u) for d in dets for u in d["unattained"]]
    if notes:
        out += ["## Operating points not attained on the calibration background", "",
                "Detection is strictly `score > threshold`. Where the achieved calibration "
                "Pfa is zero, no calibration cell scores above the threshold, so at least "
                "the stated number of cells share the threshold score and that operating "
                "point cannot be reached.", ""]
        out += _table(["Detector", "Root", "Design Pfa", "Threshold", "Cells tied (at least)"],
                      [[k, str(u["root_seed"]), pfa_text(u["pfa"]), f"{u['threshold']:.7g}",
                        f"{u['min_cells_tied_at_threshold']:,} of {u['calibration_cells']:,}"]
                       for k, u in notes])
    return "\n".join(out)


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _flag(ax, keys: list[str]) -> None:
    if keys:
        ax.text(0.02, 0.97, f"Achieved Pfa > {PFA_EXCESS:g}× design: " + ", ".join(keys),
                transform=ax.transAxes, fontsize=6.5, color=MUTED, va="top")


def _panels(n: int):
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False,
                             sharey=True)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    return fig, list(axes.flat[:n])


def _save(fig, path: Path, written: list[Path]) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path)


def render(results_paths, out_dir: Path) -> list[Path]:
    """Write figures, summary.json and summary.md for one experiment."""
    if isinstance(results_paths, (str, Path)):
        results_paths = [results_paths]
    merged = load([Path(p) for p in results_paths])
    summary = summarise(merged)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dets, rows = merged["detectors"], summary["detectors"]
    colour = {d["key"]: COLOURS[i % len(COLOURS)] for i, d in enumerate(dets)}
    p_head = summary["headline_pfa"]
    splits = [s for s in SPLITS if all(s in r["splits"] for r in rows)]
    written: list[Path] = []

    # 1. Pd at the headline point, every root seed, one panel per split.
    top = max(v for r in rows for s in splits
              for v in r["splits"][s][f"{p_head:g}"]["pd"]["per_root"])
    fig, axes = _panels(len(splits))
    for ax, split in zip(axes, splits, strict=True):
        for i, row in enumerate(rows):
            s = row["splits"][split][f"{p_head:g}"]["pd"]
            n = len(s["per_root"])
            x = i + np.linspace(-0.18, 0.18, n) if n > 1 else np.array([i])
            ax.scatter(x, s["per_root"], s=16, color=colour[row["detector"]],
                       marker=MARKERS.get(row["family"], "o"), edgecolor="white", lw=0.6,
                       zorder=3, clip_on=False)
            ax.hlines(s["mean"], i - 0.3, i + 0.3, color=INK, lw=1.4, zorder=4)
        ax.set_xticks(range(len(rows)), [r["detector"] for r in rows], rotation=30,
                      ha="right")
        ax.set_title(SPLIT_TITLES.get(split, split), fontsize=8, color=INK)
        ax.set_ylim(0, top * 1.12)
        _flag(ax, over_pfa(summary, split, p_head))
        _style(ax)
    for ax in axes[::2]:
        ax.set_ylabel(f"Pd at Pfa = {pfa_math(p_head)}", fontsize=8, color=INK)
    fig.suptitle("Pd per root seed (points) and mean over roots (bar)", fontsize=9, color=INK)
    _save(fig, out_dir / "pd_by_root.png", written)

    # 2. ROC: every root seed, one panel per split.
    fig, axes = _panels(len(splits))
    for ax, split in zip(axes, splits, strict=True):
        for det in dets:
            fam = det["describe"]["family"]
            for j, run in enumerate(sorted(det["runs"], key=lambda r: r["root_seed"])):
                roc = run["splits"][split]["roc"]
                ax.semilogx(roc["pfa"], roc["pd"], LINES.get(fam, "-"), lw=1.0, alpha=0.75,
                            color=colour[det["key"]], label=det["key"] if j == 0 else None)
        ax.axvline(p_head, color=MUTED, lw=0.8, ls=":")
        ax.set_ylim(0, 1)
        ax.set_title(SPLIT_TITLES.get(split, split), fontsize=8, color=INK)
        ax.set_xlabel("Pfa (per cell)", fontsize=8, color=INK)
        _style(ax)
    for ax in axes[::2]:
        ax.set_ylabel("Pd (per target)", fontsize=8, color=INK)
    axes[0].legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.suptitle("ROC, one line per root seed", fontsize=9, color=INK)
    _save(fig, out_dir / "roc.png", written)

    # 3. Pd by target SNR at the headline point: mean over roots, range as a band.
    fig, axes = _panels(len(splits))
    for ax, split in zip(axes, splits, strict=True):
        for det in dets:
            per_root = [run["splits"][split]["pd_by_snr"][f"{p_head:g}"] for run in det["runs"]]
            x = [(b["snr_lo"] + b["snr_hi"]) / 2 for b in per_root[0]]
            pd = np.array([[np.nan if b["pd"] is None else b["pd"] for b in bins]
                           for bins in per_root], dtype=float)
            keep = ~np.all(np.isnan(pd), axis=0)
            xs = np.array(x)[keep]
            col = colour[det["key"]]
            ax.fill_between(xs, np.nanmin(pd[:, keep], 0), np.nanmax(pd[:, keep], 0),
                            color=col, alpha=0.12, lw=0)
            ax.plot(xs, np.nanmean(pd[:, keep], 0), LINES.get(det["describe"]["family"], "-"),
                    marker=MARKERS.get(det["describe"]["family"], "o"), ms=3, lw=1.4,
                    color=col, label=det["key"])
        ax.set_ylim(0, 1)
        ax.set_title(SPLIT_TITLES.get(split, split), fontsize=8, color=INK)
        ax.set_xlabel("Target SNR (dB)", fontsize=8, color=INK)
        _flag(ax, over_pfa(summary, split, p_head))
        _style(ax)
    for ax in axes[::2]:
        ax.set_ylabel(f"Pd at Pfa = {pfa_math(p_head)}", fontsize=8, color=INK)
    axes[0].legend(fontsize=6.5, loc="upper left", frameon=False)
    fig.suptitle("Pd by target SNR: mean over roots, band = range over roots", fontsize=9,
                 color=INK)
    _save(fig, out_dir / "pd_by_snr.png", written)

    # 4. Pd against CPU latency. GPU-timed arms are named, never placed on this axis.
    first = splits[0]
    cpu = [r for r in rows if r["latency_device"] == "cpu"]
    other = [r for r in rows if r["latency_device"] != "cpu"]
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for row in cpu:
        s = row["splits"][first][f"{p_head:g}"]["pd"]
        x = row["latency_ms"]["mean"]
        ax.errorbar(x, s["mean"], yerr=s["sd"], fmt=MARKERS.get(row["family"], "o"), ms=6,
                    capsize=3, color=colour[row["detector"]], mec="white", mew=0.8)
        ax.annotate(row["detector"], (x, s["mean"]), textcoords="offset points",
                    xytext=(7, 4), fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency (ms per map, CPU)", fontsize=8, color=INK)
    ax.set_ylabel(f"Pd at Pfa = {pfa_math(p_head)}, {SPLIT_TITLES.get(first, first).lower()}",
                  fontsize=8, color=INK)
    ax.set_title("Detection performance against CPU inference cost", fontsize=9, color=INK)
    if other:
        ax.text(0.01, 0.01, "Not shown, latency measured on GPU: "
                + ", ".join(r["detector"] for r in other), transform=ax.transAxes,
                fontsize=6.5, color=MUTED, va="bottom")
    _style(ax)
    _save(fig, out_dir / "pd_vs_cost.png", written)

    p = out_dir / "summary.json"
    # LF on every platform, so a summary rendered on Windows and on the Mac is the same bytes.
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8", newline="\n")
    written.append(p)
    p = out_dir / "summary.md"
    p.write_text(markdown(summary), encoding="utf-8", newline="\n")
    written.append(p)
    return written
