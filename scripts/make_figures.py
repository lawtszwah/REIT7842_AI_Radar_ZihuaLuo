"""Render the three figures the thesis reports from a results.json.

One script, so every figure in the write-up is traceable to one results file
and one commit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"cfar": "-", "feature": "--", "deep": "-."}


def render(results_path: Path, out_dir: Path) -> list[Path]:
    res = json.loads(Path(results_path).read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    dets = res["detectors"]
    ref_pfa = res["protocol"]["pfa_points"][1]
    written = []

    # 1. ROC -- the primary comparison, seed 0 shown, all seeds in results.json.
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for d in dets:
        r = d["runs"][0]["roc"]
        ax.semilogx(r["pfa"], r["pd"], STYLE.get(d["describe"]["family"], "-"),
                    lw=1.6, label=d["key"])
    ax.axvline(ref_pfa, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("Probability of false alarm (per cell)")
    ax.set_ylabel("Probability of detection (per target)")
    ax.set_title("ROC under the shared evaluation protocol")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    p = out_dir / "roc.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    written.append(p)

    # 2. Where the gains are: Pd at fixed Pfa, stratified by target SNR.
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for d in dets:
        bins = d["runs"][0]["pd_by_snr"]
        x = [(b["snr_lo"] + b["snr_hi"]) / 2 for b in bins]
        ax.plot(x, [b["pd"] for b in bins], "o" + STYLE.get(d["describe"]["family"], "-"),
                ms=4, lw=1.5, label=d["key"])
    ax.set_xlabel("Target SNR (dB)")
    ax.set_ylabel(f"Pd at Pfa = {ref_pfa:g}")
    ax.set_title("Where detections are won or lost")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    p = out_dir / "pd_by_snr.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    written.append(p)

    # 3. Accuracy retained under an inference-cost constraint.
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    for d in dets:
        x = d["latency_s_per_map"]["mean"] * 1e3
        y = d[f"pd@pfa={ref_pfa:g}"]["mean"]
        e = d[f"pd@pfa={ref_pfa:g}"]["std"]
        ax.errorbar(x, y, yerr=e, fmt="o", ms=6, capsize=3)
        ax.annotate(d["key"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Inference latency (ms per map, CPU)")
    ax.set_ylabel(f"Pd at Pfa = {ref_pfa:g}")
    ax.set_title("Detection performance against inference cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "pd_vs_cost.png"
    fig.savefig(p, dpi=200)
    plt.close(fig)
    written.append(p)

    # Machine-readable summary table for the write-up.
    rows = []
    for d in dets:
        row = {"detector": d["key"], "family": d["describe"]["family"],
               "n_parameters": d["n_parameters"],
               "latency_ms": round(d["latency_s_per_map"]["mean"] * 1e3, 3)}
        for p_ in res["protocol"]["pfa_points"]:
            row[f"pd@{p_:g}"] = round(d[f"pd@pfa={p_:g}"]["mean"], 3)
            row[f"sd@{p_:g}"] = round(d[f"pd@pfa={p_:g}"]["std"], 3)
        rows.append(row)
    p = out_dir / "summary.json"
    p.write_text(json.dumps(rows, indent=2))
    written.append(p)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/smoke/results.json")
    ap.add_argument("--out", default="results/smoke/figures")
    a = ap.parse_args()
    for f in render(Path(a.results), Path(a.out)):
        print(f)
