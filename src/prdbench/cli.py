"""Command-line entry point: every result in the thesis is produced by one of these."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .evaluation.protocol import merge_joint_shift, run_benchmark
from .reporting import render
from .tuning.search import search


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prdbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the benchmark protocol")
    run.add_argument("--config", required=True)
    run.add_argument("--out", default="results/latest")
    run.add_argument("--simulator", default=None, choices=["stub", "prdsim"])

    tune = sub.add_parser("tune", help="equal-budget hyperparameter search")
    tune.add_argument("--config", required=True)
    tune.add_argument("--detector", required=True)
    tune.add_argument("--budget", type=int, default=12)
    tune.add_argument("--out", default=None)

    fig = sub.add_parser("report", help="render figures and tables from results.json")
    fig.add_argument("--results", required=True)
    fig.add_argument("--out", default=None)

    a = ap.parse_args(argv)

    if a.cmd == "run":
        spec = _load(a.config)
        if a.simulator:
            spec["simulator"] = a.simulator
        res = run_benchmark(merge_joint_shift(spec), Path(a.out))
        points = res["protocol"]["pfa_points"]
        reference_pfa = points[min(1, len(points) - 1)]
        for d in res["detectors"]:
            pd = d[f"pd@pfa={reference_pfa:g}"]
            print(f"{d['key']:<14} Pd[{res['run']['reporting_split']}]="
                  f"{pd['mean']:.3f}+-{pd['std']:.3f}  "
                  f"{d['latency_s_per_map']['mean']*1e3:7.2f} ms/map  "
                  f"params={d['n_parameters']}")
        print(f"\nwrote {Path(a.out) / 'results.json'}")
        return 0

    if a.cmd == "tune":
        spec = _load(a.config)
        space = spec["search_spaces"][a.detector]
        res = search(spec, a.detector, space, budget=a.budget, out=a.out)
        print(json.dumps(res["best"], indent=2))
        return 0

    if a.cmd == "report":
        results = Path(a.results)
        out = Path(a.out) if a.out else results.parent / "figures"
        for f in render(results, out):
            print(f)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
