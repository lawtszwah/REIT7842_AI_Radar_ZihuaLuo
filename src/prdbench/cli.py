"""Command-line entry point: every result in the thesis is produced by one of these."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from .detectors.cnn import DEVICE_ENV, resolve_device
from .evaluation.protocol import detector_keys, merge_joint_shift, run_benchmark
from .reporting import render
from .tuning.search import search
from .tuning.selection import load_selections


def _load(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


DEVICE_HELP = ("torch device for the learned detectors: cpu (default), cuda, cuda:N, mps or "
               f"auto; recorded per detector, not part of the frozen config (sets {DEVICE_ENV})")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="prdbench")
    sub = ap.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="run the benchmark protocol")
    run.add_argument("--config", required=True)
    run.add_argument("--out", default="results/latest")
    run.add_argument("--simulator", default=None, choices=["stub", "prdsim"])
    run.add_argument("--device", default=None, help=DEVICE_HELP)
    run.add_argument("--tuned", default=None, metavar="DIR",
                     help="directory of `prdbench tune` records (<key>_root0.json); each "
                          "detector that runs uses its selected hyperparameters, after the "
                          "record is checked against this config")
    run.add_argument("--detectors", nargs="+", default=None, metavar="KEY",
                     help="run only these detector keys, in config order; the config and "
                          "its hash are unchanged")

    tune = sub.add_parser("tune", help="equal-budget hyperparameter search")
    tune.add_argument("--config", required=True)
    tune.add_argument("--detector", required=True)
    tune.add_argument("--budget", type=int, default=None,
                      help="only for configs without protocol.tuning_budget; "
                           "must match it otherwise")
    tune.add_argument("--out", default=None)
    tune.add_argument("--device", default=None, help=DEVICE_HELP)

    fig = sub.add_parser("report", help="render figures and tables from results.json")
    fig.add_argument("--results", required=True)
    fig.add_argument("--out", default=None)

    a = ap.parse_args(argv)
    if getattr(a, "device", None):
        resolve_device(a.device)  # fail now, not after hours of data generation
        os.environ[DEVICE_ENV] = a.device

    if a.cmd == "run":
        spec = _load(a.config)
        if a.simulator:
            spec["simulator"] = a.simulator
        spec = merge_joint_shift(spec)
        keys = a.detectors or detector_keys(spec)
        unknown = sorted(set(keys) - set(detector_keys(spec)))
        if unknown:
            raise SystemExit(f"unknown detector keys {unknown}; the config has "
                             f"{detector_keys(spec)}")
        # Checked before any data are generated, which can take hours.
        tuned = load_selections(spec, Path(a.tuned), keys) if a.tuned else None
        res = run_benchmark(spec, Path(a.out), tuned=tuned, only=a.detectors)
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
        # Merged as for `run`, so the recorded config hash is the frozen one.
        spec = merge_joint_shift(_load(a.config))
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
