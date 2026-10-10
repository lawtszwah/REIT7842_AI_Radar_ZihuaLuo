"""Generate every protocol split at every root seed and check it before the freeze.

No detector runs here: this is a data check, not a result.  For each root seed and
split it generates the maps exactly as `run_benchmark` / `search` would (same spawned
stream, same materialised seed), then records

  - eligible background cells, using the evaluation geometry itself
    (metrics.split_scores with the protocol's tolerance, exclusion and guard), and
    whether the split meets protocol.min_eligible_background_cells;
  - eligible and excluded targets;
  - a SHA-256 of the generated maps, and whether regenerating gives the same bytes.

The floor applies to calibration, validation (model selection) and every test split.
Training has no floor; its counts are reported for completeness.

    python3 scripts/check_splits.py                       # all root seeds
    python3 scripts/check_splits.py --roots 0 --out results/validation/split_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import yaml

from prdbench.data.generate import build_dataset
from prdbench.evaluation import metrics
from prdbench.evaluation.protocol import ProtocolConfig, merge_joint_shift
from prdbench.evaluation.seeds import make_streams, stream_record

REPO_ROOT = Path(__file__).resolve().parents[1]

SPLIT_STREAMS = {
    "train": "train_maps",
    "validation": "validation_maps",
    "calibration": "calibration_maps",
    "test_in_distribution": "test_in_distribution_maps",
    "extended_test": "extended_test_maps",
    "joint_shift": "joint_shift_maps",
}
FLOORED = {"validation", "calibration", "test_in_distribution", "extended_test", "joint_shift"}


def check(spec: dict, roots: list[int], verify_every: bool) -> dict:
    proto = ProtocolConfig(**spec["protocol"])
    splits = [name for name in SPLIT_STREAMS if name in spec]
    rows = []
    for root in roots:
        streams = make_streams(root, proto.seed_streams)
        for name in splits:
            record = stream_record(streams[SPLIT_STREAMS[name]])
            t0 = time.perf_counter()
            ds = build_dataset(spec["simulator"], spec[name], seed=record["materialised_seed"])
            gen_s = time.perf_counter() - t0
            tgt, bg, _ = metrics.split_scores(
                ds.maps, ds.targets, tol=proto.tol_bins, exclude=proto.exclude_bins,
                zero_doppler_guard=proto.zero_doppler_guard,
            )
            n_targets = sum(len(t) for t in ds.targets)
            digest = hashlib.sha256(ds.maps.tobytes()).hexdigest()
            row = {
                "root_seed": root,
                "split": name,
                "stream": SPLIT_STREAMS[name],
                "materialised_seed": record["materialised_seed"],
                "n_maps": int(ds.maps.shape[0]),
                "eligible_background_cells": int(bg.size),
                "floor": proto.min_eligible_background_cells if name in FLOORED else None,
                "meets_floor": (bg.size >= proto.min_eligible_background_cells)
                if name in FLOORED else None,
                "targets_total": n_targets,
                "targets_eligible": int(tgt.size),
                "targets_excluded_zero_doppler": n_targets - int(tgt.size),
                "maps_sha256": digest,
                "generation_s": round(gen_s, 2),
            }
            if verify_every or name == "calibration":
                again = build_dataset(spec["simulator"], spec[name],
                                      seed=record["materialised_seed"])
                row["regenerates_identically"] = (
                    hashlib.sha256(again.maps.tobytes()).hexdigest() == digest
                    and again.targets == ds.targets
                )
            rows.append(row)
            floor = {True: "ok ", False: "LOW", None: "-- "}[row["meets_floor"]]
            print(f"root {root}  {name:<21} {row['n_maps']:>5} maps  "
                  f"{row['eligible_background_cells']:>12,} bg cells  floor {floor}"
                  f"targets {row['targets_eligible']:>6,}/{n_targets:<6,}  "
                  f"{gen_s:6.1f}s", flush=True)
            del ds, tgt, bg
    return {
        "simulator": spec["simulator"],
        "config_sha256": hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "joint_shift_source": spec.get("joint_shift_source"),
        "roots": roots,
        "all_floors_met": all(r["meets_floor"] is not False for r in rows),
        "all_regenerate_identically": all(
            r.get("regenerates_identically", True) for r in rows
        ),
        "rows": rows,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/protocol/full.yaml"))
    ap.add_argument("--roots", type=int, nargs="*", default=None,
                    help="root seeds to check (default: 0 .. n_seeds-1)")
    ap.add_argument("--verify-every-split", action="store_true",
                    help="regenerate every split, not only calibration, to check bytes")
    ap.add_argument("--out", default=str(REPO_ROOT / "results/validation/split_check.json"))
    a = ap.parse_args()
    spec = merge_joint_shift(yaml.safe_load(Path(a.config).read_text()))
    roots = a.roots if a.roots else list(range(int(spec["protocol"]["n_seeds"])))
    report = check(spec, roots, a.verify_every_split)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nall floors met: {report['all_floors_met']}   "
          f"regenerates identically: {report['all_regenerate_identically']}\nwrote {out}")
