"""Read back the hyperparameters `prdbench tune` selected, for `prdbench run`.

The frozen config names no hyperparameters: they are chosen in M3 by the frozen
selection rule (ADR-0005). A run therefore takes them from the tuning records, and
only after checking that each record is one the frozen protocol could have produced:
the same merged config, a clean commit, root seed 0, the configured budget, the full
trial list, and a best trial that really is the first one with the highest Pd.

The model seed is not taken from the record. A run draws it per root seed from the
`model_initialization` stream, exactly as before; at root 0 that is the seed the
record shows, which is checked here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..detectors import REGISTRY
from ..evaluation.protocol import ProtocolConfig, config_sha256
from ..evaluation.seeds import make_streams, stream_record
from .search import grid

TUNING_ROOT_SEED = 0


def record_path(directory: Path, key: str) -> Path:
    return Path(directory) / f"{key}_root{TUNING_ROOT_SEED}.json"


def load_selections(spec: dict, directory: Path, keys: list[str]) -> dict[str, dict]:
    """Selected parameters and their source for each detector key, or a ValueError.

    `spec` must be the merged spec the run will use, so that its hash is the one the
    tuning records were made under.
    """
    proto = ProtocolConfig(**spec.get("protocol", {}))
    entries = {entry.get("key", entry["kind"]): entry for entry in spec["detectors"]}
    expected_hash = config_sha256(spec)
    selections = {}
    for key in keys:
        entry = entries[key]
        kind = entry["kind"]
        path = record_path(directory, key)
        if not path.is_file():
            raise ValueError(f"{key}: no tuning record at {path}")
        raw = path.read_bytes()
        record = json.loads(raw)

        def check(ok: bool, what: str, key=key, path=path) -> None:
            if not ok:
                raise ValueError(f"{key}: tuning record {path} {what}")

        run = record.get("run", {})
        check(record.get("detector") == kind, f"is for {record.get('detector')!r}, not {kind!r}")
        check(record.get("root_seed") == TUNING_ROOT_SEED,
              f"is for root seed {record.get('root_seed')}, not {TUNING_ROOT_SEED}")
        check(run.get("config_sha256") == expected_hash,
              f"was made under config {run.get('config_sha256')}, not {expected_hash}")
        check(run.get("git_dirty") is False, "was made from a commit with uncommitted changes")
        check(record.get("selection_pfa") == proto.selection_pfa,
              f"selected at Pfa {record.get('selection_pfa')}, not {proto.selection_pfa}")

        space = spec.get("search_spaces", {}).get(key)
        check(space is not None, "has no search space in the config")
        budget = proto.tuning_budget
        check(budget is not None and record.get("budget") == budget,
              f"used a budget of {record.get('budget')}, not {budget}")
        n_trials = min(len(grid(space)), budget)
        trials = record.get("all", [])
        check(record.get("n_trials") == n_trials and len(trials) == n_trials,
              f"holds {len(trials)} trials, expected {n_trials}")
        best = max(trials, key=lambda trial: trial["pd"])
        check(record.get("best") == best, "names a best trial that is not the first highest Pd")

        params = dict(best["params"])
        trainable = REGISTRY[kind].trainable
        if trainable:
            model = stream_record(
                make_streams(TUNING_ROOT_SEED, proto.seed_streams,
                             {"model_initialization"})["model_initialization"],
                model=True,
            )
            check(params.pop("seed", None) == model["materialised_seed"],
                  "has a model seed that is not root 0's model_initialization seed")
        check(set(params) == set(space),
              f"selects parameters {sorted(params)}, not {sorted(space)}")
        for name, value in params.items():
            check(value in space[name], f"selects {name}={value!r}, outside {space[name]}")
        # Every trial, not only the best, must come from the frozen space; when the budget
        # covers the space, the record must hold each configuration exactly once.
        tried = [{k: v for k, v in trial["params"].items() if k != "seed"} for trial in trials]
        for t in tried:
            check(set(t) == set(space) and all(t[k] in space[k] for k in space),
                  f"holds a trial {t} outside the frozen search space")
        if len(grid(space)) <= budget:
            canonical = sorted(json.dumps(t, sort_keys=True) for t in tried)
            check(canonical == sorted(json.dumps(t, sort_keys=True) for t in grid(space)),
                  "does not hold every configuration of the search space exactly once")

        selections[key] = {
            "params": params,
            "source": {
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "git_commit": run.get("git_commit"),
                "simulator_version": run.get("simulator_version"),
                "device": run.get("device"),
                "validation_pd": best["pd"],
            },
        }
    return selections
