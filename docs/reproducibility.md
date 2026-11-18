# Reproducibility workflow

A result in this project is a triple: **(config file, seed, commit)**.

```
config + seed  ──▶  dataset            (regenerated, never stored)
      +
   commit      ──▶  detector code      (pinned by git)
      +
  protocol.py  ──▶  results.json       (stored, with provenance)
      │
      └────────▶  figures              (stored, one script)
```

## What is recorded in every `results.json`

* git commit hash of the working tree that produced it
* Python, NumPy versions and platform string
* simulator name and version (`stub` vs `prdsim`) — `stub` results are flagged unusable
* the complete spec: data config, detector configs, protocol config
* per-seed results, not just the aggregate, so variance is inspectable

## Seed discipline

* Training, validation, calibration, test, analysis-bootstrap and model-initialisation
  streams are named children spawned from each run's recorded root seed. Their frozen
  order is declared in `configs/protocol/full.yaml`; integer offsets are not used.
* Analytical simulator validation has its own two predeclared roots and named child
  streams under `simulator_validation`. The second root is reserved for Gate 3's single
  permitted repeat and is never selected after inspecting results.
* `n_seeds: 5` for reported results; repeated-run standard deviations accompany every number.

## Re-running everything

```bash
git checkout <commit>
pip install -e ".[dev]"
# one tuning record per detector, at root seed 0
prdbench tune --config configs/protocol/full.yaml --detector ca_cfar --out results/m3/tune/ca_cfar_root0.json
./scripts/run_benchmark.sh configs/protocol/full.yaml results/rerun \
    --tuned results/m3/tune --detectors ca_cfar os_cfar go_cfar feat_gbdt cnn
```

The frozen config fixes `protocol.tuning_budget` and names no hyperparameters, so `run`
refuses it without `--tuned`: a reported run takes its hyperparameters only from
`prdbench tune`, never from constructor defaults. Each record is checked before any data
are generated (`tuning/selection.py`). It must carry the run's config hash, root seed 0,
the frozen budget and selection Pfa, and a clean commit (`git_dirty: false`). Every trial
must lie in the frozen search space, and a space no larger than the budget must appear
exactly once per configuration. Its `best` must be the trial the selection rule picks, and
a learned arm's seed must be root 0's model seed. Each record's path, SHA-256, commit,
device and validation Pd are written to `results.json` under `run.tuned`, beside the
unchanged config hash. `--detectors` runs a subset of the arms; maps and bootstrap indices
depend only on the root seed, so arms run separately, on different machines, stay paired.

## What would make a result non-reproducible, and the control for it

| Threat | Control |
|---|---|
| Config edited after a run | Config is snapshotted into `results.json`, not just referenced |
| Library version drift | Versions recorded per run; `pyproject.toml` pins minimums |
| Undeclared GPU non-determinism | Default device is CPU; any GPU run is labelled and repeated |
| Protocol changed mid-project | Protocol frozen at M2; changes require an ADR and a full re-run |
| Figures redrawn by hand | One script, `scripts/make_figures.py`, from one `results.json` |
