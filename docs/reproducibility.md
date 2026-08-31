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
* simulator name and version (`stub` vs `group`) — stub results are flagged unusable
* the complete spec: data config, detector configs, protocol config
* per-seed results, not just the aggregate, so variance is inspectable

## Seed discipline

* Training data, test data and model initialisation draw from separate, derived seeds
  (`seed`, `1000 + seed`, and the detector's own `seed`), so a "different seed" changes
  data *and* initialisation together — which is what the reported standard deviation
  should cover.
* Validation data for hyperparameter search uses `5000 + seed`, disjoint from both.
* `n_seeds: 5` for reported results; repeated-run standard deviations accompany every number.

## Re-running everything

```bash
git checkout <commit>
pip install -e ".[dev]"
./scripts/run_benchmark.sh configs/protocol/full.yaml results/rerun
```

## What would make a result non-reproducible, and the control for it

| Threat | Control |
|---|---|
| Config edited after a run | Config is snapshotted into `results.json`, not just referenced |
| Library version drift | Versions recorded per run; `pyproject.toml` pins minimums |
| Undeclared GPU non-determinism | Default device is CPU; any GPU run is labelled and repeated |
| Protocol changed mid-project | Protocol frozen at M2; changes require an ADR and a full re-run |
| Figures redrawn by hand | One script, `scripts/make_figures.py`, from one `results.json` |
