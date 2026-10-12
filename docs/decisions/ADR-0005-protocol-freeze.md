# ADR-0005 — The evaluation protocol is frozen at M2

**Status:** proposed, awaiting supervisor confirmation of the data grid · **Date:** 2026-10-11,
revised 2026-10-12 for the group's generator · **Builds on:** ADR-0001, ADR-0003, ADR-0006

## Context

The proposal commits to freezing the protocol at M2, before the most flexible models are
tuned (Section 3.2, fourth commitment): "Deciding what counts as a good result after seeing
the results is the failure mode this project criticises, and freezing early is the control
against committing it." This ADR records what is frozen, the evidence that it is ready to
freeze, and the rule for changing it afterwards.

Three things had to be true first, and now are:

1. **The data source is integrated and its ground truth is calibrated.** The research
   group's generator is reached through `groupgen` (ADR-0006). Its truth conventions were
   measured, and a `matlab`-marked test checks on every run that recorded truth is within
   one cell of the target peak. The in-project `prdsim` passes Gates 1–3 and stays as a
   reference for testing the evaluation code.
2. **The harness implements the protocol the proposal describes, not the M1 prototype.**
   Thresholds are calibrated on a target-free split and applied unchanged to every test
   split; seeds are spawned children of a recorded root rather than integer offsets; model
   selection uses its own validation split; the joint-shift grid is read from its own file
   and covered by the config hash; and intervals are map-clustered.
3. **The grid is set from measurements of the generator** (`docs/data_grid.md`): background
   statistics, the reach of the zero-Doppler ridge, and the mapping from the generator's
   SNR to effective SNR, which fixes the SNR range against ideal detection theory. *Still
   to do before the tag:* run `scripts/check_splits.py` on the generator at all five root
   seeds.

## Decision

At the tag `m2-protocol-freeze`, the following are fixed.

### Data

| | Train | Validation | Calibration | In-distribution | Extended | Joint shift | Grid shift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Maps (scenes) | 4000 (400) | 1400 (140) | 1400 (140) | 3000 (300) | 3000 (300) | 2000 (200) | 1000 (100) |
| Targets per map | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| SNR_dB | −10 to 10 | −10 to 10 | (reference) | −10 to 10 | −14 to 14 | −3 to 8 | −10 to 10 |
| DNR_dB | 45 | 45 | 45 | 45 | 40–50 | 48–52 | 45 |
| CSR_dB | 20–30 | 20–30 | 20–30 | 20–30 | 15–35 | 30–40 | 20–30 |
| Speed (km/h) | 200–600 | 200–600 | 200–600 | 200–600 | 150–700 | 600–850 | 200–600 |
| Map (range × Doppler) | 99 × 81 | 99 × 81 | 99 × 81 | 99 × 81 | 99 × 81 | 99 × 81 | 199 × 121 |

Range cells are 150 m and Doppler cells are 10 Hz (−400:10:400 Hz; ±600 Hz for grid shift).
One scene is one ten-frame trajectory and never straddles splits. The rationale and the
measurements are in `docs/data_grid.md`. The grids live in `configs/protocol/full.yaml` and
`configs/data/shift_test.yaml`; both files are frozen.

### Metrics and operating points

- Pd per target, within ±1 bin; Pfa per eligible background cell. Cells within ±3 bins of
  a target and the five columns centred on zero Doppler are excluded from every count
  (ADR-0003).
- Design Pfa points 10⁻⁵, 10⁻⁴, 10⁻³ and 10⁻²; the headline figure is 10⁻⁴.
- Thresholds: `np.quantile(..., method="higher")` of the eligible calibration background;
  detection is strictly `score > τ`. Each threshold is selected once, recorded, and never
  adjusted on test data.
- Model selection: Pd on the validation split at `selection_pfa` = 10⁻³, with the
  threshold set from the validation background by the same rule. Selection uses neither
  the calibration split nor any test split.
- Tuning budget: `protocol.tuning_budget` = 12 trials per detector, identical for the CFAR
  baselines and the learned arms (RSK-03). `prdbench tune` reads it from the config and
  refuses a different `--budget`.
- Every calibration, validation and test split must hold at least 10⁷ eligible background
  cells per run, or the run is invalid.
- Intervals: 95% percentile bootstrap over whole maps, 10,000 resamples, with one set of
  resample indices per run and split shared by every detector.

### Seeds

- Five runs, root seeds 0–4.
- Streams are `SeedSequence(root).spawn(...)` children, in the order listed in
  `protocol.seed_streams`. **The order is frozen.** A name may be appended at the end;
  inserting, removing or reordering one invalidates every run made before the change.

### Simulator

- The group's generator, identified by the SHA-256 of its files in the `groupgen` version
  string, with adapter version 0.1.0 or the version current at the tag. The truth
  calibration test and the split check are re-run at that version and stored with the tag.
- `prdsim` and its Gate 1–3 records are frozen alongside it as the reference simulator.
  It is never a source of reported results.

### Recorded at freeze

| Item | Value |
|---|---|
| Merged config SHA-256 (`full.yaml` + joint shift) | *at tag* |
| `shift_test.yaml` SHA-256 | *at tag* |
| Generator fingerprint | *at tag* (currently `567f5e55df8a…`) |
| Simulator version | *at tag* |
| Commit | *at tag* |

## What is not frozen

- **Detector hyperparameters.** These are chosen in M3 by the frozen selection rule,
  inside the frozen search spaces and budget.
- **The attention arm's architecture.** It is added in M4 under the same selection rule
  and budget. If it is dropped, that is reported as a scope reduction (RSK-09).
- **Analysis and presentation**, such as which figures are drawn, provided every number
  comes from a `results.json` produced under the frozen protocol.

## Known, documented biases (not defects)

These are properties of the frozen construction. They are stated here so that they are
not discovered later and mistaken for detector behaviour:

- **Target-free calibration** may bias achieved test Pfa slightly *above* nominal,
  identically for every arm, because energy from the target can reach cells beyond the
  ±3-cell exclusion. Achieved Pfa is reported on every split, so this stays visible.
- **The generator's SNR_dB is not what limits detection.** The background is set by the
  direct path's cross-ambiguity sidelobes, and effective SNR ≈ SNR_dB − DNR_dB + 50.7 dB
  (`docs/data_grid.md`). DNR is held at 45 dB except in the shift tests; results are
  reported against SNR_dB, with the mapping stated.
- **Clutter scales with the target.** The generator sets clutter relative to the target
  (CNR = SNR + CSR). Calibration maps draw a reference SNR so that their clutter level is
  distributed as in the test splits.
- **Placement favours longer ranges.** Start positions are uniform over an area.
- **Monostatic Doppler.** The generator's Doppler follows 2v·u_Rx/λ while its delay is
  bistatic (ADR-0006). Truth follows the generator.

## Change control after the freeze

- Any change to a frozen item needs a new ADR that names the change and the reason, and
  then a re-run of **every** arm. No result produced before the change is reported beside
  results produced after it.
- A simulator change creates a new simulator version and a separate result set. It never
  amends the frozen one.
- Fixing a bug that changes no frozen value, the generated data or any score (for
  example, reporting code) needs no ADR, but is noted in the commit message.

## Open before acceptance

1. **Supervisor confirmation** of the data grid above, of the 10 Hz Doppler grid, of
   `selection_pfa` = 10⁻³ (distinct from the 10⁻⁴ headline), and of the 95% interval
   level. This is the only M2 exit criterion that needs the supervisors.
2. **Tuning budget value.** The budget is now `protocol.tuning_budget` in the config and
   cannot be overridden from the command line. The value 12 is carried over from the old
   CLI default and has not been argued for. With 12 trials, every current search space is
   subsampled: CA- and GO-CFAR have 20 configurations, OS-CFAR 36, GBDT 27, the CNN 18.
   Confirm 12, or choose a value with a reason.
3. **Search spaces.** `search_spaces` covers CA/OS/GO-CFAR, GBDT and the CNN, but not the
   attention arm. Decide whether the attention space is frozen now or added under its own
   ADR in M4.
4. **Python version.** The proposal says 3.11, CI runs 3.11, and local development runs
   3.13 (python.org build). The Anaconda interpreter on the development machine segfaults
   in scikit-learn when torch is also loaded. Record one supported interpreter.
5. **Simulator version at the tag.** Either keep 0.4.0 or bump to 1.0.0 to mark the frozen
   model. Bumping changes no generated data, but it does change `meta`, so the Gate
   records and the split check must be re-run at the tagged version in either case.

## Consequences

- From the tag on, every reported number is traceable to one config hash, one simulator
  version, one commit and a recorded root seed.
- M3 can start: CFAR, GBDT and CNN tuning under the frozen selection rule.
- The project gives up the option of adjusting splits, metrics or operating points after
  seeing detector results. That is the point of the freeze.
