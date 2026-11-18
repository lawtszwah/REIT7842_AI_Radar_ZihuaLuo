# ADR-0005 — The evaluation protocol is frozen at M2

**Status:** accepted 2026-11-18, after the freeze run (see "Recorded at freeze") ·
**Date:** 2026-10-11, revised 2026-10-12 for the group's generator, 2026-11-17 to record
the decisions and 2026-11-18 to record the freeze run · **Builds on:** ADR-0001, ADR-0003,
ADR-0006

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
   SNR to effective SNR, which fixes the SNR range against ideal detection theory.
   `scripts/check_splits.py` has been run on the generator at all five root seeds
   (2026-10-12 and 2026-10-13): all 35 split × root rows meet the 10⁷-cell floor, no
   target is lost to the zero-Doppler mask, and the calibration split regenerates
   bit-identically at every root. The tightest margin is the validation split at root 4,
   10,483,464 eligible cells, about 4.8% above the floor. The freeze run repeated the check
   at adapter 1.0.0 and reproduced every split bit for bit ("Recorded at freeze"). Details
   are in `docs/validation_report.md`.

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
- Tuning budget: `protocol.tuning_budget` = 36 trials per detector, identical for the CFAR
  baselines and the learned arms (RSK-03). 36 is the size of the largest frozen search
  space, so every arm is searched exhaustively. `prdbench tune` reads the budget from the
  config and refuses a different `--budget`.
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
  string, with adapter version 1.0.0. The truth calibration test and the split check are
  re-run at that version and stored with the tag.
- `prdsim` and its Gate 1–3 records are frozen alongside it as the reference simulator,
  at version 1.0.0. It is never a source of reported results.

### Environment

- Reported runs use Python 3.11, the version in the proposal and in CI. The interpreter
  and library versions are recorded in every `results.json`.

### Recorded at freeze

| Item | Value |
|---|---|
| Merged config SHA-256 (`full.yaml` + joint shift) | `decdc6d1259f65db8a9dd1899c4808e2c72da73f9f6e2b1b6e3711ab7017c62f` |
| `shift_test.yaml` SHA-256 | `8d6448866f2f1ab77454b0d5e7e6973b548d51725c7fc436a9735ffacada9eaf` |
| `full.yaml` SHA-256 (file bytes) | `930933cce4faa3c896df6779b23882ab8ec4cfefd4f5fa0d64ac47d778d638d6` |
| Generator fingerprint | `567f5e55df8a0c99a1a08f663de242f8cbc51315a049108c720d8707e26de4b4` |
| Simulator version | `groupgen` `1.0.0+gen.567f5e55df8a`; `prdsim` `1.0.0` |
| Commit | Freeze run at `d8a0878`, Python 3.11.16; the tag `m2-protocol-freeze` marks the commit that records this table, which changes only documentation |

The freeze run (2026-11-17 21:01 to 2026-11-18 09:05, macOS, MATLAB R2024a):

- `pytest -m matlab`: 3 passed (output contract, truth calibration, seed determinism).
- `scripts/check_splits.py`, root seeds 0–4: all 35 split × root rows meet the 10⁷ floor, no
  target is excluded by the zero-Doppler mask, and the calibration split regenerates
  bit-identically at every root.
- The version bump changed every cache key, so all 7,900 scenes were generated afresh.
  For all 35 rows, `maps_sha256`, the materialised seed, the eligible-cell count and the
  target counts equal the October run's (245 of 245 fields). Every split at every root
  is therefore reproduced bit for bit from its seed, not only the calibration split.

Record: `results/validation/split_check_groupgen_freeze.{json,log}` and
`freeze_matlab_tests.log`, summarised in `docs/validation_report.md`.

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
  reported against SNR_dB with the effective-SNR axis given alongside, so a reader can
  compare them with detection theory.
- **Background level varies between maps.** Within one map the background is exponential
  (per-map normalised CV 0.9996), but the map's mean level rises with CNR, spanning
  −0.5 to +2.1 dB across 200 maps (correlation with CNR 0.90). Pooled over maps, Gate 1
  therefore fails. CFAR estimates the level locally and adapts to this; a detector with
  one fixed threshold across maps does not, and may lose Pd or exceed its Pfa on maps at
  the ends of the CNR range. This is part of the problem the benchmark poses, not
  something to correct, and achieved Pfa is reported on every split.
- **Weak correlation between neighbouring range cells.** After per-map normalisation the
  lag-1 range correlation is 0.0096, above the 0.0046 tolerance for independent cells;
  Doppler (−0.002) is within it. A correlation this small changes CA-CFAR's effective
  number of training cells only slightly, and Gate 3 shows CA-CFAR meeting its design Pfa
  at all four points on this background.
- **Clutter scales with the target.** The generator sets clutter relative to the target
  (CNR = SNR + CSR). Calibration maps draw a reference SNR so that their clutter level is
  distributed as in the test splits.
- **Placement favours longer ranges.** Start positions are uniform over an area.
- **Monostatic Doppler.** The generator's Doppler follows 2v·u_Rx/λ while its delay is
  bistatic (ADR-0006). Truth follows the generator, matching the observed peak to
  0.01 ± 0.11 Hz, so every detector is scored against the Doppler the map actually
  contains. The inconsistency limits how far the maps stand for a real bistatic
  geometry; it does not favour one detector over another. It is reported as a threat to
  validity.

## Change control after the freeze

- Any change to a frozen item needs a new ADR that names the change and the reason, and
  then a re-run of **every** arm. No result produced before the change is reported beside
  results produced after it.
- A simulator change creates a new simulator version and a separate result set. It never
  amends the frozen one.
- Fixing a bug that changes no frozen value, the generated data or any score (for
  example, reporting code) needs no ADR, but is noted in the commit message.

## Decisions taken before acceptance

These were open in the earlier drafts. They are decided by the author, on 2026-11-17, each
with the reason recorded here.

1. **Data grid: kept as above.** It was set from measurements of the generator
   (`docs/data_grid.md`), and every split at every root seed already meets its floor.
2. **Doppler grid: 10 Hz, not the interface script's 1 Hz.** A 0.1 s frame resolves 10 Hz.
   On the 1 Hz grid neighbouring Doppler cells correlate at 0.97, which breaks CA-CFAR's
   assumption of independent training cells; on the 10 Hz grid the correlation is 0.001.
   Generation is also about 15 times faster.
3. **Monostatic Doppler: accepted as the generator's convention.** Truth follows the
   generator; recorded above as a documented bias and a threat to validity.
4. **SNR_dB: the generator's own definition, target relative to thermal noise.** The
   measured mapping to effective SNR is recorded above, and results carry both axes.
5. **`selection_pfa` = 10⁻³, distinct from the 10⁻⁴ headline.** On a validation split of
   about 1.05 × 10⁷ cells, 10⁻³ gives about 10,500 false alarms against about 1,050 at
   10⁻⁴, so the threshold that ranks configurations is ten times better determined.
   Selecting at the headline point would also tie selection to the number the thesis
   reports; selecting one decade away keeps them apart.
6. **Intervals: 95%, map-clustered percentile bootstrap.** The conventional level; the
   map is the unit of resampling because cells within a map share a background level.
7. **Tuning budget: 36.** The value 12 was inherited from an old CLI default and
   subsampled every search space. 36 is the size of the largest frozen space (OS-CFAR);
   CA- and GO-CFAR (20), GBDT (27) and the CNN (18) are then also searched exhaustively.
   No arm's result depends on which trials a random subsample drew, which is the
   strongest form of the equal-effort control in RSK-03. The cost is at most 18 CNN
   trainings per root seed.
8. **Attention arm: not frozen now.** It is still a stub. Its search space is added in M4
   under its own ADR, inside the same budget of 36 and the same selection rule.
9. **Python 3.11** for every reported run, as in the proposal and CI (see Environment).
10. **Versions at the tag: 1.0.0** for the `groupgen` adapter and for `prdsim`, to mark the
    frozen models. Bumping changes no generated data but changes `meta`.

### Steps before the tag (done)

- The adapter and `prdsim` were bumped to 1.0.0.
- Under Python 3.11, `pytest -m matlab` and `scripts/check_splits.py` at root seeds 0–4
  were re-run (the budget and the joint-shift comment had changed both config hashes).
- The "Recorded at freeze" table is filled and this ADR is accepted. The tag
  `m2-protocol-freeze` is set once CI passes.

## Consequences

- From the tag on, every reported number is traceable to one config hash, one simulator
  version, one commit and a recorded root seed.
- M3 can start: CFAR, GBDT and CNN tuning under the frozen selection rule.
- The project gives up the option of adjusting splits, metrics or operating points after
  seeing detector results. That is the point of the freeze.
