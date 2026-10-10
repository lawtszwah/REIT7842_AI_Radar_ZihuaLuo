# ADR-0005 — The evaluation protocol is frozen at M2

**Status:** proposed, awaiting supervisor confirmation of the data grid · **Date:** 2026-10-11 ·
**Builds on:** ADR-0001, ADR-0003, ADR-0004

## Context

The proposal commits to freezing the protocol at M2, before the most flexible models are
tuned (Section 3.2, fourth commitment): "Deciding what counts as a good result after seeing
the results is the failure mode this project criticises, and freezing early is the control
against committing it." This ADR records what is frozen, the evidence that it is ready to
freeze, and the rule for changing it afterwards.

Three things had to be true first, and now are:

1. **The simulator is complete and validated.** `prdsim` implements all four components of
   `docs/simulator_model.md` (receiver noise, point targets, residual zero-Doppler ridge,
   extended clutter). Gates 1–3 of Section 8 pass, and the ridge and clutter, which no
   analytic gate covers, have their own tests with tolerances derived from exact per-cell
   statistics at z = 4 (`tests/test_prdsim_ridge.py`, `tests/test_prdsim_clutter.py`).
2. **The harness implements the protocol the proposal describes, not the M1 prototype.**
   Thresholds are calibrated on a target-free split and applied unchanged to every test
   split; seeds are spawned children of a recorded root rather than integer offsets; model
   selection uses its own validation split; the joint-shift grid is read from its own file
   and covered by the config hash; and intervals are map-clustered.
3. **The data splits are checked.** `scripts/check_splits.py` generated all six splits at
   all five root seeds (30 splits, 70,000 maps) and ran no detector. Every floored split
   meets 10^7 eligible background cells with at least 1.57× margin. No target lands in the
   zero-Doppler exclusion. Every split regenerates bit-identically from config and seed,
   and all 30 seeds and map hashes are distinct.

## Decision

At the tag `m2-protocol-freeze`, the following are fixed.

### Data

| | Train | Validation | Calibration | In-distribution | Extended | Joint shift |
|---|---:|---:|---:|---:|---:|---:|
| Maps | 4000 | 1000 | 1000 | 3000 | 3000 | 2000 |
| Targets per map | 1–3 | 1–3 | 0 | 1–3 | 1–3 | 1–3 |
| SNR (dB) | 0–16 | 0–16 | — | 0–16 | −4–16 | −6 to −1 |
| Noise power | 0.5–2.0 | 0.5–2.0 | 0.5–2.0 | 0.5–2.0 | 0.25–4.0 | 4.0–8.0 |
| Clutter patches | 0–6 | 0–6 | 0–6 | 0–6 | 0–10 | 6–12 |

All maps are 128 × 128. Ridge and clutter shape parameters take the `SimConfig` defaults,
which are the frozen settings of `simulator_model.md` Sections 5 and 6. The grids live in
`configs/protocol/full.yaml` and `configs/data/shift_test.yaml`; both files are frozen.

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

- `prdsim` is frozen at the version that is current at the tag. The Gate 1–3 records and
  the split check are re-run at that version and stored with the tag.
- The signal model is `docs/simulator_model.md` as of the tag, including Section 9 (claims
  the model does not support).

### Recorded at freeze

| Item | Value |
|---|---|
| Merged config SHA-256 (`full.yaml` + joint shift) | `7193eb3814f6bd6ad6b0c8f1ec9f4a8ce732c2baaf4c15f9effcd699a036e417`, **to be recomputed at the tag** |
| `shift_test.yaml` SHA-256 | `75cf20eefc4fbd8fbd262edb2ddaef382b7e41ce32c973df2e11c7ba006e3359` |
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

- **Target-free calibration** biases achieved test Pfa slightly *above* nominal, identically
  for every arm, because target sinc sidelobes reach cells beyond the ±3-bin exclusion
  (`simulator_model.md` Section 7).
- **Ridge leakage.** At high ridge level and short range, the ridge reaches the first
  eligible columns at d₀ ± 3. At L = 30 dB and range 0 it is about −2.3 dB relative to
  noise. Departure from design Pfa there is a benchmark result, not a simulator failure
  (Section 8).
- **SNR is signal-to-thermal-noise.** It excludes ridge and clutter, so it overstates
  detectability inside a clutter patch (Section 4.1). Clutter is recorded per map so that
  results can be stratified by it.

## Change control after the freeze

- Any change to a frozen item needs a new ADR that names the change and the reason, and
  then a re-run of **every** arm. No result produced before the change is reported beside
  results produced after it.
- A simulator change creates a new simulator version and a separate result set. It never
  amends the frozen one.
- Fixing a bug that changes no frozen value, the generated data or any score (for
  example, reporting code) needs no ADR, but is noted in the commit message.

## Open before acceptance

1. **Supervisor confirmation** of the data grid above, of `selection_pfa` = 10⁻³ (distinct
   from the 10⁻⁴ headline), and of the 95% interval level. This is the only M2 exit
   criterion that needs the supervisors.
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
