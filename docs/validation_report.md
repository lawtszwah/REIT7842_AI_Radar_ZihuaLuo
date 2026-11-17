# Validation report: the group's generator and the protocol splits

**Status:** pre-freeze record · **Date:** 2026-11-17 · **Generator:** `groupgen`
`0.1.0+gen.567f5e55df8a` · **Builds on:** ADR-0005, ADR-0006, `docs/data_grid.md`

`results/` is not under version control, so this file holds the numbers the protocol freeze
relies on. The raw records are `results/validation/groupgen_gates.json` (written by
`scripts/validate_groupgen.py`) and `results/validation/split_check_groupgen_root0.json` and
`..._roots1-4.json` (written by `scripts/check_splits.py`). Both are re-run at the tagged
version before the freeze (ADR-0005, "Remaining before the tag"), and this report is
updated with that run.

No detector other than CA-CFAR at a fixed design point is run here. Nothing below is a
benchmark result.

## 1. Gate 1: the target-free background

200 target-free frames (DNR 45 dB, CSR 20–30 dB, reference SNR −10 to 10 dB), 1,504,800
eligible cells, zero-Doppler guard excluded. Tolerances are z = 4 standard errors.

| Check | Pooled over maps | Each map normalised by its own mean | Exponential law |
|---|---:|---:|---:|
| Coefficient of variation | 1.028 ✗ | 0.9996 ✓ | 1 |
| Median | 0.683 ✗ | 0.6935 ✓ | 0.6931 |
| 0.9 quantile | 2.307 ✓ | 2.3015 ✓ | 2.3026 |
| 0.99 quantile | 4.777 ✗ | 4.6115 ✓ | 4.6052 |
| 0.999 quantile | 7.471 ✗ | 6.9145 ✓ | 6.9078 |
| Lag-1 correlation, range (tol. 0.0046) | 0.044 ✗ | 0.0096 ✗ | 0 |
| Lag-1 correlation, Doppler (tol. 0.0046) | 0.026 ✗ | −0.0020 ✓ | 0 |

**Verdict: fails as recorded** (`passed: false`), for two reasons, both stated in ADR-0005
as documented biases:

- Within a map the background is exponential. Across maps its level is not constant: the
  mean level spans −0.5 to +2.1 dB about the median and correlates with CNR at r = 0.90,
  because the generator sets clutter relative to the target (CNR = SNR + CSR). This is
  what makes the pooled distribution heavier-tailed than exponential.
- After per-map normalisation, neighbouring range cells correlate at 0.0096, about twice
  the tolerance for independent cells. Doppler is within tolerance.

## 2. Gate 2: the target peak against its true position

400 frames at SNR_dB 10, DNR 45 dB. The target-cell SNR, relative to a target centred in
its cell, binned by the fractional offset of the recorded true position:

| Offset (cells) | Doppler loss (dB) | sinc², 0.1 s window (dB) | Range loss (dB) |
|---:|---:|---:|---:|
| 0.05 | −0.09 | −0.04 | −0.24 |
| 0.15 | −0.20 | −0.32 | −0.51 |
| 0.25 | −1.00 | −0.91 | −0.78 |
| 0.35 | −1.96 | −1.83 | −1.99 |
| 0.45 | −3.18 | −3.11 | −3.10 |

**Verdict: passes.** Every Doppler bin lies within its z = 4 band of the sinc² loss of a
rectangular 0.1 s window. Range has no closed-form reference and is shown for comparison
only; it falls off monotonically to about −3 dB at half a cell, as expected. Together with
the truth-calibration test (`pytest -m matlab`, recorded truth within one cell of the
observed peak; Doppler agreement 0.01 ± 0.11 Hz, ADR-0006), this shows that the recorded
truth is where the target energy actually is.

## 3. Gate 3: CA-CFAR's false-alarm rate on target-free maps

CA-CFAR with 4 training and 2 guard cells (144 training cells), on 2220 target-free maps,
10,043,280 eligible cells, with every CFAR window kept clear of the zero-Doppler guard.
Intervals are 99% map-clustered bootstrap intervals, 10,000 resamples.

| Design Pfa | Measured Pfa | False alarms | 99% interval | |
|---:|---:|---:|---|---|
| 10⁻⁵ | 1.016 × 10⁻⁵ | 102 | 7.67 × 10⁻⁶ – 1.27 × 10⁻⁵ | ✓ |
| 10⁻⁴ | 9.52 × 10⁻⁵ | 956 | 8.74 × 10⁻⁵ – 1.03 × 10⁻⁴ | ✓ |
| 10⁻³ | 1.001 × 10⁻³ | 10,054 | 9.75 × 10⁻⁴ – 1.03 × 10⁻³ | ✓ |
| 10⁻² | 9.97 × 10⁻³ | 100,092 | 9.89 × 10⁻³ – 1.004 × 10⁻² | ✓ |

**Verdict: passes at all four design points.** The between-map level variation and the weak
range correlation found by Gate 1 do not move CA-CFAR off its design Pfa, which is the
property the baseline needs.

## 4. Split check at all five root seeds

`scripts/check_splits.py` on the generator, root seeds 0 (2026-10-12) and 1–4 (finished
2026-10-13), all under one config hash (`98445044…adae5`, joint shift `60b4c0e8…d6eb`).

| Split | Maps | Eligible background cells, min over roots | Ratio to the 10⁷ floor | Tightest root |
|---|---:|---:|---:|---:|
| Train | 4000 | 29,952,864 | (no floor) | 3 |
| Validation | 1400 | 10,483,464 | 1.048 | 4 |
| Calibration | 1400 | 10,533,600 | 1.053 | all equal |
| In-distribution test | 3000 | 22,464,564 | 2.246 | 3 |
| Extended test | 3000 | 22,464,858 | 2.246 | 1 |
| Joint shift | 2000 | 14,976,072 | 1.498 | 1 |
| Grid shift | 1000 | 23,048,120 | 2.305 | 2 |

**Verdict: passes.** All 35 split × root rows meet the floor. No target is excluded by the
zero-Doppler mask in any split. The calibration split was regenerated and is bit-identical
at every root seed; the other six splits were not regenerated (the script's default). Their
reproducibility rests on the cache being keyed by each scene's full specification and on the
separate check (2026-10-12) that MATLAB returns bit-identical signals for a given seed.

Generation took 2.31–2.55 hours per root seed, 12.1 hours for all five.

## 5. The reference simulator

`prdsim` 0.4.0 passes Gates 1–3 (`results/validation/gate3_report.json` and the Gate 1 and 2
figures). It is used only to test the evaluation code and is never a source of reported
results.
