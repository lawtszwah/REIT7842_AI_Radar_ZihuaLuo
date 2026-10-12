# ADR-0006 — Thesis data come from the research group's generator

**Status:** accepted · **Date:** 2026-10-12 · **Supersedes in part:** ADR-0004 ·
**Affects:** ADR-0005 (draft)

## Context

ADR-0004 recorded that the research group would not supply a simulator, and on that basis
the project built its own, `prdsim`. That premise was a misreading. At the 9 September
meeting the instruction "generate the simulator data by yourself" meant *run the group's
generator yourself*, not *write a simulator*. The generator, by Dr Lu Zhang, was supplied
on 10 September, but the decision of ADR-0004 had already been taken and was not revisited.
The co-supervisor's reply to the data-grid email on 12 October made the misreading plain:
the grid used parameters (noise power, clutter patches) that the group's generator does not
have.

The generator is MATLAB: three student interface scripts around protected P-code cores. A
trajectory core produces one target's positions. A signal core produces two-channel
reference and surveillance signals (2 MHz sampling, 220 MHz carrier, fixed transmitter and
receiver positions; SNR, clutter-to-signal and direct-path-to-noise ratios; zero-Doppler
clutter). A processing core produces one range-Doppler frame per 0.1 s.

## Decision

1. **The group's generator is the data source for every reported result.** It is reached
   through `src/prdbench/simulation/groupgen.py` (simulator name `groupgen`), which calls the
   cores directly from `matlab/prd_groupgen.m`. The interface scripts call `clear` and use
   fixed file names, so they cannot be driven programmatically.
2. **The generator stays outside this repository** until the group confirms it may be
   published. The adapter finds it with `PRDBENCH_GENERATOR_DIR`, and records a SHA-256 of
   its files in every dataset's version string.
3. **Ground truth follows the generator's behaviour, as measured** on 2026-10-12, over six
   strong-target scenes in different geometries (60 frames):
   * range: the peak sits at delay + 1 on the `gridx` axis, where delay = (|T − Tx| +
     |T − Rx| − |Tx − Rx|) · fs / c, because `cut` is columns 2 to 100 of the full delay
     axis (mean offset +1.01, s.d. 0.20 cells);
   * Doppler: the peak follows 2 v · u_Rx / λ, the *monostatic* form about the receiver,
     to within 0.01 ± 0.11 Hz. The bistatic form v · (u_Tx + u_Rx) / λ misses by up to
     13 Hz. Delay is bistatic and Doppler is monostatic: this inconsistency belongs to the
     generator, and has been raised with the supervisors rather than corrected here.
   A `matlab`-marked test checks, on every run with MATLAB available, that the recorded
   truth is within one cell of the strongest off-ridge peak at 20 dB.
4. **The Doppler grid is −400:10:400 Hz, not −250:1:400.** The interface script's 1 Hz step
   oversamples the 10 Hz resolution of a 0.1 s frame tenfold: background correlation
   between neighbouring Doppler cells is 0.97. CA-CFAR's design Pfa assumes independent
   training cells, and the 1 Hz grid is also 15 times slower. The grid is symmetric so that
   zero Doppler is at index n_doppler // 2, the convention the metrics already use. Both
   choices are to be confirmed with the supervisors.
5. **One scene is one trajectory, giving ten frames, and scenes never straddle splits.**
   Frames of a trajectory are strongly dependent. Each split draws its own scenes from its
   own seed stream.
6. **`prdsim` is kept as a reference simulator.** It is not a data source for reported
   results. Its analytic gates still test the evaluation code and the validation method.

## Consequences

* **The harness is unchanged.** Calibration, seeds, splits, intervals and metrics never
  depended on the data source; only the simulator name and the grid in the protocol change.
* **The data grid must be redesigned** around the generator's own parameters (SNR, CSR,
  DNR, target speed and heading, map size). ADR-0005 stays a draft until that grid is
  confirmed, and M2 moves from 31 October into its buffer, to 30 November.
* **Author circularity is largely gone.** The generator was not written by the author of
  the detectors, so the main validity risk of ADR-0004 (old RSK-02) no longer applies.
* **Cost.** About 5.5 s per ten-frame scene on the 10 Hz grid (2 s for the signals, 3.5 s for
  processing), plus about 10 s of MATLAB start-up per batch. At the old split sizes
  (14,000 maps per root seed) that is roughly two hours per root seed, before any caching.
* **CI cannot run the generator.** The generator tests skip there; the adapter's planning,
  placement and truth formulas are pure Python and run everywhere.
