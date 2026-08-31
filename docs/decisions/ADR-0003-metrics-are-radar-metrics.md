# ADR-0003 — Pd is per target, Pfa is per cell, and zero Doppler is guarded

**Status:** accepted · **Date:** 2026-08-31

## Context

Per-cell classification accuracy on a range-Doppler map is close to meaningless: targets
occupy a handful of cells out of ~16,000, so predicting "no target" everywhere scores
above 99.9%. The literature's use of balanced accuracy is the flaw the project is
measuring against.

## Decision

* **Pd is per target.** A target counts as detected if any cell within `tol_bins` (1 bin) of
  its true position exceeds the threshold. This tolerates straddle loss, which is a
  property of the map, not of the detector.
* **Pfa is per cell**, counted over background cells only, with an exclusion annulus of
  `exclude_bins` (3 bins) around every true target so a near-miss is neither a hit nor a
  false alarm.
* **Zero Doppler is guarded.** Columns within `zero_doppler_guard` (2 bins) of zero Doppler
  are removed from both counts. Direct-path leakage dominates there and is removed by
  clutter cancellation before any detector runs in a real passive-radar chain; leaving it
  in would let leakage strength, not detector quality, set the threshold at every Pfa.
  Targets inside the guard are excluded rather than counted as misses.

## Consequences

* Reported Pfa is per resolution cell, so it converts directly to false alarms per map and
  per second — the number a system designer actually needs.
* The guard is a parameter, recorded in every `results.json`, so its effect can be shown by
  re-running with it disabled rather than argued about.
* Comparison with published numbers requires care: papers reporting per-cell accuracy are
  not directly comparable, and the thesis says so rather than tabulating them together.
