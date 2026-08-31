# ADR-0001 — Detectors return a per-cell score, not a decision

**Status:** accepted · **Date:** 2026-08-31 · **Supersedes:** —

## Context

The project compares CFAR detectors with machine-learning detectors. CFAR is normally
described as producing a binary decision at a designed false-alarm rate; classifiers are
normally reported with accuracy or F1 on a balanced set. Comparing them directly in
either of those forms is what makes the published literature incomparable — the CFAR
detector is evaluated at one operating point chosen by its designer, and the classifier at
one chosen by a 0.5 probability cut.

## Decision

`Detector.score(maps) -> (N, R, D)` returns a continuous statistic, higher meaning more
target-like. No detector applies a threshold. The evaluation harness sweeps a threshold
over the empirical background-score distribution and reports Pd at fixed Pfa.

For CFAR the score is the ratio of the cell under test to its local noise estimate, which
is monotone in the CFAR threshold multiplier α — so sweeping it traces exactly the
detector's own design curve. For the CNN it is the raw logit, not the sigmoid, to avoid
saturation collapsing the low-Pfa end of the ROC into ties.

## Consequences

* Every detector is comparable at any operating point, and at the *same* operating point.
* The ROC, not a single accuracy number, is the primary result — matching radar practice.
* Cost: a detector that could exploit an internal threshold (e.g. a CFAR-constrained loss
  as in CFARnet) must expose its pre-threshold statistic. This is a mild constraint and
  worth it.
* Any future detector must implement one method. This is what keeps `protocol.py` a single
  code path, which is the structural guarantee that the arms are treated identically.
