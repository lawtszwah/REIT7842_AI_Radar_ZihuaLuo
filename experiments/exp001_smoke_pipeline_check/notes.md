# exp001 — does the apparatus run end to end and produce the three reported figures?

* **Research question:** none — this is an apparatus check, not a result.
* **Config:** `configs/protocol/smoke.yaml` (snapshot in `results.json`)
* **Simulator:** `stub` v0.2.0 — ⚠ **not the research group's function; nothing here is reportable**
* **Seeds:** 2 · **Machine:** macOS laptop, CPU only

## Reading

The pipeline runs: five detectors across three families are generated, tuned-by-config,
scored, evaluated under one protocol and reported with cost, in about three minutes.
Two things are worth noting as apparatus behaviour rather than findings:

1. The feature-based classifier leads at low Pfa. On the stand-in simulator that is
   expected — it sees three window scales plus Doppler geometry, where CA-CFAR sees one.
2. The small CNN sits below the CFAR baselines. At 400 training maps that is an
   undertrained model, not evidence about the family. It is exactly the comparison the
   full protocol exists to make properly, at scale and at equal tuning budget.

## Follow-ups

* M2: integrate the group's simulator through `adapter.py`; re-run this config against it.
* M2: run `prdbench tune` for every arm before any comparison is reported.
