# ADR-0002 — The simulator sits behind an interface, with a stand-in

**Status:** accepted · **Date:** 2026-08-31

## Context

The parameterised range-Doppler simulation function is supplied by the research group and
was not available when implementation began. Waiting for it would idle the project;
building against a guess at its signature would mean a rewrite when it arrives.

## Decision

Define `simulation/interface.py` — `SimConfig` in, `SimBatch` out — and write every other
module against that protocol. Provide `stub.py`, a deliberately simple stand-in, and
`adapter.py`, an empty shell whose only job is to translate the group's function into the
interface. Make `tests/test_simulator_contract.py` the acceptance test both must pass.

## Consequences

* Detectors, metrics, tuning, CI and reporting were all built and tested before the
  simulator existed. Integration is one file.
* Risk of divergence: if the group's function is parameterised differently (bistatic
  geometry rather than range/Doppler bins), `SimConfig` needs extending. `SimConfig.extra`
  absorbs unknown parameters without breaking callers. Tracked as RSK-02.
* Risk of contamination: stub results could be mistaken for findings. Every `SimBatch`
  carries `simulator` and a warning string into `results.json`, and the README states that
  stub results are not reportable.
