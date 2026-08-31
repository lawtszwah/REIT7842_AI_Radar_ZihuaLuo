# Risk register

Reviewed at each weekly supervision meeting. Severity = impact × likelihood.

| ID | Risk | Severity | Mitigation in place |
|---|---|---|---|
| **RSK-01** | The group's simulation function is not received, or arrives late | **High** | The whole benchmark is written against `simulation/interface.py`, not against the function. `stub.py` lets every other component be built and tested now; `tests/test_simulator_contract.py` is the acceptance test for handover. Worst case, the project reports on the stub and states the limitation explicitly. |
| RSK-02 | The group's function has a different parameterisation than assumed (e.g. bistatic geometry instead of range/Doppler bins) | High | `adapter.py` is the only file that would change. `SimConfig` fields are named after the four quantities the topic definition commits to sweeping, and unknown extras go in `SimConfig.extra`. |
| RSK-03 | CFAR baselines are under-tuned, making ML look better than it is — the exact flaw the project criticises | High | `tuning/search.py` gives CFAR the same trial budget as the networks, over `configs/protocol/full.yaml::search_spaces`. CFAR tuning results are reported alongside ML tuning results. |
| RSK-04 | Results are an artefact of the evaluation protocol rather than the detector | High | One protocol, one code path (`evaluation/protocol.py`). Protocol frozen at M2, before the deep models are tuned; any later change requires an ADR and a re-run of every arm. |
| RSK-05 | Deep models overfit the simulator's specific noise and clutter model | Medium | Test configs include SNR, noise-power and clutter regimes outside the training grid (`configs/data/shift_test.yaml`), reported separately from in-distribution results. |
| RSK-06 | Latency measurements are not comparable across model families | Medium | `evaluation/cost.py` measures all detectors the same way — same machine, same batch, median of repeated runs, CPU by default — and parameter count is reported alongside. GPU numbers, if any, are reported separately and never mixed. |
| RSK-07 | Compute budget insufficient for 5 seeds × 6 detectors at 128×128 | Medium | Cost is bounded by config: map size, dataset size and seed count are all parameters. Smoke → full is a config change, so the run can be scaled to whatever compute is available and the scale is recorded in `results.json`. |
| RSK-08 | Results not reproducible months later at write-up time | Medium | Datasets regenerated from `(config, seed)`; commit hash, library versions and platform recorded in every `results.json`; CI runs the smoke benchmark on every push. |
| RSK-09 | Attention arm (M3) proves too expensive to train in the time available | Low | The research question is posed over *model families*; if the attention arm is dropped it is reported as a scope reduction with its cost measurement, not silently omitted. |
