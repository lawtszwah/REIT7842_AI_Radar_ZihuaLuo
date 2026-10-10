# Risk register

Reviewed at each weekly supervision meeting. Severity = impact × likelihood.

| ID | Risk | Severity | Mitigation in place |
|---|---|---|---|
| **RSK-01** | The simulator's signal model is too idealised for the comparison to transfer to measured passive radar | **High** | The model follows the published passive-radar processing chain and its impairments rather than being chosen for convenience, and is validated before the protocol is frozen: the target-free noise floor against the distribution it should follow, a single target's peak against the SNR it was generated at, and CA-CFAR's measured Pfa on target-free maps against its design Pfa. Findings are reported as conditional on the model, and the simulator is released with the benchmark so that the condition is inspectable and the study repeatable under a different model. |
| RSK-02 | The simulator and the detectors have the same author, so an apparent gain reflects the generator rather than the physics | High | The shift-test set (`configs/data/shift_test.yaml`) lies entirely outside the training grid and is reported separately: a gain that does not survive there is reported as an artefact of the generator, not as a finding. CFAR learns nothing from the generator and is tuned at equal budget as the reference. The simulator is frozen at M2 and its version recorded in every `results.json`. |
| RSK-03 | CFAR baselines are under-tuned, making ML look better than it is — the exact flaw the project criticises | High | `tuning/search.py` gives CFAR the same trial budget as the networks, fixed in the config as `protocol.tuning_budget` and not overridable from the command line, over `configs/protocol/full.yaml::search_spaces`. CFAR tuning results are reported alongside ML tuning results. |
| RSK-04 | Results are an artefact of the evaluation protocol rather than the detector | High | One protocol, one code path (`evaluation/protocol.py`). Protocol frozen at M2, before the deep models are tuned; any later change requires an ADR and a re-run of every arm. |
| RSK-05 | Deep models overfit the particular SNR, noise-power and clutter regimes used for training | Medium | Test configs include SNR, noise-power and clutter regimes outside the training grid (`configs/data/shift_test.yaml`), reported separately from in-distribution results. |
| RSK-06 | Latency measurements are not comparable across model families | Medium | `evaluation/cost.py` measures all detectors the same way — same machine, same batch, median of repeated runs, CPU by default — and parameter count is reported alongside. GPU numbers, if any, are reported separately and never mixed. |
| RSK-07 | Compute budget insufficient for 5 seeds × 6 detectors at 128×128 | Medium | Cost is bounded by config: map size, dataset size and seed count are all parameters. Smoke → full is a config change, so the run can be scaled to whatever compute is available and the scale is recorded in `results.json`. |
| RSK-08 | Results not reproducible months later at write-up time | Medium | Datasets regenerated from `(config, seed)`; commit hash, library versions and platform recorded in every `results.json`; CI runs the smoke benchmark on every push. |
| RSK-09 | Attention arm (M3) proves too expensive to train in the time available | Low | The research question is posed over *model families*; if the attention arm is dropped it is reported as a scope reduction with its cost measurement, not silently omitted. |

## Changes

**2026-09-09** — the research group confirmed it will not supply the range-Doppler
simulation function; the simulator is built in-project (ADR-0004). The two risks that
followed from the external dependency — non-delivery (old RSK-01) and an unexpected
parameterisation (old RSK-02) — no longer exist. The identifiers are reused for the two
risks that replace them: the fidelity of a signal model the project now owns, and the
circularity of one author writing both the generator and the detectors. RSK-05 was
narrowed to regime shift so that it does not restate RSK-01.
