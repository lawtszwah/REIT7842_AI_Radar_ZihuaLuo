# Risk register

Reviewed at each weekly supervision meeting. Severity = impact × likelihood.

| ID | Risk | Severity | Mitigation in place |
|---|---|---|---|
| **RSK-01** | The generator's signal model is too idealised, or idealised in the wrong places, for the comparison to transfer to measured passive radar | **High** | The generator is the research group's, not written for this thesis, and was characterised before the protocol was frozen (`docs/data_grid.md`, `docs/validation_report.md`): background statistics, the reach of the zero-Doppler ridge, peak loss against true position, CA-CFAR's measured Pfa against design, and the mapping from its SNR to effective SNR. Its known departures — background level varying with CNR between maps, weak range correlation, monostatic Doppler on a bistatic delay — are recorded in ADR-0005 as documented biases. Findings are reported as conditional on this generator. |
| RSK-02 | An apparent gain reflects something specific to the generator rather than to the detection problem | Medium | The generator was written by the research group, so the earlier concern that one author writes both the data and the detectors no longer applies. The shift tests (extended, joint shift in `configs/data/shift_test.yaml`, grid shift) lie outside the training grid and are reported separately: a gain that does not survive there is reported as an artefact of the training regime, not as a finding. CFAR learns nothing from the generator and is the reference. The generator's file fingerprint is recorded in every `results.json`. |
| RSK-03 | CFAR baselines are under-tuned, making ML look better than it is — the exact flaw the project criticises | High | `tuning/search.py` gives every arm the same trial budget, fixed in the config as `protocol.tuning_budget` = 36 and not overridable from the command line. 36 is the size of the largest frozen search space, so every arm, CFAR included, is searched exhaustively over `configs/protocol/full.yaml::search_spaces` (ADR-0005). CFAR tuning results are reported alongside ML tuning results. |
| RSK-04 | Results are an artefact of the evaluation protocol rather than the detector | High | One protocol, one code path (`evaluation/protocol.py`). Protocol frozen at M2, before the deep models are tuned; any later change requires an ADR and a re-run of every arm. |
| RSK-05 | Deep models overfit the particular SNR, noise-power and clutter regimes used for training | Medium | Test configs include SNR, noise-power and clutter regimes outside the training grid (`configs/data/shift_test.yaml`), reported separately from in-distribution results. |
| RSK-06 | Latency measurements are not comparable across model families | Medium | `evaluation/cost.py` measures all detectors the same way — same machine, same batch, median of repeated runs, CPU by default — and parameter count is reported alongside. GPU numbers, if any, are reported separately and never mixed. |
| RSK-07 | Compute budget insufficient for 5 seeds × 6 detectors | Medium | Data generation is the measured cost: 12.1 hours for all seven splits at five root seeds, on 99 × 81 maps (199 × 121 for grid shift). Scenes are cached on disk, so this is paid once per protocol version. Training cost is bounded by config: dataset size, seed count and search spaces are parameters, and the scale used is recorded in `results.json`. |
| RSK-08 | Results not reproducible months later at write-up time | Medium | Datasets regenerated from `(config, seed)`; commit hash, library versions and platform recorded in every `results.json`; CI runs the smoke benchmark on every push. |
| RSK-09 | Attention arm (M4) proves too expensive to train in the time available | Low | The research question is posed over *model families*; if the attention arm is dropped it is reported as a scope reduction with its cost measurement, not silently omitted. |
| RSK-10 | The generator cannot be run or changed: its cores are obfuscated MATLAB P-code, and it needs a MATLAB licence | Medium | The cores are treated as a fixed black box, identified by the SHA-256 of their files. Every generated scene is cached, so results can be re-analysed without MATLAB. The adapter (`simulation/groupgen.py`) is the only code that calls MATLAB, and its truth calibration is checked by a `matlab`-marked test. Moving to another machine (planned for M3) is checked by comparing cache bytes on both platforms. |
| RSK-11 | Generation is slow (about 5.5 s per scene), so any change to the grid after the freeze costs days | Medium | The grid was set from measurements and frozen with ADR-0005; any change needs a new ADR. Re-generating and re-checking every split takes about 12 hours plus 3–4 days of rework, which is why the grid was settled before M3. |

## Changes

**2026-09-09** — the research group confirmed it will not supply the range-Doppler
simulation function; the simulator is built in-project (ADR-0004). The two risks that
followed from the external dependency — non-delivery (old RSK-01) and an unexpected
parameterisation (old RSK-02) — no longer exist. The identifiers are reused for the two
risks that replace them: the fidelity of a signal model the project now owns, and the
circularity of one author writing both the generator and the detectors. RSK-05 was
narrowed to regime shift so that it does not restate RSK-01.

**2026-10-12** — the 2026-09-09 entry above rested on a misreading: the research group had
supplied a generator, and the thesis data now come from it (ADR-0006). The in-project
simulator stays as a reference for testing the evaluation code only.

**2026-11-17** — rewritten for the group's generator. RSK-01 now concerns the generator's
fidelity and its measured departures; RSK-02 drops the single-author circularity, which no
longer applies, and is lowered to Medium; RSK-03 records the budget of 36; RSK-07 uses
measured generation cost and the current map size; RSK-09 moves the attention arm to M4.
RSK-10 (P-code and MATLAB dependency) and RSK-11 (generation cost of a grid change) are new.
