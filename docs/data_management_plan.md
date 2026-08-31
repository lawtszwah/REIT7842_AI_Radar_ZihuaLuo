# Data management, ethics and governance

## Ethics

The project uses **synthetic range-Doppler maps only**. There are no human participants,
no personal or identifiable data, no animal subjects, and no physical transmission of
radio-frequency energy — passive radar transmits nothing, and in any case this project is
entirely computational. On the UQ human-ethics decision path this is *not* human-subject
research and no HREC application is required. This will be confirmed in writing with the
supervisor and recorded here.

Where a dual-use question could arise — target detection has surveillance applications —
the work is a methodological comparison of published detection algorithms on synthetic
data, contributes no new detection capability, and is reported openly.

## What data exists

| Asset | Volume | Where it lives | Retention |
|---|---|---|---|
| Simulator function (group-supplied) | small | supplied by the research group; **not redistributed** in this repository unless the group agrees in writing | per group's terms |
| Generated range-Doppler datasets | 10s of GB at full scale | regenerated on demand, never committed (`.gitignore`) | transient |
| Configs and seeds | KB | version-controlled | permanent |
| `results.json` per run + figures | MB | version-controlled under `experiments/` | permanent |
| Trained model checkpoints | 100s of MB | local + UQ RDM, not in git | end of project + 5 years |

Because datasets are a deterministic function of `(config, seed)`, storing the config is
storing the data. This is the design choice that keeps the project reproducible without a
data repository.

## Licensing and sharing

* Code: released under an OSI licence at submission (`LICENSE`), so results can be re-run.
* Simulator: **check before publishing.** It is the research group's, and its licence and
  any embargo govern what can be shared. `adapter.py` deliberately keeps it out of this
  repository so that the benchmark can be released even if the simulator cannot.
* Figures and `results.json`: released with the thesis.

## Version control and integrity

* Git, one repository, `main` protected; work on branches, merged via pull request.
* Every run records its commit hash. Figures in the thesis cite the commit that produced them.
* CI (`.github/workflows/ci.yml`) runs the tests and the smoke benchmark on every push, so
  a change that silently breaks the protocol fails before it can contaminate results.
* Architecture Decision Records under `docs/decisions/` capture *why* each protocol choice
  was made, so a decision cannot be quietly reversed mid-project.

## Security

Nothing here is sensitive. The one control that matters: the group's simulator must not be
committed or pushed to a public remote. `.gitignore` covers the expected drop location and
this is checked at every merge.
