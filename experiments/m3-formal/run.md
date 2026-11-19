# Experiment m3-formal — five arms under the frozen protocol, five root seeds

* **Research questions:** RQ1 and RQ2 for five arms; RQ3 for the four CPU-timed arms only
  (the CNN was timed on a GPU, see below). The attention arm is not in this run: ADR-0005
  places it in M4.
* **Config:** `configs/protocol/full.yaml` merged with `configs/data/shift_test.yaml`,
  `config_sha256` `decdc6d1259f65db8a9dd1899c4808e2c72da73f9f6e2b1b6e3711ab7017c62f`
  (tag `m2-protocol-freeze`). The merged config is snapshotted as `spec` in each
  `results.json`.
* **Commit:** `3113a75` for both runs, `git_dirty: false`. The tuning records were made at
  `869aec8`; each record names its own commit.
* **Simulator:** group generator `1.0.0+gen.567f5e55df8a` (ADR-0006). Every scene was read
  from the cache with MATLAB disabled, so no data were regenerated on this machine.
* **Seeds:** root seeds 0–4. Hyperparameters were selected on root 0 only, at equal budget
  (36 trials or the whole grid if smaller), by Pd at the selection Pfa 1e-3 on the
  validation split.
* **Machine:** HP Omen laptop, Windows 11 (10.0.26200), Intel CPU (family 6 model 183),
  NVIDIA RTX 4070 Laptop GPU; Python 3.11.4, numpy 2.4.6, torch 2.11.0+cu128,
  scikit-learn 1.9.1.
* **Wall time:** the CNN on CUDA, 454.5 s (2026-11-18 20:05–20:13 AEST); then the four CPU
  arms, 11,353.4 s (20:13–23:22 AEST). The two runs were sequential, so latency was never
  measured under contention.

## Files

| Path | What it is |
|---|---|
| `runs/cnn/results.json`, `runs/cnn.log` | `prdbench run --tuned … --detectors cnn --device cuda` |
| `runs/cpu_arms/results.json`, `runs/cpu_arms.log` | `prdbench run --tuned … --detectors ca_cfar os_cfar go_cfar feat_gbdt` |
| `tune/<key>_root0.{json,log}` | the five tuning records the runs read |
| `figures/` | `prdbench report` over both results files: four figures, `summary.json`, `summary.md` |
| `SHA256SUMS` | checksums of `runs/` and `tune/`; run `shasum -a 256 -c SHA256SUMS` here |

The runs read the tuning records from `results/m3/tune/` and recorded that path with each
file's SHA-256 under `run.tuned`. The files in `tune/` are the same bytes, as `SHA256SUMS`
shows. `.gitattributes` keeps git from converting line endings, so the checksums hold on
every checkout.

To regenerate `figures/` from the repository root:

```bash
prdbench report --results experiments/m3-formal/runs/cnn/results.json \
                experiments/m3-formal/runs/cpu_arms/results.json --out experiments/m3-formal/figures
```

## Selected hyperparameters

| Detector | Selected | Validation Pd at 1e-3 (root 0) |
|---|---|---|
| ca_cfar | train 12, guard 2 | 0.397 |
| os_cfar | train 8, guard 3, rank 87.5 % | 0.390 |
| go_cfar | train 12, guard 2 | 0.333 |
| feat_gbdt | 160 iterations, learning rate 0.05, depth 4 | 0.436 |
| cnn | width 8, depth 4, learning rate 3e-3 (12 epochs, the detector default) | 0.426 |

## Result

In-distribution test, mean ± population SD over the five root seeds. 1e-4 is the headline
point (ADR-0005) and 1e-3 the selection point. The shift splits, achieved Pfa and per-root
values are in `figures/summary.md`.

| Detector | Pd@1e-5 | Pd@1e-4 | Pd@1e-3 | Pd@1e-2 | ms/map | Measured on | params |
|---|---|---|---|---|---|---|---|
| ca_cfar | 0.219 ± 0.008 | 0.284 ± 0.006 | 0.380 ± 0.006 | 0.573 ± 0.003 | 0.334 | CPU | 0 |
| os_cfar | 0.224 ± 0.008 | 0.291 ± 0.006 | 0.392 ± 0.005 | 0.585 ± 0.002 | 40.142 | CPU | 0 |
| go_cfar | 0.147 ± 0.007 | 0.226 ± 0.008 | 0.345 ± 0.006 | 0.558 ± 0.005 | 0.324 | CPU | 0 |
| feat_gbdt | 0.212 ± 0.009 | 0.307 ± 0.011 | 0.425 ± 0.005 | 0.626 ± 0.009 | 9.760 | CPU | 4,924 |
| cnn | 0.186 ± 0.093 | 0.246 ± 0.123 | 0.408 ± 0.009 | 0.588 ± 0.008 | 0.256 | **GPU** | 1,905 |

## Reading

* **In distribution at 1e-4,** the feature classifier has the highest mean Pd, followed by
  OS- and CA-CFAR; GO-CFAR is lowest. The gaps between the top three are of the order of
  one to three root-seed SDs, so a ranking claim needs the paired per-root values and the
  bootstrap intervals in `results.json`, not these means alone.
* **CNN root 4 cannot reach 1e-4 or 1e-5.** Its calibration thresholds at both points are
  1.0935317. The achieved calibration Pfa, the achieved test Pfa and Pd are all zero there.
  Detection is strictly `score > threshold` and the threshold is the "higher" empirical
  quantile, so at least 1,054 of the 10,533,600 calibration background cells score exactly
  that value. The CNN's maximum score on background is tied over at least that many cells.
  Why this model saturates is not established; the scores and weights were not saved. At
  1e-3 and 1e-2 root 4 is in line with the other roots (Pd 0.420 and 0.600). Over the other
  four roots the CNN's Pd at 1e-4 is 0.294–0.316.
* **Under shift, achieved Pfa has to be read with Pd.** The feature classifier's thresholds,
  set on in-distribution calibration background, give an achieved Pfa of about 4e-3 on the
  extended test and 2.5e-2 on the joint shift at a design Pfa of 1e-4. Its Pd on those
  splits is therefore not at the stated Pfa. The CFAR arms hold their design Pfa on every
  split; the CNN stays within a factor of 1.4.
* **Cost:** among the CPU-timed arms, CA- and GO-CFAR take about 0.33 ms per map, the
  feature classifier 9.8 ms and OS-CFAR 40 ms. The CNN's 0.26 ms was measured on the GPU and
  is not comparable with these (RSK-06).

## Decisions and follow-ups

* **CNN root 4 is kept.** Dropping a root after seeing the results would be a post-hoc
  choice (RSK-04). It is reported with the five per-root points and the tie count above.
  A diagnosis needs the model retrained from the same seed outside this archive.
* **CNN CPU latency is not measured.** Latency depends on the architecture, not the
  weights, so an inference-only CPU timing of width 8, depth 4 is possible without
  retraining. Until that is decided, the CNN's latency is reported as GPU latency, apart
  from the CPU figure.
* **Attention arm:** M4, under the same selection rule and budget (ADR-0005). The frozen
  config has no search space for it, and `configs/` is frozen, so the space has to be
  fixed in a separate, hashed file under its own ADR before any tuning.
* The comments `# M3` beside the attention arm in `configs/protocol/full.yaml` and
  `configs/detectors/learned.yaml` predate ADR-0005. They are left as they are because
  `configs/` is frozen.
