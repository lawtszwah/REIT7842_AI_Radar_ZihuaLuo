# Results summary

Config `decdc6d1259f65db8a9dd1899c4808e2c72da73f9f6e2b1b6e3711ab7017c62f`, simulator `1.0.0+gen.567f5e55df8a`, root seeds [0, 1, 2, 3, 4]. Values are mean ± population SD over the root seeds; per-root values are in summary.json.

| Results file | SHA-256 | Commit | Dirty | Detectors |
|---|---|---|---|---|
| experiments/m3-formal/runs/cnn/results.json | `b5027c15e77f8ffe` | `3113a75` | False | cnn |
| experiments/m3-formal/runs/cpu_arms/results.json | `f6bec8ddb3dfa57b` | `3113a75` | False | ca_cfar, os_cfar, go_cfar, feat_gbdt |

## Pd at design Pfa = 1e-4 (headline operating point)

| Detector | In-distribution test | Extended test | Joint shift | Grid shift |
|---|---|---|---|---|
| ca_cfar | 0.284 ± 0.006 | 0.339 ± 0.018 | 0.038 ± 0.003 | 0.300 ± 0.038 |
| os_cfar | 0.291 ± 0.006 | 0.345 ± 0.017 | 0.038 ± 0.003 | 0.298 ± 0.039 |
| go_cfar | 0.226 ± 0.008 | 0.294 ± 0.017 | 0.022 ± 0.003 | 0.239 ± 0.027 |
| feat_gbdt | 0.307 ± 0.011 | 0.338 ± 0.032 | 0.257 ± 0.138 | 0.321 ± 0.047 |
| cnn | 0.246 ± 0.123 | 0.283 ± 0.142 | 0.018 ± 0.010 | 0.264 ± 0.137 |

Achieved test Pfa at design Pfa = 1e-4:

| Detector | In-distribution test | Extended test | Joint shift | Grid shift |
|---|---|---|---|---|
| ca_cfar | 1.00e-04 | 1.01e-04 | 9.99e-05 | 1.03e-04 |
| os_cfar | 9.92e-05 | 1.02e-04 | 1.00e-04 | 9.74e-05 |
| go_cfar | 9.94e-05 | 9.90e-05 | 9.65e-05 | 8.70e-05 |
| feat_gbdt | 1.00e-04 | 3.98e-03 | 2.54e-02 | 7.34e-05 |
| cnn | 8.16e-05 | 8.62e-05 | 7.24e-05 | 1.38e-04 |

Achieved Pfa is more than 2× the design Pfa for feat_gbdt on extended test; feat_gbdt on joint shift. Those thresholds, set on in-distribution calibration background, do not hold there, so the Pd beside them is not at the stated Pfa.

## Pd at design Pfa = 1e-3 (selection operating point)

| Detector | In-distribution test | Extended test | Joint shift | Grid shift |
|---|---|---|---|---|
| ca_cfar | 0.380 ± 0.006 | 0.414 ± 0.015 | 0.111 ± 0.010 | 0.398 ± 0.039 |
| os_cfar | 0.392 ± 0.005 | 0.422 ± 0.016 | 0.110 ± 0.007 | 0.398 ± 0.040 |
| go_cfar | 0.345 ± 0.006 | 0.386 ± 0.020 | 0.086 ± 0.006 | 0.353 ± 0.032 |
| feat_gbdt | 0.425 ± 0.005 | 0.536 ± 0.047 | 0.583 ± 0.142 | 0.439 ± 0.041 |
| cnn | 0.408 ± 0.009 | 0.431 ± 0.019 | 0.075 ± 0.006 | 0.441 ± 0.044 |

Achieved test Pfa at design Pfa = 1e-3:

| Detector | In-distribution test | Extended test | Joint shift | Grid shift |
|---|---|---|---|---|
| ca_cfar | 9.84e-04 | 9.84e-04 | 9.74e-04 | 1.02e-03 |
| os_cfar | 9.91e-04 | 9.92e-04 | 9.85e-04 | 9.80e-04 |
| go_cfar | 9.93e-04 | 9.85e-04 | 9.79e-04 | 9.31e-04 |
| feat_gbdt | 9.84e-04 | 4.32e-02 | 2.03e-01 | 7.07e-04 |
| cnn | 9.77e-04 | 1.01e-03 | 8.63e-04 | 1.58e-03 |

Achieved Pfa is more than 2× the design Pfa for feat_gbdt on extended test; feat_gbdt on joint shift. Those thresholds, set on in-distribution calibration background, do not hold there, so the Pd beside them is not at the stated Pfa.

## Pd per root seed, In-distribution test, Pfa = 1e-4

| Detector | root 0 | root 1 | root 2 | root 3 | root 4 |
|---|---|---|---|---|---|
| ca_cfar | 0.288 | 0.279 | 0.289 | 0.274 | 0.290 |
| os_cfar | 0.292 | 0.285 | 0.296 | 0.282 | 0.298 |
| go_cfar | 0.219 | 0.222 | 0.233 | 0.217 | 0.237 |
| feat_gbdt | 0.303 | 0.308 | 0.314 | 0.288 | 0.321 |
| cnn | 0.310 | 0.294 | 0.316 | 0.308 | 0.000 |

## Cost

Latency is the median over repeated runs per root, then mean ± SD over roots. GPU and CPU latency are not comparable and are never plotted together (RSK-06).

| Detector | Parameters | Measured on | Latency (ms per map) |
|---|---|---|---|
| ca_cfar | 0 | CPU | 0.334 ± 0.009 |
| os_cfar | 0 | CPU | 40.142 ± 0.543 |
| go_cfar | 0 | CPU | 0.324 ± 0.007 |
| feat_gbdt | 4924 | CPU | 9.760 ± 0.244 |
| cnn | 1905 | CUDA | 0.256 ± 0.171 |

## Operating points not attained on the calibration background

Detection is strictly `score > threshold`. Where the achieved calibration Pfa is zero, no calibration cell scores above the threshold, so at least the stated number of cells share the threshold score and that operating point cannot be reached.

| Detector | Root | Design Pfa | Threshold | Cells tied (at least) |
|---|---|---|---|---|
| cnn | 4 | 1e-5 | 1.093532 | 106 of 10,533,600 |
| cnn | 4 | 1e-4 | 1.093532 | 1,054 of 10,533,600 |
