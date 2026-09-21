# The `prdsim` signal model

**Status:** model specified; implementation and validation pending  
**Model version:** 1.0-draft  
**Milestone:** M2  
**Related:** [ADR-0004](decisions/ADR-0004-simulator-is-built-in-project.md)

## 1. Purpose and scope

`prdsim` generates synthetic range–Doppler (RD) power maps for the benchmark. This
document fixes the model before the reportable simulator is implemented. That ordering
matters because the simulator and the detectors have the same author: the data model must
not be adjusted after seeing which detector it favours.

The simulator is an **experimental instrument, not a research contribution**. It is a
map-level approximation of the output of a passive-radar cross-ambiguity processor after
direct-signal and clutter cancellation. It does not synthesize a transmitter waveform,
receiver front end, bistatic geometry, or cancellation algorithm. The literature cited
below supports the component types and the statistical assumptions; it does not validate
the numerical parameter ranges chosen for this benchmark.

## 2. Output contract and conventions

| Item | Convention |
|---|---|
| Output shape | `(N, R, D)`; the reportable protocol uses `R = D = 128` |
| Output type | `float32` |
| Output quantity | Linear power, never amplitude or dB |
| Range indexing | Zero-based; range bin 0 represents zero bistatic delay |
| Doppler indexing | Zero-based and fft-shifted; zero Doppler is `d_0 = D // 2` |
| Target coordinates | Floating-point bin coordinates |
| Target truth | `range_bin`, `doppler_bin`, nominal `snr_db`, and signal-only `peak_snr_db` |
| Randomness | A batch is a pure function of its complete config and root seed |
| Provenance | Simulator name/version, complete config, root seed, and derived seeds are recorded |

The reportable map is

$$
P[r,d] = |z[r,d]|^2,
$$

where the complex field is assembled before square-law detection:

$$
z[r,d] = n[r,d] + g[r,d] + \sum_{k=1}^{K}c_k[r,d]
          + \sum_{q=1}^{Q}t_q[r,d].
$$

Noise, residual ridge, clutter patches, and targets therefore add **coherently**. In
particular, the simulator must not replace $|t+n|^2$ by $|t|^2+|n|^2$.

The following are implementation requirements rather than optional conventions:

- `ridge_enabled` and `clutter_enabled` must be explicit booleans. Validation must not
  rely on magic values such as `ridge_db = -inf`.
- Map generation, dataset splitting, and model initialization use separate child seed
  streams derived from one recorded root seed. Within map generation, each map and each
  component use deterministic child streams so that changing one component does not
  silently change all later random draws.
- The default target-placement margin is 10% of each axis. Reportable datasets reject
  target placements whose scoring neighbourhoods overlap or intersect the five-column
  zero-Doppler exclusion band. Purpose-built validation cases may override both rules.

## 3. Receiver noise

For every cell,

$$
n[r,d] = n_\mathrm{R}[r,d] + j n_\mathrm{I}[r,d], \qquad
n_\mathrm{R},n_\mathrm{I}\overset{\mathrm{i.i.d.}}{\sim}
\mathcal N(0,\sigma^2/2).
$$

Consequently,

$$
E\{|n|^2\}=\sigma^2, \qquad |n|^2\sim
\operatorname{Exponential}(\text{mean}=\sigma^2).
$$

`noise_power` is exactly $\sigma^2$, the mean power of a noise-only cell. Complex white
Gaussian receiver noise followed by square-law detection gives the exponential cell-power
PDF in radar detection theory [1, Sec. 6.5.2, eq. (6.129), pp. 394–395]. It is also the
assumption under which the CA-CFAR multiplier

$$
\alpha=N_\mathrm{tr}\left(P_\mathrm{fa}^{-1/N_\mathrm{tr}}-1\right)
$$

achieves its design false-alarm probability when the cell under test and the
$N_\mathrm{tr}$ training cells are independent and identically distributed
[1, Sec. 6.5.3, eq. (6.140), p. 397].
This is why the noise distribution and independence are validation gates rather than
cosmetic implementation details.

The benchmark regimes are:

| Regime | `noise_power` |
|---|---:|
| Training | 0.5–2.0 |
| Extended test | 0.25–4.0 |
| Joint shift | 4.0–8.0 |

Values are sampled uniformly in linear power unless the config pins a scalar.

## 4. Point-target response

For target $q$ at fractional position $(r_q,d_q)$, with random phase
$\phi_q\sim\mathcal U[0,2\pi)$,

$$
t_q[r,d] = A_q\,\chi(r-r_q,d-d_q)e^{j\phi_q},
$$

with the normalized, separable point-spread response

$$
\chi(\Delta r,\Delta d)
= \operatorname{sinc}(\Delta r)\operatorname{sinc}(\Delta d),
\qquad
\operatorname{sinc}(x)=\frac{\sin(\pi x)}{\pi x}.
$$

The model assumes rectangular weighting, locally uncoupled delay and Doppler axes, and no
range or Doppler migration during a coherent processing interval. Finite observation in
each coordinate then gives a sinc-shaped response. Passive radar normally forms an RD map
through cross-ambiguity/matched-filter processing, while the actual sidelobe structure and
resolution depend on the illuminator and processing [2, chs. 2–3], [3]. The separable sinc
is therefore an analytically testable **surrogate**, not the ambiguity function of DVB-T,
FM, LTE, GNSS, or any other named illuminator.

### 4.1 Nominal and peak SNR

The target amplitude is set from its nominal post-integration SNR:

$$
\mathrm{SNR}_{q,\mathrm{dB}}
=10\log_{10}\!\left(\frac{A_q^2}{\sigma^2}\right),
\qquad
A_q=\sqrt{\sigma^2 10^{\mathrm{SNR}_{q,\mathrm{dB}}/10}}.
$$

This SNR is defined **after** cross-ambiguity integration and at integer-bin placement.
It excludes the residual ridge and clutter from the denominator. It is consequently a
signal-to-thermal-noise ratio, not SINR, and can overstate detectability inside a clutter
patch. Clutter density and level are reported separately.

Fractional placement reduces the signal-only peak. Let $(\delta_r,\delta_d)$ be the
offset from the nearest grid cell. Ground truth also records

$$
\mathrm{peak\_snr\_db}
=\mathrm{snr\_db}
 +20\log_{10}|\chi(\delta_r,\delta_d)|.
$$

At a half-bin offset in one coordinate the loss is
$20\log_{10}(\operatorname{sinc}(0.5))=-3.92$ dB. At a half-bin offset in both
coordinates it is $-7.84$ dB. The evaluation protocol's ±1-bin matching tolerance
accommodates this straddling; it does not remove the loss.

The target SNR regimes are 0–16 dB for training, −4–16 dB for the extended test, and
−6 to −1 dB for the joint-shift test. Multiple targets have independent phases and are
summed coherently. Reportable placement rejects overlapping ±3-bin truth-exclusion
neighbourhoods so that one peak cannot be counted as two target detections.

## 5. Residual zero-Doppler ridge

The map represents a post-cancellation product, but cancellation is not assumed perfect.
One residual ridge is added per map:

$$
g[r,d]=A_g
\exp\!\left[-\frac{1}{2}\left(\frac{d-d_0}{w_g}\right)^2\right]
\exp\!\left[-\frac{r}{\lambda_gR}\right]e^{j\phi_g},
$$

where

$$
A_g=\sqrt{\sigma^2 10^{L_g/10}},
\qquad \phi_g\sim\mathcal U[0,2\pi).
$$

The initial frozen settings are $w_g=1.1$ Doppler bins,
$\lambda_g=0.35$, and $L_g\sim\mathcal U(18,30)$ dB. A single phase is used over the
ridge so that it is a coherent residual structure.

Passive-radar surveillance data contain direct-path and multipath/clutter components that
can mask weak target returns, making cancellation a necessary upstream stage [4]–[6]. The
papers demonstrate the need for cancellation and evaluate suppression; they do **not**
establish the Gaussian Doppler profile, exponential range decay, or 18–30 dB interval used
here. Those are deliberately simple, versioned stress-test parameters. Sensitivity to
their width and level must be reported; they must not be described as measured
post-cancellation statistics.

The evaluation protocol excludes columns $d_0-2$ through $d_0+2$ from both target and
background counts. This is an explicit scope boundary, not a claim that real systems have
no useful targets near zero Doppler.

## 6. Extended-clutter surrogate

Each map contains an integer $K$ drawn uniformly from its configured inclusive patch-count
range. Patch $k$ is

$$
c_k[r,d] = A_k G_k[r,d]
\left(\sqrt{1-\beta^2}\,e^{j\phi_k}+\beta\nu_k[r,d]\right),
$$

where

$$
G_k[r,d]=\exp\!\left[-\frac{1}{2}\left(
\frac{(r-c_{r,k})^2}{s_{r,k}^2}+
\frac{(d-c_{d,k})^2}{s_{d,k}^2}\right)\right],
$$

$$
\nu_k[r,d]\overset{\mathrm{i.i.d.}}{\sim}\mathcal{CN}(0,1),
\qquad \phi_k\sim\mathcal U[0,2\pi),
\qquad A_k=\sqrt{\sigma^2 10^{L_k/10}}.
$$

The normalization makes $E[|c_k|^2]=A_k^2G_k^2$. The initial frozen settings are:

| Parameter | Distribution/value |
|---|---|
| Centre $(c_{r,k},c_{d,k})$ | Uniform over the map |
| Widths $s_{r,k},s_{d,k}$ | Independent uniform 1.5–4.0 bins |
| Centre level $L_k$ | Uniform 4–11 dB above `noise_power` |
| Speckle amplitude fraction $\beta$ | 0.35 |
| Patch count $K$ | Train 0–6; extended test 0–10; joint shift 6–12 |

Radar clutter is generally neither spatially independent nor described adequately by one
universal amplitude law. The review cited here as [7] is a **sea**-clutter review, and it
is used for its statistical observations rather than for its scene: nothing in this model
represents a sea surface, and no parameter below is taken from maritime measurements. What
is borrowed is the general picture that review sets out — that compound statistics are
usually needed rather than a single amplitude law, that the texture and speckle components
carry different spatial and temporal correlations, that Doppler spectra vary with range and
time, and that no general parametric model for bistatic amplitude statistics has been
published [7, Secs. 3.3–3.5]. Each of those points argues that structured, spatially
correlated local non-homogeneity is the realistic case and that no off-the-shelf parametric
law is available to impose here, least of all for a bistatic geometry. The model above
therefore uses correlated bright regions with a smaller random component to create
controlled local non-homogeneity, and claims nothing beyond that.

This component is intentionally called a **surrogate**. It is not a compound-Gaussian,
K-distributed, Pareto, or measured passive-radar clutter model. Its Gaussian envelope and
parameter ranges are benchmark design choices. They test whether a detector remains
stable around structured, locally elevated backgrounds without claiming to reproduce a
particular land or sea scene. A future measured-data study may replace this component
behind the same simulator interface; doing so creates a new simulator version and a
separate result set.

## 7. Data splits, parameter regimes, and threshold calibration

The model-selection validation split, threshold-calibration split, and all test splits are
disjoint from training and from each other. “Validation” in this section means
hyperparameter/model selection; it is not one of the analytical simulator-validation gates
in Section 8.

| Quantity | Training | Model-selection validation | Calibration | In-distribution test | Extended test | Joint shift |
|---|---:|---:|---:|---:|---:|---:|
| Maps | 4000 | 1000 | 1000 | 3000 | 3000 | 2000 |
| Map size | 128 × 128 | 128 × 128 | 128 × 128 | 128 × 128 | 128 × 128 | 128 × 128 |
| Targets per map | 1–3 | 1–3 | **0** | 1–3 | 1–3 | 1–3 |
| Target SNR (dB) | 0 to 16 | 0 to 16 | — | 0 to 16 | −4 to 16 | −6 to −1 |
| Noise power | 0.5 to 2.0 | 0.5 to 2.0 | 0.5 to 2.0 | 0.5 to 2.0 | 0.25 to 4.0 | 4.0 to 8.0 |
| Clutter patches | 0–6 | 0–6 | 0–6 | 0–6 | 0–10 | 6–12 |
| Ridge level (dB) | 18–30 | 18–30 | 18–30 | 18–30 | 18–30 | 18–30 |

Calibration maps are target-free but retain the configured noise, residual ridge, and
clutter surrogate. For each detector configuration selected on the validation split, its
scores on eligible calibration cells set four empirical thresholds:

$$
\tau_p = Q^{\mathrm{higher}}_{1-p}\!\left(S_\mathrm{cal,bg}\right),
\qquad p\in\{10^{-5},10^{-4},10^{-3},10^{-2}\}.
$$

Here $Q^{\mathrm{higher}}$ means the empirical quantile with NumPy's
`method="higher"` rule. A detection uses the fixed rule $s>\tau_p$. Each threshold is
selected once from the calibration split, recorded, and then applied unchanged to the
in-distribution, extended, and joint-shift tests. Test scores or labels never select or
adjust a threshold. The achieved calibration and test false-alarm probabilities are both
reported, so threshold drift under shift remains visible.

After applying the target exclusions and five-column zero-Doppler mask, the calibration
split and **each reported test split** must contain at least $10^7$ eligible background
cells per run. The configured 1000 calibration maps provide margin above this floor; the
manifest records the actual eligible-cell count, and a run below the floor is invalid.

Because the calibration split is target-free while the test splits are not, the two
background populations are not identical. A target at a fractional position has sinc
sidelobes that extend past its $\pm 3$-bin exclusion square, so a small amount of target
energy reaches cells that count as eligible background at test time but has no counterpart
in calibration. The effect is confined to the thin cross through each target and is small:
at $(\delta_r,\delta_d)=(0.5,0.5)$ the cell at $(\Delta r,\Delta d)=(3.5,0.5)$ receives
$|\chi|^2\approx-24.7$ dB relative to the peak, which at 16 dB nominal SNR is about 13% of
the noise power. It is nonetheless systematic and one-directional: it biases achieved test
false-alarm probability slightly **above** nominal, identically for every arm. This is a
known property of the data construction and must not be read as a failure of a detector's
false-alarm control.

Continuous ranges are sampled uniformly and integer ranges uniformly over both endpoints.
The joint-shift grid is deliberately outside the training grid on SNR, noise power, and
clutter count simultaneously. Ridge parameters are held fixed across these grids so that
the joint-shift result has a defined interpretation; a separate ridge sensitivity study
varies $L_g$ and $w_g$ one at a time.

The training, model-selection validation, calibration, in-distribution and extended-test
grids are defined in `configs/protocol/full.yaml`. The joint-shift grid is defined in
`configs/data/shift_test.yaml` and is referenced from there by the README and the risk
register; it is not duplicated into the protocol file, because two editable copies of one
frozen grid is a freeze hazard. Both files are covered by the protocol freeze.

The split and model-initialization streams are **spawned children of the run's recorded
root seed**, not integer offsets from it:

```python
names = [
    "train_maps",
    "test_in_distribution_maps",
    "extended_test_maps",
    "joint_shift_maps",
    "validation_maps",
    "calibration_maps",
    "analysis_bootstrap",
    "model_initialization",
]
children = numpy.random.SeedSequence(root_seed).spawn(len(names))
streams = dict(zip(names, children))
```

Each stream is the child at its position in this list. Integer offsets from a root seed
were rejected because they collide: with offsets of 1000 and root seeds chosen 1000 apart,
one run's test stream is bit-identical to another run's training stream. Spawned children
cannot collide by construction, whatever root seeds are chosen.

The consequence is that **the order of this list is frozen with the protocol**. A new
stream may be appended at the end, which leaves every existing stream unchanged. Inserting,
removing or reordering a name changes the seeds of every stream after it and invalidates
every run made under the previous order. The implementation must not reuse any stream for
a different purpose.

## 8. Validation gates

All gates are run on the implemented `prdsim`, recorded with its version and config, and
must pass before its output is reportable. Noise-only and single-target validation disable
the ridge and clutter explicitly.

### Gate 1 — noise distribution and independence

For each `noise_power` in $\{0.25,1.0,4.0,8.0\}$, generate 64 target-free,
ridge-free, clutter-free 128 × 128 maps. Convert the flattened values to float64 for the
statistics below and let $M=64(128)^2$. The fixed acceptance multiplier is $z=4.0$;
it must not be relaxed after seeing the results.

For $X=|n|^2$, verify all of the following:

1. **Mean**

   $$
   |\bar X-\sigma^2|\leq 4\frac{\sigma^2}{\sqrt M}.
   $$

2. **Coefficient of variation**

   $$
   \left|\frac{s_X}{\bar X}-1\right|\leq\frac{4}{\sqrt M}.
   $$

   The $1/\sqrt M$ standard error is the large-sample delta-method result for an
   exponential population.

3. **Tail quantiles.** For $q\in\{0.5,0.9,0.99,0.999\}$, let
   $x_q=-\sigma^2\log(1-q)$. The sample quantile must satisfy

   $$
   |\hat x_q-x_q|
   \leq 4\sigma^2\sqrt{\frac{q}{M(1-q)}}.
   $$

4. **Independence.** Form non-overlapping adjacent-cell pairs separately in range and
   Doppler, let $M_\mathrm{pair}$ be the pair count, and require

   $$
   |\hat\rho_1|\leq\frac{4}{\sqrt{M_\mathrm{pair}}}
   $$

   on each axis. Non-overlapping pairs are required so that the nominal correlation
   standard error is not understated by reusing the same cell in adjacent pairs.

Moment and tail checks are preferred to using a large-sample Kolmogorov–Smirnov test as a
binary oracle: with millions of float32 samples, a KS test can reject deviations too small
to matter to the intended detector experiment.

### Gate 2 — single-target peak and straddle loss

Use $\sigma^2=1$, nominal SNR 10 dB, and 4096 independent maps per offset. Disable the
ridge and clutter, place one target at $(64+\delta_r,64+\delta_d)$, and evaluate every
point in

$$
(\delta_r,\delta_d)\in
\{0,0.125,0.25,0.375,0.5\}^2.
$$

For cell (64, 64),

$$
E\{|A_q\chi+n|^2\}
=\sigma^2\left(1+\mathrm{SNR}_{\mathrm{lin}}|\chi|^2\right),
$$

and, over independent realizations,

$$
\operatorname{SD}(|A_q\chi+n|^2)
=\sigma^2\sqrt{1+2\mathrm{SNR}_{\mathrm{lin}}|\chi|^2}.
$$

At every offset, with $M_t=4096$, the fixed pass rule is

$$
\left|\overline{P}_{64,64}
-\sigma^2\left(1+\mathrm{SNR}_{\mathrm{lin}}|\chi|^2\right)\right|
\leq
4\frac{\sigma^2\sqrt{1+2\mathrm{SNR}_{\mathrm{lin}}|\chi|^2}}{\sqrt{M_t}}.
$$

Thus the complete sampled $|\chi|^2$ loss surface is tested with the same predeclared
$z=4.0$ multiplier. In addition, the stored signal-only `peak_snr_db` must agree with
$\mathrm{snr\_db}+20\log_{10}|\chi|$ to within $10^{-5}$ dB at every point, including
the −7.8448 dB loss at (0.5, 0.5).

### Gate 3 — CA-CFAR design false-alarm probability

On independent noise-only maps, run CA-CFAR with

$$
\alpha=N_\mathrm{tr}\left(P_\mathrm{fa}^{-1/N_\mathrm{tr}}-1\right)
$$

at design points $10^{-5}$, $10^{-4}$, $10^{-3}$, and $10^{-2}$. Exclude the border
affected by the detector's padding. Do not apply the five-column zero-Doppler mask in this
gate because the ridge is disabled and the purpose is to test the noise model over all valid
cells.

For every design point, record the number of alarms, eligible cells, measured rate,
simulator version, seed, CA-CFAR window, $N_\mathrm{tr}$, and $\alpha$. Use at least
$10^7$ eligible cells. Because overlapping CFAR windows make neighbouring decisions
dependent, form the uncertainty interval by resampling whole maps with replacement for
10,000 bootstrap replicates using the analysis-bootstrap seed stream. The reported 99%
percentile interval uses the 0.5th and 99.5th percentiles. A binomial interval may also be
shown as an ideal-i.i.d. reference, but it is not the pass criterion. The predeclared pass
rule is that the design value lies inside the 99% map-clustered bootstrap interval.

A 99% interval excludes the true value 1% of the time by chance, and this gate has four
design points, so a correct simulator fails a run about 4% of the time. The re-run rule is
therefore fixed in advance: **one** repeat is permitted, at the next seed in the run's seed
list, and only when the first failure is confined to a single design point. The gate fails
if the repeat also falls outside, if two or more design points fail together, or if a
second repeat is attempted. Both the failing and the repeated run are recorded. Deciding
whether to re-run after seeing which seed failed would be selecting seeds, which is the
same objection that fixes $z=4.0$ ahead of time.

Cluttered maps are a separate stress test and are **not** expected to satisfy the ideal
CA-CFAR formula. Departure from the design false-alarm rate under the ridge or clutter
surrogate is a benchmark result, not a simulator failure.

## 9. Claims this model does not support

- It does not reproduce the ambiguity function of a specific transmitter.
- It does not model antenna patterns, receiver filtering, quantization, propagation,
  bistatic geometry, target radar cross-section fluctuation, range/Doppler migration, or
  a cancellation algorithm.
- It does not turn bin coordinates into metres, hertz, position, or velocity.
- It does not establish realism through its analytic validation gates. Those gates show
  that the implementation matches this specification, not that the specification matches
  every deployment environment.
- Parameter-shift tests only probe changes inside this generator. They cannot distinguish
  genuine physical robustness from robustness to the generator's own artefacts.
- In-distribution detector gains are insufficient evidence of generalisation because the
  same project defines the generator and trains the detectors.

Any material change to a component distribution, numeric parameter, SNR definition, bin
convention, or placement rule increments the simulator version and requires all validation
gates and affected benchmark runs to be repeated.

## References

[1] M. A. Richards, *Fundamentals of Radar Signal Processing*, 3rd ed. New York,
NY, USA: McGraw-Hill, 2022, ch. 6. [Online]. Available:
[publisher page](https://www.mheducation.com/highered/mhp/product/fundamentals-radar-signal-processing-third-edition.html);
[Chapter 6 sample](https://radarsp.weebly.com/uploads/2/1/4/7/21471216/richards_ch006_pr.pdf).

[2] H. D. Griffiths and C. J. Baker, *An Introduction to Passive Radar*, 2nd ed.
Norwood, MA, USA: Artech House, 2022. [Online]. Available:
[publisher page](https://uk.artechhouse.com/An-Introduction-to-Passive-Radar-Second-Edition-P2177.aspx)

[3] X. Bai, H. Guo, J. Zhao, and T. Shan, “Compressed sensing-based range-Doppler
processing method for passive radar,” *Wireless Communications and Mobile Computing*,
vol. 2021, Art. no. 5570498, 2021, doi:
[10.1155/2021/5570498](https://doi.org/10.1155/2021/5570498).

[4] D. Zhao, J. Wang, G. Chen, J. Wang, and S. Guo, “Clutter cancellation based on
frequency domain analysis in passive bistatic radar,” *IEEE Access*, vol. 8,
pp. 43956–43964, 2020, doi:
[10.1109/ACCESS.2020.2977961](https://doi.org/10.1109/ACCESS.2020.2977961).

[5] X. Bai, J. Han, J. Zhao, Y. Feng, and R. Tao, “Clutter cancellation in passive radar
using batch-based CLEAN technique,” *EURASIP Journal on Advances in Signal Processing*,
vol. 2021, Art. no. 63, 2021, doi:
[10.1186/s13634-021-00769-9](https://doi.org/10.1186/s13634-021-00769-9).

[6] B. Wang, H. Cha, Z. Zhou, and B. Tian, “Clutter cancellation and long time
integration for GNSS-based passive bistatic radar,” *Remote Sensing*, vol. 13, no. 4,
Art. no. 701, 2021, doi:
[10.3390/rs13040701](https://doi.org/10.3390/rs13040701).

[7] S. Watts and L. Rosenberg, “Challenges in radar sea clutter modelling,” *IET Radar,
Sonar & Navigation*, vol. 16, no. 9, pp. 1403–1414, 2022, doi:
[10.1049/rsn2.12272](https://doi.org/10.1049/rsn2.12272).
