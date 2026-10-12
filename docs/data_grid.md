# The data grid, and why it is this grid

**Status:** proposed with ADR-0005 · **Date:** 2026-10-12 · **Generator:** the research
group's, through `groupgen` (ADR-0006).

The grid is set from measurements of the generator, not chosen in advance. The measurements
are produced by `scripts/characterise_groupgen.py`, which runs no detector, and are stored in
`results/validation/groupgen_characterisation.json`, recorded with the generator version
`0.1.0+gen.567f5e55df8a`.

## 1. The map

| | Value | Reason |
|---|---|---|
| Range | delay 1–99 samples, 150 m per cell, 99 cells | The generator's `gridx = 2:100`. |
| Doppler | −400:10:400 Hz, 81 cells | A 0.1 s frame resolves 10 Hz, so this grid has one cell per resolution cell. The generator's 1 Hz grid oversamples tenfold. The grid is symmetric so that zero Doppler is at index 40. |
| Frames | 10 per trajectory, 0.1 s each | The generator's default; one trajectory is one *scene*. |

Measured on 200 target-free maps:

- **Background** outside the zero-Doppler guard is exponential: coefficient of variation
  1.002 (exponential: 1). Neighbouring cells are independent: correlation 0.001 in Doppler
  and 0.002 in range. On the 1 Hz grid the Doppler correlation is 0.97, which breaks
  CA-CFAR's assumption of independent training cells. That is the main reason for the
  10 Hz grid.
- **The zero-Doppler ridge** (direct path and clutter, not cancelled) occupies only the 0 Hz
  column. On average that column is 3.9 times the noise floor; the ±10 Hz columns are at
  the floor (0.99). Along range, the ridge is above 3 dB out to cell 15. The protocol's
  ±2-cell zero-Doppler guard therefore removes all of it, with margin.
- **Eligible background cells per map:** 99 × 76 = 7524.

## 2. SNR: range, step, and the reason for both

`snr_db` is the generator's own `SNR_dB`. Measured at the recorded target cell on the
10 Hz grid (80 frames per level), the effective SNR, mean(target cell) / mean(background) − 1,
is about **SNR_dB + 5.6 dB**: −10 → −4.5, −5 → +1.4, 0 → +5.6, +5 → +11.1 dB. Below
−10 dB the target cell cannot be told from noise with 80 frames, so those levels give no
estimate. The offset already includes the average straddle loss on this grid.

**What range separates detectors.** For a non-fluctuating target in exponential noise, an
ideal single-cell detector has Pd = Q₁(√(2·snr), √(2·ln(1/Pfa))). At the four design points it
moves from Pd = 0.05 to Pd = 0.99 over these effective SNRs:

| Design Pfa | Pd 0.05 | Pd 0.5 | Pd 0.95 | Pd 0.99 |
|---|---:|---:|---:|---:|
| 10⁻² | −2.0 dB | 6.25 | 10.25 | 11.5 |
| 10⁻³ | 2.5 | 8.25 | 11.5 | 12.5 |
| 10⁻⁴ | 5.0 | 9.5 | 12.5 | 13.5 |
| 10⁻⁵ | 6.75 | 10.5 | 13.25 | 14.0 |

Below the transition every detector misses; above it every detector detects. Maps there
cost time and say nothing about which detector is better. Mapped back through the +5.6 dB
offset, the transition at all four design points spans generator SNR **−7.6 to +8.4 dB**.

**Chosen:** training, validation and in-distribution test use **−10 to +10 dB**: the
transition plus about 2 dB on each side. The margin covers the spread of straddle loss from
map to map, and detectors that gather energy from several cells, which can detect below the
single-cell curve. The extended test uses −14 to +14 dB; the joint-shift test uses −14 to
−4 dB.

**Step.** There is no step. SNR is drawn from a continuous uniform distribution, so no
detector is tuned to a lattice of values. Results are reported in 2 dB bands
(`protocol.snr_edges`), about 300 targets per band in the in-distribution test.

## 3. The other swept quantities

| Quantity | Train / validation / in-distribution | Extended test | Joint shift |
|---|---|---|---|
| SNR_dB | −10 to 10 | −14 to 14 | −14 to −4 |
| CSR_dB (clutter re target) | 20 to 30 | 15 to 35 | 30 to 40 |
| DNR_dB (direct path re noise) | 40 to 50 | 35 to 55 | 50 to 55 |
| Speed (km/h) | 200 to 600 | 150 to 700 | 600 to 850 |
| Heading | uniform | uniform | uniform |

CSR and DNR are centred on the generator's defaults (25 and 45 dB). The generator sets the
clutter level relative to the target (CNR = SNR + CSR). Target-free calibration maps
therefore draw a reference SNR from the same range, so that their clutter level is
distributed as in maps with targets. Because the ridge is confined to the masked 0 Hz column
(section 1), CSR and DNR mostly change cells that are not scored. That is a property of
the generator's processing, and it is worth confirming with the group.

**Placement.** Each trajectory's start position is drawn uniformly over a square: ±16 km
for 99-cell maps, ±30 km for the grid-shift split. A trajectory is redrawn unless every one
of its ten frames is at least 3 cells inside the window and 4 Doppler cells (40 Hz) away
from zero Doppler. Uniform placement over an area favours longer ranges; this is stated,
not corrected.

## 4. Splits and the 10⁷-cell floor

| Split | Maps | Scenes | Eligible cells per run |
|---|---:|---:|---:|
| Train | 4000 | 400 | (no floor) |
| Validation | 1400 | 140 | 10.5 M |
| Calibration (target-free) | 1400 | 140 | 10.5 M |
| In-distribution test | 3000 | 300 | 22.6 M |
| Extended test | 3000 | 300 | 22.6 M |
| Joint shift | 2000 | 200 | 15.0 M |
| Grid shift (199 × 121 maps) | 1000 | 100 | 23.1 M |

Validation and calibration were 1000 maps in the 128 × 128 design. With 7524 eligible cells
per map, 1000 maps would give 7.5 M cells, below the floor, so both are 1400.

**Grid shift** answers the supervisor's point that the map size should not be fixed.
Detectors trained on 99 × 81 maps are scored on 199 × 121 maps (30 km, ±600 Hz), with
thresholds calibrated on 99 × 81 maps applied unchanged. CFAR detectors and the fully
convolutional CNN take any input size.

**Cost.** About 5.5 s per scene, 1580 scenes per root seed: roughly 2.5 hours per root seed,
12 hours for all five. Generated scenes are cached on disk, keyed by their full
specification and the generator's files, so every scene is generated once.

## 5. Open with the supervisors

1. The generator's Doppler follows the monostatic form 2v·u_Rx/λ while its delay is
   bistatic (ADR-0006). Is that intended?
2. Confirm the meaning of `SNR_dB`: measured here as about 5.6 dB below the effective SNR at
   the target cell on this grid.
3. Confirm the 10 Hz Doppler grid in place of the interface script's 1 Hz grid.
