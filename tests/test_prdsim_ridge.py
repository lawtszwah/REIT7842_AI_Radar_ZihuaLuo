"""Fast ridge checks against simulator_model.md section 5; seeds and z are fixed."""

from dataclasses import replace

import numpy as np
import pytest

from prdbench.simulation import SimConfig, get_simulator

Z = 4.0
RIDGE_SEED = 20261101
N_MAPS = 512
LEVEL_DB = 24.0
SIGMA2 = 1.0


def _config(**overrides) -> SimConfig:
    params = {
        "n_maps": N_MAPS,
        "n_range": 128,
        "n_doppler": 128,
        "n_targets": (0, 0),
        "noise_power": (SIGMA2, SIGMA2),
        "clutter_ridge_db": (LEVEL_DB, LEVEL_DB),
        "ridge_enabled": True,
        "clutter_enabled": False,
        "seed": RIDGE_SEED,
    }
    params.update(overrides)
    return SimConfig(**params)


@pytest.fixture(scope="module")
def ridge_config():
    return _config()


@pytest.fixture(scope="module")
def ridge_batch(ridge_config):
    return get_simulator("prdsim").generate(ridge_config)


def _ridge_snr(config, ranges, dopplers):
    """Signal power divided by sigma2, for the pinned 24 dB profile tests."""
    return (
        10.0 ** (LEVEL_DB / 10.0)
        * np.exp(-((dopplers - config.n_doppler // 2) / config.ridge_width_bins) ** 2)
        * np.exp(-2.0 * ranges / (config.ridge_range_decay * config.n_range))
    )


def _check_profile(config, batch):
    ranges = np.array([0, 16, 32, 64])
    dopplers = config.n_doppler // 2 + np.array([-2, -1, 0, 1, 2])
    measured = batch.maps[:, ranges][:, :, dopplers].astype(np.float64).mean(axis=0)
    s = _ridge_snr(config, ranges[:, None], dopplers[None, :])
    expected = SIGMA2 * (1.0 + s)
    tolerance = Z * SIGMA2 * np.sqrt(1.0 + 2.0 * s) / np.sqrt(config.n_maps)
    assert np.all(np.abs(measured - expected) <= tolerance), (
        f"measured={measured}, expected={expected}, tolerance={tolerance}"
    )


def test_ridge_profile_matches_theory(ridge_config, ridge_batch):
    _check_profile(ridge_config, ridge_batch)


def test_ridge_is_confined(ridge_config, ridge_batch):
    columns = np.abs(np.arange(ridge_config.n_doppler) - ridge_config.n_doppler // 2) >= 8
    cells = ridge_batch.maps[:, :, columns].astype(np.float64)
    tolerance = Z * SIGMA2 / np.sqrt(cells.size)
    assert abs(cells.mean() - SIGMA2) <= tolerance, (
        f"off-ridge mean={cells.mean()}, expected={SIGMA2}, tolerance={tolerance}"
    )


def test_doubling_ridge_width_widens_profile(ridge_config, ridge_batch):
    wide = replace(ridge_config, ridge_width_bins=2.0 * ridge_config.ridge_width_bins)
    wide_batch = get_simulator("prdsim").generate(wide)
    ratios = []
    for config, batch in ((ridge_config, ridge_batch), (wide, wide_batch)):
        d0 = config.n_doppler // 2
        center = batch.maps[:, 0, d0].astype(np.float64).mean()
        sides = batch.maps[:, 0, [d0 - 2, d0 + 2]].astype(np.float64).mean()
        measured = sides / center
        s0 = _ridge_snr(config, 0, d0)
        s2 = _ridge_snr(config, 0, d0 + 2)
        expected_center = SIGMA2 * (1.0 + s0)
        expected = (1.0 + s2) / (1.0 + s0)
        # Delta-method ratio SE: the two side cells are averaged; powers at distinct
        # cells are independent even conditional on the common phase, by circular noise.
        variance_sides = SIGMA2**2 * (1.0 + 2.0 * s2) / (2.0 * config.n_maps)
        variance_center = SIGMA2**2 * (1.0 + 2.0 * s0) / config.n_maps
        tolerance = Z * np.sqrt(variance_sides + expected**2 * variance_center) / expected_center
        assert abs(measured - expected) <= tolerance, (
            f"width={config.ridge_width_bins}: ratio={measured}, "
            f"expected={expected}, tolerance={tolerance}"
        )
        ratios.append(measured)
    assert ratios[1] > ratios[0]


def test_ridge_range_decay_is_honoured(ridge_config):
    config = replace(ridge_config, ridge_range_decay=2.0 * ridge_config.ridge_range_decay)
    _check_profile(config, get_simulator("prdsim").generate(config))


def test_ridge_switch_preserves_noise_and_target_streams():
    config = _config(n_maps=8, n_targets=(1, 3), noise_power=(0.5, 2.0), ridge_enabled=False)
    sim = get_simulator("prdsim")
    before = sim.generate(config)
    enabled = sim.generate(replace(config, ridge_enabled=True))
    after = sim.generate(config)
    fresh = get_simulator("prdsim").generate(config)
    repeated = sim.generate(replace(config, ridge_enabled=True))

    assert before.targets == enabled.targets == after.targets == fresh.targets
    assert before.maps.tobytes() == after.maps.tobytes() == fresh.maps.tobytes()
    assert before.meta == after.meta == fresh.meta
    assert not np.array_equal(before.maps, enabled.maps)
    assert enabled.maps.tobytes() == repeated.maps.tobytes()
    assert enabled.targets == repeated.targets and enabled.meta == repeated.meta
    assert enabled.maps.dtype == np.float32
    assert np.isfinite(enabled.maps).all() and (enabled.maps >= 0).all()
    for index, (off_keys, on_keys) in enumerate(
        zip(before.meta["derived_seeds"], enabled.meta["derived_seeds"], strict=True)
    ):
        assert "ridge" not in off_keys
        assert on_keys["ridge"] == [index, 3]
        assert {key: value for key, value in on_keys.items() if key != "ridge"} == off_keys


def test_ridge_envelope_has_no_speckle():
    config = _config(n_maps=1, clutter_ridge_db=40.0, seed=20261102)
    batch = get_simulator("prdsim").generate(config)
    ranges = np.arange(32, dtype=np.float64)
    log_power = np.log(batch.maps[0, :32, config.n_doppler // 2].astype(np.float64))
    centered = ranges - ranges.mean()
    slope = np.dot(centered, log_power) / np.dot(centered, centered)
    expected = -2.0 / (config.ridge_range_decay * config.n_range)

    # Chosen analytically before running: at 40 dB the weakest tested cell has
    # SNR > 2500, so log-power linearisation is accurate. From Var(P)=1+2s,
    # Var(log P) ~= (1+2s)/(1+s)^2; independent cells give this OLS slope SE.
    # The tolerance is four SEs, with no empirical fit or adjustable multiplier.
    s = 10.0**4 * np.exp(expected * ranges)
    variance_log = (1.0 + 2.0 * s) / (1.0 + s) ** 2
    slope_se = np.sqrt(np.sum(centered**2 * variance_log)) / np.sum(centered**2)
    tolerance = Z * slope_se
    assert abs(slope - expected) <= tolerance, (
        f"log-power slope={slope}, expected={expected}, tolerance={tolerance}"
    )


@pytest.mark.parametrize("name", ["range_bin", "doppler_bin", "snr_db", "noise_power",
                                  "clutter_ridge_db"])
def test_scalar_pair_and_list_bounds_give_identical_batches(name):
    config = _config(n_maps=2, n_targets=(1, 1), range_bin=(40.125, 40.125),
                     doppler_bin=(80.375, 80.375), snr_db=(8.0, 8.0))
    value = getattr(config, name)[0]
    sim = get_simulator("prdsim")
    scalar = sim.generate(replace(config, **{name: value}))
    pair = sim.generate(replace(config, **{name: (value, value)}))
    listed = sim.generate(replace(config, **{name: [value, value]}))
    assert scalar.maps.tobytes() == pair.maps.tobytes() == listed.maps.tobytes()
    assert scalar.targets == pair.targets == listed.targets
    assert scalar.targets[0][0].range_bin == 40.125
    assert scalar.targets[0][0].doppler_bin == 80.375
    # Config provenance retains the supplied representation, so compare stream records.
    assert (
        scalar.meta["derived_seeds"] == pair.meta["derived_seeds"] == listed.meta["derived_seeds"]
    )


@pytest.mark.parametrize("name", ["range_bin", "doppler_bin", "snr_db", "noise_power",
                                  "clutter_ridge_db"])
@pytest.mark.parametrize("bounds", [(2.0, 1.0), (0.0, np.inf), (-np.inf, 0.0), np.nan])
def test_invalid_continuous_bounds_are_rejected(name, bounds):
    with pytest.raises(ValueError, match=name):
        get_simulator("prdsim").generate(_config(n_maps=1, **{name: bounds}))


@pytest.mark.parametrize("name", ["ridge_width_bins", "ridge_range_decay"])
@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_invalid_ridge_shape_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        get_simulator("prdsim").generate(_config(n_maps=1, **{name: value}))
