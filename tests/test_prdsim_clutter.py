"""Fast clutter checks against simulator_model.md section 6; seeds and z are fixed."""

import json
from dataclasses import replace

import numpy as np
import pytest

from prdbench.simulation import SimConfig, get_simulator

Z = 4.0
CLUTTER_SEED = 20261111
N_MAPS = 512
SIGMA2 = 1.0
CENTRE_R = 64.0
CENTRE_D = 32.0
WIDTH = 3.0
LEVEL_DB = 11.0
PATCH_KEYS = {
    "centre_range_bin", "centre_doppler_bin", "width_range_bins",
    "width_doppler_bins", "level_db",
}


def _config(**overrides) -> SimConfig:
    params = {
        "n_maps": N_MAPS,
        "n_range": 128,
        "n_doppler": 128,
        "n_targets": (0, 0),
        "noise_power": (SIGMA2, SIGMA2),
        "ridge_enabled": False,
        "clutter_enabled": True,
        "clutter_patches": (1, 1),
        "clutter_centre_range_bin": CENTRE_R,
        "clutter_centre_doppler_bin": CENTRE_D,
        "clutter_width_bins": WIDTH,
        "clutter_level_db": LEVEL_DB,
        "clutter_speckle_fraction": 0.35,
        "seed": CLUTTER_SEED,
    }
    params.update(overrides)
    return SimConfig(**params)


def _envelope_power(ranges, dopplers):
    """G squared for the independently specified pinned profile."""
    return np.exp(-((ranges - CENTRE_R) ** 2 + (dopplers - CENTRE_D) ** 2) / WIDTH**2)


def _single_patch_statistics(level_db, beta, g2):
    patch_power = SIGMA2 * 10.0 ** (level_db / 10.0) * g2
    variance_complex = SIGMA2 + patch_power * beta**2
    mean_squared = patch_power * (1.0 - beta**2)
    return (
        SIGMA2 + patch_power,
        variance_complex**2 + 2.0 * mean_squared * variance_complex,
    )


def test_clutter_mean_profile_matches_theory():
    config = _config()
    batch = get_simulator("prdsim").generate(config)
    offsets = np.array([-5, -2, 0, 2, 5])
    ranges = int(CENTRE_R) + offsets
    dopplers = int(CENTRE_D) + offsets
    measured = batch.maps[:, ranges][:, :, dopplers].astype(np.float64).mean(axis=0)
    g2 = _envelope_power(ranges[:, None], dopplers[None, :])
    expected, variance = _single_patch_statistics(LEVEL_DB, config.clutter_speckle_fraction, g2)
    tolerance = Z * np.sqrt(variance / N_MAPS)
    assert np.all(np.abs(measured - expected) <= tolerance), (
        f"measured={measured}, expected={expected}, tolerance={tolerance}"
    )


def test_pure_clutter_speckle_is_exponential():
    config = _config(clutter_speckle_fraction=1.0, clutter_level_db=30.0, seed=20261112)
    batch = get_simulator("prdsim").generate(config)
    g2 = _envelope_power(np.arange(128)[:, None], np.arange(128)[None, :])
    keep = g2 >= 0.1
    variance_complex = SIGMA2 + SIGMA2 * 10.0**3 * g2
    x = (batch.maps[:, keep].astype(np.float64) / variance_complex[keep]).reshape(-1)
    tolerance = Z / np.sqrt(x.size)
    cv = x.std(ddof=1) / x.mean()
    assert abs(x.mean() - 1.0) <= tolerance, (
        f"normalised mean={x.mean()}, expected=1, tolerance={tolerance}"
    )
    assert abs(cv - 1.0) <= tolerance, f"CV={cv}, expected=1, tolerance={tolerance}"


def test_zero_speckle_has_a_deterministic_envelope():
    config = _config(n_maps=1, clutter_speckle_fraction=0.0,
                     clutter_level_db=30.0, seed=20261113)
    batch = get_simulator("prdsim").generate(config)
    g2 = _envelope_power(np.arange(128)[:, None], np.arange(128)[None, :])
    keep = g2 >= 0.01
    expected, variance = _single_patch_statistics(30.0, 0.0, g2[keep])
    measured = batch.maps[0, keep].astype(np.float64)
    tolerance = Z * np.sqrt(variance)
    assert np.all(np.abs(measured - expected) <= tolerance), (
        f"measured={measured}, expected={expected}, tolerance={tolerance}"
    )


def test_overlapping_patches_have_independent_phases():
    config = _config(n_maps=1024, clutter_patches=(2, 2), seed=20261114)
    batch = get_simulator("prdsim").generate(config)
    measured = batch.maps[:, int(CENTRE_R), int(CENTRE_D)].astype(np.float64).mean()
    amplitude_squared = SIGMA2 * 10.0 ** (LEVEL_DB / 10.0)
    a2 = amplitude_squared * (1.0 - config.clutter_speckle_fraction**2)
    v = SIGMA2 + 2.0 * amplitude_squared * config.clutter_speckle_fraction**2
    # The coherent power is a2 * |exp(j phi1) + exp(j phi2)|^2:
    # independent phases give mean 2*a2 and variance 2*a2^2. Total variance is
    # E[Var(P | phases)] + Var(E[P | phases]) = v^2 + 4*a2*v + 2*a2^2.
    variance = v**2 + 4.0 * a2 * v + 2.0 * a2**2
    expected = SIGMA2 + 2.0 * amplitude_squared
    tolerance = Z * np.sqrt(variance / config.n_maps)
    assert abs(measured - expected) <= tolerance, (
        f"overlap mean={measured}, expected={expected}, tolerance={tolerance}"
    )
    assert all(len(patches) == 2 for patches in batch.meta["clutter"])


@pytest.mark.parametrize("default_centres", [True, False])
def test_clutter_counts_and_recorded_ranges(default_centres):
    range_bounds = (0.0, 31.0) if default_centres else (5.0, 25.0)
    doppler_bounds = (0.0, 47.0) if default_centres else (3.0, 42.0)
    config = _config(
        n_maps=128, n_range=32, n_doppler=48, clutter_patches=(0, 6),
        clutter_centre_range_bin=None if default_centres else range_bounds,
        clutter_centre_doppler_bin=None if default_centres else doppler_bounds,
        clutter_width_bins=(1.5, 4.0), clutter_level_db=(4.0, 11.0), seed=20261115,
    )
    batch = get_simulator("prdsim").generate(config)
    records = batch.meta["clutter"]
    assert len(records) == config.n_maps
    assert {len(patches) for patches in records} == set(range(7))
    for patches in records:
        for patch in patches:
            assert set(patch) == PATCH_KEYS
            assert range_bounds[0] <= patch["centre_range_bin"] <= range_bounds[1]
            assert doppler_bounds[0] <= patch["centre_doppler_bin"] <= doppler_bounds[1]
            assert 1.5 <= patch["width_range_bins"] <= 4.0
            assert 1.5 <= patch["width_doppler_bins"] <= 4.0
            assert 4.0 <= patch["level_db"] <= 11.0
    assert any(patch["width_range_bins"] != patch["width_doppler_bins"]
               for patches in records for patch in patches)
    assert batch.meta == json.loads(json.dumps(batch.meta))


def test_clutter_switch_preserves_targets_and_ridge_stream():
    config = _config(
        n_maps=8, n_targets=(1, 3), range_bin=(15.0, 100.0), doppler_bin=(70.0, 110.0),
        ridge_enabled=True, clutter_enabled=False, noise_power=(0.5, 2.0),
        clutter_patches=(1, 3), seed=20261116,
    )
    sim = get_simulator("prdsim")
    before = sim.generate(config)
    enabled = sim.generate(replace(config, clutter_enabled=True))
    after = sim.generate(config)
    fresh = get_simulator("prdsim").generate(config)
    repeated = sim.generate(replace(config, clutter_enabled=True))
    empty_enabled = sim.generate(replace(config, clutter_enabled=True, clutter_patches=(0, 0)))

    assert (
        before.targets == enabled.targets == after.targets == fresh.targets == empty_enabled.targets
    )
    assert (
        before.maps.tobytes() == after.maps.tobytes() == fresh.maps.tobytes()
        == empty_enabled.maps.tobytes()
    )
    assert before.meta == after.meta == fresh.meta
    assert before.meta["clutter"] == [[] for _ in range(config.n_maps)]
    assert not np.array_equal(before.maps, enabled.maps)
    assert enabled.maps.tobytes() == repeated.maps.tobytes()
    assert enabled.targets == repeated.targets and enabled.meta == repeated.meta
    for index, (off_keys, on_keys) in enumerate(
        zip(before.meta["derived_seeds"], enabled.meta["derived_seeds"], strict=True)
    ):
        assert "clutter" not in off_keys
        assert on_keys["clutter"] == [index, 4]
        assert {key: value for key, value in on_keys.items() if key != "clutter"} == off_keys


def test_clutter_can_overlap_targets_and_the_zero_doppler_ridge():
    config = _config(
        n_maps=2, n_targets=(1, 1), range_bin=64.5, doppler_bin=64.5,
        ridge_enabled=True, clutter_centre_range_bin=64.5, clutter_centre_doppler_bin=64.5,
    )
    sim = get_simulator("prdsim")
    enabled = sim.generate(config)
    disabled = sim.generate(replace(config, clutter_enabled=False))
    assert enabled.targets == disabled.targets
    for patches in enabled.meta["clutter"]:
        assert patches[0]["centre_range_bin"] == 64.5
        assert patches[0]["centre_doppler_bin"] == 64.5


@pytest.mark.parametrize("beta", [-0.1, 1.1, np.inf, -np.inf, np.nan])
def test_invalid_clutter_speckle_fraction_is_rejected(beta):
    with pytest.raises(ValueError, match="clutter_speckle_fraction"):
        get_simulator("prdsim").generate(_config(n_maps=1, clutter_speckle_fraction=beta))


@pytest.mark.parametrize("widths", [0.0, -1.0, (-1.0, 3.0), (0.0, 3.0)])
def test_non_positive_clutter_width_is_rejected(widths):
    with pytest.raises(ValueError, match="clutter_width_bins"):
        get_simulator("prdsim").generate(_config(n_maps=1, clutter_width_bins=widths))


@pytest.mark.parametrize("name", ["clutter_level_db", "clutter_width_bins",
                                  "clutter_centre_range_bin", "clutter_centre_doppler_bin"])
@pytest.mark.parametrize("bounds", [(2.0, 1.0), (0.0, np.inf), (-np.inf, 0.0), np.nan])
def test_invalid_clutter_bounds_are_rejected(name, bounds):
    with pytest.raises(ValueError, match=name):
        get_simulator("prdsim").generate(_config(n_maps=1, **{name: bounds}))


@pytest.mark.parametrize("name", ["clutter_level_db", "clutter_width_bins",
                                  "clutter_centre_range_bin", "clutter_centre_doppler_bin"])
def test_scalar_pair_and_list_clutter_bounds_are_equivalent(name):
    config = _config(n_maps=2)
    value = getattr(config, name)
    sim = get_simulator("prdsim")
    scalar = sim.generate(config)
    pair = sim.generate(replace(config, **{name: (value, value)}))
    listed = sim.generate(replace(config, **{name: [value, value]}))
    assert scalar.maps.tobytes() == pair.maps.tobytes() == listed.maps.tobytes()
    assert scalar.meta["clutter"] == pair.meta["clutter"] == listed.meta["clutter"]


@pytest.mark.parametrize("counts", [(-1, 1), (2, 1), (0.0, 1.5)])
def test_invalid_clutter_counts_are_rejected(counts):
    with pytest.raises(ValueError, match="clutter_patches"):
        get_simulator("prdsim").generate(_config(n_maps=1, clutter_patches=counts))
