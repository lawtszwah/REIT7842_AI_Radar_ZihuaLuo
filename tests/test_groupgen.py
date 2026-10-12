"""The adapter for the research group's MATLAB generator.

Planning, placement and the truth formulas are pure Python and always run. Tests that
call the generator are marked `matlab` and skip where MATLAB or the generator is absent
(CI has neither).
"""

from dataclasses import replace

import numpy as np
import pytest

from prdbench.simulation import SimConfig, get_simulator
from prdbench.simulation.groupgen import DEFAULTS, _grids, delay_samples, doppler_hz

SIM = get_simulator("groupgen")
FMAP, GRIDX = _grids(DEFAULTS)
FRAMES = DEFAULTS["range_size"]

needs_matlab = pytest.mark.skipif(not SIM.available(), reason="MATLAB or generator absent")


def _config(**overrides) -> SimConfig:
    params = dict(n_maps=2 * FRAMES, n_range=len(GRIDX), n_doppler=len(FMAP),
                  n_targets=(1, 1), snr_db=(20.0, 20.0), seed=11)
    params.update(overrides)
    return SimConfig(**params)


# -- truth formulas, measured 2026-10-12 -------------------------------------


def test_delay_is_bistatic_path_minus_baseline_in_samples():
    p = DEFAULTS
    pos = np.array([[5000.0, 3000.0]])
    tx = np.array(p["tx"])
    expected = (np.linalg.norm(pos[0] - tx) + np.linalg.norm(pos[0]) - np.linalg.norm(tx)) \
        * p["fs"] / p["c"]
    assert delay_samples(pos, p)[0] == pytest.approx(expected)
    assert expected == pytest.approx(71.61, abs=0.01)  # the default scene's first frame


def test_doppler_is_monostatic_about_the_receiver():
    p = DEFAULTS
    lam = p["c"] / p["fc"]
    pos = np.array([[3000.0, 4000.0]])                     # unit vector (0.6, 0.8) from Rx
    receding = np.array([60.0, 80.0])                      # 100 m/s straight away from Rx
    assert doppler_hz(pos, receding, p)[0] == pytest.approx(2 * 100 / lam)
    assert doppler_hz(pos, -receding, p)[0] == pytest.approx(-2 * 100 / lam)
    tangential = np.array([-80.0, 60.0])
    assert doppler_hz(pos, tangential, p)[0] == pytest.approx(0.0, abs=1e-9)


# -- planning ----------------------------------------------------------------


def test_plan_is_deterministic_and_seeded():
    a, b = SIM.plan(_config()), SIM.plan(_config())
    c = SIM.plan(_config(seed=12))
    assert a == b
    assert a != c
    assert len(a) == 2
    assert all(0 <= s["seed"] < 2**32 for s in a)


def test_every_planned_frame_sits_inside_the_window_and_off_zero_doppler():
    p = DEFAULTS
    scenes = SIM.plan(_config(n_maps=40 * FRAMES, snr_db=(0.0, 20.0)))
    step = FMAP[1] - FMAP[0]
    for s in scenes:
        v = s["speed_kmh"] / 3.6 * np.array([np.cos(np.radians(s["heading_deg"])),
                                             np.sin(np.radians(s["heading_deg"]))])
        t = np.arange(FRAMES) * p["update_interval"]
        pos = np.asarray(s["initial_position"]) + t[:, None] * v
        cols = delay_samples(pos, p) - (GRIDX[0] - 1)
        fd = doppler_hz(pos, v, p)
        assert cols.min() >= p["edge_margin_cells"] - 1e-9
        assert cols.max() <= len(GRIDX) - 1 - p["edge_margin_cells"] + 1e-9
        assert np.abs(fd).max() <= FMAP[-1] - p["edge_margin_cells"] * step + 1e-9
        assert np.abs(fd).min() >= p["zero_doppler_margin_cells"] * step - 1e-9
        assert 0.0 <= s["snr_db"] <= 20.0


@pytest.mark.parametrize("overrides, message", [
    (dict(n_range=64), "n_range x n_doppler"),
    (dict(n_doppler=64), "n_range x n_doppler"),
    (dict(n_targets=(1, 3)), "one target"),
    (dict(n_maps=15), "multiple of"),
    (dict(extra={"groupgen": {"fmap": [-250.0, 1.0, 400.0]}}), "symmetric"),
])
def test_plan_rejects_configs_the_generator_cannot_honour(overrides, message):
    with pytest.raises(ValueError, match=message):
        SIM.plan(_config(**overrides))


def test_version_names_the_exact_generator_files():
    assert SIM.version.startswith(f"{SIM.adapter_version}+gen.")


# -- against the generator itself ---------------------------------------------


@pytest.fixture(scope="module")
def strong_batch():
    """Five strong-target scenes (50 frames). About a minute of MATLAB."""
    return SIM.generate(_config(n_maps=5 * FRAMES, seed=2026))


@pytest.mark.matlab
@needs_matlab
def test_output_contract(strong_batch):
    assert strong_batch.maps.shape == (5 * FRAMES, len(GRIDX), len(FMAP))
    assert strong_batch.maps.dtype == np.float32
    assert np.isfinite(strong_batch.maps).all() and (strong_batch.maps >= 0).all()
    assert all(len(t) == 1 for t in strong_batch.targets)
    assert strong_batch.meta["simulator"] == "groupgen"
    assert strong_batch.meta["simulator_version"] == SIM.version


@pytest.mark.matlab
@needs_matlab
def test_recorded_truth_is_the_strongest_off_ridge_peak(strong_batch):
    """The truth calibration: at 20 dB every target is the brightest thing off the ridge,
    and its recorded position must be within one cell of that peak in both axes."""
    zero = len(FMAP) // 2
    guard = DEFAULTS["zero_doppler_margin_cells"]
    for i, (power, (target,)) in enumerate(zip(strong_batch.maps, strong_batch.targets,
                                                strict=True)):
        off_ridge = power.copy()
        off_ridge[:, zero - guard:zero + guard + 1] = 0
        r, d = np.unravel_index(off_ridge.argmax(), off_ridge.shape)
        assert abs(r - target.range_bin) <= 1.0, f"map {i}: range {r} vs {target.range_bin:.2f}"
        assert abs(d - target.doppler_bin) <= 1.0, (
            f"map {i}: Doppler {d} vs {target.doppler_bin:.2f}")


@pytest.mark.matlab
@needs_matlab
def test_same_seed_same_maps_and_new_seed_new_maps():
    cfg = _config(n_maps=FRAMES, seed=7)
    a, b = SIM.generate(cfg), SIM.generate(cfg)
    c = SIM.generate(replace(cfg, seed=8))
    assert np.array_equal(a.maps, b.maps)
    assert a.targets == b.targets
    assert not np.array_equal(a.maps, c.maps)
