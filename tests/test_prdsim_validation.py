"""Executable validation specification for the reportable ``prdsim`` simulator.

The numerical settings and validation seeds are loaded from the frozen protocol
configuration. The analytical gates are intentionally separate from ordinary
unit tests: they are run and recorded before M2 can be declared complete.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from prdbench.detectors.cfar import CACFAR
from prdbench.simulation import RDMapSimulator, SimConfig, get_simulator
from prdbench.simulation.interface import SimBatch, Target

SIMULATOR = "prdsim"
REPO_ROOT = Path(__file__).resolve().parents[1]

with (REPO_ROOT / "configs/protocol/full.yaml").open() as f:
    FULL_PROTOCOL = yaml.safe_load(f)

VALIDATION_SPEC = FULL_PROTOCOL["simulator_validation"]
VALIDATION_ROOT_SEEDS = tuple(int(x) for x in VALIDATION_SPEC["root_seeds"])
VALIDATION_STREAMS = tuple(VALIDATION_SPEC["seed_streams"])
Z = float(VALIDATION_SPEC["acceptance_z"])
MAP_SIZE = int(VALIDATION_SPEC["map_size"])
PFA_POINTS = tuple(float(x) for x in FULL_PROTOCOL["protocol"]["pfa_points"])

GATE1 = VALIDATION_SPEC["gate1"]
GATE1_NOISE_POWERS = tuple(float(x) for x in GATE1["noise_powers"])
GATE1_MAPS = int(GATE1["n_maps"])
GATE1_QUANTILES = tuple(float(x) for x in GATE1["quantiles"])

GATE2 = VALIDATION_SPEC["gate2"]
GATE2_OFFSETS = tuple(float(x) for x in GATE2["offsets"])
GATE2_MAPS = int(GATE2["n_maps"])
GATE2_CHUNK = int(GATE2["chunk_size"])
GATE2_SNR_DB = float(GATE2["snr_db"])
GATE2_SIGMA2 = float(GATE2["noise_power"])
GATE2_CELL = MAP_SIZE // 2
GATE2_METADATA_TOL_DB = float(GATE2["metadata_tolerance_db"])

GATE3 = VALIDATION_SPEC["gate3"]
GATE3_TRAIN = int(GATE3["train"])
GATE3_GUARD = int(GATE3["guard"])
GATE3_CHUNK = int(GATE3["chunk_size"])
MIN_ELIGIBLE_CELLS = int(GATE3["min_eligible_cells"])
BOOTSTRAP_RESAMPLES = int(GATE3["bootstrap_resamples"])
BOOTSTRAP_PERCENTILES = tuple(float(x) for x in GATE3["interval_percentiles"])
GATE3_NOISE_POWER = float(GATE3["noise_power"])
GATE3_PERMITTED_REPEATS = int(FULL_PROTOCOL["protocol"]["gate3_permitted_repeats"])

if len(VALIDATION_ROOT_SEEDS) != GATE3_PERMITTED_REPEATS + 1:
    raise ValueError("simulator_validation.root_seeds must contain first run plus repeats")

pytestmark = pytest.mark.validation


@pytest.fixture(scope="module")
def sim():
    """The validated simulator. Raises ``NotImplementedError`` until M2 exists."""
    return get_simulator(SIMULATOR)


def seed_sequence(root_seed: int, stream_name: str, *child_path: int) -> np.random.SeedSequence:
    """Return one stable child in the frozen validation seed tree."""
    try:
        stream_index = VALIDATION_STREAMS.index(stream_name)
    except ValueError as exc:
        raise KeyError(f"unknown validation seed stream {stream_name!r}") from exc
    return np.random.SeedSequence(root_seed, spawn_key=(stream_index, *child_path))


def simulator_seed(root_seed: int, stream_name: str, *child_path: int) -> int:
    """Materialise a uint64 child seed for the integer-based simulator API."""
    seq = seed_sequence(root_seed, stream_name, *child_path)
    return int(seq.generate_state(1, dtype=np.uint64)[0])


def rng_stream(root_seed: int, stream_name: str, *child_path: int) -> np.random.Generator:
    return np.random.default_rng(seed_sequence(root_seed, stream_name, *child_path))


def seed_record(root_seed: int, stream_name: str, *child_path: int) -> dict:
    """JSON-serialisable provenance for one simulator or analysis child seed."""
    seq = seed_sequence(root_seed, stream_name, *child_path)
    return {
        "root_seed": root_seed,
        "stream": stream_name,
        "spawn_key": list(seq.spawn_key),
        "materialised_seed": int(seq.generate_state(1, dtype=np.uint64)[0]),
    }


def chi(dr: float, dd: float) -> float:
    """Normalised separable response using NumPy's pi-normalised sinc."""
    return float(np.sinc(dr) * np.sinc(dd))


def n_training_cells(train: int, guard: int) -> int:
    outer = 2 * (train + guard) + 1
    inner = 2 * guard + 1
    return outer * outer - inner * inner


def ca_cfar_alpha(pfa: float, n_tr: int) -> float:
    """CA-CFAR multiplier for i.i.d. exponential powers.

    Richards, *Fundamentals of Radar Signal Processing*, 3rd ed., Sec. 6.5.3,
    Eq. (6.140), p. 397. Exact only under that premise, which Gate 3 tests.
    """
    return n_tr * (pfa ** (-1.0 / n_tr) - 1.0)


def _base_config(**overrides) -> SimConfig:
    params = {
        "n_range": MAP_SIZE,
        "n_doppler": MAP_SIZE,
        "ridge_enabled": False,
        "clutter_enabled": False,
    }
    params.update(overrides)
    return SimConfig(**params)


def noise_only_batch(sim, *, n_maps: int, noise_power: float, seed: int) -> SimBatch:
    cfg = _base_config(
        n_maps=n_maps,
        n_targets=(0, 0),
        noise_power=(noise_power, noise_power),
        seed=seed,
    )
    return sim.generate(cfg)


# -- interface and component contract ---------------------------------------


def test_prdsim_satisfies_protocol(sim):
    assert isinstance(sim, RDMapSimulator)


def test_simconfig_exposes_explicit_component_switches():
    fields = {f.name for f in dataclasses.fields(SimConfig)}
    assert {"ridge_enabled", "clutter_enabled"} <= fields


def test_target_records_signal_only_peak_snr():
    fields = {f.name for f in dataclasses.fields(Target)}
    assert {"snr_db", "peak_snr_db"} <= fields


def test_prdsim_output_contract_and_reproducibility(sim):
    seed = simulator_seed(VALIDATION_ROOT_SEEDS[0], "contract", 0)
    cfg = _base_config(n_maps=4, n_targets=(1, 1), seed=seed)
    first = sim.generate(cfg)
    second = sim.generate(cfg)
    different_seed = simulator_seed(VALIDATION_ROOT_SEEDS[0], "contract", 1)
    third = sim.generate(dataclasses.replace(cfg, seed=different_seed))

    assert first.maps.shape == (4, MAP_SIZE, MAP_SIZE)
    assert first.maps.dtype == np.float32
    assert np.isfinite(first.maps).all()
    assert (first.maps >= 0).all(), "maps are linear power"
    assert len(first.targets) == 4
    assert all(len(targets) == 1 for targets in first.targets)
    assert all(isinstance(target, Target) for targets in first.targets for target in targets)

    assert np.array_equal(first.maps, second.maps)
    assert first.targets == second.targets
    assert first.meta == second.meta
    assert not np.array_equal(first.maps, third.maps), "the simulator must consume its seed"

    required_meta = {"simulator", "simulator_version", "config", "seed", "derived_seeds"}
    assert required_meta <= first.meta.keys()
    assert first.meta["simulator"] == SIMULATOR
    assert first.meta["simulator_version"] == sim.version
    assert first.meta["seed"] == seed


def test_component_switches_change_only_the_selected_component_stream(sim):
    """Switches must work without perturbing target placement or target SNR draws."""
    seed = simulator_seed(VALIDATION_ROOT_SEEDS[0], "component_switches", 0)
    base = _base_config(
        n_maps=4,
        n_targets=(1, 1),
        range_bin=(48.25, 48.25),
        doppler_bin=(80.25, 80.25),
        snr_db=(8.0, 8.0),
        noise_power=(1.0, 1.0),
        clutter_patches=(1, 1),
        clutter_ridge_db=(24.0, 24.0),
        seed=seed,
    )
    ridge = dataclasses.replace(base, ridge_enabled=True)
    clutter = dataclasses.replace(base, clutter_enabled=True)

    plain_batch = sim.generate(base)
    ridge_batch = sim.generate(ridge)
    clutter_batch = sim.generate(clutter)

    assert plain_batch.targets == ridge_batch.targets == clutter_batch.targets
    assert not np.array_equal(plain_batch.maps, ridge_batch.maps)
    assert not np.array_equal(plain_batch.maps, clutter_batch.maps)


# -- Gate 1: noise distribution and independence ---------------------------


@dataclass(frozen=True)
class Gate1Sample:
    maps: np.ndarray
    seed: dict


@pytest.fixture(scope="module")
def gate1_samples(sim):
    out = {}
    root_seed = VALIDATION_ROOT_SEEDS[0]
    for noise_index, sigma2 in enumerate(GATE1_NOISE_POWERS):
        record = seed_record(root_seed, "gate1_noise", noise_index)
        batch = noise_only_batch(
            sim,
            n_maps=GATE1_MAPS,
            noise_power=sigma2,
            seed=record["materialised_seed"],
        )
        out[sigma2] = Gate1Sample(batch.maps, record)
    return out


@pytest.mark.parametrize("sigma2", GATE1_NOISE_POWERS)
def test_gate1_mean_power(gate1_samples, sigma2):
    x = gate1_samples[sigma2].maps.reshape(-1).astype(np.float64)
    tolerance = Z * sigma2 / math.sqrt(x.size)
    assert abs(x.mean() - sigma2) <= tolerance, (
        f"noise floor {x.mean():.6g} vs sigma^2 {sigma2:.6g}, tolerance {tolerance:.3g}"
    )


@pytest.mark.parametrize("sigma2", GATE1_NOISE_POWERS)
def test_gate1_coefficient_of_variation(gate1_samples, sigma2):
    x = gate1_samples[sigma2].maps.reshape(-1).astype(np.float64)
    cv = x.std(ddof=1) / x.mean()
    tolerance = Z / math.sqrt(x.size)
    assert abs(cv - 1.0) <= tolerance, f"CV {cv:.6f} vs 1, tolerance {tolerance:.3g}"


@pytest.mark.parametrize("sigma2", GATE1_NOISE_POWERS)
@pytest.mark.parametrize("q", GATE1_QUANTILES)
def test_gate1_tail_quantiles(gate1_samples, sigma2, q):
    x = gate1_samples[sigma2].maps.reshape(-1).astype(np.float64)
    expected = -sigma2 * math.log(1.0 - q)
    measured = float(np.quantile(x, q))
    tolerance = Z * sigma2 * math.sqrt(q / (x.size * (1.0 - q)))
    assert abs(measured - expected) <= tolerance, (
        f"q={q}: {measured:.6g} vs {expected:.6g}, tolerance {tolerance:.3g}"
    )


@pytest.mark.parametrize("sigma2", GATE1_NOISE_POWERS)
@pytest.mark.parametrize("axis,label", [(1, "range"), (2, "doppler")])
def test_gate1_adjacent_cells_are_uncorrelated(gate1_samples, sigma2, axis, label):
    x = gate1_samples[sigma2].maps.astype(np.float64)
    n = x.shape[axis]
    first = np.take(x, np.arange(0, n - 1, 2), axis=axis).reshape(-1)
    second = np.take(x, np.arange(1, n, 2), axis=axis).reshape(-1)
    rho = float(np.corrcoef(first, second)[0, 1])
    tolerance = Z / math.sqrt(first.size)
    assert abs(rho) <= tolerance, (
        f"sigma^2={sigma2:g}: lag-1 correlation along {label} is {rho:.5f}, "
        f"tolerance {tolerance:.3g}"
    )


# -- Gate 2: single-target peak and straddle loss --------------------------


@dataclass(frozen=True)
class Gate2Result:
    mean_power: float
    targets: tuple[Target, ...]
    seeds: tuple[dict, ...]


def _single_target_peak_and_truth(
    sim, *, offset_index: int, dr: float, dd: float
) -> Gate2Result:
    total = 0.0
    targets = []
    seeds = []
    done = 0
    chunk_index = 0
    root_seed = VALIDATION_ROOT_SEEDS[0]

    while done < GATE2_MAPS:
        size = min(GATE2_CHUNK, GATE2_MAPS - done)
        record = seed_record(root_seed, "gate2_targets", offset_index, chunk_index)
        cfg = _base_config(
            n_maps=size,
            n_targets=(1, 1),
            range_bin=(GATE2_CELL + dr, GATE2_CELL + dr),
            doppler_bin=(GATE2_CELL + dd, GATE2_CELL + dd),
            snr_db=(GATE2_SNR_DB, GATE2_SNR_DB),
            noise_power=(GATE2_SIGMA2, GATE2_SIGMA2),
            seed=record["materialised_seed"],
        )
        batch = sim.generate(cfg)
        assert len(batch.targets) == size
        assert all(len(map_targets) == 1 for map_targets in batch.targets)
        total += float(batch.maps[:, GATE2_CELL, GATE2_CELL].astype(np.float64).sum())
        targets.extend(map_targets[0] for map_targets in batch.targets)
        seeds.append(record)
        done += size
        chunk_index += 1

    assert len(targets) == GATE2_MAPS
    return Gate2Result(total / GATE2_MAPS, tuple(targets), tuple(seeds))


@pytest.fixture(scope="module")
def gate2_results(sim):
    results = {}
    offset_index = 0
    for dr in GATE2_OFFSETS:
        for dd in GATE2_OFFSETS:
            results[(dr, dd)] = _single_target_peak_and_truth(
                sim, offset_index=offset_index, dr=dr, dd=dd
            )
            offset_index += 1
    return results


@pytest.mark.slow
@pytest.mark.parametrize("dd", GATE2_OFFSETS)
@pytest.mark.parametrize("dr", GATE2_OFFSETS)
def test_gate2_peak_power_matches_nominal_snr(gate2_results, dr, dd):
    measured = gate2_results[(dr, dd)].mean_power
    snr_lin = 10.0 ** (GATE2_SNR_DB / 10.0)
    snr_eff = snr_lin * chi(dr, dd) ** 2
    expected = GATE2_SIGMA2 * (1.0 + snr_eff)
    sd = GATE2_SIGMA2 * math.sqrt(1.0 + 2.0 * snr_eff)
    tolerance = Z * sd / math.sqrt(GATE2_MAPS)
    assert abs(measured - expected) <= tolerance, (
        f"offset ({dr}, {dd}): peak {measured:.6g} vs {expected:.6g}, "
        f"tolerance {tolerance:.3g}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("dd", GATE2_OFFSETS)
@pytest.mark.parametrize("dr", GATE2_OFFSETS)
def test_gate2_all_target_metadata_matches_straddle_loss(gate2_results, dr, dd):
    result = gate2_results[(dr, dd)]
    expected_peak = GATE2_SNR_DB + 20.0 * math.log10(abs(chi(dr, dd)))
    expected_range = GATE2_CELL + dr
    expected_doppler = GATE2_CELL + dd

    assert all(target.range_bin == expected_range for target in result.targets)
    assert all(target.doppler_bin == expected_doppler for target in result.targets)
    assert all(target.snr_db == GATE2_SNR_DB for target in result.targets)
    errors = np.fromiter(
        (abs(target.peak_snr_db - expected_peak) for target in result.targets),
        dtype=np.float64,
        count=GATE2_MAPS,
    )
    assert float(errors.max()) <= GATE2_METADATA_TOL_DB


def test_gate2_worst_case_straddle_loss_is_minus_7_84_db():
    worst = 20.0 * math.log10(abs(chi(0.5, 0.5)))
    assert abs(worst - (-7.8448)) < 5e-4


# -- Gate 3: CA-CFAR design false-alarm probability ------------------------


@dataclass(frozen=True)
class Gate3PointResult:
    design_pfa: float
    alarm_count: int
    eligible_cells: int
    measured_pfa: float
    interval_low: float
    interval_high: float
    binomial_se: float
    alpha: float
    passed: bool


@dataclass(frozen=True)
class Gate3RunResult:
    root_seed: int
    simulator: str
    simulator_version: str
    n_maps: int
    cells_per_map: int
    train: int
    guard: int
    n_training_cells: int
    map_config: dict
    map_seeds: tuple[dict, ...]
    bootstrap_seed: dict
    points: tuple[Gate3PointResult, ...]

    @property
    def failed_points(self) -> tuple[float, ...]:
        return tuple(point.design_pfa for point in self.points if not point.passed)


@dataclass(frozen=True)
class Gate3ValidationResult:
    first: Gate3RunResult
    repeat: Gate3RunResult | None

    @property
    def passed(self) -> bool:
        if not self.first.failed_points:
            return self.repeat is None
        return (
            len(self.first.failed_points) == 1
            and self.repeat is not None
            and not self.repeat.failed_points
        )


def _gate3_border() -> int:
    return GATE3_TRAIN + GATE3_GUARD


def _gate3_maps_needed() -> int:
    cells_per_map = (MAP_SIZE - 2 * _gate3_border()) ** 2
    return math.ceil(MIN_ELIGIBLE_CELLS / cells_per_map)


def _run_gate3_once(sim, root_seed: int) -> Gate3RunResult:
    border = _gate3_border()
    n_maps = _gate3_maps_needed()
    n_tr = n_training_cells(GATE3_TRAIN, GATE3_GUARD)
    detector = CACFAR(train=GATE3_TRAIN, guard=GATE3_GUARD)
    cells_per_map = (MAP_SIZE - 2 * border) ** 2

    alarms = {pfa: [] for pfa in PFA_POINTS}
    map_seeds = []
    done = 0
    chunk_index = 0
    while done < n_maps:
        size = min(GATE3_CHUNK, n_maps - done)
        record = seed_record(root_seed, "gate3_noise", chunk_index)
        batch = noise_only_batch(
            sim,
            n_maps=size,
            noise_power=GATE3_NOISE_POWER,
            seed=record["materialised_seed"],
        )
        scores = detector.score(batch.maps)[:, border:-border, border:-border]
        for pfa in PFA_POINTS:
            alpha = ca_cfar_alpha(pfa, n_tr)
            alarms[pfa].append((scores > alpha).sum(axis=(1, 2)))
        map_seeds.append(record)
        done += size
        chunk_index += 1

    counts_by_pfa = {pfa: np.concatenate(parts) for pfa, parts in alarms.items()}
    eligible = n_maps * cells_per_map
    assert eligible >= MIN_ELIGIBLE_CELLS

    bootstrap_record = seed_record(root_seed, "analysis_bootstrap", 0)
    bootstrap_rng = rng_stream(root_seed, "analysis_bootstrap", 0)
    indices = bootstrap_rng.integers(
        0,
        n_maps,
        size=(BOOTSTRAP_RESAMPLES, n_maps),
        dtype=np.int32,
    )

    points = []
    for pfa in PFA_POINTS:
        counts = counts_by_pfa[pfa]
        alarm_count = int(counts.sum())
        measured = alarm_count / eligible
        rates = counts[indices].sum(axis=1) / eligible
        low, high = np.percentile(rates, BOOTSTRAP_PERCENTILES)
        binomial_se = math.sqrt(pfa * (1.0 - pfa) / eligible)
        points.append(
            Gate3PointResult(
                design_pfa=pfa,
                alarm_count=alarm_count,
                eligible_cells=eligible,
                measured_pfa=measured,
                interval_low=float(low),
                interval_high=float(high),
                binomial_se=binomial_se,
                alpha=ca_cfar_alpha(pfa, n_tr),
                passed=bool(low <= pfa <= high),
            )
        )

    return Gate3RunResult(
        root_seed=root_seed,
        simulator=sim.name,
        simulator_version=sim.version,
        n_maps=n_maps,
        cells_per_map=cells_per_map,
        train=GATE3_TRAIN,
        guard=GATE3_GUARD,
        n_training_cells=n_tr,
        map_config={
            "n_range": MAP_SIZE,
            "n_doppler": MAP_SIZE,
            "n_targets": [0, 0],
            "noise_power": [GATE3_NOISE_POWER, GATE3_NOISE_POWER],
            "ridge_enabled": False,
            "clutter_enabled": False,
        },
        map_seeds=tuple(map_seeds),
        bootstrap_seed=bootstrap_record,
        points=tuple(points),
    )


def _run_gate3_with_predeclared_repeat(sim) -> Gate3ValidationResult:
    first = _run_gate3_once(sim, VALIDATION_ROOT_SEEDS[0])
    repeat = None
    if GATE3_PERMITTED_REPEATS == 1 and len(first.failed_points) == 1:
        repeat = _run_gate3_once(sim, VALIDATION_ROOT_SEEDS[1])
    return Gate3ValidationResult(first=first, repeat=repeat)


@pytest.fixture(scope="module")
def gate3_result(sim):
    return _run_gate3_with_predeclared_repeat(sim)


def _format_gate3_failure(result: Gate3ValidationResult) -> str:
    lines = []
    for label, run in (("first", result.first), ("repeat", result.repeat)):
        if run is None:
            continue
        for point in run.points:
            lines.append(
                f"{label} root={run.root_seed}, design={point.design_pfa:.0e}, "
                f"measured={point.measured_pfa:.4e}, alarms={point.alarm_count}, "
                f"99% interval=[{point.interval_low:.4e}, {point.interval_high:.4e}], "
                f"passed={point.passed}"
            )
    return "\n".join(lines)


@pytest.mark.slow
def test_gate3_ca_cfar_achieves_all_design_pfa_points(gate3_result):
    assert gate3_result.passed, _format_gate3_failure(gate3_result)


def _run_as_report_dict(run: Gate3RunResult) -> dict:
    payload = asdict(run)
    payload["failed_points"] = list(run.failed_points)
    payload["passed"] = not run.failed_points
    return payload


def _write_gate3_report(result: Gate3ValidationResult, path: Path) -> dict:
    report = {
        "schema_version": 1,
        "gate": "ca_cfar_design_pfa",
        "passed": result.passed,
        "validation_config": VALIDATION_SPEC,
        "repeat_policy": {
            "permitted_repeats": GATE3_PERMITTED_REPEATS,
            "trigger": "exactly_one_first_run_design_point_failed",
        },
        "runs": [_run_as_report_dict(result.first)],
    }
    if result.repeat is not None:
        report["runs"].append(_run_as_report_dict(result.repeat))

    # Normalise tuples and NumPy-compatible scalar values through the same JSON
    # representation that is persisted, so callers can compare the returned
    # object directly with a subsequent json.load().
    report = json.loads(json.dumps(report))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


@pytest.mark.slow
def test_gate3_records_the_numbers_not_just_a_verdict(gate3_result, tmp_path, request):
    requested_path = request.config.getoption("--prdsim-validation-report")
    path = requested_path or tmp_path / "prdsim_gate3_validation.json"
    expected = _write_gate3_report(gate3_result, path)
    recorded = json.loads(path.read_text())

    assert recorded == expected
    assert recorded["passed"] == gate3_result.passed
    assert len(recorded["runs"]) in {1, 2}
    for run in recorded["runs"]:
        assert run["simulator"] == SIMULATOR
        assert run["simulator_version"]
        assert run["map_seeds"]
        assert run["bootstrap_seed"]
        assert len(run["points"]) == len(PFA_POINTS)
        for point in run["points"]:
            assert {
                "design_pfa",
                "alarm_count",
                "eligible_cells",
                "measured_pfa",
                "interval_low",
                "interval_high",
                "alpha",
            } <= point.keys()
