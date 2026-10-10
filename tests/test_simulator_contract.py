"""Conformance tests every simulator must pass -- the development generator today,
the validated simulator from M2 onwards."""

import numpy as np

from prdbench.simulation import SimConfig, get_simulator
from prdbench.simulation.interface import RDMapSimulator, Target


def test_stub_satisfies_protocol():
    assert isinstance(get_simulator("stub"), RDMapSimulator)


def test_shapes_and_ground_truth():
    cfg = SimConfig(n_maps=8, n_range=32, n_doppler=32, seed=0)
    batch = get_simulator("stub").generate(cfg)
    assert batch.maps.shape == (8, 32, 32)
    assert batch.maps.dtype == np.float32
    assert (batch.maps >= 0).all(), "maps are linear power"
    assert len(batch.targets) == 8
    assert all(isinstance(t, Target) for tl in batch.targets for t in tl)


def test_seed_reproducibility():
    cfg = SimConfig(n_maps=4, n_range=32, n_doppler=32, seed=7)
    a = get_simulator("stub").generate(cfg)
    b = get_simulator("stub").generate(cfg)
    assert np.array_equal(a.maps, b.maps)


def test_prdsim_generates_with_every_component_enabled():
    """Defaults enable targets, ridge and clutter in the complete signal model."""
    sim = get_simulator("prdsim")
    assert isinstance(sim, RDMapSimulator)
    batch = sim.generate(SimConfig(n_maps=2, n_range=32, n_doppler=32))
    assert batch.maps.shape == (2, 32, 32)
    assert batch.maps.dtype == np.float32
    assert np.isfinite(batch.maps).all()
    assert (batch.maps >= 0).all()
