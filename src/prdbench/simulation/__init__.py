from .groupgen import GroupGenerator
from .interface import RDMapSimulator, SimBatch, SimConfig, Target
from .prdsim import PrdSim
from .stub import StubSimulator

SIMULATORS = {"stub": StubSimulator, "prdsim": PrdSim, "groupgen": GroupGenerator}


def get_simulator(name: str):
    if name not in SIMULATORS:
        raise KeyError(f"unknown simulator {name!r}; have {sorted(SIMULATORS)}")
    return SIMULATORS[name]()


__all__ = ["RDMapSimulator", "SimBatch", "SimConfig", "Target", "get_simulator"]
