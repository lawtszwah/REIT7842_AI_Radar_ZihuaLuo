from .interface import RDMapSimulator, SimBatch, SimConfig, Target
from .stub import StubSimulator

SIMULATORS = {"stub": StubSimulator}


def get_simulator(name: str):
    if name == "group":
        from .adapter import GroupSimulator

        return GroupSimulator()
    if name not in SIMULATORS:
        raise KeyError(f"unknown simulator {name!r}; have {sorted(SIMULATORS) + ['group']}")
    return SIMULATORS[name]()


__all__ = ["RDMapSimulator", "SimBatch", "SimConfig", "Target", "get_simulator"]
