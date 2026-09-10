from .interface import RDMapSimulator, SimBatch, SimConfig, Target
from .stub import StubSimulator

SIMULATORS = {"stub": StubSimulator}

# The validated simulator is milestone M2 work: the signal model it implements has to be
# justified and checked against analytically predictable behaviour before any result it
# produces is reportable.  Named here so that `configs/protocol/full.yaml` fails loudly
# rather than quietly falling back to the development generator.  See
# docs/decisions/ADR-0004-simulator-is-built-in-project.md.
PENDING = "prdsim"


def get_simulator(name: str):
    if name == PENDING:
        raise NotImplementedError(
            f"the {PENDING!r} simulator is not implemented yet (milestone M2). "
            "Use --simulator stub for development; stub results are not reportable."
        )
    if name not in SIMULATORS:
        raise KeyError(f"unknown simulator {name!r}; have {sorted(SIMULATORS) + [PENDING]}")
    return SIMULATORS[name]()


__all__ = ["RDMapSimulator", "SimBatch", "SimConfig", "Target", "get_simulator"]
