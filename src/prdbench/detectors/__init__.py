from .base import Detector
from .cfar import CACFAR, GOCFAR, OSCFAR
from .features import FeatureGBDT
from .cnn import DilatedCNN
from .attention import PatchAttention

REGISTRY = {
    "ca_cfar": CACFAR,
    "os_cfar": OSCFAR,
    "go_cfar": GOCFAR,
    "feat_gbdt": FeatureGBDT,
    "cnn": DilatedCNN,
    "attention": PatchAttention,
}


def build(kind: str, **params) -> Detector:
    if kind not in REGISTRY:
        raise KeyError(f"unknown detector {kind!r}; have {sorted(REGISTRY)}")
    return REGISTRY[kind](**params)


__all__ = ["Detector", "REGISTRY", "build"]
