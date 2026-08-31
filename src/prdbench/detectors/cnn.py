"""Fully-convolutional per-cell detector.

Kept small on purpose: the benchmark's third question is what accuracy survives
an inference-cost constraint, so the deep models start at a size that could
plausibly run in near real time and are scaled up only if the cost budget allows.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import Detector

_EPS = 1e-12


def _normalise(maps: np.ndarray) -> torch.Tensor:
    x = np.log10(np.asarray(maps, dtype=np.float32) + _EPS)
    mu = x.mean(axis=(1, 2), keepdims=True)
    sd = x.std(axis=(1, 2), keepdims=True) + 1e-6
    return torch.from_numpy((x - mu) / sd).unsqueeze(1)


class _Net(nn.Module):
    def __init__(self, width: int = 16, depth: int = 4):
        super().__init__()
        layers, c_in = [], 1
        for i in range(depth):
            dil = 2 ** i
            layers += [nn.Conv2d(c_in, width, 3, padding=dil, dilation=dil),
                       nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
            c_in = width
        layers.append(nn.Conv2d(width, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DilatedCNN(Detector):
    family = "deep"
    trainable = True

    def __init__(self, width: int = 16, depth: int = 4, epochs: int = 12,
                 lr: float = 3e-3, batch_size: int = 16, pos_weight: float = 200.0,
                 seed: int = 0, device: str = "cpu"):
        super().__init__(width=width, depth=depth, epochs=epochs, lr=lr,
                         batch_size=batch_size, pos_weight=pos_weight, seed=seed)
        self.name = f"CNN(w={width},d={depth})"
        self.epochs, self.lr, self.batch_size = epochs, lr, batch_size
        self.pos_weight, self.seed = pos_weight, seed
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.model = _Net(width, depth).to(self.device)

    def fit(self, maps: np.ndarray, labels: np.ndarray, **kw) -> DilatedCNN:
        torch.manual_seed(self.seed)
        x = _normalise(maps)
        y = torch.from_numpy(labels.astype(np.float32)).unsqueeze(1)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
        n = len(x)
        g = torch.Generator().manual_seed(self.seed)
        self.model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                opt.zero_grad()
                loss = lossf(self.model(x[idx].to(self.device)), y[idx].to(self.device))
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def score(self, maps: np.ndarray) -> np.ndarray:
        self.model.eval()
        x = _normalise(maps)
        out = []
        for s in range(0, len(x), 32):
            # Raw logits, not sigmoid: the harness owns the threshold, and
            # a saturated sigmoid would collapse the top of the ROC into ties.
            out.append(self.model(x[s : s + 32].to(self.device)).cpu().numpy())
        return np.concatenate(out)[:, 0].astype(np.float32)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
