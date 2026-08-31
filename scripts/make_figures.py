"""Thin wrapper so figures can be regenerated without installing an entry point.

The rendering itself lives in `prdbench.reporting`, which `prdbench report`
also calls -- one code path, so the two cannot drift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from prdbench.reporting import render

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.results).parent / "figures"
    for f in render(Path(a.results), out):
        print(f)
