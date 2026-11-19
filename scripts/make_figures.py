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
    ap.add_argument("--results", required=True, nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None and len(a.results) > 1:
        ap.error("--out is required when reporting more than one results file")
    out = Path(a.out) if a.out else Path(a.results[0]).parent / "figures"
    for f in render([Path(r) for r in a.results], out):
        print(f)
