#!/usr/bin/env bash
# Reproduce every number and figure in the thesis from a clean checkout.
set -euo pipefail
CONFIG="${1:-configs/protocol/full.yaml}"
OUT="${2:-results/$(date +%Y%m%d-%H%M%S)}"
export PYTHONPATH=src

python -m prdbench run --config "$CONFIG" --out "$OUT"
python scripts/make_figures.py --results "$OUT/results.json" --out "$OUT/figures"
git rev-parse HEAD > "$OUT/COMMIT"
echo "results in $OUT"
