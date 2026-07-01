#!/usr/bin/env bash
set -euo pipefail

# Sweeps |magnetization| vs. h across checkpoints via scripts/reports/plot_magnetization_vs_h.py.
# Edit the variables below to point at a different run or change sweep granularity/precision.

CKPT_DIR="mount/20260701_112324"

H_MIN=0.5
H_MAX=1.5
N_H=11

ENSEMBLE_SIZE=5
NUM_WALKERS=512
MICROBATCH_SIZE=512
SAMPLE_BUFFER_SIZE=512

cd "$(dirname "$0")/.."

uv run python scripts/reports/plot_magnetization_vs_h.py "$CKPT_DIR" \
    --h-min "$H_MIN" --h-max "$H_MAX" --n-h "$N_H" \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --num-walkers "$NUM_WALKERS" --microbatch-size "$MICROBATCH_SIZE" --sample-buffer-size "$SAMPLE_BUFFER_SIZE"
