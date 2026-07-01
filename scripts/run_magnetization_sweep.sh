#!/usr/bin/env bash
set -euo pipefail

# Sweeps |magnetization| vs. h across checkpoints via scripts/reports/plot_magnetization_vs_h.py.
# Edit the variables below to point at a different run or change sweep granularity/precision.

CKPT_DIR="mount/20260701_130543"

H_MIN=0.5
H_MAX=1.5
N_H=11

EPOCH_MIN=800
EPOCH_MAX=800

ENSEMBLE_SIZE=10
NUM_WALKERS=500000
MICROBATCH_SIZE=10000
SAMPLE_BUFFER_SIZE=500000

cd "$(dirname "$0")/.."

EPOCH_ARGS=()
[[ -n "$EPOCH_MIN" ]] && EPOCH_ARGS+=(--epoch-min "$EPOCH_MIN")
[[ -n "$EPOCH_MAX" ]] && EPOCH_ARGS+=(--epoch-max "$EPOCH_MAX")

uv run python scripts/reports/plot_magnetization_vs_h.py "$CKPT_DIR" \
    --h-min "$H_MIN" --h-max "$H_MAX" --n-h "$N_H" \
    --ensemble-size "$ENSEMBLE_SIZE" \
    --num-walkers "$NUM_WALKERS" --microbatch-size "$MICROBATCH_SIZE" --sample-buffer-size "$SAMPLE_BUFFER_SIZE" \
    "${EPOCH_ARGS[@]}"
