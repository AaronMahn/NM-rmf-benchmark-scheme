#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
MODEL_DIR="$REPO_ROOT/skill/rmf-toolkit/models"
RUNNER="$REPO_ROOT/skill/rmf-toolkit/scripts/run_model.py"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results}"

for model in tm1_paper nl3_paper fsu_delta_m6p7_paper; do
  "$PYTHON" "$RUNNER" \
    --model "$MODEL_DIR/$model.json" \
    --out "$OUT_DIR/$model"
done
