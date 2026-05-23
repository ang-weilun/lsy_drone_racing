#!/usr/bin/env bash
# Faster eval_matrix: single Python process for N levels (JIT cache + imports
# amortized). Emits the composite metric on stdout.
#
# Usage: bash scripts/eval_matrix_fast.sh <checkpoint_dir> <tag> [levels]
#        levels defaults to '0,1,2'. Pass '0,1,2,3' to include L3.
set -euo pipefail

ckpt_dir="${1:?usage: eval_matrix_fast.sh <checkpoint_dir> <tag> [levels]}"
tag="${2:?usage: eval_matrix_fast.sh <checkpoint_dir> <tag> [levels]}"
levels="${3:-0,1,2}"

if [[ ! -d "${ckpt_dir}" ]]; then
    echo "checkpoint dir not found: ${ckpt_dir}" >&2
    exit 1
fi

export PATH="${HOME}/.pixi/bin:${PATH}"
mkdir -p training_logs
log=training_logs/eval_${tag}.log

pixi run -e rl-train python scripts/eval_3level.py \
    --checkpoint "${ckpt_dir}" \
    --n-runs 20 \
    --levels "${levels}" > "${log}" 2>&1 || true

echo "== Eval matrix: ${tag} =="
grep -E '^L[0-9]:|^COMPOSITE_METRIC=|^L3_FINISHES=' "${log}" || tail -10 "${log}"
