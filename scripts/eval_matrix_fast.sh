#!/usr/bin/env bash
# Faster eval_matrix: single Python process for 3 levels (JIT cache + imports
# amortized). Emits the composite metric on stdout in the same format as
# eval_matrix.sh; logs the full output to training_logs/eval_<tag>.log.
#
# Usage: bash scripts/eval_matrix_fast.sh <checkpoint_dir> <tag>
set -euo pipefail

ckpt_dir="${1:?usage: eval_matrix_fast.sh <checkpoint_dir> <tag>}"
tag="${2:?usage: eval_matrix_fast.sh <checkpoint_dir> <tag>}"

if [[ ! -d "${ckpt_dir}" ]]; then
    echo "checkpoint dir not found: ${ckpt_dir}" >&2
    exit 1
fi

export PATH="${HOME}/.pixi/bin:${PATH}"
mkdir -p training_logs
log=training_logs/eval_${tag}.log

pixi run -e rl-train python scripts/eval_3level.py \
    --checkpoint "${ckpt_dir}" \
    --n-runs 20 > "${log}" 2>&1 || true

echo "== Eval matrix: ${tag} =="
grep -E '^L[0-9]:|^COMPOSITE_METRIC=' "${log}" || tail -10 "${log}"
