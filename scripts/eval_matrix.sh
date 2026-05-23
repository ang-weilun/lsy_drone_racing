#!/usr/bin/env bash
# Run 20-seed deterministic eval on level0/1/2 for a checkpoint dir.
# Emits the composite metric (L2_lap / max(L2_finish_frac, 0.05)) on stdout.
# Logs per-level details to training_logs/eval_<tag>_levelN_n20.log.
#
# Usage: bash scripts/eval_matrix.sh <checkpoint_dir> <tag>
set -euo pipefail

ckpt_dir="${1:?usage: eval_matrix.sh <checkpoint_dir> <tag>}"
tag="${2:?usage: eval_matrix.sh <checkpoint_dir> <tag>}"

if [[ ! -d "${ckpt_dir}" ]]; then
    echo "checkpoint dir not found: ${ckpt_dir}" >&2
    exit 1
fi

export PATH="${HOME}/.pixi/bin:${PATH}"
mkdir -p training_logs

declare -A finished_count
declare -A mean_lap
for level in 0 1 2; do
    log=training_logs/eval_${tag}_level${level}_n20.log
    pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim \
        --config level${level}.toml \
        --checkpoint "${ckpt_dir}" \
        --control_mode attitude \
        --n_runs 20 \
        --render False \
        > "${log}" 2>&1 || true
    fin=$(grep -c 'Finished: True' "${log}" || true)
    finished_count[${level}]=$fin
    mean=$(awk '
        /^INFO:.*Flight time/{ t=$0; next }
        /^Finished: True/ { print t }
    ' "${log}" | sed -E 's/.*Flight time \(s\): //' | awk '{s+=$1; n++} END{ if (n>0) printf "%.3f", s/n; else print "NA" }')
    mean_lap[${level}]=$mean
done

L0_F=${finished_count[0]}; L0_T=${mean_lap[0]}
L1_F=${finished_count[1]}; L1_T=${mean_lap[1]}
L2_F=${finished_count[2]}; L2_T=${mean_lap[2]}
L2_frac=$(awk -v f=$L2_F 'BEGIN{ printf "%.4f", f/20 }')

if [[ "${L2_T}" == "NA" ]]; then
    metric="inf"
else
    metric=$(awk -v lap=$L2_T -v frac=$L2_frac 'BEGIN{ d=(frac<0.05)?0.05:frac; printf "%.2f", lap/d }')
fi

echo "== Eval matrix: ${tag} =="
printf "L0: %s/20 @ %s\n" $L0_F $L0_T
printf "L1: %s/20 @ %s\n" $L1_F $L1_T
printf "L2: %s/20 @ %s\n" $L2_F $L2_T
echo "COMPOSITE_METRIC=${metric}"
