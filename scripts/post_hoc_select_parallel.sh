#!/usr/bin/env bash
# Parallel post-hoc eval. Selects every Nth step from a run dir, then runs
# the per-checkpoint eval_matrix_fast in batches of <concurrency> in parallel.
# Useful when training is fast (a few min) and eval dominates wall time —
# the RTX PRO 6000 has plenty of headroom for 2-3 concurrent eval_sim
# processes (mujoco env step is CPU-bound, only the policy net touches GPU).
#
# Usage: bash scripts/post_hoc_select_parallel.sh \
#   <run_name> <stride> [levels] [concurrency] [tail_only]
#
# concurrency: how many eval processes to run in parallel. Default 3.
# tail_only:   if 1, only eval the last N=stride*4 checkpoints (skip early
#              warm-up checkpoints that are usually worse). Default 0.
set -euo pipefail

run_name="${1:?usage: post_hoc_select_parallel.sh <run_name> <stride> [levels] [concurrency] [tail_only]}"
stride="${2:-4}"
levels="${3:-0,1,2}"
concurrency="${4:-3}"
tail_only="${5:-0}"

run_dir="${HOME}/lsy_drone_racing/lsy_drone_racing/control/rl_song/checkpoints/${run_name}"
if [[ ! -d "${run_dir}" ]]; then
    echo "run dir not found: ${run_dir}" >&2
    exit 1
fi
out_tsv="${HOME}/lsy_drone_racing/training_logs/ph_eval_${run_name}.tsv"
view_root="/tmp/ph_view_${run_name}"
rm -rf "${view_root}"
mkdir -p "${view_root}"

IFS=',' read -ra LEVEL_ARR <<< "$levels"
header='step_M'
for lvl in "${LEVEL_ARR[@]}"; do
    header+=$'\t'"L${lvl}_finished"$'\t'"L${lvl}_lap"
done
header+=$'\t'metric
if [[ ",${levels}," == *,3,* ]]; then
    header+=$'\t'L3_finishes
fi
printf '%s\n' "${header}" > "${out_tsv}"

mapfile -t steps < <(ls -d "${run_dir}"/step_* 2>/dev/null | sort)
total=${#steps[@]}
if (( total == 0 )); then
    echo "no step dirs found in ${run_dir}" >&2
    exit 1
fi

selected_idx=()
i=0
while (( i < total )); do
    selected_idx+=($i)
    i=$((i + stride))
done
last_idx=$((total - 1))
if [[ "${selected_idx[-1]}" != "$last_idx" ]]; then
    selected_idx+=($last_idx)
fi

if [[ "${tail_only}" == "1" ]]; then
    keep=4
    n_selected=${#selected_idx[@]}
    if (( n_selected > keep )); then
        selected_idx=("${selected_idx[@]: -keep}")
    fi
fi

mkdir -p /tmp/ph_results_${run_name}
rm -f /tmp/ph_results_${run_name}/*.txt

run_one() {
    local idx=$1
    local step_dir="${steps[idx]}"
    local step_name
    step_name=$(basename "${step_dir}")
    local step_num
    step_num=$(echo "${step_name}" | sed -E 's/step_0*([0-9]+)/\1/')
    local step_M
    step_M=$(awk -v n="$step_num" 'BEGIN{ printf "%d", n/1000000 }')
    local view_dir="${view_root}/view_step${step_M}M"
    rm -rf "${view_dir}"
    mkdir -p "${view_dir}"
    ln -s "${step_dir}" "${view_dir}/${step_name}"
    ln -sf "${run_dir}/policy_config.json" "${view_dir}/policy_config.json"
    ln -sf "${run_dir}/reward_config.json" "${view_dir}/reward_config.json"

    local tag="${run_name}_step${step_M}M"
    cd "${HOME}/lsy_drone_racing"
    local out
    out=$(bash scripts/eval_matrix_fast.sh "${view_dir}" "${tag}" "${levels}")

    local row="${step_M}"
    for lvl in "${LEVEL_ARR[@]}"; do
        local l_finished l_lap
        l_finished=$(echo "${out}" | grep "^L${lvl}:" | awk '{print $2}')
        l_lap=$(echo "${out}" | grep "^L${lvl}:" | awk '{print $4}')
        row+=$'\t'"${l_finished:-NA}"$'\t'"${l_lap:-NA}"
    done
    local metric
    metric=$(echo "${out}" | grep '^COMPOSITE_METRIC=' | cut -d= -f2)
    row+=$'\t'"${metric:-NA}"
    if [[ ",${levels}," == *,3,* ]]; then
        local l3_fin
        l3_fin=$(echo "${out}" | grep '^L3_FINISHES=' | cut -d= -f2)
        row+=$'\t'"${l3_fin:-NA}"
    fi
    # Write to a per-idx file so we can collate in order at the end.
    printf '%s\n' "${row}" > "/tmp/ph_results_${run_name}/${step_M}.txt"
    echo "[done step ${step_M}M] ${row}"
}

# Run jobs in batches of <concurrency>.
batch=()
for idx in "${selected_idx[@]}"; do
    run_one "${idx}" &
    batch+=($!)
    if (( ${#batch[@]} >= concurrency )); then
        wait "${batch[@]}"
        batch=()
    fi
done
# Drain remaining
if (( ${#batch[@]} > 0 )); then
    wait "${batch[@]}"
fi

# Collate results in numeric step order
for f in $(ls /tmp/ph_results_${run_name}/*.txt 2>/dev/null | sort -t/ -k4 -n); do
    cat "${f}" >> "${out_tsv}"
done

echo
echo "=== Post-hoc TSV: ${out_tsv} ==="
cat "${out_tsv}"
