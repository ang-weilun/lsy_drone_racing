#!/usr/bin/env bash
# Run eval_matrix_fast.sh on every Nth saved step of a run; emit TSV.
#
# Usage: bash scripts/post_hoc_select_fast.sh <run_name> <stride> [levels]
set -euo pipefail

run_name="${1:?usage: post_hoc_select_fast.sh <run_name> <stride> [levels]}"
stride="${2:-4}"
levels="${3:-0,1,2}"

run_dir="${HOME}/lsy_drone_racing/lsy_drone_racing/control/rl_song/checkpoints/${run_name}"
if [[ ! -d "${run_dir}" ]]; then
    echo "run dir not found: ${run_dir}" >&2
    exit 1
fi
out_tsv="${HOME}/lsy_drone_racing/training_logs/ph_eval_${run_name}.tsv"
view_root="/tmp/ph_view_${run_name}"
rm -rf "${view_root}"
mkdir -p "${view_root}"

# Pick header based on levels
IFS=',' read -ra LEVEL_ARR <<< "$levels"
header='step_M'
for lvl in "${LEVEL_ARR[@]}"; do
    header+=$'\t'"L${lvl}_finished"$'\t'"L${lvl}_lap"
done
header+=$'\t'metric
if [[ ',${levels},' == *,3,* ]]; then
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

for idx in "${selected_idx[@]}"; do
    step_dir="${steps[idx]}"
    step_name=$(basename "${step_dir}")
    step_num=$(echo "${step_name}" | sed -E 's/step_0*([0-9]+)/\1/')
    step_M=$(awk -v n=$step_num 'BEGIN{ printf "%d", n/1000000 }')
    view_dir="${view_root}/view_step${step_M}M"
    rm -rf "${view_dir}"
    mkdir -p "${view_dir}"
    ln -s "${step_dir}" "${view_dir}/${step_name}"
    ln -sf "${run_dir}/policy_config.json" "${view_dir}/policy_config.json"
    ln -sf "${run_dir}/reward_config.json" "${view_dir}/reward_config.json"

    tag="${run_name}_step${step_M}M"
    cd "${HOME}/lsy_drone_racing"
    out=$(bash scripts/eval_matrix_fast.sh "${view_dir}" "${tag}" "${levels}")
    echo "${out}"
    row="${step_M}"
    for lvl in "${LEVEL_ARR[@]}"; do
        l_finished=$(echo "${out}" | grep "^L${lvl}:" | awk '{print $2}')
        l_lap=$(echo "${out}" | grep "^L${lvl}:" | awk '{print $4}')
        row+=$'\t'"${l_finished:-NA}"$'\t'"${l_lap:-NA}"
    done
    metric=$(echo "${out}" | grep '^COMPOSITE_METRIC=' | cut -d= -f2)
    row+=$'\t'"${metric:-NA}"
    if [[ ',${levels},' == *,3,* ]]; then
        l3_fin=$(echo "${out}" | grep '^L3_FINISHES=' | cut -d= -f2)
        row+=$'\t'"${l3_fin:-NA}"
    fi
    printf '%s\n' "${row}" >> "${out_tsv}"
done

echo
echo "=== Post-hoc TSV: ${out_tsv} ==="
cat "${out_tsv}"
