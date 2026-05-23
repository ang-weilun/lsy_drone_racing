#!/usr/bin/env bash
# Run eval_matrix.sh on every Nth saved step of a run; emit a TSV.
#
# Usage: bash scripts/post_hoc_select.sh <run_name> <stride>
# e.g.   bash scripts/post_hoc_select.sh level2_v58_no_guide_zero_tpenalty_warm56s163_100M 4
set -euo pipefail

run_name="${1:?usage: post_hoc_select.sh <run_name> <stride>}"
stride="${2:-4}"

run_dir="${HOME}/lsy_drone_racing/lsy_drone_racing/control/rl_song/checkpoints/${run_name}"
if [[ ! -d "${run_dir}" ]]; then
    echo "run dir not found: ${run_dir}" >&2
    exit 1
fi
out_tsv="${HOME}/lsy_drone_racing/training_logs/ph_eval_${run_name}.tsv"
view_root="/tmp/ph_view_${run_name}"
rm -rf "${view_root}"
mkdir -p "${view_root}"

printf 'step_M\tL0_finished\tL0_lap\tL1_finished\tL1_lap\tL2_finished\tL2_lap\tmetric\n' > "${out_tsv}"

# List step_NNN dirs, sorted ascending. Take every Nth + the final.
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
    out=$(bash scripts/eval_matrix.sh "${view_dir}" "${tag}")
    echo "${out}"
    L0=$(echo "${out}" | grep '^L0:' | awk '{print $2,$4}')
    L1=$(echo "${out}" | grep '^L1:' | awk '{print $2,$4}')
    L2=$(echo "${out}" | grep '^L2:' | awk '{print $2,$4}')
    metric=$(echo "${out}" | grep '^COMPOSITE_METRIC=' | cut -d= -f2)
    printf '%d\t%s\t%s\t%s\t%s\n' "${step_M}" "${L0}" "${L1}" "${L2}" "${metric}" \
      | sed 's/ /\t/g' >> "${out_tsv}"
done

echo
echo "=== Post-hoc TSV: ${out_tsv} ==="
cat "${out_tsv}"
