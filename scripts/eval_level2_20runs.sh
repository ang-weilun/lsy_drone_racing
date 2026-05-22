#!/usr/bin/env bash
# Run 20 eval episodes against config/level2.toml for a given RL Song run
# directory, then summarize finish rate + lap-time statistics from the
# eval_sim.py log output.
#
# Usage: bash scripts/eval_level2_20runs.sh <run_name>
#   e.g. bash scripts/eval_level2_20runs.sh level2_cold_v38_alpha008
#
# Assumes PWD=~/lsy_drone_racing and pixi env "rl-train" is available.
set -euo pipefail

run_name="${1:?usage: eval_level2_20runs.sh <run_name>}"
run_dir="lsy_drone_racing/control/rl_song/checkpoints/${run_name}"
log_file="training_logs/eval_${run_name}_level2_n20.log"

if [[ ! -d "${run_dir}" ]]; then
    echo "Run dir not found: ${run_dir}" >&2
    exit 1
fi

export PATH="${HOME}/.pixi/bin:${PATH}"
mkdir -p training_logs

pixi run -e rl-train python -m lsy_drone_racing.control.rl_song.eval_sim \
    --config level2.toml \
    --checkpoint "${run_dir}" \
    --control_mode attitude \
    --n_runs 20 \
    --render False \
    > "${log_file}" 2>&1

# Summarize: count "Finished: True", extract lap times.
finished_count=$(grep -c "Finished: True" "${log_file}" || true)
total_count=$(grep -c "Flight time" "${log_file}" || true)
finish_rate=$(awk -v f="${finished_count}" -v t="${total_count}" 'BEGIN{ if (t>0) print f/t; else print "nan" }')

echo "=== Eval summary: ${run_name} on level2.toml ==="
echo "Finished: ${finished_count} / ${total_count}"
echo "Finish rate: ${finish_rate}"
echo ""
echo "Per-episode flight times (s) for finished episodes:"
awk '
    /^INFO:.*Flight time/{ time_line=$0; next }
    /^Finished: True/ { print time_line }
' "${log_file}" | sed -E 's/.*Flight time \(s\): //' | head -25

echo ""
echo "Mean finished lap time:"
awk '
    /^INFO:.*Flight time/{ time_line=$0; next }
    /^Finished: True/ { print time_line }
' "${log_file}" | sed -E 's/.*Flight time \(s\): //' | \
    awk '{ s+=$1; n++ } END{ if (n>0) printf "%.3f s (n=%d)\n", s/n, n; else print "no finished episodes" }'

echo ""
echo "Full log: ${log_file}"
