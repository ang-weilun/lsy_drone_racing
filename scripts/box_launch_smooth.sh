#!/usr/bin/env bash
# Smoothness finetune launcher (runs ON the vast box).
# Warm-starts from the twitchy L2 speed model spd7 and adds the r_smooth
# (action-jerk) penalty, keeping spd7's recipe otherwise — a clean A/B to
# reduce wobble for sim2real without degrading L2 speed.
#
# Usage (on box):
#   bash box_launch_smooth.sh <run_name> <r_smooth_coef> [total_steps]
#
# spd7 recipe held fixed: alpha=1.2, omega=0.002, time_penalty=0.10,
#   L2 default curriculum (stage 0), ent 0.005->0.001.
set -euo pipefail

RUN_NAME="${1:?run_name}"
RSMOOTH="${2:?r_smooth_coef}"
TOTAL="${3:-150000000}"

REPO=/root/lsy_drone_racing
WARM="$REPO/lsy_drone_racing/control/rl_sbx/checkpoints/sbx_spd7_a120_om002_tp10_200M/step_000201326592"
LOG="$REPO/training_logs/${RUN_NAME}.log"

export PATH="$HOME/.pixi/bin:$PATH"
export SCIPY_ARRAY_API=1
mkdir -p "$REPO/training_logs"
cd "$REPO"

tmux kill-session -t "$RUN_NAME" 2>/dev/null || true
tmux new-session -d -s "$RUN_NAME" "
  export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export SCIPY_ARRAY_API=1;
  cd $REPO;
  pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train \
    --run-name=$RUN_NAME \
    --init-from=$WARM \
    --alpha-max-rad=1.2 \
    --omega-coef=0.002 \
    --time-penalty=0.10 \
    --r-smooth-coef=$RSMOOTH \
    --ent-coef=0.001 --ent-coef-final=0.001 \
    --curriculum=default --stage-idx=0 \
    --total-timesteps=$TOTAL \
    2>&1 | tee $LOG
"
echo "launched $RUN_NAME (r_smooth_coef=$RSMOOTH, warm spd7, total=$TOTAL) in tmux; log=$LOG"
