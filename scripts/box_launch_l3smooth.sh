#!/usr/bin/env bash
# L3-SOTA smoothness finetune (runs ON the vast box).
# Warm-starts from the L3 speed SOTA (tp0.60 step440M, 74%/4.47s) and adds the
# r_smooth (action-jerk) penalty, keeping the full L3 recipe otherwise — to
# de-twitch the competition policy for sim2real without losing L3 SR/lap.
#
# Usage (on box):  bash box_launch_l3smooth.sh <run_name> <r_smooth_coef> [total_steps]
#
# L3 SOTA recipe held fixed: alpha=1.4, time_penalty=0.60, omega=0.005,
#   obstacle_weight=0.3 (barrier ON), gate_frame_weight=0.5 (barrier ON),
#   curriculum=full, stage_idx=5 (stage4_level3_dr).
set -euo pipefail

RUN_NAME="${1:?run_name}"
RSMOOTH="${2:?r_smooth_coef}"
TOTAL="${3:-150000000}"

REPO=/root/lsy_drone_racing
WARM="$REPO/lsy_drone_racing/control/rl_sbx/checkpoints/speedT06_a140_tp060/step_000440401920"
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
    --alpha-max-rad=1.4 --time-penalty=0.60 --omega-coef=0.005 \
    --r-smooth-coef=$RSMOOTH \
    --progress-coef=15 \
    --use-obstacle-barrier --obstacle-weight=0.3 \
    --use-gate-frame-barrier --gate-frame-weight=0.5 \
    --curriculum=full --stage-idx=5 \
    --total-timesteps=$TOTAL \
    2>&1 | tee $LOG
"
echo "launched $RUN_NAME (r_smooth_coef=$RSMOOTH, warm L3-SOTA tp0.60/s440M, total=$TOTAL) in tmux; log=$LOG"
