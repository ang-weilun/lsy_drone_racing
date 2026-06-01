#!/usr/bin/env bash
# Detached speed-sweep launcher (runs ON the vast box).
# Warm-starts from spdobs03 and overrides ONE speed lever per run.
#
# Usage (on box):
#   bash box_launch_speed.sh <run_name> <alpha> <time_penalty> <omega> [total_steps]
#
# Baseline (spdobs03) recipe held fixed unless overridden by args:
#   alpha=1.4 time_penalty=0.40 omega=0.005 progress_coef=15
#   obstacle_weight=0.3 (barrier ON) gate_frame_weight=0.5 (barrier ON)
#   curriculum=full stage_idx=5 (stage4_level3_dr)  warm=spdobs03 step_000964689920
set -euo pipefail

RUN_NAME="${1:?run_name}"
ALPHA="${2:?alpha}"
TP="${3:?time_penalty}"
OMEGA="${4:?omega}"
TOTAL="${5:-1000000000}"

REPO=/root/lsy_drone_racing
WARM="$REPO/lsy_drone_racing/control/rl_sbx/checkpoints/spdobs03_a140_obsw030_1B/step_000964689920"
LOG="$REPO/training_logs/${RUN_NAME}.log"

export PATH="$HOME/.pixi/bin:$PATH"
export SCIPY_ARRAY_API=1
mkdir -p "$REPO/training_logs"

cd "$REPO"
# Launch detached in tmux so it survives SSH drops.
tmux kill-session -t "$RUN_NAME" 2>/dev/null || true
tmux new-session -d -s "$RUN_NAME" "
  export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export SCIPY_ARRAY_API=1;
  cd $REPO;
  pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train \
    --run-name=$RUN_NAME \
    --init-from=$WARM \
    --alpha-max-rad=$ALPHA \
    --time-penalty=$TP \
    --omega-coef=$OMEGA \
    --progress-coef=15 \
    --use-obstacle-barrier --obstacle-weight=0.3 \
    --use-gate-frame-barrier --gate-frame-weight=0.5 \
    --curriculum=full --stage-idx=5 \
    --total-timesteps=$TOTAL \
    2>&1 | tee $LOG
"
echo "launched $RUN_NAME (alpha=$ALPHA tp=$TP omega=$OMEGA total=$TOTAL) in tmux; log=$LOG"
