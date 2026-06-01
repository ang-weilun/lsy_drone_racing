#!/usr/bin/env bash
# High-velocity seg-init probe launcher (runs ON the vast box).
#
# Hypothesis: the L3 policy flies a uniform conservative cruise (~1.8 m/s) with
# huge unused accel headroom (TWR 1.88 → ~1.6 g horizontal; rotational authority
# ~6% used) because the FINAL DR stage (stage4_level3_dr, stage_idx=5) starts
# every episode from rest at the track origin (segment_init_prob=0, reset_vel=0)
# — so PPO never sees the high-speed regime. This probe turns seg-init ON at that
# stage and injects forward velocity, forcing the policy to experience + learn
# fast flight. REWARD-NEUTRAL (no reward term changed; pure exploration/curriculum
# intervention — no Song-philosophy conflict, distinct from the discarded exit-vel).
#
# Warm-starts from the L3 speed SOTA (tp0.60 / step440M) and holds the tp0.60
# recipe fixed; overrides only segment_init_prob + segment_init_vel_mps.
#
# Usage (on box):
#   bash box_launch_segvel.sh <run_name> <seg_vel_mps> [seg_prob] [total_steps]
set -euo pipefail

RUN_NAME="${1:?run_name}"
SEG_VEL="${2:?seg_vel_mps}"
SEG_PROB="${3:-0.5}"
TOTAL="${4:-200000000}"

REPO=/root/lsy_drone_racing
WARM="$REPO/lsy_drone_racing/control/rl_sbx/checkpoints/speedT06_a140_tp060/step_000440401920"
LOG="$REPO/training_logs/${RUN_NAME}.log"

export PATH="$HOME/.pixi/bin:$PATH"
export SCIPY_ARRAY_API=1
mkdir -p "$REPO/training_logs"

if [ ! -d "$WARM" ]; then
  echo "WARM checkpoint missing: $WARM" >&2
  exit 2
fi

cd "$REPO"
tmux kill-session -t "$RUN_NAME" 2>/dev/null || true
tmux new-session -d -s "$RUN_NAME" "
  export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export SCIPY_ARRAY_API=1;
  cd $REPO;
  pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train \
    --run-name=$RUN_NAME \
    --init-from=$WARM \
    --segment-init-prob=$SEG_PROB \
    --segment-init-vel-mps=$SEG_VEL \
    --alpha-max-rad=1.4 \
    --time-penalty=0.60 \
    --omega-coef=0.005 \
    --progress-coef=15 \
    --use-obstacle-barrier --obstacle-weight=0.3 \
    --use-gate-frame-barrier --gate-frame-weight=0.5 \
    --curriculum=full --stage-idx=5 \
    --total-timesteps=$TOTAL \
    2>&1 | tee $LOG
"
echo "launched $RUN_NAME (seg_vel=$SEG_VEL seg_prob=$SEG_PROB total=$TOTAL warm=tp0.60/440M) in tmux; log=$LOG"
