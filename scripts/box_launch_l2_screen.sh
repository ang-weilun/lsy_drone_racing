#!/usr/bin/env bash
# L2 cold-start screen launcher (runs ON the vast box).
# Cold-trains one cell of the omega / 512 screen on the single-stage L2
# curriculum. No --init-from (cold); --curriculum=default is the single L2
# stage (rl_song.config.default_curriculum / stage1_level2_phase12). The reward
# is the minimal Song recipe (r_prog + r_omega + r_crash + r_finish; NO obstacle
# / gate-frame barriers, NO time penalty) with alpha_max=0.36 -- the gentle
# cold-start substrate, NOT the L3 SOTA speed recipe (alpha=1.4 + barriers +
# time penalty, which thrashes a from-scratch policy). The only varying factor
# is the cell's toggle (RL_OBS_ANG_VEL / RL_HIDDEN_SIZE).
#
# Usage (on box):
#   bash box_launch_l2_screen.sh <run_name> <ang_vel 0|1> <hidden_size> [n_envs] [total_steps]
# Cells (n_envs=4096 fits a 32GB 5090; total_steps=300M):
#   ref:    bash box_launch_l2_screen.sh l2scr_ref    0 256
#   omegaA: bash box_launch_l2_screen.sh l2scr_omega  1 256
#   capB:   bash box_launch_l2_screen.sh l2scr_cap512 0 512
#   combo:  bash box_launch_l2_screen.sh l2scr_combo  1 512
set -euo pipefail

RUN_NAME="${1:?run_name}"
ANG_VEL="${2:?ang_vel 0|1}"
HIDDEN="${3:?hidden_size}"
NENVS="${4:-4096}"
TOTAL="${5:-300000000}"

REPO=/root/lsy_drone_racing
LOG="$REPO/training_logs/${RUN_NAME}.log"

export PATH="$HOME/.pixi/bin:$PATH"
export SCIPY_ARRAY_API=1
mkdir -p "$REPO/training_logs"
cd "$REPO"

# Pre-flight: the JAX/numpy encoders must agree at this cell's obs toggle before
# we burn compute. set -e aborts the launch if parity fails.
RL_OBS_ANG_VEL=$ANG_VEL pixi run -e rl-train python scripts/check_obs_encoder_parity.py

tmux kill-session -t "$RUN_NAME" 2>/dev/null || true
tmux new-session -d -s "$RUN_NAME" "
  export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export SCIPY_ARRAY_API=1;
  export RL_OBS_ANG_VEL=$ANG_VEL; export RL_HIDDEN_SIZE=$HIDDEN;
  cd $REPO;
  pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train \
    --run-name=$RUN_NAME \
    --curriculum=default \
    --n-envs=$NENVS \
    --alpha-max-rad=0.36 \
    --time-penalty=0.0 \
    --omega-coef=0.005 \
    --progress-coef=15 \
    --diag-every-n-rollouts=20 \
    --total-timesteps=$TOTAL \
    2>&1 | tee $LOG
"
echo "launched $RUN_NAME (ang_vel=$ANG_VEL hidden=$HIDDEN n_envs=$NENVS total=$TOTAL) in tmux; log=$LOG"
