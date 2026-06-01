#!/usr/bin/env bash
# Discount-horizon (gamma) probe launcher (runs ON the vast box).
#
# Warm-starts from the L3 speed SOTA (speedT06_a140_tp060 / step440M, the
# 74% SR / 4.47 s pick) and overrides ONLY the discount gamma, holding the
# tp0.60 recipe fixed. Tests whether shortening the RL horizon buys lap time
# WITHIN the pure gate-progress objective (Song 2023) -- gamma is the RL
# objective's own lap-time valuation, NOT a trajectory/contour-tracking term.
#
# Caveat (config.py PPOConfig): gamma was deliberately raised 0.997 -> 0.998
# so the terminal finish_bonus back-propagates through the ~10 s episode.
# Lowering gamma underweights that finish signal, so this is expected to trade
# SR for speed. The warm-start-from-a-finishing-policy regime is what makes it
# worth probing: the finish behaviour is already baked in, so a short low-gamma
# fine-tune may shave lap time without forgetting how to finish. EVAL EARLY and
# across the SR/speed Pareto.
#
# Usage (on box):
#   bash box_launch_gamma.sh <run_name> <gamma> [total_steps]
set -euo pipefail

RUN_NAME="${1:?run_name}"
GAMMA="${2:?gamma}"
TOTAL="${3:-200000000}"

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
# Launch detached in tmux so it survives SSH drops.
tmux kill-session -t "$RUN_NAME" 2>/dev/null || true
tmux new-session -d -s "$RUN_NAME" "
  export PATH=\"\$HOME/.pixi/bin:\$PATH\"; export SCIPY_ARRAY_API=1;
  cd $REPO;
  pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train \
    --run-name=$RUN_NAME \
    --init-from=$WARM \
    --gamma=$GAMMA \
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
echo "launched $RUN_NAME (gamma=$GAMMA total=$TOTAL warm=tp0.60/440M) in tmux; log=$LOG"
