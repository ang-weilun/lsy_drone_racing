#!/usr/bin/env bash
# Throughput A/B: seg-init ON vs OFF on the stage-5 recipe, with profiling.
# Runs ON the vast box. Each arm recompiles; drop the first two log dumps.
# Usage: bash ab_throughput.sh <on|off> [total_steps]
set -euo pipefail

ARM="${1:?arm: on|off}"
TOTAL="${2:-30000000}"

REPO=/root/lsy_drone_racing
WARM="$REPO/lsy_drone_racing/control/rl_sbx/checkpoints/speedT06_a140_tp060/step_000440401920"
LOG="$REPO/training_logs/ab_seg${ARM}.log"

export PATH="$HOME/.pixi/bin:$PATH"
export SCIPY_ARRAY_API=1
mkdir -p "$REPO/training_logs"

if [ ! -d "$WARM" ]; then echo "WARM checkpoint missing: $WARM" >&2; exit 2; fi

if [ "$ARM" = "on" ]; then
  SEG_FLAGS=(--segment-init-prob=0.5 --segment-init-vel-mps=2.5)
elif [ "$ARM" = "off" ]; then
  SEG_FLAGS=(--segment-init-prob=0 --phase2-prob=0)
else
  echo "arm must be on|off" >&2; exit 2
fi

cd "$REPO"
pixi run -e rl-train python -m lsy_drone_racing.control.rl_sbx.train \
  --run-name="ab_seg${ARM}" \
  --init-from="$WARM" \
  "${SEG_FLAGS[@]}" \
  --alpha-max-rad=1.4 --time-penalty=0.60 --omega-coef=0.005 --progress-coef=15 \
  --use-obstacle-barrier --obstacle-weight=0.3 \
  --use-gate-frame-barrier --gate-frame-weight=0.5 \
  --curriculum=full --stage-idx=5 --seed=0 \
  --total-timesteps="$TOTAL" --no-wandb --profile-throughput \
  2>&1 | tee "$LOG"
echo "arm=$ARM done; log=$LOG"
