#!/usr/bin/env bash
# Seed-matched eval sweep for a speed-sweep run (runs ON the vast box).
# Evals the last N checkpoints at base-seed 0 (select) and base-seed 100
# (held-out validate), printing a compact SR / lap table parseable downstream.
#
# Usage (on box):  bash box_eval_speed.sh <run_name> [n_last] [n_runs]
#   run_name : checkpoint sub-dir under .../rl_sbx/checkpoints/
#   n_last   : how many of the latest step_* dirs to eval (default 3)
#   n_runs   : episodes per seed block (default 100)
set -euo pipefail

RUN_NAME="${1:?run_name}"
N_LAST="${2:-3}"
N_RUNS="${3:-100}"

REPO=/root/lsy_drone_racing
CKROOT="$REPO/lsy_drone_racing/control/rl_sbx/checkpoints/$RUN_NAME"
OUTDIR="$REPO/eval_out/$RUN_NAME"
export PATH="$HOME/.pixi/bin:$PATH"
export SCIPY_ARRAY_API=1
mkdir -p "$OUTDIR"
cd "$REPO"

mapfile -t STEPS < <(ls -1d "$CKROOT"/step_* 2>/dev/null | sort -t_ -k2 -n | tail -"$N_LAST")
if [ "${#STEPS[@]}" -eq 0 ]; then echo "NO_CHECKPOINTS in $CKROOT"; exit 3; fi

parse() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{d['success_rate']*100:.0f} {d['lap_time_finished_only'].get('mean', float('nan')):.3f}\")" "$1"; }

echo "=== EVAL $RUN_NAME (last $N_LAST ckpts, n=$N_RUNS/block) ==="
for STEP in "${STEPS[@]}"; do
  SID=$(basename "$STEP" | sed 's/step_0*//')
  for BASE in 0 100; do
    OUT="$OUTDIR/$(basename "$STEP")_b${BASE}.json"
    pixi run -e rl-train python scripts/eval_l3_seed_matched.py \
      --checkpoint "$STEP" --config level3.toml \
      --controller rl_sbx/controller_numpy.py --control-mode attitude \
      --n-runs "$N_RUNS" --base-seed "$BASE" --out "$OUT" >/dev/null 2>&1 || { echo "EVAL_FAIL $STEP b$BASE"; continue; }
    read -r SR LAP < <(parse "$OUT")
    echo "RESULT step=$SID b$BASE SR=${SR}% lap=${LAP}s"
  done
done
echo "=== EVAL DONE $RUN_NAME ==="
