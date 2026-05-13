#!/usr/bin/env bash
# Local smoke test: start the trainer, kill it with SIGTERM, restart it,
# and verify it resumes from where it left off. Run this on your laptop
# before paying for spot time.
#
# Usage:
#   bash scripts/smoke_resume.sh
#
# What it asserts:
#   * The trainer writes at least one ckpt before being killed.
#   * SIGTERM triggers a graceful emergency save (exit code == 0).
#   * On restart the trainer reports a non-zero starting global_step.
#   * No remote-sync (LSY_REMOTE unset) is required to pass.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT

CKPT_DIR="${TMP}/ckpts"
LOG1="${TMP}/run1.log"
LOG2="${TMP}/run2.log"

# Use a very small workload so the test finishes in seconds.
COMMON_ARGS=(
    --builder noop_smoke
    --total_steps 5000
    --save_every 200
    --log_every 100
    --ckpt_dir "${CKPT_DIR}"
    --wandb_mode disabled
    --seed 7
)

cd "${ROOT}"

PY="${PYTHON:-python}"

echo "=== run 1: start trainer in background, kill with SIGTERM ==="
"${PY}" scripts/train_resumable.py "${COMMON_ARGS[@]}" > "${LOG1}" 2>&1 &
PID=$!
echo "trainer pid=${PID}, logs=${LOG1}"

# Wait for at least one ckpt to appear (or the process to die unexpectedly).
DEADLINE=$(( $(date +%s) + 60 ))
while : ; do
    if ls "${CKPT_DIR}"/ckpt-*.pt >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "FAIL: trainer exited before writing a ckpt"; cat "${LOG1}"; exit 1
    fi
    if [ "$(date +%s)" -gt "${DEADLINE}" ]; then
        echo "FAIL: timed out waiting for first ckpt"; cat "${LOG1}"; kill "${PID}" || true; exit 1
    fi
    sleep 0.2
done
FIRST_CKPT=$(ls "${CKPT_DIR}"/ckpt-*.pt | tail -n1)
echo "first ckpt written: $(basename "${FIRST_CKPT}")"

# Send SIGTERM and wait for graceful exit.
kill -TERM "${PID}"
if ! wait "${PID}"; then
    rc=$?
    echo "FAIL: trainer did not exit cleanly after SIGTERM (rc=${rc})"
    cat "${LOG1}"; exit 1
fi
echo "trainer exited cleanly after SIGTERM"

# Capture the step from the latest ckpt.
LATEST=$(ls "${CKPT_DIR}"/ckpt-*.pt | tail -n1)
STEP_BEFORE=$(basename "${LATEST}" | sed -E 's/ckpt-0*([0-9]+)\.pt/\1/')
echo "latest ckpt before restart: $(basename "${LATEST}") (step=${STEP_BEFORE})"
if [ "${STEP_BEFORE}" -lt 200 ]; then
    echo "FAIL: expected at least 200 steps before SIGTERM, got ${STEP_BEFORE}"; exit 1
fi

echo "=== run 2: restart, expect resume from step=${STEP_BEFORE} ==="
"${PY}" scripts/train_resumable.py "${COMMON_ARGS[@]}" > "${LOG2}" 2>&1

# Verify the resume log line is present.
if ! grep -q "Resumed from step=${STEP_BEFORE}" "${LOG2}"; then
    echo "FAIL: run2 did not log 'Resumed from step=${STEP_BEFORE}'"
    echo "--- run2 log ---"; cat "${LOG2}"; exit 1
fi

# Verify the final ckpt is at total_steps.
FINAL=$(ls "${CKPT_DIR}"/ckpt-*.pt | tail -n1)
STEP_AFTER=$(basename "${FINAL}" | sed -E 's/ckpt-0*([0-9]+)\.pt/\1/')
if [ "${STEP_AFTER}" -ne 5000 ]; then
    echo "FAIL: run2 ended at step=${STEP_AFTER}, expected 5000"; exit 1
fi

# Verify pruning: at most 3 local ckpts.
COUNT=$(ls "${CKPT_DIR}"/ckpt-*.pt | wc -l)
if [ "${COUNT}" -gt 3 ]; then
    echo "FAIL: local ckpt count=${COUNT}, expected <= 3 (prune broken)"; exit 1
fi

echo "PASS: smoke resume test (resumed from ${STEP_BEFORE}, finished at ${STEP_AFTER}, ${COUNT} ckpts kept)"
