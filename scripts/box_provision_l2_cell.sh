#!/usr/bin/env bash
# Provision ONE RTX 5090 on vast.ai, bootstrap the current local branch, and
# launch one L2-screen cell on it (detached tmux). Self-contained so several
# instances can run in parallel (one per cell). Retries instance creation to
# tolerate offer-collision when multiple provisioners pick the same cheapest
# offer at once, and destroys the instance if bootstrap/launch fails so a
# half-provisioned box never bills silently.
#
# Run on the dev box:
#   bash scripts/box_provision_l2_cell.sh <run_name> <ang_vel 0|1> <hidden> [n_envs] [total_steps]
#
# Emits "PROVISIONED <run> instance=<id> ssh=<host>:<port>" then
# "LAUNCHED <run> ..." on success.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
SCRIPTS_DIR="/home/exedev/scripts"
REPO_ROOT="${REPO_ROOT:-/home/exedev/lsy_drone_racing}"

RUN_NAME="${1:?run_name}"
ANGVEL="${2:?ang_vel 0|1}"
HIDDEN="${3:?hidden_size}"
NENVS="${4:-4096}"
TOTAL="${5:-300000000}"

BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"

# 1. Rent a 5090, retrying on offer-collision / transient create failure.
create_out=""
for attempt in 1 2 3 4 5 6; do
    if create_out=$(VAST_GPU=RTX_5090 bash "$SCRIPTS_DIR/vast_create_instance.sh" "$RUN_NAME" \
        2>"/tmp/${RUN_NAME}.create.err"); then
        break
    fi
    echo "[$RUN_NAME] create attempt $attempt failed; retrying in 15s" >&2
    create_out=""
    sleep 15
done
if [ -z "$create_out" ]; then
    echo "[$RUN_NAME] FAILED to create instance after retries" >&2
    cat "/tmp/${RUN_NAME}.create.err" >&2 || true
    exit 1
fi
iid=$(echo "$create_out" | sed -n 's/^instance_id=//p' | head -1)
host=$(echo "$create_out" | sed -n 's/^ssh_host=//p' | head -1)
port=$(echo "$create_out" | sed -n 's/^ssh_port=//p' | head -1)
echo "PROVISIONED $RUN_NAME instance=$iid ssh=$host:$port"

# 2. Bootstrap our branch; destroy the box on failure (don't leave it billing).
if ! REPO_ROOT="$REPO_ROOT" bash "$SCRIPTS_DIR/vast_bootstrap.sh" "$host" "$port" "$BRANCH" \
    >"/tmp/${RUN_NAME}.bootstrap.log" 2>&1; then
    echo "[$RUN_NAME] bootstrap FAILED; destroying instance $iid" >&2
    echo y | vastai destroy instance "$iid" >/dev/null 2>&1 || true
    exit 1
fi

# 3. Launch the cell on the box via the in-repo launcher (detached tmux).
if ! ssh -p "$port" -o StrictHostKeyChecking=accept-new "root@$host" \
    "cd /root/lsy_drone_racing && bash scripts/box_launch_l2_screen.sh '$RUN_NAME' '$ANGVEL' '$HIDDEN' '$NENVS' '$TOTAL'"; then
    echo "[$RUN_NAME] launch FAILED; destroying instance $iid" >&2
    echo y | vastai destroy instance "$iid" >/dev/null 2>&1 || true
    exit 1
fi

echo "LAUNCHED $RUN_NAME instance=$iid ssh=$host:$port"
