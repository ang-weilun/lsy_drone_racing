#!/usr/bin/env bash
# Provision ONE RTX 5090 on vast.ai, bootstrap the current local branch, and
# launch one L2-screen cell on it (detached tmux). Self-contained so several
# instances can run in parallel (one per cell).
#
# Robustness: loops until it gets a box that is (a) creatable (retries on
# offer-collision), (b) SSH-reachable -- vast_wait_for_ssh can return before
# sshd accepts connections -- and (c) has enough FREE disk for the rl-train env
# (some cheap offers hand out a tiny allocatable slice despite large total
# disk; jaxlib install then dies with ENOSPC). Bad boxes are destroyed and a
# fresh one is rented. Bootstrap/launch failures also destroy the box so a
# half-provisioned instance never bills silently.
#
# Run on the dev box:
#   bash scripts/box_provision_l2_cell.sh <run_name> <ang_vel 0|1> <hidden> [n_envs] [total_steps]
#
# Emits "PROVISIONED ...", "LAUNCHED <run> instance=<id> ssh=<host>:<port>".
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
SCRIPTS_DIR="/home/exedev/scripts"
REPO_ROOT="${REPO_ROOT:-/home/exedev/lsy_drone_racing}"

RUN_NAME="${1:?run_name}"
ANGVEL="${2:?ang_vel 0|1}"
HIDDEN="${3:?hidden_size}"
NENVS="${4:-4096}"
TOTAL="${5:-300000000}"
MIN_FREE_GB=40

BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"

destroy() { echo y | vastai destroy instance "$1" >/dev/null 2>&1 || true; }
box_ssh() {  # box_ssh <host> <port> <cmd...>
    local h="$1" p="$2"
    shift 2
    ssh -p "$p" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 \
        -o BatchMode=yes "root@$h" "$@"
}

# Acquire a usable box (create -> ssh-ready -> enough free disk).
iid=""
host=""
port=""
for box_try in 1 2 3 4 5 6 7 8; do
    create_out=""
    for attempt in 1 2 3 4 5; do
        if create_out=$(VAST_GPU=RTX_5090 bash "$SCRIPTS_DIR/vast_create_instance.sh" \
            "$RUN_NAME" 2>>"/tmp/${RUN_NAME}.create.err"); then
            break
        fi
        create_out=""
        sleep 15
    done
    if [ -z "$create_out" ]; then
        echo "[$RUN_NAME] create failed (box_try $box_try)" >&2
        continue
    fi
    iid=$(echo "$create_out" | sed -n 's/^instance_id=//p' | head -1)
    host=$(echo "$create_out" | sed -n 's/^ssh_host=//p' | head -1)
    port=$(echo "$create_out" | sed -n 's/^ssh_port=//p' | head -1)
    echo "PROVISIONED $RUN_NAME instance=$iid ssh=$host:$port (box_try $box_try)"

    ready=""
    for _ in $(seq 1 30); do
        if box_ssh "$host" "$port" true 2>/dev/null; then
            ready=1
            break
        fi
        sleep 10
    done
    if [ -z "$ready" ]; then
        echo "[$RUN_NAME] sshd never came up; destroying $iid" >&2
        destroy "$iid"
        iid=""
        continue
    fi

    free_gb=$(box_ssh "$host" "$port" "df -BG --output=avail / | tail -1 | tr -dc 0-9" \
        2>/dev/null || echo 0)
    if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
        echo "[$RUN_NAME] only ${free_gb}GB free (<${MIN_FREE_GB}); destroying $iid" >&2
        destroy "$iid"
        iid=""
        continue
    fi
    echo "[$RUN_NAME] usable box: ${free_gb}GB free"
    break
done
if [ -z "$iid" ]; then
    echo "[$RUN_NAME] FAILED to acquire a usable box after retries" >&2
    exit 1
fi

# Bootstrap our branch; destroy the box on failure.
if ! REPO_ROOT="$REPO_ROOT" bash "$SCRIPTS_DIR/vast_bootstrap.sh" "$host" "$port" "$BRANCH" \
    >"/tmp/${RUN_NAME}.bootstrap.log" 2>&1; then
    echo "[$RUN_NAME] bootstrap FAILED; destroying instance $iid" >&2
    destroy "$iid"
    exit 1
fi

# Launch the cell via the in-repo launcher (detached tmux); destroy on failure.
if ! ssh -p "$port" -o StrictHostKeyChecking=accept-new "root@$host" \
    "cd /root/lsy_drone_racing && bash scripts/box_launch_l2_screen.sh '$RUN_NAME' '$ANGVEL' '$HIDDEN' '$NENVS' '$TOTAL'"; then
    echo "[$RUN_NAME] launch FAILED; destroying instance $iid" >&2
    destroy "$iid"
    exit 1
fi

echo "LAUNCHED $RUN_NAME instance=$iid ssh=$host:$port"
