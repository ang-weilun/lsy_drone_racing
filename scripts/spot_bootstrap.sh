#!/usr/bin/env bash
# Spot-instance entrypoint. Provisions the env from a pre-built pixi-pack
# tarball on R2 instead of solving pixi from scratch (~30s vs 3min).
#
# Designed to run on a stock vast.ai CUDA image with nothing else on it.
# Set this as the on-start command (or paste it into a screen/tmux on
# manual launch). It will:
#   1. Write rclone config from $RCLONE_CONFIG_B64.
#   2. Download the env tarball from $LSY_PACK_REMOTE/train-<tag>.tar.
#   3. Unpack it once; subsequent boots of the same disk skip this.
#   4. Clone / fast-forward the repo at $LSY_REPO_REF.
#   5. Exec the trainer using the unpacked env's python so SIGTERM hits
#      python directly (no shell wrapper in between).
#
# Required env:
#   LSY_REPO_URL         git URL to clone
#   LSY_PACK_REMOTE      rclone remote with env packs, e.g. r2:bucket/envs
#   LSY_CKPT_REMOTE      rclone remote for ckpts, e.g. r2:bucket/runs/abc
#   RCLONE_CONFIG_B64    base64-encoded rclone.conf
#
# Optional:
#   LSY_REPO_REF         git ref to train on (default: main)
#   LSY_PACK_TAG         specific pack tag; otherwise reads -latest.txt
#   LSY_REPO_DIR         clone dir (default: /workspace/lsy_drone_racing)
#   LSY_ENV_DIR          unpack dir (default: /workspace/train-env)
#   LSY_TRAIN_ARGS       extra args to train_resumable.py
#   WANDB_API_KEY        wandb auth

set -euo pipefail

LSY_REPO_DIR="${LSY_REPO_DIR:-/workspace/lsy_drone_racing}"
LSY_ENV_DIR="${LSY_ENV_DIR:-/workspace/train-env}"
LSY_REPO_REF="${LSY_REPO_REF:-main}"
LSY_TRAIN_ARGS="${LSY_TRAIN_ARGS:-}"

log() { printf '[bootstrap %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
require() {
    if [ -z "${!1:-}" ]; then
        echo "ERROR: required env var $1 is not set" >&2
        exit 64
    fi
}

require LSY_REPO_URL
require LSY_PACK_REMOTE
require LSY_CKPT_REMOTE

# --- minimal system deps (rclone + git). vast.ai images usually have ----
# git already; rclone we always need.
if ! command -v rclone >/dev/null 2>&1; then
    log "installing rclone"
    curl -fsSL https://rclone.org/install.sh | bash
fi
command -v git >/dev/null 2>&1 || { log "installing git"; apt-get update && apt-get install -y --no-install-recommends git; }
command -v zstd >/dev/null 2>&1 || { log "installing zstd"; apt-get update && apt-get install -y --no-install-recommends zstd; }

# --- rclone config from env so creds aren't on disk in the image. ------
if [ -n "${RCLONE_CONFIG_B64:-}" ]; then
    mkdir -p "${HOME}/.config/rclone"
    echo "${RCLONE_CONFIG_B64}" | base64 -d > "${HOME}/.config/rclone/rclone.conf"
    chmod 600 "${HOME}/.config/rclone/rclone.conf"
    log "wrote rclone config ($(wc -c <"${HOME}/.config/rclone/rclone.conf") bytes)"
else
    log "WARNING: RCLONE_CONFIG_B64 unset — sync will fail"
fi

# --- env pack: download + unpack if not already there for this tag. ----
mkdir -p "${LSY_ENV_DIR}"
if [ -z "${LSY_PACK_TAG:-}" ]; then
    rclone copyto "${LSY_PACK_REMOTE}/train-latest.txt" /tmp/train-latest.txt
    LSY_PACK_TAG="$(cat /tmp/train-latest.txt | tr -d '[:space:]')"
fi
TAG_MARKER="${LSY_ENV_DIR}/.tag"
if [ -f "${TAG_MARKER}" ] && [ "$(cat "${TAG_MARKER}")" = "${LSY_PACK_TAG}" ]; then
    log "env pack ${LSY_PACK_TAG} already unpacked at ${LSY_ENV_DIR}"
else
    log "downloading env pack tag=${LSY_PACK_TAG}"
    rclone copyto "${LSY_PACK_REMOTE}/train-${LSY_PACK_TAG}.tar" /tmp/train-pack.tar
    rm -rf "${LSY_ENV_DIR:?}"/*
    log "unpacking to ${LSY_ENV_DIR}"
    tar -xf /tmp/train-pack.tar -C "${LSY_ENV_DIR}"
    bash "${LSY_ENV_DIR}/activate.sh" >/dev/null 2>&1 || true  # extract + link
    echo "${LSY_PACK_TAG}" > "${TAG_MARKER}"
    rm -f /tmp/train-pack.tar
fi

# Source the pixi-pack activate.sh so PATH/CONDA_PREFIX/LD_LIBRARY_PATH
# point at the bundled python. This sets $PATH so plain `python` resolves
# to the unpacked interpreter.
# shellcheck disable=SC1091
source "${LSY_ENV_DIR}/activate.sh"

# --- code: clone if missing, then fast-forward to the requested ref. ---
if [ ! -d "${LSY_REPO_DIR}/.git" ]; then
    log "cloning ${LSY_REPO_URL} -> ${LSY_REPO_DIR}"
    git clone --filter=blob:none "${LSY_REPO_URL}" "${LSY_REPO_DIR}"
fi
cd "${LSY_REPO_DIR}"
git fetch --quiet origin "${LSY_REPO_REF}"
git checkout --quiet "${LSY_REPO_REF}"
git reset --hard "origin/${LSY_REPO_REF}" 2>/dev/null || true
log "HEAD: $(git rev-parse --short HEAD) ($(git log -1 --pretty=%s))"

# --- exec the trainer. exec replaces this shell so python is PID 1's
# direct child (or PID 1 itself if no init), so SIGTERM goes straight to
# the python signal handler.
log "launching trainer"
# shellcheck disable=SC2086
exec python scripts/train_resumable.py \
    --remote "${LSY_CKPT_REMOTE}" \
    ${LSY_TRAIN_ARGS}
