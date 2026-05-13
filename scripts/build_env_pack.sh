#!/usr/bin/env bash
# Dev-side: build a self-contained pixi-pack tarball of the `train` env
# and upload it to the rclone remote so spot instances can grab it on
# cold-start. Run this whenever pyproject.toml or pixi.lock changes.
#
# Usage:
#   bash scripts/build_env_pack.sh                  # auto-tag from pixi.lock
#   bash scripts/build_env_pack.sh my-tag           # explicit tag
#
# Required env:
#   LSY_PACK_REMOTE   rclone remote, e.g. r2:lsy-drone-racing/envs
#
# Produces:
#   <repo>/dist/train-<tag>.tar      uploaded as $LSY_PACK_REMOTE/train-<tag>.tar
#   <repo>/dist/train-latest.txt     pointer file, uploaded as ...-latest.txt
#
# Prereq: install pixi-pack on this machine:
#   curl -fsSL https://pixi.sh/install.sh | bash
#   pixi global install pixi-pack

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT}"

if [ -z "${LSY_PACK_REMOTE:-}" ]; then
    echo "ERROR: LSY_PACK_REMOTE must be set (e.g. r2:lsy-drone-racing/envs)" >&2
    exit 64
fi

if ! command -v pixi-pack >/dev/null 2>&1; then
    echo "ERROR: pixi-pack not on PATH. Install it with:" >&2
    echo "  pixi global install pixi-pack" >&2
    exit 65
fi

TAG="${1:-$(sha256sum pixi.lock | cut -c1-12)}"
DIST="${ROOT}/dist"
PACK="${DIST}/train-${TAG}.tar"
POINTER="${DIST}/train-latest.txt"
mkdir -p "${DIST}"

echo "[pack] building train env pack tag=${TAG}"
pixi-pack pack --environment train --output "${PACK}"

echo "[pack] pack size: $(du -h "${PACK}" | cut -f1)"
echo "${TAG}" > "${POINTER}"

echo "[pack] uploading to ${LSY_PACK_REMOTE}/"
rclone copy "${PACK}" "${LSY_PACK_REMOTE}/"
rclone copyto "${POINTER}" "${LSY_PACK_REMOTE}/train-latest.txt"

echo "[pack] done. spot bootstrap will use tag=${TAG}"
