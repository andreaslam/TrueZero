#!/bin/sh
set -eu

IMAGE="${1:-ghcr.io/andreaslam/truezero:latest}"
MODEL="${TZ_MODEL_PATH:-/data/nets/tz_0.pt}"

docker run --rm --gpus all \
  -e TZ_DEVICE=cuda \
  -e TZ_MODEL_PATH="$MODEL" \
  -v "${TRUEZERO_DATA_DIR:-$PWD/.truezero-data}:/data" \
  "$IMAGE" getinference
