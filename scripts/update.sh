#!/bin/sh
set -eu

usage() {
  echo "Usage: $0 [--tag TAG] [--force] [--rollback]"
}

TAG="latest"
FORCE=0
ROLLBACK=0
while [ "$#" -gt 0 ]; do
  arg="$1"
  case "$arg" in
    --tag)
      [ "$#" -ge 2 ] || { echo "--tag requires a value" >&2; exit 2; }
      TAG="$2"
      shift
      ;;
    --force) FORCE=1 ;;
    --rollback) ROLLBACK=1 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

command -v docker >/dev/null 2>&1 || {
  echo "Docker is required" >&2
  exit 1
}
[ -f compose.yaml ] || {
  echo "Run this script from the TrueZero deployment directory" >&2
  exit 1
}

STATE_DIR="${TRUEZERO_STATE_DIR:-.truezero-deploy}"
mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/previous-image"

if [ "$ROLLBACK" -eq 1 ]; then
  [ -s "$STATE_FILE" ] || {
    echo "No previous image is recorded; rollback is unavailable" >&2
    exit 1
  }
  export TRUEZERO_IMAGE="$(cat "$STATE_FILE")"
  echo "Rolling back to $TRUEZERO_IMAGE"
  docker compose up -d --no-build --force-recreate
  docker compose ps
  exit 0
fi

if docker inspect truezero >/dev/null 2>&1; then
  running="$(docker inspect -f '{{.State.Running}}' truezero)"
  image_id="$(docker inspect -f '{{.Image}}' truezero)"
  command_line="$(docker inspect -f '{{join .Config.Cmd " "}}' truezero)"
  case "$command_line" in
    *main*|*train*|*client*)
      if [ "$running" = "true" ] && [ "$FORCE" -ne 1 ]; then
        echo "Refusing to replace active training/self-play container '$command_line'." >&2
        echo "Stop it deliberately or rerun with --force (which may interrupt training)." >&2
        exit 1
      fi
      ;;
  esac
  docker tag "$image_id" truezero:rollback
  printf '%s\n' "truezero:rollback" > "$STATE_FILE"
fi

export TRUEZERO_IMAGE="ghcr.io/andreaslam/truezero:$TAG"
echo "Pulling $TRUEZERO_IMAGE"
docker compose pull truezero
docker image inspect "$TRUEZERO_IMAGE" >/dev/null
docker compose up -d --no-build --force-recreate
sleep 2
docker inspect -f '{{.State.Running}}' truezero | grep -q '^true$' || {
  echo "Updated container did not remain running; previous image is recorded in $STATE_FILE" >&2
  docker compose logs --tail=80 truezero >&2 || true
  exit 1
}

digest="$(docker inspect -f '{{.Image}}' truezero)"
echo "Deployed $TRUEZERO_IMAGE"
echo "Image ID: $digest"
docker compose ps
