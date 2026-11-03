#!/bin/sh
set -eu

mkdir -p \
  "${TZ_DATA_ROOT}/nets" \
  "${TZ_DATA_ROOT}/games" \
  "${TZ_DATA_ROOT}/hidden" \
  "${TZ_DATA_ROOT}/logs" \
  "${TZ_DATA_ROOT}/runs" \
  "${TZ_DATA_ROOT}/python_client_games"

if [ "$#" -eq 0 ]; then
  set -- ucimain
fi

case "$1" in
  main|server|ucimain|lichess|battle|enginetest|experiment|getdecode|getgame|getinference|getinferencetime|getmove|mps|trainenginetest)
    set -- "/opt/truezero/bin/$@"
    ;;
  python)
    shift
    set -- /opt/venv/bin/python "$@"
    ;;
esac

exec "$@"
