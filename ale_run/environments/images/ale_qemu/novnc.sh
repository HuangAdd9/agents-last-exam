#!/usr/bin/env bash
set -Eeuo pipefail

target="${ALE_VNC_TARGET:-127.0.0.1:5900}"
if [[ ! "$target" =~ ^[a-zA-Z0-9.-]+:[0-9]+$ ]]; then
  echo "Invalid ALE_VNC_TARGET: $target" >&2
  exit 64
fi

host="${target%:*}"
port="${target##*:}"
ln -sf vnc.html /usr/share/novnc/index.html

exec websocketd \
  --binary \
  --address 0.0.0.0 \
  --port="${WEB_PORT:-8006}" \
  --staticdir=/usr/share/novnc \
  nc "$host" "$port"
