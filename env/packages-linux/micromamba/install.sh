#!/usr/bin/env bash
# micromamba at ~/.local/bin/micromamba for the uid-1000 desktop user.
set -euo pipefail
TARGET_HOME="$(getent passwd 1000 | cut -d: -f6)"
[ -n "$TARGET_HOME" ] || { echo "[pkg micromamba] FATAL: uid 1000 has no passwd entry" >&2; exit 1; }
if [ ! -x "$TARGET_HOME/.local/bin/micromamba" ]; then
  mkdir -p "$TARGET_HOME/.local/bin"
  curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar -xj -C /tmp bin/micromamba
  install -m0755 /tmp/bin/micromamba "$TARGET_HOME/.local/bin/micromamba"
  chown -R 1000:0 "$TARGET_HOME/.local"
fi
mkdir -p "$TARGET_HOME/.cache/mamba"; chown -R 1000:0 "$TARGET_HOME/.cache" 2>/dev/null || true
test -x "$TARGET_HOME/.local/bin/micromamba" || { echo "[pkg micromamba] FATAL: micromamba missing" >&2; exit 1; }
echo "[pkg micromamba] OK ($("$TARGET_HOME/.local/bin/micromamba" --version 2>&1))"
