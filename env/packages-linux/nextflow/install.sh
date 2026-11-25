#!/usr/bin/env bash
# Nextflow (self-installing launcher) -> /usr/local/bin/nextflow. Needs Java.
set -euo pipefail
TARGET_HOME="$(getent passwd 1000 | cut -d: -f6)"
[ -n "$TARGET_HOME" ] || { echo "[pkg nextflow] FATAL: uid 1000 has no passwd entry" >&2; exit 1; }
if ! command -v nextflow >/dev/null 2>&1; then
  command -v java >/dev/null 2>&1 || { echo "[pkg nextflow] FATAL: java (jdk-default) required first" >&2; exit 1; }
  curl -fsSL https://get.nextflow.io -o /usr/local/bin/nextflow
  chmod +x /usr/local/bin/nextflow
fi
command -v nextflow >/dev/null 2>&1 || { echo "[pkg nextflow] FATAL: nextflow missing" >&2; exit 1; }
# Bootstrap the framework jars as the desktop user so first agent invocation
# does not need network access or write to a root-owned runtime directory.
export NXF_HOME="$TARGET_HOME/.nextflow"
mkdir -p "$NXF_HOME"
HOME="$TARGET_HOME" NXF_HOME="$NXF_HOME" nextflow -version >/dev/null 2>&1 || { echo "[pkg nextflow] FATAL: framework bootstrap failed" >&2; exit 1; }
chown -R 1000:0 "$NXF_HOME" 2>/dev/null || true
echo "[pkg nextflow] OK"
