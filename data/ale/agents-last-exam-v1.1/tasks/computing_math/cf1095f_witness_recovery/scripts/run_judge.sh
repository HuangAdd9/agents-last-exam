#!/usr/bin/env bash
set -u
SOURCE=$1
REFERENCE=$2
WORK=$3
python3 "$REFERENCE/judge.py" --source "$SOURCE" --reference "$REFERENCE" \
    --report "$WORK/report.json" > "$WORK/stdout" 2> "$WORK/stderr"
STATUS=$?
printf '%s\n' "$STATUS" > "$WORK/rc"
