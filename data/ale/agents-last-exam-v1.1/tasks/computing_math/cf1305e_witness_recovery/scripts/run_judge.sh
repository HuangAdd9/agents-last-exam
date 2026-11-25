#!/usr/bin/env bash
set -u
RECEIPT_SOURCE=$1
RECEIPT_REFERENCE=$2
RECEIPT_WORK=$3
sudo -n /usr/bin/python3 "$RECEIPT_REFERENCE/judge.py" --source "$RECEIPT_SOURCE" \
    --reference "$RECEIPT_REFERENCE" --report "$RECEIPT_WORK/report.json" \
    > "$RECEIPT_WORK/stdout" 2> "$RECEIPT_WORK/stderr"
RECEIPT_STATUS=$?
printf '%s\n' "$RECEIPT_STATUS" > "$RECEIPT_WORK/rc"
