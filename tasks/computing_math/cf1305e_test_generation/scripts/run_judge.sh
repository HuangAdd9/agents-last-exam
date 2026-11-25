#!/usr/bin/env bash
set -u
dispatch_source=$1
dispatch_reference=$2
dispatch_work=$3
sudo -n /usr/bin/python3 -I "$dispatch_reference/judge.py" \
  --source "$dispatch_source" --reference "$dispatch_reference" \
  --report "$dispatch_work/report.json" > "$dispatch_work/stdout" 2> "$dispatch_work/stderr"
dispatch_status=$?
printf '%s\n' "$dispatch_status" > "$dispatch_work/rc"
