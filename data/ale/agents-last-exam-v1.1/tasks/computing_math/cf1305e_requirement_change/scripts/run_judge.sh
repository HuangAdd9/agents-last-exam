#!/usr/bin/env bash
set -u
sudo -n /usr/bin/python3 -I "$2/judge.py" --source "$1" --reference "$2" --report "$3/report.json" > "$3/stdout" 2> "$3/stderr"
STATUS=$?
printf '%s\n' "$STATUS" > "$3/rc"
