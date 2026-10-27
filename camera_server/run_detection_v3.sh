#!/usr/bin/env bash
# Start camera_v3 in the background. Runs the code in this repo folder.
# Hackathon-temporary: use the host's ORIGINAL server venv (reverts after the hackathon).
cd "$(dirname "$0")" || exit 1
VENV="${ORIGINAL_VENV:-$HOME/camera_server/.venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
nohup python -m camera_v3 >> detect_log.log 2>&1 &
echo "camera_v3 started (pid $!), log: $(pwd)/detect_log.log (venv: ${VIRTUAL_ENV:-system python})"
