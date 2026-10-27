#!/usr/bin/env bash
# Start camera_v3 in the background. Runs the code in this repo folder.
# Hackathon-temporary: use the host's ORIGINAL server venv (reverts after the hackathon).
cd "$(dirname "$0")" || exit 1
# Stop any instance that is already running (also one started from the old folder).
if pkill -f "python -m camera_v3"; then
  echo "stopped existing camera_v3"
  for _ in $(seq 10); do pgrep -f "python -m camera_v3" > /dev/null || break; sleep 1; done
  pkill -9 -f "python -m camera_v3"
fi
VENV="${ORIGINAL_VENV:-$HOME/camera_server/.venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
nohup python -m camera_v3 >> detect_log.log 2>&1 &
echo "camera_v3 started (pid $!), log: $(pwd)/detect_log.log (venv: ${VIRTUAL_ENV:-system python})"
