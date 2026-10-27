#!/usr/bin/env bash
# Start detection_ingest_server.py (WebSocket 8000 + tracking MCP 10002) in the background.
# Runs the code in this repo folder with the host's python, or ORIGINAL_VENV if set.
cd "$(dirname "$0")" || exit 1
# Stop any instance that is already running (also one started from the old folder).
if pkill -f "python.* detection_ingest_server.py"; then
  echo "stopped existing detection_ingest_server.py"
  for _ in $(seq 10); do pgrep -f "python.* detection_ingest_server.py" > /dev/null || break; sleep 1; done
  pkill -9 -f "python.* detection_ingest_server.py"
fi
[ -n "$ORIGINAL_VENV" ] && [ -f "$ORIGINAL_VENV/bin/activate" ] && source "$ORIGINAL_VENV/bin/activate"
nohup python detection_ingest_server.py >> ingest_server.log 2>&1 &
echo "detection_ingest_server.py started (pid $!), log: $(pwd)/ingest_server.log (venv: ${VIRTUAL_ENV:-system python})"
