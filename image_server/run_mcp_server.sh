#!/usr/bin/env bash
# Start the MCP image server in the background. Runs the code in this repo folder.
# Hackathon-temporary: use the host's ORIGINAL server venv (reverts after the hackathon).
cd "$(dirname "$0")" || exit 1
# Stop any instance that is already running (also one started from the old folder).
if pkill -f "python -m mcp_image_server"; then
  echo "stopped existing mcp_image_server"
  for _ in $(seq 10); do pgrep -f "python -m mcp_image_server" > /dev/null || break; sleep 1; done
  pkill -9 -f "python -m mcp_image_server"
fi
VENV="${ORIGINAL_VENV:-$HOME/camera_server/.venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
nohup python -m mcp_image_server >> mcp_server.log 2>&1 &
echo "mcp_image_server started (pid $!), log: $(pwd)/mcp_server.log (venv: ${VIRTUAL_ENV:-system python})"
# Stream the log; Ctrl-C stops only the tail, the server keeps running.
tail -f mcp_server.log
