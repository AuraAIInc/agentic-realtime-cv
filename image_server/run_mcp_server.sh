#!/usr/bin/env bash
# Start the MCP image server in the background. Run from the image_server folder.
# Hackathon-temporary: use the host's ORIGINAL server venv (reverts after the hackathon).
cd "$(dirname "$0")" || exit 1
VENV="${ORIGINAL_VENV:-$HOME/camera_server/.venv}"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
nohup python -m mcp_image_server >> mcp_server.log 2>&1 &
echo "mcp_image_server started (pid $!), log: $(pwd)/mcp_server.log (venv: ${VIRTUAL_ENV:-system python})"
