#!/usr/bin/env bash
# Start the MCP image server in the background. Run from the image_server folder.
cd "$(dirname "$0")" || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate
nohup python -m mcp_image_server >> mcp_server.log 2>&1 &
echo "mcp_image_server started (pid $!), log: $(pwd)/mcp_server.log"
