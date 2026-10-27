#!/usr/bin/env bash
cd /home/muyang/camera_server || exit 1
export PATH="$HOME/.local/bin:$PATH"  # default location of the uv installer
nohup uv run --frozen python -m camera_v3 >> /home/muyang/camera_server/detect_log.log 2>&1 &
