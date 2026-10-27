#!/usr/bin/env bash
# Hackathon-temporary: run with the host's ORIGINAL server venv (reverts after the hackathon).
cd /home/muyang/camera_server || exit 1
VENV="${ORIGINAL_VENV:-$HOME/camera_server/.venv}"
nohup "$VENV/bin/python" -m camera_v3 >> /home/muyang/camera_server/detect_log.log 2>&1 &
