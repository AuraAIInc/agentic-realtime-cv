#!/usr/bin/env bash
# Run camera_sim from this folder. Uses the project .venv at the repo root.
cd "$(dirname "$0")/.." || exit 1
VENV="$(dirname "$PWD")/.venv"
[ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"
exec python -m camera_sim "$@"
