"""Repo-root .env loading for every ``agent.*`` entry point.

``load_dotenv()`` with no arguments searches from the current working
directory, so it silently loads nothing when a command runs from anywhere
but the repo root. Every entry point must call :func:`load_repo_env`
instead, which always reads ``<repo-root>/.env``.

Like the camera and image servers, values from the .env win over already
exported variables (``override=True``), so a stale export cannot silently
replace the configured token.

``AGENT_ENV_FILE`` (set in the process environment, not in a .env) points
every entry point at a different env file instead. ``run_demo.py`` uses it to
run the pipeline against a self-contained demo config without touching the
shared servers the repo-root .env names.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ENV = REPO_ROOT / ".env"


def env_file() -> Path:
    """The env file to load: ``$AGENT_ENV_FILE`` when set, else the repo-root ``.env``."""
    override = os.environ.get("AGENT_ENV_FILE")
    return Path(override) if override else REPO_ENV


def load_repo_env(*, override: bool = True) -> bool:
    """Load the repo-root ``.env`` (or ``$AGENT_ENV_FILE``). Returns True when the file was found."""
    from dotenv import load_dotenv

    return bool(load_dotenv(env_file(), override=override))
