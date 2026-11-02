"""Repo-root .env loading for every ``agent.*`` entry point.

``load_dotenv()`` with no arguments searches from the current working
directory, so it silently loads nothing when a command runs from anywhere
but the repo root. Every entry point must call :func:`load_repo_env`
instead, which always reads ``<repo-root>/.env``.

Like the camera and image servers, values from the .env win over already
exported variables (``override=True``), so a stale export cannot silently
replace the configured token.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_ENV = REPO_ROOT / ".env"


def load_repo_env(*, override: bool = True) -> bool:
    """Load the repo-root ``.env``. Returns True when the file was found."""
    from dotenv import load_dotenv

    return bool(load_dotenv(REPO_ENV, override=override))
