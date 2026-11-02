"""Repo-root .env loading (agent/env.py). Run from the repo root:
    .venv/bin/python -m pytest agent/tests -q
"""

from __future__ import annotations

import os
from pathlib import Path

import agent
import agent.env as env


def test_repo_env_points_at_repo_root():
    assert env.REPO_ENV == Path(agent.__file__).resolve().parent.parent / ".env"
    assert env.REPO_ENV.name == ".env"


def test_missing_env_returns_false(monkeypatch, tmp_path):
    monkeypatch.setattr(env, "REPO_ENV", tmp_path / "no-such.env")
    assert env.load_repo_env() is False


def test_loads_values_and_overrides_stale_exports(monkeypatch, tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("RCAT_ENV_PROBE=from-file\n")
    monkeypatch.setattr(env, "REPO_ENV", dotenv)
    monkeypatch.setenv("RCAT_ENV_PROBE", "stale-export")
    assert env.load_repo_env() is True
    assert os.environ["RCAT_ENV_PROBE"] == "from-file"
