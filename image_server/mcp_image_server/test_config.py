"""The image MCP server's bearer token is the shared $MCP_AUTH_TOKEN.

Run from the image_server/ folder:
    python -m mcp_image_server.test_config
(pytest also collects the test_* functions below.)
"""

from __future__ import annotations

import os

from mcp_image_server.config import Settings


def _set_env(**vars: str | None) -> dict:
    old = {k: os.environ.get(k) for k in vars}
    for k, v in vars.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return old


def _restore(old: dict) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_token_comes_from_shared_mcp_auth_token():
    old = _set_env(MCP_AUTH_TOKEN="secret-123", IMAGE_MCP_AUTH_TOKEN=None)
    try:
        settings = Settings.from_env()
    finally:
        _restore(old)
    assert settings.auth_token == "secret-123"


def test_legacy_image_token_is_ignored():
    old = _set_env(MCP_AUTH_TOKEN="shared", IMAGE_MCP_AUTH_TOKEN="legacy")
    try:
        settings = Settings.from_env()
    finally:
        _restore(old)
    assert settings.auth_token == "shared"


def test_no_token_means_no_auth():
    old = _set_env(MCP_AUTH_TOKEN=None, IMAGE_MCP_AUTH_TOKEN=None)
    try:
        settings = Settings.from_env()
    finally:
        _restore(old)
    assert settings.auth_token is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
