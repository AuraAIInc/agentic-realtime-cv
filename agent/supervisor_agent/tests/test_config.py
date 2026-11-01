"""Single-token MCP auth: every server uses $MCP_AUTH_TOKEN.

Run from the agent/ folder:
    python -m pytest supervisor_agent/tests/test_config.py -q
or without pytest:
    python -m supervisor_agent.tests.test_config
"""

from __future__ import annotations

import os

from supervisor_agent.config import AgentConfig


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


def test_all_servers_share_the_single_mcp_token():
    old = _set_env(MCP_AUTH_TOKEN="secret-123",
                   CAMERA_MCP_TOKEN=None, IMAGE_MCP_TOKEN=None, TRACKING_MCP_TOKEN=None)
    try:
        cfg = AgentConfig.from_env()
    finally:
        _restore(old)
    assert [s.name for s in cfg.servers] == ["camera", "images", "tracking"]
    assert {s.token for s in cfg.servers} == {"secret-123"}


def test_legacy_per_server_tokens_are_ignored():
    old = _set_env(MCP_AUTH_TOKEN="shared",
                   CAMERA_MCP_TOKEN="cam", IMAGE_MCP_TOKEN="img", TRACKING_MCP_TOKEN="trk")
    try:
        cfg = AgentConfig.from_env()
    finally:
        _restore(old)
    assert {s.token for s in cfg.servers} == {"shared"}


def test_no_token_means_no_auth():
    old = _set_env(MCP_AUTH_TOKEN=None,
                   CAMERA_MCP_TOKEN=None, IMAGE_MCP_TOKEN=None, TRACKING_MCP_TOKEN=None)
    try:
        cfg = AgentConfig.from_env()
    finally:
        _restore(old)
    assert {s.token for s in cfg.servers} == {None}


def test_default_chat_hides_write_and_rare_tools():
    from types import SimpleNamespace

    from supervisor_agent.cli import CHAT_HIDDEN_TOOLS, chat_tool_names

    hub = SimpleNamespace(tools={n: None for n in [
        "camera__get_status", "camera__set_schedule", "camera__reset_schedule",
        "camera__list_models", "camera__get_camera_modes", "tracking__get_raw_detections",
        "tracking__get_live_status", "local__add_label", "local__save_zone",
        "local__get_cat_profiles", "skill__check-live-detections",
    ]})
    names = chat_tool_names(hub)
    assert set(names).isdisjoint(CHAT_HIDDEN_TOOLS)
    assert set(names) == {"camera__get_status", "tracking__get_live_status",
                          "local__get_cat_profiles", "skill__check-live-detections"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
