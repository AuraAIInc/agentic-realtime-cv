"""Skill tools: auto-scan finds every SKILL.md; invoking one loads its instructions.

Skills live in their own `skill__*` toolset (SkillTools), separate from LocalTools.
Task-only skills are not registered in chat surfaces. Run from the agent/ folder:
    python -m pytest supervisor_agent/tests/test_skills.py -q
or without pytest:
    python -m supervisor_agent.tests.test_skills
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from supervisor_agent.local_tools import skills as skill_tools
from supervisor_agent.local_tools.server import LocalTools
from supervisor_agent.local_tools.skills import SkillTools
from supervisor_agent.local_tools.store import Store

SHIPPED = {"assign-activity-label", "check-live-detections", "check-trajectory", "manage-labels", "validate-cat-identity"}
CHAT = {"check-live-detections", "manage-labels"}


def test_scan_finds_all_shipped_skills():
    skills = skill_tools.scan()
    assert {s.name for s in skills} == SHIPPED
    for s in skills:
        spec = s.spec()
        assert spec["name"] == s.name  # tool name is the skill name
        assert s.description in spec["description"]
        assert "skill" in spec["description"].lower()  # marked as a skill, not a plain tool
        assert spec["input_schema"] == {"type": "object", "properties": {}}
        assert s.surface in ("task", "chat", "both")


def test_default_toolset_is_chat_surface_only():
    tools = SkillTools()
    assert set(tools.skills) == CHAT
    assert {s["name"] for s in tools.specs()} == CHAT
    everything = SkillTools(surfaces=("task", "chat", "both"))
    assert set(everything.skills) == SHIPPED


def test_scan_skips_unreadable_skills():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "good").mkdir()
        (root / "good" / "SKILL.md").write_text("---\nname: good-skill\ndescription: A fine skill.\nsurface: chat\n---\n\nDo this.\n")
        (root / "no_file").mkdir()
        (root / "no_frontmatter").mkdir()
        (root / "no_frontmatter" / "SKILL.md").write_text("Just prose, no frontmatter.\n")
        (root / "no_name").mkdir()
        (root / "no_name" / "SKILL.md").write_text("---\ndescription: Nameless.\n---\n\nNothing.\n")
        (root / "bad_surface").mkdir()
        (root / "bad_surface" / "SKILL.md").write_text("---\nname: bad-surface\ndescription: Bad surface.\nsurface: everywhere\n---\n\nNothing.\n")
        (root / "notes.txt").write_text("not a skill dir")
        assert [s.name for s in skill_tools.scan(root)] == ["good-skill"]
        assert [s["name"] for s in SkillTools(root).specs()] == ["good-skill"]


def _result_text(result) -> str:
    return result.content[0].text


def test_invoke_returns_plain_instructions_without_frontmatter():
    tools = SkillTools()
    out = _result_text(asyncio.run(tools.call("check-live-detections", {})))
    assert out.startswith("# Check the live camera")
    assert "---" not in out.splitlines()[0]
    assert "surface:" not in out


def test_invoke_unknown_or_task_skill_errors():
    tools = SkillTools()
    out = _result_text(asyncio.run(tools.call("no-such-skill", {})))
    assert "unknown skill" in out
    gated = _result_text(asyncio.run(tools.call("validate-cat-identity", {})))
    assert "unknown skill" in gated  # task-only: not registered in chat
    everything = SkillTools(surfaces=("task", "chat", "both"))
    ok = _result_text(asyncio.run(everything.call("validate-cat-identity", {})))
    assert ok.startswith("# Validate a detection") and "send_to_human" in ok


def test_local_tools_carry_no_skills():
    with tempfile.TemporaryDirectory() as tmp:
        names = {s["name"] for s in LocalTools(Store(root=tmp)).specs()}
        assert names.isdisjoint(SHIPPED)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
