"""Expose every skills/<name>/SKILL.md as a callable local__<name> tool.

A skill tool keeps the skill's own name and description (marked as a skill).
It takes no arguments: invoking it returns the skill file's full instructions,
so the model can follow them from there. New skills are picked up automatically
by scanning the skills folder; no registration needed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from mcp.types import CallToolResult, TextContent

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parents[1] / "skills"


@dataclass(frozen=True)
class Skill:
    """One parsed SKILL.md: tool name, tool description source, and file to load on invoke."""

    name: str
    description: str
    path: Path

    def spec(self) -> dict:
        return {
            "name": self.name,
            "description": f"Skill '{self.name}': {self.description} "
            "Invoke it (no arguments) to load the full skill instructions, then follow them step by step.",
            "input_schema": {"type": "object", "properties": {}},
        }

    def read(self) -> str:
        return self.path.read_text()


def parse(path: Path) -> Skill | None:
    """SKILL.md -> Skill, or None (with a warning) when name/description are missing."""
    try:
        text = path.read_text()
    except OSError as e:
        logger.warning(f"skill {path}: unreadable ({e})")
        return None
    if not text.startswith("---\n"):
        logger.warning(f"skill {path}: no frontmatter, skipped")
        return None
    front, _, _ = text[4:].partition("\n---\n")
    meta = {}
    for line in front.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip()
    if not meta.get("name") or not meta.get("description"):
        logger.warning(f"skill {path}: frontmatter needs name and description, skipped")
        return None
    return Skill(name=meta["name"], description=meta["description"], path=path)


def scan(directory: Path | None = None) -> list[Skill]:
    """All valid skills in a folder (sorted by name); the factory for skill tools."""
    directory = Path(directory) if directory is not None else SKILLS_DIR
    skills = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        skill = parse(child / "SKILL.md")
        if skill is None:
            continue
        if any(s.name == skill.name for s in skills):
            logger.warning(f"skill {child}: duplicate name '{skill.name}', skipped")
            continue
        skills.append(skill)
    return skills


class SkillTools:
    """In-process `skill__*` toolset: one tool per SKILL.md in the skills folder.

    Register on the hub next to LocalTools. Invoking a tool (no arguments)
    returns its skill instructions for the model to follow from there.
    """

    server_name = "skill"

    def __init__(self, directory: Path | None = None):
        self.skills: dict[str, Skill] = {s.name: s for s in scan(directory)}

    def specs(self) -> list[dict]:
        return [s.spec() for s in self.skills.values()]

    async def call(self, name: str, args: dict) -> CallToolResult:
        skill = self.skills.get(name)
        if skill is None:
            return _json({"error": f"unknown skill {name}"}, error=True)
        try:
            return _json({"skill": skill.name, "instructions": skill.read()})
        except Exception as e:  # report to the model, don't crash the run
            logger.exception(f"skill__{name}")
            return _json({"error": f"{type(e).__name__}: {e}"}, error=True)


def _json(payload: dict, error: bool = False) -> CallToolResult:
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, default=str))],
        is_error=error,
    )
