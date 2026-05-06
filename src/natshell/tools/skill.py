"""Skill tool — load instructions for a named skill."""

from __future__ import annotations

from typing import TYPE_CHECKING

from natshell.tools.registry import ToolDefinition, ToolResult

if TYPE_CHECKING:
    from natshell.skills import SkillRegistry

_SKILL_REGISTRY: SkillRegistry | None = None


def set_skill_registry(registry: SkillRegistry) -> None:
    global _SKILL_REGISTRY
    _SKILL_REGISTRY = registry


DEFINITION = ToolDefinition(
    name="skill",
    description=(
        "Load the full instructions for a named skill. "
        "Call this when the user's task matches a skill listed in <available_skills>."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the skill to load (e.g. 'spreadsheet').",
            }
        },
        "required": ["name"],
    },
    requires_confirmation=False,
)


async def skill(name: str) -> ToolResult:
    if _SKILL_REGISTRY is None:
        return ToolResult(output="", error="skill registry not initialized", exit_code=1)

    s = _SKILL_REGISTRY.get(name)
    if s is None or s.name in _SKILL_REGISTRY._disabled:
        available = ", ".join(x.name for x in _SKILL_REGISTRY.enabled()) or "(none)"
        return ToolResult(
            output="",
            error=f"unknown or disabled skill: {name!r}. Available: {available}",
            exit_code=1,
        )

    body = _SKILL_REGISTRY.load_body(name) or ""
    assets = _SKILL_REGISTRY.list_assets(name)

    parts = [f"# Skill: {name}", body]
    if assets["references"] or assets["scripts"]:
        parts.append("\n---\n## Bundled assets\nUse read_file or run_code on these as needed.\n")
        for ref in assets["references"]:
            parts.append(f"- reference: {ref}")
        for sc in assets["scripts"]:
            parts.append(f"- script: {sc}")

    return ToolResult(output="\n".join(parts), error=None, exit_code=0)
