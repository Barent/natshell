"""Skill discovery, registration, and access."""

from __future__ import annotations

import importlib.resources
import importlib.util
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from natshell.config import NatShellConfig
    from natshell.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

USER_SKILLS_DIR = Path.home() / ".config" / "natshell" / "skills"
PROJECT_SKILLS_DIRNAME = ".natshell/skills"

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n(.*)$", re.DOTALL)
_KV_RE = re.compile(r"^([a-z_][a-z0-9_]*):\s*(.*?)\s*$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (metadata, body). Raises ValueError on malformed frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("missing or malformed frontmatter")
    meta_block, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in meta_block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        kv = _KV_RE.match(line)
        if not kv:
            raise ValueError(f"bad frontmatter line: {line!r}")
        key, val = kv.group(1), kv.group(2)
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        meta[key] = val
    return meta, body


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    source: str  # "builtin" | "user" | "project"


class SkillRegistry:
    def __init__(self, skills: list[Skill], disabled: set[str]) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in skills}
        self._disabled: set[str] = disabled

    @staticmethod
    def _normalize(name: str) -> str:
        """Fold a skill name for tolerant lookup (hyphen/underscore/case agnostic)."""
        return name.strip().lower().replace("_", "-")

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def enabled(self) -> list[Skill]:
        return [s for s in self._skills.values() if s.name not in self._disabled]

    def get(self, name: str) -> Skill | None:
        s = self._skills.get(name)
        if s is not None:
            return s
        # Tolerate hyphen/underscore/case variants — models often emit e.g.
        # "web_research" for a skill registered as "web-research".
        target = self._normalize(name)
        for skill in self._skills.values():
            if self._normalize(skill.name) == target:
                return skill
        return None

    def load_body(self, name: str) -> str | None:
        s = self.get(name)
        if s is None:
            return None
        text = (s.path / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        _meta, body = _parse_frontmatter(text)
        return body.strip()

    def list_assets(self, name: str) -> dict[str, list[Path]]:
        s = self.get(name)
        if s is None:
            return {"references": [], "scripts": []}
        out: dict[str, list[Path]] = {"references": [], "scripts": []}
        for sub in ("references", "scripts"):
            d = s.path / sub
            if d.is_dir():
                out[sub] = sorted(p for p in d.iterdir() if p.is_file())
        return out

    def enable(self, name: str) -> bool:
        """Remove name from disabled set. Returns True if it was disabled.

        Tolerant of hyphen/underscore/case variants. Falls back to the raw name
        so a skill disabled in config but not currently present can still be
        re-enabled.
        """
        s = self.get(name)
        canonical = s.name if s is not None else name
        if canonical in self._disabled:
            self._disabled.discard(canonical)
            return True
        return False

    def disable(self, name: str) -> bool:
        """Add name to disabled set. Returns True if it was enabled.

        Tolerant of hyphen/underscore/case variants.
        """
        s = self.get(name)
        if s is not None and s.name not in self._disabled:
            self._disabled.add(s.name)
            return True
        return False


def _iter_skill_dirs() -> Iterable[tuple[Path, str]]:
    """Yield (skill_dir, source_label). Later sources override earlier on name collision."""
    # 1. builtin — iterate the skills package directory itself
    try:
        builtin_root = importlib.resources.files(__name__)
        for entry in builtin_root.iterdir():
            if entry.is_dir() and (entry / "SKILL.md").is_file():  # type: ignore[operator]
                yield Path(str(entry)), "builtin"
    except (ModuleNotFoundError, FileNotFoundError, TypeError, AttributeError):
        pass
    # 2. user
    if USER_SKILLS_DIR.is_dir():
        for entry in USER_SKILLS_DIR.iterdir():
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                yield entry, "user"
    # 3. project
    project_dir = Path.cwd() / PROJECT_SKILLS_DIRNAME
    if project_dir.is_dir():
        for entry in project_dir.iterdir():
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                yield entry, "project"


def load_skills(tool_registry: ToolRegistry, config: NatShellConfig) -> SkillRegistry:
    """Discover and load skills from builtin, user, and project directories.

    Errors are logged as warnings and never crash startup.
    """
    seen: dict[str, Skill] = {}
    disabled: set[str] = set(getattr(config.skills, "disabled", None) or [])
    if not getattr(config.skills, "enabled", True):
        return SkillRegistry([], disabled)

    old_plugin_dir = Path.home() / ".config" / "natshell" / "plugins"
    if old_plugin_dir.is_dir() and any(old_plugin_dir.glob("*.py")):
        import sys
        print(
            f"\nWARNING: {old_plugin_dir} is no longer loaded.\n"
            "Wrap your register() function in a skill: "
            "~/.config/natshell/skills/<name>/tools.py\n"
            "See SKILL_AUTHORING.md for migration instructions.\n",
            file=sys.stderr,
        )

    for skill_dir, source in _iter_skill_dirs():
        try:
            text = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="replace")
            meta, _body = _parse_frontmatter(text)
        except Exception as e:
            logger.warning("skipping skill at %s: %s", skill_dir, e)
            continue

        name = meta.get("name", "").strip()
        desc = meta.get("description", "").strip()

        if not _NAME_RE.match(name):
            logger.warning("skipping skill at %s: invalid name %r", skill_dir, name)
            continue
        if name != skill_dir.name:
            logger.warning(
                "skipping skill at %s: name %r != dir %r", skill_dir, name, skill_dir.name
            )
            continue
        if not desc or len(desc) > 200:
            logger.warning("skipping skill %s: description missing or > 200 chars", name)
            continue
        if "\n" in desc:
            logger.warning("skipping skill %s: description must be one line", name)
            continue

        if name in seen:
            logger.info("skill %s: %s overrides %s", name, source, seen[name].source)
        seen[name] = Skill(name=name, description=desc, path=skill_dir, source=source)

        tools_py = skill_dir / "tools.py"
        if tools_py.is_file():
            try:
                spec = importlib.util.spec_from_file_location(
                    f"natshell_skill_tools_{name.replace('-', '_')}", tools_py
                )
                if spec is None or spec.loader is None:
                    logger.warning("skill %s tools.py: invalid module spec", name)
                else:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)  # type: ignore[union-attr]
                    if hasattr(module, "register") and callable(module.register):
                        module.register(tool_registry)
                    else:
                        logger.warning("skill %s tools.py has no register() callable", name)
            except Exception as e:
                logger.warning("skill %s tools.py load failed: %s", name, e)

    return SkillRegistry(list(seen.values()), disabled)
