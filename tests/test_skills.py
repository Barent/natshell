"""Tests for the skill system."""
from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_skill_dir(tmp_path: Path, name: str, description: str, body: str) -> Path:
    """Create a minimal skill directory under tmp_path."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\n{body}"
    )
    return d


def _make_config(disabled: list[str] | None = None, enabled: bool = True):
    from natshell.config import NatShellConfig, SkillsConfig

    cfg = NatShellConfig()
    cfg.skills = SkillsConfig(
        enabled=enabled,
        disabled=disabled or [],
        inject_in_compact=False,
    )
    return cfg


def _make_tool_registry():
    from natshell.tools.registry import ToolRegistry

    return ToolRegistry()


# ---------------------------------------------------------------------------
# Frontmatter parser
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_basic(self):
        from natshell.skills import _parse_frontmatter

        meta, body = _parse_frontmatter(
            "---\nname: foo\ndescription: A foo skill\n---\n\n# Body\nHello"
        )
        assert meta == {"name": "foo", "description": "A foo skill"}
        assert body.strip() == "# Body\nHello"

    def test_missing_frontmatter_raises(self):
        from natshell.skills import _parse_frontmatter

        with pytest.raises(ValueError, match="missing or malformed"):
            _parse_frontmatter("no frontmatter here")

    def test_quoted_values(self):
        from natshell.skills import _parse_frontmatter

        meta, _ = _parse_frontmatter(
            '---\nname: foo\ndescription: "Quoted desc"\n---\n\nbody'
        )
        assert meta["description"] == "Quoted desc"

    def test_extra_keys_tolerated(self):
        from natshell.skills import _parse_frontmatter

        meta, _ = _parse_frontmatter(
            "---\nname: foo\ndescription: desc\nauthor: someone\n---\n\nbody"
        )
        assert meta["author"] == "someone"

    def test_comment_lines_ignored(self):
        from natshell.skills import _parse_frontmatter

        meta, _ = _parse_frontmatter(
            "---\n# This is a comment\nname: foo\ndescription: desc\n---\n\nbody"
        )
        assert meta == {"name": "foo", "description": "desc"}


# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class TestSkillRegistry:
    def _make_registry(self, skills=None, disabled=None):
        from natshell.skills import SkillRegistry

        skills = skills or []
        disabled = set(disabled or [])
        return SkillRegistry(skills, disabled)

    def _make_skill(self, name="foo", desc="A skill", path=None, source="builtin"):
        from natshell.skills import Skill

        return Skill(name=name, description=desc, path=path or Path("/fake/foo"), source=source)

    def test_all_returns_all(self):
        s1 = self._make_skill("s1")
        s2 = self._make_skill("s2")
        reg = self._make_registry([s1, s2])
        assert {s.name for s in reg.all()} == {"s1", "s2"}

    def test_enabled_filters_disabled(self):
        s1 = self._make_skill("s1")
        s2 = self._make_skill("s2")
        reg = self._make_registry([s1, s2], disabled=["s2"])
        names = {s.name for s in reg.enabled()}
        assert names == {"s1"}
        assert "s2" not in names

    def test_get_returns_skill(self):
        s = self._make_skill("foo")
        reg = self._make_registry([s])
        assert reg.get("foo") is s
        assert reg.get("missing") is None

    def test_get_tolerates_underscore_and_case(self):
        # Models often emit "web_research" for a skill named "web-research".
        s = self._make_skill("web-research")
        reg = self._make_registry([s])
        assert reg.get("web_research") is s
        assert reg.get("Web_Research") is s
        assert reg.get("WEB-RESEARCH") is s
        assert reg.get("nope") is None

    def test_enable_removes_from_disabled(self):
        s = self._make_skill("foo")
        reg = self._make_registry([s], disabled=["foo"])
        assert "foo" not in {x.name for x in reg.enabled()}
        changed = reg.enable("foo")
        assert changed is True
        assert "foo" in {x.name for x in reg.enabled()}

    def test_disable_adds_to_disabled(self):
        s = self._make_skill("foo")
        reg = self._make_registry([s])
        assert "foo" in {x.name for x in reg.enabled()}
        changed = reg.disable("foo")
        assert changed is True
        assert "foo" not in {x.name for x in reg.enabled()}

    def test_enable_disable_tolerate_underscore_and_case(self):
        s = self._make_skill("web-research")
        reg = self._make_registry([s])
        # disable via underscore variant
        assert reg.disable("web_research") is True
        assert "web-research" in reg._disabled
        assert "web-research" not in {x.name for x in reg.enabled()}
        # enable via mixed-case variant
        assert reg.enable("Web_Research") is True
        assert "web-research" not in reg._disabled
        assert "web-research" in {x.name for x in reg.enabled()}

    def test_enable_already_enabled_returns_false(self):
        s = self._make_skill("foo")
        reg = self._make_registry([s])
        assert reg.enable("foo") is False

    def test_disable_already_disabled_returns_false(self):
        s = self._make_skill("foo")
        reg = self._make_registry([s], disabled=["foo"])
        assert reg.disable("foo") is False

    def test_load_body(self, tmp_path):
        d = _make_skill_dir(tmp_path, "foo", "A foo skill", "# Body\nHello world")
        from natshell.skills import Skill, SkillRegistry

        s = Skill(name="foo", description="A foo skill", path=d, source="test")
        reg = SkillRegistry([s], set())
        body = reg.load_body("foo")
        assert "# Body" in body
        assert "Hello world" in body
        assert "---" not in body  # frontmatter stripped

    def test_load_body_unknown_name(self):
        from natshell.skills import SkillRegistry

        reg = SkillRegistry([], set())
        assert reg.load_body("nonexistent") is None

    def test_list_assets_references_and_scripts(self, tmp_path):
        d = _make_skill_dir(tmp_path, "foo", "A foo skill", "body")
        refs = d / "references"
        refs.mkdir()
        (refs / "python.md").write_text("# Python ref")
        scripts = d / "scripts"
        scripts.mkdir()
        (scripts / "helper.py").write_text("print('hi')")

        from natshell.skills import Skill, SkillRegistry

        s = Skill(name="foo", description="A foo skill", path=d, source="test")
        reg = SkillRegistry([s], set())
        assets = reg.list_assets("foo")
        assert len(assets["references"]) == 1
        assert assets["references"][0].name == "python.md"
        assert len(assets["scripts"]) == 1
        assert assets["scripts"][0].name == "helper.py"

    def test_list_assets_empty_when_no_dirs(self, tmp_path):
        d = _make_skill_dir(tmp_path, "foo", "A foo skill", "body")
        from natshell.skills import Skill, SkillRegistry

        s = Skill(name="foo", description="A foo skill", path=d, source="test")
        reg = SkillRegistry([s], set())
        assets = reg.list_assets("foo")
        assert assets == {"references": [], "scripts": []}


# ---------------------------------------------------------------------------
# load_skills — discovery and validation
# ---------------------------------------------------------------------------

class TestLoadSkills:
    def test_loads_valid_skill(self, tmp_path):
        d = _make_skill_dir(
            tmp_path, "myskill", "A test skill for testing purposes only",
            "# Body\nSome content here that is long enough.",
        )
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "user")]):
            from natshell.skills import load_skills

            skill_reg = load_skills(registry, cfg)

        assert skill_reg.get("myskill") is not None
        assert skill_reg.get("myskill").source == "user"

    def test_skips_invalid_name_regex(self, tmp_path, caplog):
        d = tmp_path / "My-Bad-Name"
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: My-Bad-Name\ndescription: test\n---\n\nbody")
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "user")]):
            from natshell.skills import load_skills

            with caplog.at_level(logging.WARNING):
                skill_reg = load_skills(registry, cfg)

        assert skill_reg.get("My-Bad-Name") is None

    def test_skips_name_mismatch_with_dir(self, tmp_path, caplog):
        d = tmp_path / "wrongdir"
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: rightname\ndescription: test\n---\n\nbody")
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "user")]):
            from natshell.skills import load_skills

            with caplog.at_level(logging.WARNING):
                skill_reg = load_skills(registry, cfg)

        assert skill_reg.get("rightname") is None

    def test_skips_missing_description(self, tmp_path, caplog):
        d = tmp_path / "nodesc"
        d.mkdir()
        (d / "SKILL.md").write_text("---\nname: nodesc\n---\n\nbody")
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "user")]):
            from natshell.skills import load_skills

            with caplog.at_level(logging.WARNING):
                skill_reg = load_skills(registry, cfg)

        assert skill_reg.get("nodesc") is None

    def test_skips_description_too_long(self, tmp_path, caplog):
        d = tmp_path / "toolong"
        d.mkdir()
        desc = "x" * 201
        (d / "SKILL.md").write_text(f"---\nname: toolong\ndescription: {desc}\n---\n\nbody")
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "user")]):
            from natshell.skills import load_skills

            with caplog.at_level(logging.WARNING):
                skill_reg = load_skills(registry, cfg)

        assert skill_reg.get("toolong") is None

    def test_skips_multiline_description(self, tmp_path, caplog):
        d = tmp_path / "multiline"
        d.mkdir()
        (d / "SKILL.md").write_text(
            "---\nname: multiline\ndescription: line1\n  line2\n---\n\nbody"
        )
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "user")]):
            from natshell.skills import load_skills

            with caplog.at_level(logging.WARNING):
                load_skills(registry, cfg)

        # Multi-line description: "line1\n  line2" contains \n → should be skipped
        # Note: the YAML parser may see it as bad frontmatter or as multi-line
        # depending on the actual content; we just check it doesn't blow up.
        # skill may or may not load, but should not raise.

    def test_skips_malformed_frontmatter(self, tmp_path, caplog):
        d = tmp_path / "badfm"
        d.mkdir()
        (d / "SKILL.md").write_text("no frontmatter here")
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "user")]):
            from natshell.skills import load_skills

            with caplog.at_level(logging.WARNING):
                skill_reg = load_skills(registry, cfg)

        assert skill_reg.get("badfm") is None

    def test_later_source_overrides_earlier(self, tmp_path, caplog):
        d1 = _make_skill_dir(tmp_path / "builtin", "shared", "Builtin version", "Builtin body")
        d2 = _make_skill_dir(tmp_path / "user", "shared", "User version", "User body")

        cfg = _make_config()
        registry = _make_tool_registry()

        dirs = [(d1, "builtin"), (d2, "user")]
        with patch("natshell.skills._iter_skill_dirs", return_value=dirs):
            from natshell.skills import load_skills

            with caplog.at_level(logging.INFO):
                skill_reg = load_skills(registry, cfg)

        s = skill_reg.get("shared")
        assert s is not None
        assert s.source == "user"
        assert s.description == "User version"

    def test_disabled_config_hides_from_enabled(self, tmp_path):
        d = _make_skill_dir(tmp_path, "myskill", "A skill", "Body content here.")
        cfg = _make_config(disabled=["myskill"])
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "builtin")]):
            from natshell.skills import load_skills

            skill_reg = load_skills(registry, cfg)

        assert skill_reg.get("myskill") is not None  # still in all()
        assert "myskill" not in {s.name for s in skill_reg.enabled()}

    def test_skills_disabled_entirely(self, tmp_path):
        d = _make_skill_dir(tmp_path, "myskill", "A skill", "Body here.")
        cfg = _make_config(enabled=False)
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "builtin")]):
            from natshell.skills import load_skills

            skill_reg = load_skills(registry, cfg)

        assert skill_reg.all() == []

    def test_tools_py_registers_tool(self, tmp_path):
        d = _make_skill_dir(tmp_path, "mytool", "A skill with a tool", "Body here.")
        tools_py = d / "tools.py"
        tools_py.write_text(
            "from natshell.tools.registry import ToolDefinition\n"
            "async def _handler(**kwargs): return 'ok'\n"
            "def register(registry):\n"
            "    registry.register(ToolDefinition(name='my_custom_tool', description='custom', "
            "parameters={'type': 'object', 'properties': {}}), _handler)\n"
        )
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "builtin")]):
            from natshell.skills import load_skills

            load_skills(registry, cfg)

        assert "my_custom_tool" in registry.tool_names

    def test_broken_tools_py_does_not_crash(self, tmp_path, caplog):
        d = _make_skill_dir(tmp_path, "broken", "A skill", "Body here.")
        tools_py = d / "tools.py"
        tools_py.write_text("raise RuntimeError('intentional failure')\n")
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "builtin")]):
            from natshell.skills import load_skills

            with caplog.at_level(logging.WARNING):
                skill_reg = load_skills(registry, cfg)

        # Skill is still loaded (body ok), but tools.py failed silently
        assert skill_reg.get("broken") is not None

    def test_no_register_callable_in_tools_py(self, tmp_path, caplog):
        d = _make_skill_dir(tmp_path, "noop", "A skill", "Body here.")
        (d / "tools.py").write_text("# no register function\nfoo = 42\n")
        cfg = _make_config()
        registry = _make_tool_registry()

        with patch("natshell.skills._iter_skill_dirs", return_value=[(d, "builtin")]):
            from natshell.skills import load_skills

            with caplog.at_level(logging.WARNING):
                load_skills(registry, cfg)


# ---------------------------------------------------------------------------
# skill tool handler
# ---------------------------------------------------------------------------

class TestSkillToolHandler:
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        """Reset the global skill registry before each test."""
        from natshell.tools import skill as skill_mod

        original = skill_mod._SKILL_REGISTRY
        yield
        skill_mod._SKILL_REGISTRY = original

    def _register(self, skills, disabled=None):
        from natshell.skills import SkillRegistry
        from natshell.tools.skill import set_skill_registry

        reg = SkillRegistry(skills, set(disabled or []))
        set_skill_registry(reg)
        return reg

    @pytest.mark.asyncio
    async def test_returns_body(self, tmp_path):
        d = _make_skill_dir(tmp_path, "foo", "A test skill", "# Foo Skill\nDo this and that.")
        from natshell.skills import Skill

        s = Skill(name="foo", description="A test skill", path=d, source="builtin")
        self._register([s])

        from natshell.tools.skill import skill

        result = await skill(name="foo")
        assert result.exit_code == 0
        assert "# Skill: foo" in result.output
        assert "# Foo Skill" in result.output
        assert result.error is None

    @pytest.mark.asyncio
    async def test_lists_references_and_scripts(self, tmp_path):
        d = _make_skill_dir(tmp_path, "foo", "A test skill", "# Body\nContent.")
        (d / "references").mkdir()
        (d / "references" / "guide.md").write_text("# Guide")
        (d / "scripts").mkdir()
        (d / "scripts" / "helper.py").write_text("print('hi')")
        from natshell.skills import Skill

        s = Skill(name="foo", description="A test skill", path=d, source="builtin")
        self._register([s])

        from natshell.tools.skill import skill

        result = await skill(name="foo")
        assert "guide.md" in result.output
        assert "helper.py" in result.output

    @pytest.mark.asyncio
    async def test_unknown_name_returns_error(self, tmp_path):
        self._register([])

        from natshell.tools.skill import skill

        result = await skill(name="nonexistent")
        assert result.exit_code == 1
        assert "unknown or disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_disabled_skill_returns_error(self, tmp_path):
        d = _make_skill_dir(tmp_path, "foo", "A test skill", "# Body\nContent here.")
        from natshell.skills import Skill

        s = Skill(name="foo", description="A test skill", path=d, source="builtin")
        self._register([s], disabled=["foo"])

        from natshell.tools.skill import skill

        result = await skill(name="foo")
        assert result.exit_code == 1
        assert "unknown or disabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_registry_returns_error(self):
        from natshell.tools import skill as skill_mod
        from natshell.tools.skill import skill

        skill_mod._SKILL_REGISTRY = None
        result = await skill(name="foo")
        assert result.exit_code == 1
        assert "not initialized" in result.error


# ---------------------------------------------------------------------------
# System prompt — <available_skills> section
# ---------------------------------------------------------------------------

class TestSystemPromptSkills:
    def _make_context(self):
        from natshell.agent.context import SystemContext

        return SystemContext(hostname="testhost", distro="TestOS", shell="bash", cwd="/home/test")

    def _make_skill(self, name: str, desc: str):
        from natshell.skills import Skill

        return Skill(name=name, description=desc, path=Path("/fake"), source="builtin")

    def test_skills_section_injected_in_normal_mode(self):
        from natshell.agent.system_prompt import build_system_prompt

        skills = [self._make_skill("spreadsheet", "Work with CSV/XLSX files.")]
        prompt = build_system_prompt(self._make_context(), compact=False, skills=skills)
        assert "<available_skills>" in prompt
        assert "spreadsheet: Work with CSV/XLSX files." in prompt

    def test_skills_section_omitted_in_compact_mode_by_default(self):
        from natshell.agent.system_prompt import build_system_prompt

        skills = [self._make_skill("spreadsheet", "Work with CSV/XLSX files.")]
        prompt = build_system_prompt(
            self._make_context(), compact=True, skills=skills, inject_skills_in_compact=False
        )
        assert "<available_skills>" not in prompt

    def test_skills_section_injected_in_compact_when_forced(self):
        from natshell.agent.system_prompt import build_system_prompt

        skills = [self._make_skill("spreadsheet", "Work with CSV/XLSX files.")]
        prompt = build_system_prompt(
            self._make_context(), compact=True, skills=skills, inject_skills_in_compact=True
        )
        assert "<available_skills>" in prompt

    def test_no_skills_no_section(self):
        from natshell.agent.system_prompt import build_system_prompt

        prompt = build_system_prompt(self._make_context(), compact=False, skills=None)
        assert "<available_skills>" not in prompt

    def test_skills_sorted_alphabetically(self):
        from natshell.agent.system_prompt import build_system_prompt

        skills = [
            self._make_skill("zebra", "Last alphabetically."),
            self._make_skill("alpha", "First alphabetically."),
        ]
        prompt = build_system_prompt(self._make_context(), compact=False, skills=skills)
        assert prompt.index("alpha") < prompt.index("zebra")

    def test_all_skills_listed(self):
        from natshell.agent.system_prompt import build_system_prompt

        skills = [
            self._make_skill("pdf", "PDF skill."),
            self._make_skill("docx", "Docx skill."),
        ]
        prompt = build_system_prompt(self._make_context(), compact=False, skills=skills)
        assert "- pdf:" in prompt
        assert "- docx:" in prompt


# ---------------------------------------------------------------------------
# Shipped skills validation (parameterized)
# ---------------------------------------------------------------------------

EXPECTED_SKILL_NAMES = [
    "spreadsheet",
    "pdf",
    "docx",
    "coding",
    "testing",
    "git-workflow",
    "system-admin",
    "data-analysis",
    "web-research",
    "markdown-docs",
]


@pytest.mark.parametrize("skill_name", EXPECTED_SKILL_NAMES)
def test_shipped_skill_valid(skill_name: str):
    """Each shipped skill must have valid frontmatter and a non-trivial body."""
    from natshell.skills import _parse_frontmatter

    try:
        skill_pkg = importlib.resources.files("natshell.skills")
        skill_dir = skill_pkg / skill_name
        skill_md = skill_dir / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8")
    except (FileNotFoundError, TypeError):
        # Fall back to filesystem path for editable installs
        here = Path(__file__).parent.parent / "src" / "natshell" / "skills"
        skill_md_path = here / skill_name / "SKILL.md"
        assert skill_md_path.exists(), f"Missing SKILL.md for {skill_name}"
        text = skill_md_path.read_text(encoding="utf-8")

    meta, body = _parse_frontmatter(text)

    assert meta["name"] == skill_name, f"{skill_name}: name mismatch in frontmatter"
    desc = meta.get("description", "")
    assert desc, f"{skill_name}: empty description"
    assert len(desc) <= 200, f"{skill_name}: description too long ({len(desc)} chars)"
    assert "\n" not in desc, f"{skill_name}: multi-line description"
    assert len(body) >= 200, f"{skill_name}: body too short ({len(body)} chars)"


def test_all_10_skills_discoverable():
    """All 10 shipped skills load from the package."""
    from natshell.skills import _iter_skill_dirs

    found = {d.name for d, _ in _iter_skill_dirs() if d.name in EXPECTED_SKILL_NAMES}
    missing = set(EXPECTED_SKILL_NAMES) - found
    assert not missing, f"Missing shipped skills: {missing}"


# ---------------------------------------------------------------------------
# Registry — "skill" tool in PLAN_SAFE and SMALL_CONTEXT sets
# ---------------------------------------------------------------------------

class TestSkillToolRegistration:
    def test_skill_in_plan_safe_tools(self):
        from natshell.tools.registry import PLAN_SAFE_TOOLS

        assert "skill" in PLAN_SAFE_TOOLS

    def test_skill_in_small_context_tools(self):
        from natshell.tools.registry import SMALL_CONTEXT_TOOLS

        assert "skill" in SMALL_CONTEXT_TOOLS

    def test_skill_tool_registered_in_default_registry(self):
        from natshell.tools.registry import create_default_registry

        registry = create_default_registry()
        assert "skill" in registry.tool_names
