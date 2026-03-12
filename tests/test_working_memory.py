"""Tests for working memory (agents.md) support."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from natshell.agent.working_memory import (
    effective_memory_chars,
    find_memory_file,
    find_project_root,
    load_working_memory,
    memory_file_path,
    should_inject_memory,
)

# ── find_project_root ──────────────────────────────────────────────────


def test_find_project_root_git(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    assert find_project_root(sub) == tmp_path


def test_find_project_root_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").touch()
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").touch()
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_cargo(tmp_path: Path) -> None:
    (tmp_path / "Cargo.toml").touch()
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_go_mod(tmp_path: Path) -> None:
    (tmp_path / "go.mod").touch()
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_makefile(tmp_path: Path) -> None:
    (tmp_path / "Makefile").touch()
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_cmake(tmp_path: Path) -> None:
    (tmp_path / "CMakeLists.txt").touch()
    assert find_project_root(tmp_path) == tmp_path


def test_find_project_root_none_in_tmp(tmp_path: Path) -> None:
    sub = tmp_path / "empty"
    sub.mkdir()
    assert find_project_root(sub) is None


def test_find_project_root_picks_nearest(tmp_path: Path) -> None:
    """When nested markers exist, the innermost wins."""
    outer = tmp_path / "outer"
    outer.mkdir()
    (outer / ".git").mkdir()
    inner = outer / "inner"
    inner.mkdir()
    (inner / "pyproject.toml").touch()
    assert find_project_root(inner) == inner


# ── find_memory_file ──────────────────────────────────────────────────


def test_find_memory_file_project_local(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    mem_dir = tmp_path / ".natshell"
    mem_dir.mkdir()
    mem_file = mem_dir / "agents.md"
    mem_file.write_text("# Notes\nstuff")
    assert find_memory_file(tmp_path) == mem_file


def test_find_memory_file_global_fallback(tmp_path: Path) -> None:
    """Falls back to ~/.config/natshell/agents.md when no project-local file."""
    (tmp_path / ".git").mkdir()
    global_dir = tmp_path / "fakehome" / ".config" / "natshell"
    global_dir.mkdir(parents=True)
    global_file = global_dir / "agents.md"
    global_file.write_text("global notes")

    with patch("natshell.agent.working_memory.Path.home", return_value=tmp_path / "fakehome"):
        result = find_memory_file(tmp_path)
    assert result == global_file


def test_find_memory_file_prefers_local_over_global(tmp_path: Path) -> None:
    """Project-local agents.md takes priority over global."""
    (tmp_path / ".git").mkdir()
    # Create project-local
    local_dir = tmp_path / ".natshell"
    local_dir.mkdir()
    local_file = local_dir / "agents.md"
    local_file.write_text("local notes")
    # Create global
    global_dir = tmp_path / "fakehome" / ".config" / "natshell"
    global_dir.mkdir(parents=True)
    (global_dir / "agents.md").write_text("global notes")

    with patch("natshell.agent.working_memory.Path.home", return_value=tmp_path / "fakehome"):
        result = find_memory_file(tmp_path)
    assert result == local_file


def test_find_memory_file_none_when_missing(tmp_path: Path) -> None:
    sub = tmp_path / "empty"
    sub.mkdir()
    with patch("natshell.agent.working_memory.Path.home", return_value=tmp_path / "fakehome"):
        assert find_memory_file(sub) is None


# ── load_working_memory ──────────────────────────────────────────────


def test_load_working_memory_basic(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    mem_dir = tmp_path / ".natshell"
    mem_dir.mkdir()
    mem_file = mem_dir / "agents.md"
    mem_file.write_text("# Deploy target\nstaging.example.com")

    result = load_working_memory(tmp_path)
    assert result is not None
    assert result.content == "# Deploy target\nstaging.example.com"
    assert result.source == mem_file
    assert result.is_project_local is True


def test_load_working_memory_truncates(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    mem_dir = tmp_path / ".natshell"
    mem_dir.mkdir()
    (mem_dir / "agents.md").write_text("x" * 5000)

    result = load_working_memory(tmp_path, max_chars=100)
    assert result is not None
    assert len(result.content) < 200
    assert result.content.endswith("... (truncated)")


def test_load_working_memory_none_when_missing(tmp_path: Path) -> None:
    sub = tmp_path / "empty"
    sub.mkdir()
    with patch("natshell.agent.working_memory.Path.home", return_value=tmp_path / "fakehome"):
        assert load_working_memory(sub) is None


def test_load_working_memory_none_when_empty(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    mem_dir = tmp_path / ".natshell"
    mem_dir.mkdir()
    (mem_dir / "agents.md").write_text("   \n  ")
    assert load_working_memory(tmp_path) is None


def test_load_working_memory_global(tmp_path: Path) -> None:
    """Global memory file is loaded and flagged as not project-local."""
    global_dir = tmp_path / "fakehome" / ".config" / "natshell"
    global_dir.mkdir(parents=True)
    global_file = global_dir / "agents.md"
    global_file.write_text("global state")

    sub = tmp_path / "noproject"
    sub.mkdir()

    with patch("natshell.agent.working_memory.Path.home", return_value=tmp_path / "fakehome"):
        result = load_working_memory(sub)
    assert result is not None
    assert result.is_project_local is False
    assert result.source == global_file


# ── memory_file_path ─────────────────────────────────────────────────


def test_memory_file_path_with_project(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    assert memory_file_path(tmp_path) == tmp_path / ".natshell" / "agents.md"


def test_memory_file_path_without_project(tmp_path: Path) -> None:
    sub = tmp_path / "empty"
    sub.mkdir()
    expected = Path.home() / ".config" / "natshell" / "agents.md"
    assert memory_file_path(sub) == expected


# ── should_inject_memory ─────────────────────────────────────────────


def test_should_inject_memory_above_threshold() -> None:
    assert should_inject_memory(32768) is True
    assert should_inject_memory(16384) is True
    assert should_inject_memory(131072) is True


def test_should_inject_memory_below_threshold() -> None:
    assert should_inject_memory(4096) is False
    assert should_inject_memory(8192) is False
    assert should_inject_memory(16383) is False


def test_should_inject_memory_custom_min() -> None:
    assert should_inject_memory(8192, min_ctx=8192) is True
    assert should_inject_memory(4096, min_ctx=8192) is False


# ── System prompt injection ──────────────────────────────────────────


def test_memory_appears_in_system_prompt() -> None:
    from natshell.agent.context import SystemContext
    from natshell.agent.system_prompt import build_system_prompt

    ctx = SystemContext(hostname="test", distro="TestOS", kernel="5.0")
    prompt = build_system_prompt(
        ctx,
        working_memory="deploy target is staging.example.com",
        memory_path="/project/.natshell/agents.md",
    )
    assert "deploy target is staging.example.com" in prompt
    assert "/project/.natshell/agents.md" in prompt
    assert "<working_memory>" in prompt


def test_memory_omitted_for_small_ctx() -> None:
    from natshell.agent.context import SystemContext
    from natshell.agent.system_prompt import build_system_prompt

    ctx = SystemContext(hostname="test", distro="TestOS", kernel="5.0")
    prompt = build_system_prompt(ctx, compact=True)
    assert "<working_memory>" not in prompt


# ── Config ───────────────────────────────────────────────────────────


def test_memory_config_defaults() -> None:
    from natshell.config import MemoryConfig

    mc = MemoryConfig()
    assert mc.enabled is True
    assert mc.max_chars == 4000
    assert mc.min_ctx == 16384


def test_memory_config_in_natshell_config() -> None:
    from natshell.config import NatShellConfig

    cfg = NatShellConfig()
    assert hasattr(cfg, "memory")
    assert cfg.memory.enabled is True


# ── Safety exemption ─────────────────────────────────────────────────


def test_safety_exempts_agents_md_write() -> None:
    from natshell.config import SafetyConfig
    from natshell.safety.classifier import Risk, SafetyClassifier

    sc = SafetyClassifier(SafetyConfig())
    risk = sc.classify_tool_call(
        "write_file", {"path": "/home/user/project/.natshell/agents.md"}
    )
    assert risk == Risk.SAFE


def test_safety_exempts_agents_md_edit() -> None:
    from natshell.config import SafetyConfig
    from natshell.safety.classifier import Risk, SafetyClassifier

    sc = SafetyClassifier(SafetyConfig())
    risk = sc.classify_tool_call(
        "edit_file", {"path": "/home/user/.config/natshell/agents.md"}
    )
    assert risk == Risk.SAFE


def test_safety_still_confirms_other_writes() -> None:
    from natshell.config import SafetyConfig
    from natshell.safety.classifier import Risk, SafetyClassifier

    sc = SafetyClassifier(SafetyConfig())
    risk = sc.classify_tool_call("write_file", {"path": "/home/user/foo.txt"})
    assert risk == Risk.CONFIRM


# ── effective_memory_chars ─────────────────────────────────────────


def test_effective_memory_chars_default() -> None:
    """Below 32K threshold, returns the base value."""
    assert effective_memory_chars(4096) == 4000
    assert effective_memory_chars(16384) == 4000
    assert effective_memory_chars(31999) == 4000


def test_effective_memory_chars_scaling() -> None:
    """Each tier returns the expected scaled value."""
    assert effective_memory_chars(32768) == 8000
    assert effective_memory_chars(65536) == 12000
    assert effective_memory_chars(131072) == 16000
    assert effective_memory_chars(262144) == 24000
    assert effective_memory_chars(524288) == 32000
    assert effective_memory_chars(1048576) == 32000


def test_effective_memory_chars_custom_base() -> None:
    """Custom base_chars is used below 32K threshold."""
    assert effective_memory_chars(4096, base_chars=2000) == 2000
    assert effective_memory_chars(16384, base_chars=6000) == 6000
    # Above 32K, base_chars is ignored
    assert effective_memory_chars(32768, base_chars=2000) == 8000


def test_memory_injected_at_exactly_16384() -> None:
    """At n_ctx=16384, memory should be injected (compact=False)."""
    assert should_inject_memory(16384) is True
    # And the system prompt should NOT be compact at 16384
    from natshell.agent.context import SystemContext
    from natshell.agent.system_prompt import build_system_prompt

    ctx = SystemContext(hostname="test", distro="TestOS", kernel="5.0")
    # compact=False at 16384 means working_memory section is included
    prompt = build_system_prompt(
        ctx,
        compact=False,
        working_memory="test content",
        memory_path="/test/agents.md",
    )
    assert "<working_memory>" in prompt
