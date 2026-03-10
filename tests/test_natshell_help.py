"""Tests for the natshell_help tool."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from natshell.config import SafetyConfig
from natshell.tools.natshell_help import (
    DEFINITION,
    VALID_TOPICS,
    natshell_help,
    set_safety_config,
)
from natshell.tools.registry import create_default_registry

# ─── static topics ────────────────────────────────────────────────────────


class TestStaticTopics:
    @pytest.mark.parametrize(
        "topic",
        [
            "overview",
            "commands",
            "tools",
            "models",
            "troubleshooting",
            "getting_started",
            "profiles",
            "sessions",
            "plans",
            "plugins",
            "headless",
            "mcp",
            "backup",
            "keyboard_shortcuts",
        ],
    )
    async def test_static_topic_returns_content(self, topic: str):
        result = await natshell_help(topic)
        assert result.exit_code == 0
        assert len(result.output) > 0

    async def test_overview_mentions_natshell(self):
        result = await natshell_help("overview")
        assert "NatShell" in result.output

    async def test_commands_lists_slash_commands(self):
        result = await natshell_help("commands")
        assert "/help" in result.output
        assert "/clear" in result.output
        assert "/model" in result.output
        assert "/cmd" in result.output
        assert "/compact" in result.output
        assert "/profile" in result.output
        assert "/undo" in result.output
        assert "/save" in result.output
        assert "/load" in result.output
        assert "/sessions" in result.output
        assert "/keys" in result.output
        assert "/history" in result.output
        assert "/model download" in result.output

    async def test_tools_lists_all_tools(self):
        result = await natshell_help("tools")
        assert "execute_shell" in result.output
        assert "read_file" in result.output
        assert "edit_file" in result.output
        assert "run_code" in result.output
        assert "git_tool" in result.output
        assert "natshell_help" in result.output

    async def test_models_mentions_qwen(self):
        result = await natshell_help("models")
        assert "Qwen" in result.output

    async def test_troubleshooting_has_gpu_hint(self):
        result = await natshell_help("troubleshooting")
        assert "GPU" in result.output or "gpu" in result.output

    async def test_getting_started_has_setup_info(self):
        result = await natshell_help("getting_started")
        assert "wizard" in result.output
        assert "/model download" in result.output

    async def test_profiles_has_config_example(self):
        result = await natshell_help("profiles")
        assert "/profile" in result.output
        assert "config.toml" in result.output

    async def test_sessions_has_save_load(self):
        result = await natshell_help("sessions")
        assert "/save" in result.output
        assert "/load" in result.output

    async def test_plans_has_exeplan(self):
        result = await natshell_help("plans")
        assert "/plan" in result.output
        assert "/exeplan" in result.output

    async def test_plugins_has_register(self):
        result = await natshell_help("plugins")
        assert "register" in result.output
        assert "plugins/" in result.output

    async def test_headless_has_flags(self):
        result = await natshell_help("headless")
        assert "--headless" in result.output
        assert "--danger-fast" in result.output

    async def test_mcp_has_protocol(self):
        result = await natshell_help("mcp")
        assert "--mcp" in result.output
        assert "JSON-RPC" in result.output

    async def test_backup_has_undo(self):
        result = await natshell_help("backup")
        assert "/undo" in result.output
        assert "backup" in result.output.lower()

    async def test_keyboard_shortcuts_has_keys(self):
        result = await natshell_help("keyboard_shortcuts")
        assert "Ctrl+C" in result.output
        assert "Ctrl+P" in result.output
        assert "Enter" in result.output


# ─── dynamic: config ──────────────────────────────────────────────────────


class TestConfigTopic:
    async def test_config_no_file(self, tmp_path: Path):
        with patch("natshell.tools.natshell_help.Path.home", return_value=tmp_path):
            result = await natshell_help("config")
        assert result.exit_code == 0
        assert "No user config" in result.output

    async def test_config_reads_file(self, tmp_path: Path):
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_file = config_dir / "config.toml"
        config_file.write_text("[agent]\nmax_steps = 10\n")

        with patch("natshell.tools.natshell_help.Path.home", return_value=tmp_path):
            result = await natshell_help("config")
        assert result.exit_code == 0
        assert "max_steps" in result.output


# ─── dynamic: config_reference ────────────────────────────────────────────


class TestConfigReferenceTopic:
    async def test_config_reference_returns_content(self):
        result = await natshell_help("config_reference")
        assert result.exit_code == 0
        # Should contain actual config.default.toml content or a fallback message
        assert len(result.output) > 0


# ─── dynamic: safety ─────────────────────────────────────────────────────


class TestSafetyTopic:
    async def test_safety_without_injection(self):
        # Reset global state
        import natshell.tools.natshell_help as mod

        original = mod._safety_config
        mod._safety_config = None
        try:
            result = await natshell_help("safety")
            assert result.exit_code == 0
            assert "not available" in result.output
        finally:
            mod._safety_config = original

    async def test_safety_with_injection(self):
        config = SafetyConfig(
            mode="confirm",
            always_confirm=["^rm\\s", "^sudo\\s"],
            blocked=[":(){ :|:& };:"],
        )
        set_safety_config(config)
        result = await natshell_help("safety")
        assert result.exit_code == 0
        assert "confirm" in result.output
        assert "^rm\\s" in result.output
        assert ":(){ :|:& };:" in result.output
        assert "2 patterns" in result.output
        assert "1 patterns" in result.output


# ─── invalid topic ───────────────────────────────────────────────────────


class TestInvalidTopic:
    async def test_unknown_topic_returns_error(self):
        result = await natshell_help("nonexistent")
        assert result.exit_code == 1
        assert "Unknown topic" in result.error
        assert "nonexistent" in result.error


# ─── definition and registry ─────────────────────────────────────────────


class TestDefinition:
    def test_definition_name(self):
        assert DEFINITION.name == "natshell_help"

    def test_definition_has_enum(self):
        enum_values = DEFINITION.parameters["properties"]["topic"]["enum"]
        assert set(enum_values) == set(VALID_TOPICS)

    def test_valid_topics_complete(self):
        expected = {
            "overview",
            "commands",
            "tools",
            "models",
            "troubleshooting",
            "getting_started",
            "profiles",
            "sessions",
            "plans",
            "plugins",
            "headless",
            "mcp",
            "backup",
            "keyboard_shortcuts",
            "config",
            "config_reference",
            "safety",
            "kiwix",
        }
        assert set(VALID_TOPICS) == expected

    def test_registered_in_default_registry(self):
        registry = create_default_registry()
        assert "natshell_help" in registry.tool_names

    def test_schema_in_registry(self):
        registry = create_default_registry()
        schemas = registry.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "natshell_help" in names
