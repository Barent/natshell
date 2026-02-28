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
        "topic", ["overview", "commands", "tools", "models", "troubleshooting"]
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

    async def test_tools_lists_all_tools(self):
        result = await natshell_help("tools")
        assert "execute_shell" in result.output
        assert "read_file" in result.output
        assert "natshell_help" in result.output

    async def test_models_mentions_qwen(self):
        result = await natshell_help("models")
        assert "Qwen" in result.output

    async def test_troubleshooting_has_gpu_hint(self):
        result = await natshell_help("troubleshooting")
        assert "GPU" in result.output or "gpu" in result.output


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
            "config",
            "config_reference",
            "safety",
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
