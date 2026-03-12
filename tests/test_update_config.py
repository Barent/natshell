"""Tests for update_config tool and config infrastructure."""

from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

from natshell.config import (
    CONFIG_ENUMS,
    VALID_CONFIG_KEYS,
    NatShellConfig,
    save_config_value,
)
from natshell.tools.update_config import (
    _apply_to_live_config,
    _coerce_value,
    set_live_config,
    update_config,
)

# ── TestValidConfigKeys ──────────────────────────────────────────────────


class TestValidConfigKeys:
    """Verify VALID_CONFIG_KEYS covers all config sections."""

    def test_all_sections_present(self):
        expected = {
            "model", "remote", "ollama", "agent", "safety",
            "ui", "backup", "engine", "mcp", "kiwix", "prompt", "memory",
        }
        assert set(VALID_CONFIG_KEYS.keys()) == expected

    def test_model_keys(self):
        keys = VALID_CONFIG_KEYS["model"]
        assert "hf_repo" in keys
        assert "hf_file" in keys
        assert "n_ctx" in keys
        assert "n_gpu_layers" in keys
        assert keys["n_ctx"] == "int"
        assert keys["hf_repo"] == "str"

    def test_agent_keys(self):
        keys = VALID_CONFIG_KEYS["agent"]
        assert "temperature" in keys
        assert "max_steps" in keys
        assert keys["temperature"] == "float"
        assert keys["max_steps"] == "int"

    def test_safety_keys(self):
        keys = VALID_CONFIG_KEYS["safety"]
        assert "mode" in keys
        assert keys["mode"] == "str"

    def test_engine_keys(self):
        keys = VALID_CONFIG_KEYS["engine"]
        assert "preferred" in keys

    def test_backup_keys(self):
        keys = VALID_CONFIG_KEYS["backup"]
        assert "enabled" in keys
        assert keys["enabled"] == "bool"

    def test_remote_keys(self):
        keys = VALID_CONFIG_KEYS["remote"]
        assert "url" in keys
        assert "api_key" in keys

    def test_ollama_keys(self):
        keys = VALID_CONFIG_KEYS["ollama"]
        assert "default_model" in keys

    def test_ui_keys(self):
        keys = VALID_CONFIG_KEYS["ui"]
        assert "theme" in keys

    def test_mcp_keys(self):
        keys = VALID_CONFIG_KEYS["mcp"]
        assert "safety_mode" in keys


# ── TestConfigEnums ──────────────────────────────────────────────────────


class TestConfigEnums:
    def test_safety_mode_enum(self):
        assert "confirm" in CONFIG_ENUMS["safety"]["mode"]
        assert "warn" in CONFIG_ENUMS["safety"]["mode"]
        assert "danger" in CONFIG_ENUMS["safety"]["mode"]

    def test_engine_preferred_enum(self):
        assert "auto" in CONFIG_ENUMS["engine"]["preferred"]
        assert "local" in CONFIG_ENUMS["engine"]["preferred"]
        assert "remote" in CONFIG_ENUMS["engine"]["preferred"]

    def test_mcp_safety_mode_enum(self):
        assert "strict" in CONFIG_ENUMS["mcp"]["safety_mode"]
        assert "permissive" in CONFIG_ENUMS["mcp"]["safety_mode"]


# ── TestSaveConfigValue ──────────────────────────────────────────────────


class TestSaveConfigValue:
    def test_create_new_file(self, tmp_path: Path):
        config_path = tmp_path / ".config" / "natshell" / "config.toml"
        with patch(
            "natshell.config.Path.home", return_value=tmp_path
        ):
            save_config_value("agent", "temperature", 0.7)

        assert config_path.exists()
        content = config_path.read_text()
        assert "[agent]" in content
        assert "temperature = 0.7" in content

    def test_update_existing_key(self, tmp_path: Path):
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.toml"
        config_path.write_text("[agent]\ntemperature = 0.3\n")

        with patch(
            "natshell.config.Path.home", return_value=tmp_path
        ):
            save_config_value("agent", "temperature", 0.7)

        content = config_path.read_text()
        assert "temperature = 0.7" in content
        assert "temperature = 0.3" not in content

    def test_insert_new_key_in_existing_section(self, tmp_path: Path):
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.toml"
        config_path.write_text("[agent]\ntemperature = 0.3\n")

        with patch(
            "natshell.config.Path.home", return_value=tmp_path
        ):
            save_config_value("agent", "max_steps", 20)

        content = config_path.read_text()
        assert "temperature = 0.3" in content
        assert "max_steps = 20" in content

    def test_preserve_other_sections(self, tmp_path: Path):
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.toml"
        config_path.write_text(
            '[model]\nhf_repo = "Qwen/Qwen3-4B-GGUF"\n\n'
            "[agent]\ntemperature = 0.3\n"
        )

        with patch(
            "natshell.config.Path.home", return_value=tmp_path
        ):
            save_config_value("agent", "temperature", 0.7)

        content = config_path.read_text()
        assert 'hf_repo = "Qwen/Qwen3-4B-GGUF"' in content
        assert "temperature = 0.7" in content

    def test_bool_formatting(self, tmp_path: Path):
        with patch(
            "natshell.config.Path.home", return_value=tmp_path
        ):
            save_config_value("backup", "enabled", True)

        config_path = tmp_path / ".config" / "natshell" / "config.toml"
        content = config_path.read_text()
        assert "enabled = true" in content

    def test_string_formatting(self, tmp_path: Path):
        with patch(
            "natshell.config.Path.home", return_value=tmp_path
        ):
            save_config_value("engine", "preferred", "remote")

        config_path = tmp_path / ".config" / "natshell" / "config.toml"
        content = config_path.read_text()
        assert 'preferred = "remote"' in content

    def test_round_trip_valid_toml(self, tmp_path: Path):
        """Written config should be valid TOML."""
        with patch(
            "natshell.config.Path.home", return_value=tmp_path
        ):
            save_config_value("agent", "temperature", 0.7)
            save_config_value("agent", "max_steps", 20)
            save_config_value("engine", "preferred", "local")

        config_path = tmp_path / ".config" / "natshell" / "config.toml"
        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        assert data["agent"]["temperature"] == 0.7
        assert data["agent"]["max_steps"] == 20
        assert data["engine"]["preferred"] == "local"

    def test_add_new_section_to_existing_file(self, tmp_path: Path):
        config_dir = tmp_path / ".config" / "natshell"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.toml"
        config_path.write_text('[model]\nhf_repo = "test"\n')

        with patch(
            "natshell.config.Path.home", return_value=tmp_path
        ):
            save_config_value("engine", "preferred", "local")

        content = config_path.read_text()
        assert "[model]" in content
        assert "[engine]" in content
        assert 'preferred = "local"' in content


# ── TestCoerceValue ──────────────────────────────────────────────────────


class TestCoerceValue:
    def test_int_coercion(self):
        assert _coerce_value("42", "int") == 42
        assert _coerce_value("-1", "int") == -1

    def test_int_error(self):
        with pytest.raises(ValueError, match="integer"):
            _coerce_value("abc", "int")

    def test_float_coercion(self):
        assert _coerce_value("0.7", "float") == 0.7
        assert _coerce_value("1", "float") == 1.0

    def test_float_error(self):
        with pytest.raises(ValueError, match="float"):
            _coerce_value("abc", "float")

    def test_bool_true_variants(self):
        for v in ("true", "True", "1", "yes", "on"):
            assert _coerce_value(v, "bool") is True

    def test_bool_false_variants(self):
        for v in ("false", "False", "0", "no", "off"):
            assert _coerce_value(v, "bool") is False

    def test_bool_error(self):
        with pytest.raises(ValueError, match="boolean"):
            _coerce_value("maybe", "bool")

    def test_str_passthrough(self):
        assert _coerce_value("hello", "str") == "hello"


# ── TestUpdateConfigTool ─────────────────────────────────────────────────


class TestUpdateConfigTool:
    @pytest.fixture(autouse=True)
    def _patch_home(self, tmp_path: Path):
        with patch(
            "natshell.config.Path.home", return_value=tmp_path
        ):
            yield

    @pytest.mark.asyncio
    async def test_valid_update(self):
        result = await update_config("agent", "temperature", "0.7")
        assert result.exit_code == 0
        assert "temperature" in result.output
        assert "0.7" in result.output

    @pytest.mark.asyncio
    async def test_unknown_section(self):
        result = await update_config("nonexistent", "key", "val")
        assert result.exit_code == 1
        assert "Unknown config section" in result.error

    @pytest.mark.asyncio
    async def test_unknown_key(self):
        result = await update_config("agent", "nonexistent", "val")
        assert result.exit_code == 1
        assert "Unknown key" in result.error

    @pytest.mark.asyncio
    async def test_type_mismatch(self):
        result = await update_config("agent", "max_steps", "abc")
        assert result.exit_code == 1
        assert "integer" in result.error

    @pytest.mark.asyncio
    async def test_enum_validation(self):
        result = await update_config("safety", "mode", "invalid")
        assert result.exit_code == 1
        assert "Allowed values" in result.error

    @pytest.mark.asyncio
    async def test_enum_valid(self):
        result = await update_config("safety", "mode", "warn")
        assert result.exit_code == 0

    @pytest.mark.asyncio
    async def test_live_config_update(self):
        config = NatShellConfig()
        set_live_config(config)
        assert config.agent.temperature == 0.3

        await update_config("agent", "temperature", "0.7")
        assert config.agent.temperature == 0.7

        # Clean up
        set_live_config(None)

    @pytest.mark.asyncio
    async def test_bool_update(self):
        result = await update_config("backup", "enabled", "false")
        assert result.exit_code == 0
        assert "False" in result.output


# ── TestApplyToLiveConfig ────────────────────────────────────────────────


class TestApplyToLiveConfig:
    def test_apply_agent_temperature(self):
        config = NatShellConfig()
        _apply_to_live_config(config, "agent", "temperature", 0.7)
        assert config.agent.temperature == 0.7

    def test_apply_engine_preferred(self):
        config = NatShellConfig()
        _apply_to_live_config(config, "engine", "preferred", "remote")
        assert config.engine.preferred == "remote"

    def test_apply_nonexistent_section(self):
        config = NatShellConfig()
        # Should not raise
        _apply_to_live_config(config, "nonexistent", "key", "val")


# ── TestRegistration ─────────────────────────────────────────────────────


class TestRegistration:
    def test_registry_has_update_config(self):
        from natshell.tools.registry import create_default_registry

        registry = create_default_registry()
        assert "update_config" in registry.tool_names

    def test_in_plan_safe_tools(self):
        from natshell.tools.registry import PLAN_SAFE_TOOLS

        assert "update_config" in PLAN_SAFE_TOOLS

    def test_not_in_small_context_tools(self):
        from natshell.tools.registry import SMALL_CONTEXT_TOOLS

        assert "update_config" not in SMALL_CONTEXT_TOOLS

    def test_requires_confirmation(self):
        from natshell.tools.registry import create_default_registry

        registry = create_default_registry()
        defn = registry.get_definition("update_config")
        assert defn is not None
        assert defn.requires_confirmation is True
