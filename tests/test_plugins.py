"""Tests for the plugin discovery and loading system."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from natshell.plugins import ENTRY_POINT_GROUP, _load_file_plugins, load_plugins
from natshell.tools.registry import ToolDefinition, ToolRegistry


def _make_entry_point(name: str, load_return=None, load_side_effect=None):
    """Create a mock entry point."""
    ep = MagicMock()
    ep.name = name
    if load_side_effect is not None:
        ep.load.side_effect = load_side_effect
    else:
        ep.load.return_value = load_return if load_return is not None else MagicMock()
    return ep


# ─── No plugins ──────────────────────────────────────────────────────────


class TestNoPlugins:
    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points", return_value=[])
    def test_returns_zero_when_no_plugins(self, mock_eps):
        registry = ToolRegistry()
        assert load_plugins(registry) == 0

    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points", return_value=[])
    def test_does_not_crash_with_empty_registry(self, mock_eps):
        registry = ToolRegistry()
        load_plugins(registry)
        assert registry.tool_names == []


# ─── Successful plugin loading ───────────────────────────────────────────


class TestSuccessfulPlugins:
    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points")
    def test_calls_plugin_with_registry(self, mock_eps):
        plugin_fn = MagicMock()
        ep = _make_entry_point("my_plugin", load_return=plugin_fn)
        mock_eps.return_value = [ep]

        registry = ToolRegistry()
        load_plugins(registry)

        plugin_fn.assert_called_once_with(registry)

    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points")
    def test_returns_count_of_loaded_plugins(self, mock_eps):
        ep1 = _make_entry_point("plugin_a")
        ep2 = _make_entry_point("plugin_b")
        mock_eps.return_value = [ep1, ep2]

        registry = ToolRegistry()
        assert load_plugins(registry) == 2

    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points")
    def test_plugin_can_register_a_tool(self, mock_eps):
        async def dummy_handler(**kwargs):
            pass  # pragma: no cover

        def plugin_fn(reg: ToolRegistry):
            reg.register(
                ToolDefinition(
                    name="my_custom_tool",
                    description="A plugin-provided tool",
                    parameters={"type": "object", "properties": {}},
                ),
                dummy_handler,
            )

        ep = _make_entry_point("registering_plugin", load_return=plugin_fn)
        mock_eps.return_value = [ep]

        registry = ToolRegistry()
        load_plugins(registry)

        assert "my_custom_tool" in registry.tool_names


# ─── Error handling ──────────────────────────────────────────────────────


class TestPluginErrors:
    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points")
    def test_load_error_does_not_crash(self, mock_eps):
        ep = _make_entry_point(
            "bad_plugin", load_side_effect=ImportError("missing dep")
        )
        mock_eps.return_value = [ep]

        registry = ToolRegistry()
        result = load_plugins(registry)
        assert result == 0

    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points")
    def test_plugin_runtime_error_does_not_crash(self, mock_eps):
        def exploding_plugin(reg):
            raise RuntimeError("plugin broke")

        ep = _make_entry_point("exploding", load_return=exploding_plugin)
        mock_eps.return_value = [ep]

        registry = ToolRegistry()
        result = load_plugins(registry)
        assert result == 0

    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points")
    def test_error_logged_as_warning(self, mock_eps, caplog):
        ep = _make_entry_point(
            "bad_plugin", load_side_effect=ImportError("missing dep")
        )
        mock_eps.return_value = [ep]

        registry = ToolRegistry()
        with caplog.at_level(logging.WARNING, logger="natshell.plugins"):
            load_plugins(registry)

        assert any("bad_plugin" in record.message for record in caplog.records)

    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points")
    def test_good_plugins_still_load_after_bad_one(self, mock_eps):
        bad_ep = _make_entry_point(
            "bad_plugin", load_side_effect=ImportError("missing dep")
        )
        good_fn = MagicMock()
        good_ep = _make_entry_point("good_plugin", load_return=good_fn)
        mock_eps.return_value = [bad_ep, good_ep]

        registry = ToolRegistry()
        result = load_plugins(registry)

        assert result == 1
        good_fn.assert_called_once_with(registry)


# ─── Entry point group ───────────────────────────────────────────────────


class TestEntryPointGroup:
    @patch("natshell.plugins.PLUGIN_DIR", Path("/nonexistent/plugins"))
    @patch("natshell.plugins.entry_points")
    def test_uses_correct_group(self, mock_eps):
        mock_eps.return_value = []

        registry = ToolRegistry()
        load_plugins(registry)

        mock_eps.assert_called_once_with(group=ENTRY_POINT_GROUP)


# ─── File-based plugins ──────────────────────────────────────────────────


def _write_plugin(plugin_dir: Path, name: str, source: str) -> Path:
    """Write a plugin file and return its path."""
    plugin_dir.mkdir(parents=True, exist_ok=True)
    p = plugin_dir / name
    p.write_text(source)
    return p


class TestFilePlugins:
    def test_missing_directory_returns_zero(self, tmp_path):
        nonexistent = tmp_path / "plugins"
        with patch("natshell.plugins.PLUGIN_DIR", nonexistent):
            assert _load_file_plugins(ToolRegistry()) == 0

    def test_empty_directory_returns_zero(self, tmp_path):
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        with patch("natshell.plugins.PLUGIN_DIR", plugin_dir):
            assert _load_file_plugins(ToolRegistry()) == 0

    def test_valid_plugin_loads(self, tmp_path):
        plugin_dir = tmp_path / "plugins"
        _write_plugin(plugin_dir, "hello.py", "def register(registry): registry._test_marker = True\n")

        registry = ToolRegistry()
        with patch("natshell.plugins.PLUGIN_DIR", plugin_dir):
            count = _load_file_plugins(registry)

        assert count == 1
        assert getattr(registry, "_test_marker", False) is True

    def test_plugin_registers_tool(self, tmp_path):
        plugin_dir = tmp_path / "plugins"
        source = """
from natshell.tools.registry import ToolDefinition
import asyncio

async def _handler(**kwargs):
    return "hello"

def register(registry):
    registry.register(
        ToolDefinition(
            name="file_plugin_tool",
            description="From a file plugin",
            parameters={"type": "object", "properties": {}},
        ),
        _handler,
    )
"""
        _write_plugin(plugin_dir, "tool_plugin.py", source)

        registry = ToolRegistry()
        with patch("natshell.plugins.PLUGIN_DIR", plugin_dir):
            _load_file_plugins(registry)

        assert "file_plugin_tool" in registry.tool_names

    def test_no_register_callable_skipped(self, tmp_path, caplog):
        plugin_dir = tmp_path / "plugins"
        _write_plugin(plugin_dir, "no_register.py", "x = 42\n")

        registry = ToolRegistry()
        with patch("natshell.plugins.PLUGIN_DIR", plugin_dir):
            with caplog.at_level(logging.WARNING, logger="natshell.plugins"):
                count = _load_file_plugins(registry)

        assert count == 0
        assert any("no register()" in r.message for r in caplog.records)

    def test_import_error_skipped_others_still_load(self, tmp_path, caplog):
        plugin_dir = tmp_path / "plugins"
        _write_plugin(plugin_dir, "bad.py", "raise ImportError('boom')\n")
        _write_plugin(plugin_dir, "good.py", "def register(registry): registry._good = True\n")

        registry = ToolRegistry()
        with patch("natshell.plugins.PLUGIN_DIR", plugin_dir):
            with caplog.at_level(logging.WARNING, logger="natshell.plugins"):
                count = _load_file_plugins(registry)

        assert count == 1
        assert getattr(registry, "_good", False) is True
        assert any("bad.py" in r.message for r in caplog.records)

    def test_register_raises_skipped(self, tmp_path):
        plugin_dir = tmp_path / "plugins"
        _write_plugin(plugin_dir, "explode.py", "def register(registry): raise RuntimeError('kaboom')\n")

        with patch("natshell.plugins.PLUGIN_DIR", plugin_dir):
            count = _load_file_plugins(ToolRegistry())

        assert count == 0

    def test_underscore_files_skipped(self, tmp_path):
        plugin_dir = tmp_path / "plugins"
        _write_plugin(plugin_dir, "__init__.py", "def register(registry): registry._init = True\n")
        _write_plugin(plugin_dir, "_helpers.py", "def register(registry): registry._helper = True\n")

        registry = ToolRegistry()
        with patch("natshell.plugins.PLUGIN_DIR", plugin_dir):
            count = _load_file_plugins(registry)

        assert count == 0
        assert not hasattr(registry, "_init")
        assert not hasattr(registry, "_helper")

    def test_non_py_files_ignored(self, tmp_path):
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        (plugin_dir / "notes.txt").write_text("not a plugin")
        (plugin_dir / "data.json").write_text("{}")

        with patch("natshell.plugins.PLUGIN_DIR", plugin_dir):
            count = _load_file_plugins(ToolRegistry())

        assert count == 0

    @patch("natshell.plugins.entry_points", return_value=[])
    def test_load_plugins_combines_both_loaders(self, mock_eps, tmp_path):
        """load_plugins() sums entry-point and file-based plugin counts."""
        plugin_dir = tmp_path / "plugins"
        _write_plugin(plugin_dir, "fp.py", "def register(r): r._fp = True\n")

        registry = ToolRegistry()
        with patch("natshell.plugins.PLUGIN_DIR", plugin_dir):
            count = load_plugins(registry)

        assert count == 1
        assert getattr(registry, "_fp", False) is True
