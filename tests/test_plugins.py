"""Tests for the plugin discovery and loading system."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from natshell.plugins import ENTRY_POINT_GROUP, load_plugins
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
    @patch("natshell.plugins.entry_points", return_value=[])
    def test_returns_zero_when_no_plugins(self, mock_eps):
        registry = ToolRegistry()
        assert load_plugins(registry) == 0

    @patch("natshell.plugins.entry_points", return_value=[])
    def test_does_not_crash_with_empty_registry(self, mock_eps):
        registry = ToolRegistry()
        load_plugins(registry)
        assert registry.tool_names == []


# ─── Successful plugin loading ───────────────────────────────────────────


class TestSuccessfulPlugins:
    @patch("natshell.plugins.entry_points")
    def test_calls_plugin_with_registry(self, mock_eps):
        plugin_fn = MagicMock()
        ep = _make_entry_point("my_plugin", load_return=plugin_fn)
        mock_eps.return_value = [ep]

        registry = ToolRegistry()
        load_plugins(registry)

        plugin_fn.assert_called_once_with(registry)

    @patch("natshell.plugins.entry_points")
    def test_returns_count_of_loaded_plugins(self, mock_eps):
        ep1 = _make_entry_point("plugin_a")
        ep2 = _make_entry_point("plugin_b")
        mock_eps.return_value = [ep1, ep2]

        registry = ToolRegistry()
        assert load_plugins(registry) == 2

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
    @patch("natshell.plugins.entry_points")
    def test_load_error_does_not_crash(self, mock_eps):
        ep = _make_entry_point(
            "bad_plugin", load_side_effect=ImportError("missing dep")
        )
        mock_eps.return_value = [ep]

        registry = ToolRegistry()
        result = load_plugins(registry)
        assert result == 0

    @patch("natshell.plugins.entry_points")
    def test_plugin_runtime_error_does_not_crash(self, mock_eps):
        def exploding_plugin(reg):
            raise RuntimeError("plugin broke")

        ep = _make_entry_point("exploding", load_return=exploding_plugin)
        mock_eps.return_value = [ep]

        registry = ToolRegistry()
        result = load_plugins(registry)
        assert result == 0

    @patch("natshell.plugins.entry_points")
    def test_error_logged_as_warning(self, mock_eps, caplog):
        ep = _make_entry_point(
            "bad_plugin", load_side_effect=ImportError("missing dep")
        )
        mock_eps.return_value = [ep]

        import logging

        registry = ToolRegistry()
        with caplog.at_level(logging.WARNING, logger="natshell.plugins"):
            load_plugins(registry)

        assert any("bad_plugin" in record.message for record in caplog.records)

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
    @patch("natshell.plugins.entry_points")
    def test_uses_correct_group(self, mock_eps):
        mock_eps.return_value = []

        registry = ToolRegistry()
        load_plugins(registry)

        mock_eps.assert_called_once_with(group=ENTRY_POINT_GROUP)
