"""Tests for tab completion in the NatShell TUI."""

from __future__ import annotations

from unittest.mock import MagicMock

from natshell.config import NatShellConfig, ProfileConfig
from natshell.ui.widgets import HistoryInput

# ── TestTabCompleteMessage ───────────────────────────────────────────────


class TestTabCompleteMessage:
    def test_message_class_exists(self):
        msg = HistoryInput.TabComplete()
        assert msg.reverse is False

    def test_message_reverse(self):
        msg = HistoryInput.TabComplete(reverse=True)
        assert msg.reverse is True


# ── TestHistoryInputBindings ─────────────────────────────────────────────


class TestHistoryInputBindings:
    def test_tab_binding_exists(self):
        bindings = {b.key for b in HistoryInput.BINDINGS}
        assert "tab" in bindings

    def test_shift_tab_binding_exists(self):
        bindings = {b.key for b in HistoryInput.BINDINGS}
        assert "shift+tab" in bindings


# ── TestBuildCompletions ─────────────────────────────────────────────────


class TestBuildCompletions:
    """Test the _build_completions method on NatShellApp."""

    def _make_app(self, config=None):
        """Create a minimal NatShellApp-like object for testing."""
        from natshell.app import NatShellApp
        from natshell.session import SessionManager

        app = MagicMock(spec=NatShellApp)
        app._config = config or NatShellConfig()
        app._session_mgr = SessionManager()
        app._build_completions = (
            NatShellApp._build_completions.__get__(app)
        )
        return app

    def test_empty_string_no_matches(self):
        app = self._make_app()
        assert app._build_completions("") == []

    def test_non_slash_no_matches(self):
        app = self._make_app()
        assert app._build_completions("hello") == []

    def test_slash_matches_all_base_commands(self):
        app = self._make_app()
        matches = app._build_completions("/")
        assert len(matches) > 0
        assert all(m.startswith("/") for m in matches)

    def test_slash_h_matches_help_and_history(self):
        app = self._make_app()
        matches = app._build_completions("/h")
        assert "/help" in matches
        assert "/history" in matches

    def test_slash_m_matches_model(self):
        app = self._make_app()
        matches = app._build_completions("/m")
        assert "/model" in matches

    def test_no_duplicates(self):
        app = self._make_app()
        matches = app._build_completions("/")
        assert len(matches) == len(set(matches))

    def test_model_subcommands(self):
        app = self._make_app()
        matches = app._build_completions("/model ")
        assert "/model list" in matches
        assert "/model use" in matches
        assert "/model switch" in matches
        assert "/model local" in matches
        assert "/model default" in matches

    def test_model_subcommand_prefix(self):
        app = self._make_app()
        matches = app._build_completions("/model l")
        assert "/model list" in matches
        assert "/model local" in matches
        assert "/model use" not in matches

    def test_profile_names(self):
        config = NatShellConfig()
        config.profiles["fast"] = ProfileConfig(temperature=0.1)
        config.profiles["creative"] = ProfileConfig(temperature=0.9)
        app = self._make_app(config)
        matches = app._build_completions("/profile ")
        assert "/profile creative" in matches
        assert "/profile fast" in matches

    def test_profile_prefix_filter(self):
        config = NatShellConfig()
        config.profiles["fast"] = ProfileConfig(temperature=0.1)
        config.profiles["creative"] = ProfileConfig(temperature=0.9)
        app = self._make_app(config)
        matches = app._build_completions("/profile f")
        assert "/profile fast" in matches
        assert "/profile creative" not in matches


# ── TestCompletionCycling ────────────────────────────────────────────────


class TestCompletionCycling:
    """Test the cycling behavior when Tab is pressed multiple times."""

    def _make_app_with_state(self):
        from natshell.app import NatShellApp

        app = MagicMock(spec=NatShellApp)
        app._config = NatShellConfig()
        app._session_mgr = MagicMock()
        app._completion_matches = []
        app._completion_index = -1
        app._completion_prefix = ""
        app._build_completions = (
            NatShellApp._build_completions.__get__(app)
        )
        app.on_history_input_tab_complete = (
            NatShellApp.on_history_input_tab_complete.__get__(app)
        )
        return app

    def test_forward_cycling(self):
        app = self._make_app_with_state()
        app._completion_matches = ["/help", "/history"]
        app._completion_index = -1

        input_widget = MagicMock()
        app.query_one = MagicMock(return_value=input_widget)
        input_widget.value = "/h"

        msg = HistoryInput.TabComplete(reverse=False)

        app.on_history_input_tab_complete(msg)
        assert app._completion_index == 0

    def test_backward_cycling(self):
        app = self._make_app_with_state()
        app._completion_matches = ["/help", "/history"]
        app._completion_index = -1

        input_widget = MagicMock()
        app.query_one = MagicMock(return_value=input_widget)
        input_widget.value = "/h"

        msg = HistoryInput.TabComplete(reverse=True)

        app.on_history_input_tab_complete(msg)
        # Reverse from -1 wraps to last item
        assert app._completion_index == 1

    def test_wrap_forward(self):
        app = self._make_app_with_state()
        app._completion_matches = ["/help", "/history"]
        app._completion_index = 1  # at last

        input_widget = MagicMock()
        app.query_one = MagicMock(return_value=input_widget)
        input_widget.value = "/history"

        msg = HistoryInput.TabComplete(reverse=False)

        app.on_history_input_tab_complete(msg)
        assert app._completion_index == 0

    def test_single_match(self):
        app = self._make_app_with_state()
        app._completion_matches = ["/sessions"]
        app._completion_index = -1

        input_widget = MagicMock()
        app.query_one = MagicMock(return_value=input_widget)
        input_widget.value = "/se"

        msg = HistoryInput.TabComplete(reverse=False)

        app.on_history_input_tab_complete(msg)
        assert app._completion_index == 0

    def test_no_matches_does_nothing(self):
        app = self._make_app_with_state()
        # Override _build_completions to return empty
        app._build_completions = MagicMock(return_value=[])

        input_widget = MagicMock()
        app.query_one = MagicMock(return_value=input_widget)
        input_widget.value = "hello"

        msg = HistoryInput.TabComplete(reverse=False)

        app.on_history_input_tab_complete(msg)
        assert app._completion_index == -1
