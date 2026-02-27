"""Tests for HistoryInput widget history logic."""

from natshell.ui.widgets import HistoryInput


class _TestInput(HistoryInput):
    """Test double that bypasses Textual's reactive value/cursor system."""

    def __init__(self):
        # Only init history state, skip Textual Input.__init__
        self._history: list[str] = []
        self._history_index: int = -1
        self._draft: str = ""
        self._val: str = ""
        self._cur: int = 0

    @property
    def value(self) -> str:
        return self._val

    @value.setter
    def value(self, v: str) -> None:
        self._val = v

    @property
    def cursor_position(self) -> int:
        return self._cur

    @cursor_position.setter
    def cursor_position(self, v: int) -> None:
        self._cur = v


def _make() -> _TestInput:
    return _TestInput()


class TestAddToHistory:
    """Tests for add_to_history()."""

    def test_basic_add(self):
        w = _make()
        w.add_to_history("hello")
        assert w._history == ["hello"]

    def test_whitespace_stripping(self):
        w = _make()
        w.add_to_history("  hello  ")
        assert w._history == ["hello"]

    def test_empty_rejected(self):
        w = _make()
        w.add_to_history("")
        w.add_to_history("   ")
        assert w._history == []

    def test_consecutive_dedup(self):
        w = _make()
        w.add_to_history("hello")
        w.add_to_history("hello")
        assert w._history == ["hello"]

    def test_non_consecutive_duplicates_allowed(self):
        w = _make()
        w.add_to_history("hello")
        w.add_to_history("world")
        w.add_to_history("hello")
        assert w._history == ["hello", "world", "hello"]

    def test_resets_nav_state(self):
        w = _make()
        w.add_to_history("first")
        w._history_index = 0  # Simulate navigating
        w.add_to_history("second")
        assert w._history_index == -1


class TestNavigation:
    """Tests for action_history_back/forward."""

    def test_back_empty_history(self):
        w = _make()
        w.action_history_back()
        assert w.value == ""
        assert w._history_index == -1

    def test_back_single_entry(self):
        w = _make()
        w.add_to_history("hello")
        w.action_history_back()
        assert w.value == "hello"
        assert w._history_index == 0

    def test_back_saves_draft(self):
        w = _make()
        w.add_to_history("hello")
        w.value = "current typing"
        w.action_history_back()
        assert w._draft == "current typing"
        assert w.value == "hello"

    def test_back_multiple(self):
        w = _make()
        w.add_to_history("first")
        w.add_to_history("second")
        w.add_to_history("third")
        w.action_history_back()
        assert w.value == "third"
        w.action_history_back()
        assert w.value == "second"
        w.action_history_back()
        assert w.value == "first"

    def test_back_stops_at_oldest(self):
        w = _make()
        w.add_to_history("only")
        w.action_history_back()
        w.action_history_back()
        w.action_history_back()
        assert w.value == "only"
        assert w._history_index == 0

    def test_forward_without_navigating(self):
        w = _make()
        w.add_to_history("hello")
        w.action_history_forward()
        assert w.value == ""  # No change
        assert w._history_index == -1

    def test_forward_restores_draft(self):
        w = _make()
        w.add_to_history("first")
        w.add_to_history("second")
        w.value = "my draft"
        w.action_history_back()  # saves draft, shows "second"
        w.action_history_back()  # shows "first"
        w.action_history_forward()  # shows "second"
        assert w.value == "second"
        w.action_history_forward()  # restores draft
        assert w.value == "my draft"
        assert w._history_index == -1

    def test_forward_past_newest(self):
        w = _make()
        w.add_to_history("hello")
        w.value = "draft"
        w.action_history_back()
        w.action_history_forward()
        # Now past newest, further forward is no-op
        w.action_history_forward()
        assert w.value == "draft"
        assert w._history_index == -1

    def test_cursor_at_end(self):
        w = _make()
        w.add_to_history("hello world")
        w.action_history_back()
        assert w.cursor_position == len("hello world")


class TestClearHistory:
    """Tests for clear_history()."""

    def test_clears_everything(self):
        w = _make()
        w.add_to_history("a")
        w.add_to_history("b")
        w._history_index = 1
        w._draft = "draft"
        w.clear_history()
        assert w._history == []
        assert w._history_index == -1
        assert w._draft == ""


class TestSlashCommands:
    """Slash commands should be stored in history."""

    def test_slash_commands_in_history(self):
        w = _make()
        w.add_to_history("/help")
        w.add_to_history("/clear")
        w.add_to_history("/model list")
        assert w._history == ["/help", "/clear", "/model list"]
