"""Tests: left-drag release auto-copies the selection, matching terminal behavior.

Before this change, releasing a left click after highlighting did nothing —
the user had to press Ctrl+C or right-click.  Now a left MouseUp with a
non-empty selection copies it (same handler, same backend, same notification)
exactly like the right-click path that already shipped.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from textual.events import MouseUp

import natshell.ui.clipboard as clipboard
from natshell.app import NatShellApp


def _make_event(button: int) -> MouseUp:
    """A MouseUp event — the handler only reads .button, so a minimal one suffices."""
    return MouseUp(
        widget=None,
        x=1.0,
        y=1.0,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        ctrl=False,
        meta=False,
    )


class _FakeScreen:
    """Stand-in for `app.screen` exposing only get_selected_text()."""

    def __init__(self, selected: str | None) -> None:
        self._selected = selected

    def get_selected_text(self) -> str | None:
        return self._selected


class _App:
    """Just enough of a NatShellApp instance to drive on_mouse_up."""

    def __init__(self, selected: str | None) -> None:
        self._notify_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._notify_result_calls: list[tuple[str | None, None]] = []
        self.screen = _FakeScreen(selected)

    def notify(self, *args: Any, **kwargs: Any) -> None:
        self._notify_calls.append((args, kwargs))

    def _notify_copy_result(self, what: str = "Copied") -> None:
        """Record the call — production behavior is already covered by test_copy_notify.py."""
        self._notify_result_calls.append((what, None))
        self.notify(f"Copied (via {what})", timeout=2)


@pytest.mark.parametrize(
    "button, selected, copy_landed, copy_calls, notify_calls",
    [
        (1, "hello world", True, 1, 1),
        (1, "hello world", False, 1, 1),
        (1, None, True, 0, 0),
        (2, "x", True, 0, 0),
        (3, "x", True, 1, 1),
    ],
)
def test_release_copies_when_selected(button, selected, copy_landed, copy_calls, notify_calls):
    """A left-drag release or right-click copies the selection; other buttons don't."""
    app = _App(selected)
    with patch.object(clipboard, "copy", return_value=copy_landed) as fake_copy:
        NatShellApp.on_mouse_up(app, _make_event(button))
    assert fake_copy.call_count == copy_calls
    if copy_calls:
        assert fake_copy.call_args.args[0] == selected
    assert len(app._notify_calls) == notify_calls


def test_left_release_routes_notify_for_both_outcomes():
    """A left-drag release routes success/error through _notify_copy_result.

    Production behavior of _notify_copy_result itself is covered by
    test_copy_notify.py; here we only assert the release path reaches it.
    """
    # Success: one notify call for the backend.
    ok = _App("abc def")
    with patch.object(clipboard, "copy", return_value=True):
        NatShellApp.on_mouse_up(ok, _make_event(1))
    assert len(ok._notify_result_calls) == 1
    assert len(ok._notify_calls) == 1

    # Failure: one attempted copy (which failed), then an error notification
    # path via _notify_copy_result (already covered by test_copy_notify.py).
    fail = _App("abc def")
    with patch.object(clipboard, "copy", return_value=False) as fake_copy:
        NatShellApp.on_mouse_up(fail, _make_event(1))
    assert fake_copy.call_count == 1
    assert len(fail._notify_result_calls) == 1
