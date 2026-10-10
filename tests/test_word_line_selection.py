"""Tests for double-click (word) and triple-click (line) selection+copy.

Covers:
  - The pure helpers `_expand_to_word` and `_expand_to_line` (deterministic,
    no TUI needed).
  - The app-level `on_mouse_up` integration: a real ``MouseUp`` event fired at
    a widget position expands to the word / whole line and copies it, while a
    single click and the right-click path are unchanged.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("natshell")

from textual.events import MouseUp  # noqa: E402

import natshell.ui.clipboard as clipboard  # noqa: E402
from natshell.app import NatShellApp, _expand_to_line, _expand_to_word  # noqa: E402
from natshell.ui.widgets import UserMessage  # noqa: E402

# ---- Pure helpers ----------------------------------------------------------

@pytest.mark.parametrize(
    "text, row, col, expected",
    [
        # Plain words.
        ("hello world foo", 0, 1, (0, 0, 5, 0)),   # 'hello'
        ("hello world foo", 0, 7, (6, 0, 11, 0)),  # 'world'
        ("hello world foo", 0, 13, (12, 0, 15, 0)),  # 'foo'
        # Multiline: pick a word on the second line.
        ("first line\nsecond line here", 1, 10, (7, 1, 11, 1)),  # 'line'
        # Hyphenated token is one "word" (no whitespace inside).
        ("foo-bar baz", 0, 5, (0, 0, 7, 0)),
        # Numeric.
        ("answer is 42 now", 0, 14, (13, 0, 16, 0)),  # 'now'
        # Space between two words snaps to the next word.
        ("answer is 42 now", 0, 12, (13, 0, 16, 0)),
    ],
)
def test_expand_to_word(text, row, col, expected):
    assert _expand_to_word(text, row, col) == expected


def test_expand_to_word_returns_none_out_of_range():
    assert _expand_to_word("hello", 5, 0) is None
    assert _expand_to_word("", 0, 0) is None


@pytest.mark.parametrize(
    "text, row, expected",
    [
        ("hello world", 0, (0, 0, 11, 0)),
        ("one\ntwo\nthree", 2, (0, 2, 5, 2)),
    ],
)
def test_expand_to_line(text, row, expected):
    assert _expand_to_line(text, row) == expected


def test_expand_to_line_out_of_range():
    assert _expand_to_line("one\ntwo", 3) is None


# ---- on_mouse_up integration ---------------------------------------------

def _mouse_up(button: int, x: float, y: float) -> MouseUp:
    """Build a MouseUp at screen (x, y)."""
    return MouseUp(
        widget=None,
        x=x,
        y=y,
        delta_x=0,
        delta_y=0,
        button=button,
        shift=False,
        ctrl=False,
        meta=False,
        screen_x=x,
        screen_y=y,
    )


async def _run(app: Any, body) -> None:
    """Mount a user message, lay out, then run *body* inside the app."""
    async with app.run_test() as pilot:
        conv = pilot.app.query_one("#conversation")
        for c in list(conv.children):
            c.remove()
        conv.mount(UserMessage("the quick brown fox jumps over"))
        await pilot.pause()
        await body(pilot)


async def test_double_click_copies_word() -> None:
    async def body(pilot) -> None:
        app = pilot.app
        static = pilot.app.query_one(UserMessage).query_one(".msg-text")
        rendered = str(static._render())
        # The visual line for UserMessage("the quick brown fox jumps over")
        # is "You: the quick brown fox jumps over".  'brown' is one of the
        # middle words.
        idx = rendered.find("brown")
        row = rendered.count("\n", 0, idx)
        col = idx - (rendered.rfind("\n", 0, idx) + 1)
        tx, ty = float(static.region.x + col), float(static.region.y + row)

        spy: list = []
        app.notify = lambda *a, **k: spy.append(a)  # type: ignore[assignment]

        with (
            patch.object(clipboard, "copy", return_value=True) as fake,
            patch.object(clipboard, "last_backend", return_value="xclip"),
        ):
            app.on_mouse_up(_mouse_up(1, tx, ty))
            app.on_mouse_up(_mouse_up(1, tx, ty))

        assert fake.call_count >= 1
        last = fake.call_args_list[-1]
        assert last.args[0] == "brown", last.args
        # Success notification (routed through _notify_copy_result).
        assert any("Copied" in a[0] for a in spy), spy

    await _run(NatShellApp(agent=None), body)


async def test_triple_click_copies_whole_line() -> None:
    async def body(pilot) -> None:
        app = pilot.app
        static = pilot.app.query_one(UserMessage).query_one(".msg-text")
        rendered = str(static._render())
        # The visible line (what the user sees) — this is what triple-click
        # should copy.
        expected_line = rendered.split("\n")[0]
        tx, ty = float(static.region.x + 5), float(static.region.y)

        with patch.object(clipboard, "copy", return_value=True) as fake:
            for _ in range(3):
                app.on_mouse_up(_mouse_up(1, tx, ty))

        # Click sequence: #1 (single → no copy), #2 (double → word "the"),
        # #3 (triple → whole visible line).  The last copy is the line.
        fake.assert_called()
        last = fake.call_args_list[-1]
        assert last.args[0] == expected_line, (last.args[0], expected_line)

    await _run(NatShellApp(agent=None), body)


async def test_single_click_does_not_copy() -> None:
    async def body(pilot) -> None:
        app = pilot.app
        static = pilot.app.query_one(UserMessage).query_one(".msg-text")
        tx, ty = float(static.region.x + 5), float(static.region.y)
        with patch.object(clipboard, "copy", return_value=True) as fake:
            app.on_mouse_up(_mouse_up(1, tx, ty))
        fake.assert_not_called()

    await _run(NatShellApp(agent=None), body)


def test_on_mouse_up_ignores_middle_button() -> None:
    app = NatShellApp(agent=None)
    with patch.object(clipboard, "copy", return_value=True) as fake:
        app.on_mouse_up(_mouse_up(2, 5.0, 5.0))
    fake.assert_not_called()
