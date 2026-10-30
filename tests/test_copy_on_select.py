"""Tests for copy-on-select: a mouse selection is copied when it is released.

Unlike the earlier attempts (PRs #55/#57), these drive real mouse gestures
through Textual's event pipeline with Pilot instead of calling the handler
directly, so Screen's selection handling, Click-chain synthesis and event
bubbling all run as they do in the real app.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import patch

import natshell.ui.clipboard as clipboard
from natshell.app import NatShellApp
from natshell.ui.widgets import AssistantMessage, SystemMessage, UserMessage

_SIZE = (100, 30)


class _FakeCopy:
    """Stands in for clipboard.copy and records which thread ran it."""

    def __init__(self, ok: bool = True) -> None:
        self.ok = ok
        self.texts: list[str] = []
        self.threads: list[threading.Thread] = []

    def __call__(self, text: str, app=None) -> bool:
        self.texts.append(text)
        self.threads.append(threading.current_thread())
        return self.ok


async def _settle(app: NatShellApp, pilot) -> None:
    await pilot.pause()
    await app.workers.wait_for_complete()
    await pilot.pause()


async def _mount(app: NatShellApp, pilot, *messages) -> None:
    conv = app.query_one("#conversation")
    for child in list(conv.children):
        child.remove()
    for message in messages:
        conv.mount(message)
    await pilot.pause()


async def _drag(pilot, widget, start: int, end: int) -> None:
    await pilot.mouse_down(widget, offset=(start, 0))
    await pilot.hover(widget, offset=(end, 0))
    await pilot.mouse_up(widget, offset=(end, 0))


def _patches(fake: _FakeCopy, backend: str = "xclip"):
    return (
        patch.object(clipboard, "copy", fake),
        patch.object(clipboard, "detect_backend", return_value=backend),
        patch.object(clipboard, "last_backend", return_value=backend if fake.ok else None),
    )


async def test_drag_release_copies_selection_off_the_event_loop() -> None:
    app = NatShellApp(agent=None)
    fake = _FakeCopy()
    p1, p2, p3 = _patches(fake)
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, UserMessage("the quick brown fox"))
        text = app.query_one(UserMessage).query_one(".msg-text")
        notes: list[str] = []
        app.notify = lambda message, **kw: notes.append(message)  # type: ignore[method-assign]
        with p1, p2, p3:
            await _drag(pilot, text, 0, 9)
            await _settle(app, pilot)
            selected = app.screen.get_selected_text()
    assert fake.texts == [selected]
    assert selected and selected.startswith("You:")
    # Real clipboard tools are subprocesses: they must not run on the UI thread.
    assert fake.threads[0] is not threading.main_thread()
    assert notes == ["Selection copied via xclip"]


async def test_plain_click_does_not_copy() -> None:
    app = NatShellApp(agent=None)
    fake = _FakeCopy()
    p1, p2, p3 = _patches(fake)
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, UserMessage("the quick brown fox"))
        text = app.query_one(UserMessage).query_one(".msg-text")
        with p1, p2, p3:
            await pilot.click(text, offset=(5, 0))
            await _settle(app, pilot)
    assert fake.texts == []


async def test_double_click_copies_what_textual_highlights() -> None:
    """Textual's own double-click selection is what lands on the clipboard."""
    app = NatShellApp(agent=None)
    fake = _FakeCopy()
    p1, p2, p3 = _patches(fake)
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, UserMessage("the quick brown fox"), SystemMessage("system note"))
        highlighted = []
        for cls in (UserMessage, SystemMessage):
            text = app.query_one(cls).query_one(".msg-text")
            with p1, p2, p3:
                await pilot.click(text, offset=(3, 0), times=2)
                await _settle(app, pilot)
            highlighted.append(app.screen.get_selected_text())
    assert all(highlighted)
    # Clipboard holds exactly what the user sees highlighted, nothing else.
    assert fake.texts == highlighted
    assert "brown" in highlighted[0]
    assert "system note" in highlighted[1]


async def test_empty_selection_is_not_copied() -> None:
    """Rich Markdown replies are not text-selectable in Textual; never copy ''."""
    app = NatShellApp(agent=None)
    fake = _FakeCopy()
    p1, p2, p3 = _patches(fake)
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, AssistantMessage("**bold** reply"))
        text = app.query_one(AssistantMessage).query_one(".msg-text")
        with p1, p2, p3:
            await pilot.click(text, offset=(3, 0), times=2)
            await _drag(pilot, text, 0, 9)
            await _settle(app, pilot)
    assert fake.texts == []


async def test_same_selection_is_not_recopied_but_a_new_one_is() -> None:
    app = NatShellApp(agent=None)
    fake = _FakeCopy()
    p1, p2, p3 = _patches(fake)
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, UserMessage("the quick brown fox"))
        text = app.query_one(UserMessage).query_one(".msg-text")
        with p1, p2, p3:
            await _drag(pilot, text, 0, 9)
            await _settle(app, pilot)
            # A second TextSelected while the selection is unchanged (what a
            # scrollbar click produces) must not copy again.
            app._copy_on_select()
            await _settle(app, pilot)
            assert len(fake.texts) == 1
            # A plain click clears the selection; the next drag copies again.
            await pilot.click(text, offset=(15, 0))
            await _settle(app, pilot)
            await _drag(pilot, text, 0, 9)
            await _settle(app, pilot)
    assert len(fake.texts) == 2


async def test_disabled_by_config() -> None:
    app = NatShellApp(agent=None)
    app._config.ui.copy_on_select = False
    fake = _FakeCopy()
    p1, p2, p3 = _patches(fake)
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, UserMessage("the quick brown fox"))
        text = app.query_one(UserMessage).query_one(".msg-text")
        with p1, p2, p3:
            await _drag(pilot, text, 0, 9)
            await _settle(app, pilot)
            assert app.screen.get_selected_text()  # selection itself still works
    assert fake.texts == []


async def test_osc52_runs_on_event_loop_and_warns_once() -> None:
    app = NatShellApp(agent=None)
    fake = _FakeCopy()
    p1, p2, p3 = _patches(fake, backend="osc52")
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, UserMessage("the quick brown fox jumps"))
        text = app.query_one(UserMessage).query_one(".msg-text")
        notes: list[str] = []
        app.notify = lambda message, **kw: notes.append(message)  # type: ignore[method-assign]
        with p1, p2, p3:
            await _drag(pilot, text, 0, 9)
            await _settle(app, pilot)
            await _drag(pilot, text, 0, 14)
            await _settle(app, pilot)
    assert len(fake.texts) == 2
    # OSC52 writes to the terminal driver, so it stays on the main thread.
    assert all(t is threading.main_thread() for t in fake.threads)
    assert "truncate" in notes[0]
    assert notes[1] == "Selection copied via osc52"


async def test_failed_copy_reports_error_and_allows_retry() -> None:
    app = NatShellApp(agent=None)
    fake = _FakeCopy(ok=False)
    p1, p2, p3 = _patches(fake)
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, UserMessage("the quick brown fox"))
        text = app.query_one(UserMessage).query_one(".msg-text")
        severities: list = []
        app.notify = lambda message, **kw: severities.append(kw.get("severity"))  # type: ignore[method-assign]
        with p1, p2, p3:
            await _drag(pilot, text, 0, 9)
            await _settle(app, pilot)
            app._copy_on_select()
            await _settle(app, pilot)
    assert severities == ["error", "error"]
    assert len(fake.texts) == 2


async def test_triple_click_copies_what_textual_highlights() -> None:
    app = NatShellApp(agent=None)
    fake = _FakeCopy()
    p1, p2, p3 = _patches(fake)
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, UserMessage("the quick brown fox"))
        text = app.query_one(UserMessage).query_one(".msg-text")
        with p1, p2, p3:
            await pilot.click(text, offset=(3, 0), times=3)
            await _settle(app, pilot)
            highlighted = app.screen.get_selected_text()
    assert highlighted and "brown" in highlighted
    assert fake.texts[-1] == highlighted


async def test_copy_does_not_cancel_the_agent_worker() -> None:
    """The agent runs as an exclusive worker in the default group; copying must not cancel it."""
    app = NatShellApp(agent=None)
    fake = _FakeCopy()
    p1, p2, p3 = _patches(fake)
    async with app.run_test(size=_SIZE) as pilot:
        await _mount(app, pilot, UserMessage("the quick brown fox"))
        text = app.query_one(UserMessage).query_one(".msg-text")
        release = asyncio.Event()

        async def fake_agent() -> None:
            await release.wait()

        agent_worker = app.run_worker(fake_agent(), exclusive=True)
        with p1, p2, p3:
            await _drag(pilot, text, 0, 9)
            for _ in range(100):
                if fake.texts:
                    break
                await pilot.pause(0.01)
            await pilot.pause()
            assert agent_worker.is_running
        release.set()
        await app.workers.wait_for_complete()
    assert len(fake.texts) == 1
