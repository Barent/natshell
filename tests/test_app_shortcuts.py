"""Tests for issue #48: copy-last, edit-last, regenerate, and stop-when-busy.

All four are pure additions to `NatShellApp` (new bindings + new action
methods); no existing binding or action is changed.  Tests invoke the actions
unbound with a stub `self`, using *real* message widgets as the conversation
children so the production ``isinstance`` matching in ``_last_message_of`` is
exercised end-to-end without booting the Textual app.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from natshell.app import NatShellApp, _find_last
from natshell.ui import clipboard
from natshell.ui.widgets import AssistantMessage, UserMessage


def _app_stub() -> MagicMock:
    """Stub app: every method/attr we touch is a fresh mock.

    We call the actions unbound (``NatShellApp.<method>(app)``) so `app` plays
    the role of ``self``.
    """
    return MagicMock()


def _user(text: str = "do something cool") -> UserMessage:
    return UserMessage(text)


def _asst(text: str = "hello-world") -> AssistantMessage:
    return AssistantMessage(text)


def _conv(*messages) -> MagicMock:
    conv = MagicMock()
    conv.children = list(messages)
    return conv


def test_last_message_of_returns_most_recent_instance():
    conv = _conv(_user(), _asst(), _asst(), _user("later prompt"))
    assert _find_last(conv, AssistantMessage) is conv.children[2]
    assert _find_last(conv, UserMessage) is conv.children[3]


def test_last_message_of_none_when_absent():
    conv = _conv(_user())
    assert _find_last(conv, AssistantMessage) is None


def test_copy_last_response_uses_last_assistant_message():
    app = _app_stub()
    conv = _conv(_user(), _asst("first"), _asst("fresh answer"))
    app.query_one = lambda *a, **k: conv
    with patch.object(clipboard, "copy", return_value=True) as fake_copy:
        NatShellApp.action_copy_last_response(app)
    fake_copy.assert_called_once_with("fresh answer", app)
    app.notify.assert_called_once()
    assert "Copied last response" in app.notify.call_args.args[0]


def test_copy_last_response_none_notifies():
    app = _app_stub()
    conv = _conv(_user())
    app.query_one = lambda *a, **k: conv
    with patch.object(clipboard, "copy") as fake_copy:
        NatShellApp.action_copy_last_response(app)
    fake_copy.assert_not_called()
    app.notify.assert_called_once()
    assert "No previous response" in app.notify.call_args.args[0]


def test_edit_last_prompt_populates_input():
    app = _app_stub()
    conv = _conv(_asst(), _user("tweak me please"))
    input_stub = MagicMock()
    app.query_one.side_effect = {
        "#conversation": conv,
        "#user-input": input_stub,
    }.get
    NatShellApp.action_edit_last_prompt(app)
    assert input_stub.value == "tweak me please"
    input_stub.cursor_position == len(input_stub.value)
    input_stub.focus.assert_called_once()
    input_stub.clear_paste.assert_called()


def test_edit_last_prompt_none_notifies():
    app = _app_stub()
    conv = _conv(_asst())
    app.query_one.side_effect = {
        "#conversation": conv,
        "#user-input": MagicMock(),
    }.get
    NatShellApp.action_edit_last_prompt(app)
    app.notify.assert_called_once()
    assert "No previous prompt" in app.notify.call_args.args[0]


def test_regenerate_re_runs_last_user_message():
    app = _app_stub()
    conv = _conv(_asst("a"), _user("restart that thing"), _asst("b"))
    input_stub = MagicMock()
    app.query_one.side_effect = {
        "#conversation": conv,
        "#user-input": input_stub,
    }.get
    app._busy = False
    app.run_agent = MagicMock()
    NatShellApp.action_regenerate_last(app)
    app.run_agent.assert_called_once_with("restart that thing")
    conv.mount.assert_called()
    conv.scroll_end.assert_called()
    input_stub.add_to_history.assert_called_once_with("restart that thing")


def test_regenerate_none_notifies():
    app = _app_stub()
    conv = _conv(_asst())
    app.query_one = lambda *a, **k: conv
    app.run_agent = MagicMock()
    NatShellApp.action_regenerate_last(app)
    app.run_agent.assert_not_called()
    app.notify.assert_called_once()
    assert "No previous prompt" in app.notify.call_args.args[0]


def test_regenerate_blocked_when_busy():
    app = _app_stub()
    conv = _conv(_user("x"))
    app.query_one = lambda *a, **k: conv
    app._busy = True
    app.run_agent = MagicMock()
    NatShellApp.action_regenerate_last(app)
    app.run_agent.assert_not_called()
    app.notify.assert_called_once()
    assert "stop" in app.notify.call_args.args[0].lower()


def test_stop_when_busy_cancels_workers():
    app = _app_stub()
    app._busy = True
    app.workers = MagicMock()
    NatShellApp.action_stop(app)
    app.workers.cancel_all.assert_called_once()
    assert app._busy is False
    app.notify.assert_called_once()
    assert "Stopped" in app.notify.call_args.args[0]


def test_stop_when_idle_is_a_noop():
    app = _app_stub()
    app._busy = False
    app.workers = MagicMock()
    NatShellApp.action_stop(app)
    app.workers.cancel_all.assert_not_called()
    app.notify.assert_not_called()  # Esc left for dialogs when idle


def test_bindings_registered_for_issue_48_keys():
    keys = {b.key for b in NatShellApp.BINDINGS}
    actions = {b.action for b in NatShellApp.BINDINGS}
    assert "esc" in keys
    assert "ctrl+y" in keys
    assert "ctrl+r" in keys
    assert "stop" in actions
    assert "copy_last_response" in actions
    assert "regenerate_last" in actions
    assert "edit_last_prompt" in actions
