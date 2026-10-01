"""Tests for issue #50: copy-result notifications report the *actual* backend.

`action_copy_chat` / `action_copy_selection` / right-click copy now route their
success notifications through `_notify_copy_result`, which reports the backend
that landed the copy.  A verified real-tool copy and an OSC52 escape (which many
terminals silently truncate on large payloads) must produce *different*, honest
messages so the user can tell a working copy from a silently-truncated one.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import natshell.ui.clipboard as clipboard
from natshell.app import NatShellApp


class _NotifySpy:
    """Just enough of the app to drive _notify_copy_result."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def notify(self, *args, **kwargs) -> None:
        # Textual notify accepts (message, **opts) or (message, timeout=, severity=).
        message = args[0] if args else kwargs.get("title") or kwargs.get("message")
        self.calls.append({"message": message, "severity": kwargs.get("severity")})


def _run(seeded_backend: str | None, landed: bool) -> list[dict]:
    """Drive _notify_copy_result as if copy() just reported backend X."""
    self = _NotifySpy()
    with patch.object(clipboard, "last_backend", return_value=seeded_backend):
        if landed:
            NatShellApp._notify_copy_result(self, "Copied")
        else:
            NatShellApp._notify_copy_result(self)
    return self.calls


@pytest.mark.parametrize(
    "landed_backend, landed, needle, severity",
    [
        ("xclip", True, "xclip", None),
        ("wl-copy", True, "wl-copy", None),
        ("pbcopy", True, "pbcopy", None),
        ("osc52", True, "OSC52", None),
        (None, False, "no working", "error"),
    ],
)
def test_notification_matches_backend(landed_backend, landed, needle, severity):
    calls = _run(landed_backend, landed)
    assert len(calls) == 1
    call = calls[0]
    assert needle in call["message"], f"expected {needle!r} in {call['message']!r}"
    assert call["severity"] == severity


def test_osc52_warns_about_truncation():
    """The OSC52 success path must proactively warn about truncation (issue #50)."""
    calls = _run("osc52", landed=True)
    assert "truncat" in calls[0]["message"].lower()


def test_real_tool_success_is_brief_and_positive():
    calls = _run("xclip", landed=True)
    msg = calls[0]["message"]
    assert "Copied via xclip" == msg
    # A verified tool copy must not carry the OSC52 truncation warning.
    assert "truncat" not in msg.lower()


def test_failure_reports_error_severity():
    calls = _run(None, landed=False)
    assert calls[0]["severity"] == "error"
