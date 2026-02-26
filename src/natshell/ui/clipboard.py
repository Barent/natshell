"""Clipboard support for NatShell.

Tries real clipboard tools (xclip, xsel, wl-copy) before falling back
to OSC52 terminal escape sequences.  VM consoles typically don't support
OSC52, so having a real tool available is important.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)

_backend: str | None = None  # cached after first detection
_wayland_warned: bool = False  # one-time warning flag


def detect_backend() -> str:
    """Detect the best available clipboard backend.

    On Wayland sessions, prefers wl-copy before falling back to X11 tools.
    On X11 sessions, prefers xclip/xsel before wl-copy.
    Falls back to "osc52" if no tool is found.
    Result is cached for the process lifetime.
    """
    global _backend
    if _backend is not None:
        return _backend

    session_type = os.environ.get("XDG_SESSION_TYPE", "")

    if session_type == "wayland":
        order = ("wl-copy", "xclip", "xsel")
    else:
        order = ("xclip", "xsel", "wl-copy")

    for tool in order:
        if shutil.which(tool):
            _backend = tool
            return _backend

    _backend = "osc52"
    return _backend


def copy(text: str, app=None) -> bool:
    """Copy *text* to the system clipboard.

    Returns True on success, False on failure.  When the backend is
    "osc52", delegates to ``app.copy_to_clipboard()`` (which may silently
    fail on terminals that don't support OSC52).
    """
    global _wayland_warned
    backend = detect_backend()

    # Warn once if using an X11 clipboard tool on a Wayland session
    session_type = os.environ.get("XDG_SESSION_TYPE", "")
    if session_type == "wayland" and backend in ("xclip", "xsel") and not _wayland_warned:
        _wayland_warned = True
        logger.warning(
            "Wayland session detected but using %s (X11). "
            "Clipboard may not work in Wayland-native apps. "
            "Install wl-clipboard for proper Wayland support.",
            backend,
        )

    if backend == "osc52":
        if app is not None:
            try:
                app.copy_to_clipboard(text)
                return True
            except Exception:
                return False
        return False

    cmd = _build_command(backend)
    try:
        proc = subprocess.run(
            cmd,
            input=text,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.returncode == 0
    except Exception:
        return False


def backend_name() -> str:
    """Return a human-readable name for the active clipboard backend."""
    backend = detect_backend()
    if backend == "osc52":
        return "OSC52 (terminal escape — may not work in all terminals)"
    return backend


def _build_command(backend: str) -> list[str]:
    """Build the subprocess command list for the given backend."""
    match backend:
        case "xclip":
            return ["xclip", "-selection", "clipboard"]
        case "xsel":
            return ["xsel", "--clipboard", "--input"]
        case "wl-copy":
            return ["wl-copy"]
        case _:
            raise ValueError(f"Unknown clipboard backend: {backend}")


def _reset() -> None:
    """Reset cached backend (for testing)."""
    global _backend, _wayland_warned
    _backend = None
    _wayland_warned = False
