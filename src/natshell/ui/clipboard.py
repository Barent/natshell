"""Clipboard support for NatShell.

Tries real clipboard tools (xclip, xsel, wl-copy) before falling back
to OSC52 terminal escape sequences.  VM consoles typically don't support
OSC52, so having a real tool available is important.
"""

from __future__ import annotations

import shutil
import subprocess

_backend: str | None = None  # cached after first detection


def detect_backend() -> str:
    """Detect the best available clipboard backend.

    Checks for xclip, xsel, wl-copy in order.  Falls back to "osc52"
    if none are found.  Result is cached for the process lifetime.
    """
    global _backend
    if _backend is not None:
        return _backend

    for tool in ("xclip", "xsel", "wl-copy"):
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
    backend = detect_backend()

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
    global _backend
    _backend = None
