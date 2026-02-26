"""Platform detection for cross-platform support (Linux, macOS, WSL)."""

from __future__ import annotations

import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def current_platform() -> str:
    """Detect the current platform.

    Returns ``"macos"``, ``"wsl"``, or ``"linux"``.
    WSL is detected by checking for ``"microsoft"`` in ``/proc/version``.
    Result is cached for the process lifetime.
    """
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "linux":
        try:
            with open("/proc/version", "r") as f:
                if "microsoft" in f.read().lower():
                    return "wsl"
        except OSError:
            pass
    return "linux"


def is_macos() -> bool:
    return current_platform() == "macos"


def is_wsl() -> bool:
    return current_platform() == "wsl"


def is_linux() -> bool:
    """True for native Linux *and* WSL."""
    return current_platform() in ("linux", "wsl")
