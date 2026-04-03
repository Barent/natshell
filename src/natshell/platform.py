"""Platform detection for cross-platform support (Linux, macOS, WSL, Windows)."""

from __future__ import annotations

import os
import platform as _platform
import sys
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def current_platform() -> str:
    """Detect the current platform.

    Returns ``"macos"``, ``"windows"``, ``"wsl"``, or ``"linux"``.
    WSL is detected by checking for ``"microsoft"`` in ``/proc/version``.
    Result is cached for the process lifetime.
    """
    if sys.platform == "darwin":
        return "macos"
    if sys.platform == "win32":
        return "windows"
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


def is_windows() -> bool:
    return current_platform() == "windows"


def is_wsl() -> bool:
    return current_platform() == "wsl"


def is_linux() -> bool:
    """True for native Linux *and* WSL."""
    return current_platform() in ("linux", "wsl")


@lru_cache(maxsize=1)
def is_arm64() -> bool:
    """True if the CPU architecture is ARM64 (aarch64 on Linux, ARM64 on Windows)."""
    return _platform.machine().lower() in ("aarch64", "arm64")


# ── Platform-aware directory helpers ────────────────────────────────────────


def data_dir() -> Path:
    """Return the platform-appropriate data directory for NatShell.

    - Windows: ``%LOCALAPPDATA%\\natshell``
    - Unix:    ``~/.local/share/natshell``
    """
    if is_windows():
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "natshell"
    return Path.home() / ".local" / "share" / "natshell"


def config_dir() -> Path:
    """Return the platform-appropriate config directory for NatShell.

    - Windows: ``%APPDATA%\\natshell``
    - Unix:    ``~/.config/natshell``
    """
    if is_windows():
        base = os.environ.get("APPDATA")
        if base:
            return Path(base) / "natshell"
    return Path.home() / ".config" / "natshell"
