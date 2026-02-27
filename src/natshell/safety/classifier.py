"""Command risk classification — pattern-based, deterministic, fast."""

from __future__ import annotations

import logging
import re
from enum import Enum

from natshell.config import SafetyConfig

logger = logging.getLogger(__name__)


class Risk(Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    BLOCKED = "blocked"


_CMD_SEPARATORS = re.compile(r'\s*(?:&&|\|\||[;&|])\s*')

# Paths that should require user confirmation before read_file accesses them
_SENSITIVE_PATH_PATTERNS = [
    "/.ssh/", "/id_rsa", "/id_ed25519",
    "/etc/shadow", "/etc/sudoers",
    "/proc/self/environ",
    ".env",
]


class SafetyClassifier:
    """Classify tool calls by risk level using regex patterns."""

    def __init__(self, config: SafetyConfig) -> None:
        self.mode = config.mode
        self._confirm_patterns = [re.compile(p) for p in config.always_confirm]
        self._blocked_patterns = [re.compile(p) for p in config.blocked]

    def classify_command(self, command: str) -> Risk:
        """Classify a shell command string by risk level.

        Checks the full command against blocked patterns first (to catch
        multi-token patterns like fork bombs), then splits on shell operators
        (&&, ||, ;, &, |) and classifies each sub-command independently,
        returning the highest risk found.
        Also flags subshells and backtick expansions as CONFIRM.
        """
        # Check patterns against the full command first
        # (some patterns like fork bombs or pipe-based patterns span operators)
        for pattern in self._blocked_patterns:
            if pattern.search(command):
                logger.warning(f"BLOCKED command: {command}")
                return Risk.BLOCKED
        for pattern in self._confirm_patterns:
            if pattern.search(command):
                return Risk.CONFIRM

        # Flag commands using subshells or backtick expansion
        if re.search(r'`[^`]+`|\$\([^)]+\)', command):
            return Risk.CONFIRM

        sub_commands = _CMD_SEPARATORS.split(command)
        worst_risk = Risk.SAFE
        for sub in sub_commands:
            sub = sub.strip()
            if not sub:
                continue
            risk = self._classify_single(sub)
            if risk == Risk.BLOCKED:
                return Risk.BLOCKED
            if risk == Risk.CONFIRM:
                worst_risk = Risk.CONFIRM
        return worst_risk

    def _classify_single(self, command: str) -> Risk:
        """Classify a single command (no chaining operators)."""
        # Check blocked first
        for pattern in self._blocked_patterns:
            if pattern.search(command):
                logger.warning(f"BLOCKED command: {command}")
                return Risk.BLOCKED

        # Check confirmation-required patterns
        for pattern in self._confirm_patterns:
            if pattern.search(command):
                return Risk.CONFIRM

        # Heuristic: sudo always requires confirmation
        if command.strip().startswith("sudo "):
            return Risk.CONFIRM

        # Heuristic: redirecting to system paths
        if re.search(r">\s*/(?:etc|boot|usr|var/lib)/", command):
            return Risk.CONFIRM

        return Risk.SAFE

    def classify_tool_call(self, tool_name: str, arguments: dict) -> Risk:
        """Classify any tool call by risk level."""
        if tool_name == "execute_shell":
            command = arguments.get("command", "")
            risk = self.classify_command(command)
            # In yolo mode, downgrade CONFIRM to SAFE (but not BLOCKED)
            if self.mode == "yolo" and risk == Risk.CONFIRM:
                return Risk.SAFE
            return risk

        if tool_name == "write_file":
            return Risk.CONFIRM

        if tool_name == "edit_file":
            # Always confirm edits; also check sensitive paths
            path = arguments.get("path", "")
            for pattern in _SENSITIVE_PATH_PATTERNS:
                if pattern in path:
                    return Risk.CONFIRM
            if self.mode == "yolo":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "run_code":
            if self.mode == "yolo":
                return Risk.SAFE
            return Risk.CONFIRM

        if tool_name == "read_file":
            path = arguments.get("path", "")
            for pattern in _SENSITIVE_PATH_PATTERNS:
                if pattern in path:
                    return Risk.CONFIRM
            return Risk.SAFE

        # list_directory, search_files are always safe
        return Risk.SAFE
