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


class SafetyClassifier:
    """Classify tool calls by risk level using regex patterns."""

    def __init__(self, config: SafetyConfig) -> None:
        self.mode = config.mode
        self._confirm_patterns = [re.compile(p) for p in config.always_confirm]
        self._blocked_patterns = [re.compile(p) for p in config.blocked]

    def classify_command(self, command: str) -> Risk:
        """Classify a shell command string by risk level."""
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

        # read_file, list_directory, search_files are always safe
        return Risk.SAFE
