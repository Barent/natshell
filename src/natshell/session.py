"""Session persistence — save and restore conversation history."""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SESSION_ID_RE = re.compile(r"^[a-f0-9]{32}$")

# Default max serialized session size: 10 MB
_DEFAULT_MAX_SIZE = 10 * 1024 * 1024

SESSION_DIR = Path.home() / ".local" / "share" / "natshell" / "sessions"


class SessionManager:
    """Manage saved conversation sessions as JSON files.

    Sessions are stored in ``~/.local/share/natshell/sessions/{uuid}.json``
    with the schema::

        {
            "id": "<uuid>",
            "name": "<user-supplied or auto-generated>",
            "created": "<ISO-8601>",
            "updated": "<ISO-8601>",
            "engine_info": {engine_type, model_name, base_url, n_ctx, ...},
            "messages": [...]
        }
    """

    def __init__(
        self, session_dir: Path | None = None, max_size: int = _DEFAULT_MAX_SIZE
    ) -> None:
        self._dir = session_dir or SESSION_DIR
        self._max_size = max_size

    # ── validation ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        """Reject session IDs that aren't 32-char lowercase hex (UUID hex)."""
        if not _SESSION_ID_RE.match(session_id):
            raise ValueError(f"Invalid session ID: {session_id!r}")

    # ── public API ────────────────────────────────────────────────────

    def save(
        self,
        messages: list[dict[str, Any]],
        engine_info: dict[str, Any] | None = None,
        name: str = "",
        session_id: str | None = None,
    ) -> str:
        """Save a conversation to disk.

        Parameters
        ----------
        messages:
            The full message list from the agent loop.
        engine_info:
            Optional engine metadata dict (from ``EngineInfo``).
        name:
            Human-readable session name.  When empty an auto-generated
            name is derived from the first user message.
        session_id:
            If provided, overwrite an existing session file.  Otherwise
            a new UUID is generated.

        Returns
        -------
        str
            The session ID (uuid).
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        self._dir.chmod(0o700)

        now = datetime.now(timezone.utc).isoformat()
        sid = session_id or uuid.uuid4().hex
        self._validate_session_id(sid)

        # Auto-generate a name from the first user message
        if not name:
            name = self._auto_name(messages)

        path = self._dir / f"{sid}.json"

        # Preserve original created timestamp on overwrite
        created = now
        if path.exists():
            try:
                existing = json.loads(path.read_text())
                created = existing.get("created", now)
            except (json.JSONDecodeError, OSError):
                pass

        data = {
            "id": sid,
            "name": name,
            "created": created,
            "updated": now,
            "engine_info": engine_info or {},
            "messages": messages,
        }

        serialized = json.dumps(data, indent=2, default=str)
        if len(serialized.encode()) > self._max_size:
            raise RuntimeError(
                f"Session too large ({len(serialized.encode())} bytes, "
                f"limit {self._max_size} bytes)"
            )
        path.write_text(serialized)
        logger.info("Session saved: %s (%s)", sid, name)
        return sid

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a session by ID.  Returns ``None`` if not found."""
        self._validate_session_id(session_id)
        path = self._dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load session %s: %s", session_id, exc)
            return None

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return a summary list of all saved sessions.

        Each entry contains: ``id``, ``name``, ``created``, ``updated``,
        ``message_count``.  Sorted by *updated* descending (newest first).
        """
        if not self._dir.is_dir():
            return []

        sessions: list[dict[str, Any]] = []
        for path in self._dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                sessions.append(
                    {
                        "id": data["id"],
                        "name": data.get("name", ""),
                        "created": data.get("created", ""),
                        "updated": data.get("updated", ""),
                        "message_count": len(data.get("messages", [])),
                    }
                )
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                logger.warning("Skipping corrupt session file %s: %s", path.name, exc)
        sessions.sort(key=lambda s: s["updated"], reverse=True)
        return sessions

    def delete(self, session_id: str) -> bool:
        """Delete a session file.  Returns ``True`` if the file existed."""
        self._validate_session_id(session_id)
        path = self._dir / f"{session_id}.json"
        if path.exists():
            path.unlink()
            logger.info("Session deleted: %s", session_id)
            return True
        return False

    # ── helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _auto_name(messages: list[dict[str, Any]]) -> str:
        """Derive a short name from the first user message."""
        for msg in messages:
            if msg.get("role") == "user":
                text = str(msg.get("content", ""))
                # Strip leading slash commands that may have been injected
                if text.startswith("["):
                    continue
                # Truncate to a readable length
                if len(text) > 60:
                    return text[:57] + "..."
                return text
        return f"session-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
