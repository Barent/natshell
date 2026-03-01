"""Session persistence — save and restore conversation history."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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

    def __init__(self, session_dir: Path | None = None) -> None:
        self._dir = session_dir or SESSION_DIR

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

        now = datetime.now(timezone.utc).isoformat()
        sid = session_id or uuid.uuid4().hex

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

        path.write_text(json.dumps(data, indent=2, default=str))
        logger.info("Session saved: %s (%s)", sid, name)
        return sid

    def load(self, session_id: str) -> dict[str, Any] | None:
        """Load a session by ID.  Returns ``None`` if not found."""
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
