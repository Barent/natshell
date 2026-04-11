"""Recall content from the retrieval-augmented compaction store.

When ``compact_history`` writes dropped messages to a ``MemoryStore``, the
short summary that replaces them contains ``[mem:<hash>]`` references.  This
tool lets the agent fetch the full content of any reference (or search by
keyword) on demand, so detail lost to compaction can be re-introduced into
the conversation only when actually needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from natshell.tools.registry import ToolDefinition, ToolResult

if TYPE_CHECKING:
    from natshell.agent.memory_store import MemoryStore

logger = logging.getLogger(__name__)

# Module-level state injected by AgentLoop on init.  Tool handlers are
# called by the registry without explicit context, so we follow the same
# injection pattern used by other tools (e.g. fetch_url.set_limits).
_store: "MemoryStore | None" = None
_session_id: str = ""

# Cap on returned content per call to keep model context usage bounded.
_MAX_RECALL_BYTES = 8 * 1024


def configure(store: "MemoryStore | None", session_id: str) -> None:
    """Bind the AgentLoop's MemoryStore + session id to this tool."""
    global _store, _session_id
    _store = store
    _session_id = session_id


def reset() -> None:
    """Test helper — clear injected state."""
    global _store, _session_id
    _store = None
    _session_id = ""


DEFINITION = ToolDefinition(
    name="recall_memory",
    description=(
        "Retrieve full content from the conversation memory store. Use this "
        "when the visible context contains a [mem:<hash>] reference (created "
        "during compaction) and you need the original content. You can fetch "
        "by exact hash, or search by keyword across all stored chunks from "
        "this session. Returns the chunk content with its hash for chaining."
    ),
    parameters={
        "type": "object",
        "properties": {
            "hash": {
                "type": "string",
                "description": (
                    "Hash prefix from a [mem:<hash>] reference. Either the "
                    "12-char short prefix or the full 64-char SHA256."
                ),
            },
            "query": {
                "type": "string",
                "description": (
                    "Free-text keyword search over stored chunks. Use this "
                    "when you don't have a specific hash but want to find "
                    "earlier content (e.g. 'nmap output', 'config file')."
                ),
            },
            "kind": {
                "type": "string",
                "description": (
                    "Optional filter by tool name (e.g. 'execute_shell', "
                    "'read_file') or chunk kind ('user_message', "
                    "'assistant_text', 'tool_result')."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max search results to return (default 5).",
            },
        },
    },
)


async def recall_memory(
    hash: str = "",
    query: str = "",
    kind: str = "",
    limit: int = 5,
) -> ToolResult:
    """Fetch chunk content by hash or keyword search."""
    if _store is None or not _store.healthy:
        return ToolResult(
            error=(
                "Memory store is unavailable. The recall_memory tool only "
                "works when the chunk store is enabled in config."
            ),
            exit_code=1,
        )

    hash = (hash or "").strip()
    query = (query or "").strip()
    kind = (kind or "").strip() or None
    try:
        limit = max(1, min(20, int(limit or 5)))
    except (TypeError, ValueError):
        limit = 5

    # Hash lookup takes priority — it's deterministic.
    if hash:
        row = _store.get(hash, session_id=_session_id)
        if row is None:
            return ToolResult(
                error=f"No chunk found for hash '{hash}' in this session.",
                exit_code=1,
            )
        return ToolResult(output=_format_chunk(row))

    if not query:
        return ToolResult(
            error=(
                "recall_memory needs either a 'hash' or a 'query' argument."
            ),
            exit_code=1,
        )

    rows = _store.search(query, session_id=_session_id, kind=kind, limit=limit)
    if not rows:
        return ToolResult(
            output=f"No stored chunks matched query '{query}'.",
        )

    parts: list[str] = [f"Found {len(rows)} chunk(s) matching '{query}':\n"]
    for row in rows:
        parts.append(_format_chunk(row, brief=True))
        parts.append("")  # blank separator
    return ToolResult(output="\n".join(parts).rstrip())


def _format_chunk(row: dict, brief: bool = False) -> str:
    """Render a chunk row for the agent.  Truncates content to a sane cap."""
    short = (row.get("hash") or "")[:12]
    role = row.get("role", "")
    tool_name = row.get("tool_name") or ""
    kind = row.get("kind") or ""
    size = row.get("size_bytes", 0)
    content = row.get("content", "") or ""

    header_bits = [f"[mem:{short}]"]
    if tool_name:
        header_bits.append(f"tool={tool_name}")
    if kind:
        header_bits.append(f"kind={kind}")
    header_bits.append(f"role={role}")
    header_bits.append(f"size={size}B")
    header = " ".join(header_bits)

    if brief and len(content) > 600:
        content = content[:600] + f"\n[... truncated, use hash='{short}' for full content ...]"
    elif len(content) > _MAX_RECALL_BYTES:
        content = (
            content[:_MAX_RECALL_BYTES]
            + f"\n[... {len(content) - _MAX_RECALL_BYTES} bytes truncated ...]"
        )

    return f"{header}\n{content}"
