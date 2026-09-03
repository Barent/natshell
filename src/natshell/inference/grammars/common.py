"""Family-agnostic helpers shared by the tool-call grammar modules.

Each grammar module in this package (``qwen.py``, ``mistral.py``, ``gemma.py``)
owns one model family's *native* tool-call rendering and parsing.  Everything
they share — the regexes for common chat-template residue, the parse of
structured ``tool_calls``, the bare-JSON recovery path, the multi-stage strip
of markers/think-blocks from content — lives here so it's implemented exactly
once instead of once-per-family.

No model-specific code belongs in this file.  If a helper starts making sense
only for one model, it belongs in that model's module.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Any

from natshell.inference.engine import ToolCall

__all__ = [
    "ToolCall",
    "CODE_FENCE_JSON_RE",
    "THINK_RE",
    "THINK_UNCLOSED_RE",
    "parse_structured_tool_calls",
    "try_bare_json_recovery",
    "is_bare_tool_json",
    "strip_prose_markers",
    "is_degenerate_output",
    "new_tool_call_id",
]

# JSON wrapped in markdown code fences (Mistral / Gemma both sometimes emit this)
CODE_FENCE_JSON_RE = re.compile(
    r"```(?:json)?\s*\n?\s*(\{.*?\}|\[.*?\])\s*\n?\s*```", re.DOTALL
)
#  closed / unclosed think blocks
THINK_RE = re.compile(r" .*?", re.DOTALL)
THINK_UNCLOSED_RE = re.compile(r"(?:(?!>).)*$", re.DOTALL)


def new_tool_call_id() -> str:
    """Allocate a fresh tool-call id (short, collision-resistant)."""
    return str(uuid.uuid4())[:9]


def parse_structured_tool_calls(
    structured: list[Any] | None,
) -> list[ToolCall]:
    """Parse an OpenAI-style ``tool_calls`` array into ``ToolCall`` objects.

    This is the universal first step across all families: if the runtime (or
    a wrapper) already produced structured tool calls, use them verbatim and
    skip native parsing entirely.
    """
    tool_calls: list[ToolCall] = []
    for tc in structured or []:
        func = tc.get("function", {})
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(
            ToolCall(
                id=tc.get("id", new_tool_call_id()),
                name=func.get("name", ""),
                arguments=args,
            )
        )
    return tool_calls


def try_bare_json_recovery(json_text: str | None) -> tuple[list[ToolCall], bool]:
    """Recover tool calls from bare JSON (i.e. the family's native wrapper
    is missing but the JSON body is present).

    Handles all three common shapes:
      1. ``{"name": "tool", "arguments": {...}}``   — standard
      2. ``[{"name": ..., "arguments": ...}, ...]`` — array of calls
      3. ``{"name": "tool", "key": "val"}``         — flat (args at top level)

    Returns ``(tool_calls, recovered)``.  ``recovered`` is only true when
    every candidate looked like a valid tool call — so the caller can strip
    the JSON blob from the prose.
    """
    if json_text is None:
        return [], False
    try:
        parsed = json.loads(json_text)
        candidates = parsed if isinstance(parsed, list) else [parsed]
        if not all(isinstance(c, dict) and "name" in c for c in candidates):
            return [], False
        tool_calls: list[ToolCall] = []
        for call in candidates:
            if "arguments" in call:
                arguments = call["arguments"]
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
            else:
                arguments = {k: v for k, v in call.items() if k != "name"}
            tool_calls.append(
                ToolCall(id=new_tool_call_id(), name=call["name"], arguments=arguments)
            )
        return tool_calls, True
    except (json.JSONDecodeError, KeyError, TypeError):
        return [], False


def is_bare_tool_json(text: str, *, require_arguments: bool = False) -> bool:
    """True if *text* is a bare JSON object/array whose entries all look like
    tool-call dicts (all have ``"name"``; optionally also ``"arguments"``).
    """
    if not text.lstrip().startswith(("{", "[")):
        return False
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False
    candidates = parsed if isinstance(parsed, list) else [parsed]
    if not candidates:
        return False
    for c in candidates:
        if not isinstance(c, dict) or "name" not in c:
            return False
        if require_arguments and "arguments" not in c:
            return False
    return True


# ---------------------------------------------------------------------------
# Shared stripping pipeline
# ---------------------------------------------------------------------------


def strip_prose_markers(content: str, *, family_regexes: list[re.Pattern[str]]) -> str:
    """Strip common chat-template residue from model output.

    Applies (in order, all DOTALL):

    - Qwen / generic ``...`` and its unclosed variant,
    - every family-specific marker regex passed in ``family_regexes``
      (think blocks, special tokens, native tool-call syntax, etc.).

    The caller is responsible for the *final* step (stripping any
    recovered bare JSON / code-fenced JSON) because whether to run that and
    what shape to demand depends on which parser fired.
    """
    content = THINK_RE.sub("", content)
    content = THINK_UNCLOSED_RE.sub("", content)
    for pattern in family_regexes:
        content = pattern.sub("", content)
    return content


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def is_degenerate_output(text: str) -> bool:
    """Detect degenerate repetitive output from local models.

    Returns True when the output is dominated by a single repeated character,
    which indicates context exhaustion or model collapse. Only triggers on
    outputs longer than 100 characters to avoid false positives on short
    valid responses.
    """
    if len(text) < 100:
        return False
    non_ws = text.replace(" ", "").replace("\n", "").replace("\t", "")
    if not non_ws:
        return False
    from collections import Counter

    counts = Counter(non_ws)
    _char, top_count = counts.most_common(1)[0]
    return top_count / len(non_ws) > 0.5
