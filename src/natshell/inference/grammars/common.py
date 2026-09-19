"""Family-agnostic helpers shared by the tool-call grammar modules.

Each grammar module in this package (``qwen.py``, ``mistral.py``, ``gemma.py``)
owns one model family's *native* tool-call rendering and parsing.  Everything
they share — the ``Grammar`` base class, the regexes for common chat-template
residue, the parse of structured ``tool_calls``, the shared strip of
markers/think-blocks from content, and the shared message normalizers — lives
here so it's implemented exactly once instead of once-per-family.

No model-specific code belongs in this file.  If a helper starts making sense
only for one model, it belongs in that model's module.

The two generic think-block markers are assembled from ``chr()`` fragments
(see ``_THINK_OPEN`` / ``_THINK_CLOSE``) because the agent write pipeline we
generate this source through mangles the raw literal markers.  The resulting
compiled regexes are byte-for-byte identical to the original ``_THINK_RE`` /
``_THINK_UNCLOSED_RE`` that used to live in ``inference/local.py``.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

from natshell.inference.engine import ToolCall

__all__ = [
    "ToolCall",
    "Grammar",
    "CODE_FENCE_JSON_RE",
    "THINK_RE",
    "THINK_UNCLOSED_RE",
    "parse_structured_tool_calls",
    "try_bare_json_recovery",
    "is_bare_tool_json",
    "strip_prose_markers",
    "normalize_messages_strict_alternation",
    "normalize_tool_results_to_user",
    "format_tool_entries",
    "is_degenerate_output",
    "new_tool_call_id",
]

# JSON wrapped in markdown code fences (Mistral / Gemma both sometimes emit this)
CODE_FENCE_JSON_RE = re.compile(
    r"```(?:json)?\s*\n?\s*(\{.*?\}|\[.*?\])\s*\n?\s*```", re.DOTALL
)

# Generic think-block markers.  Assembled from chr() fragments so the raw
# ``<``/``>`` literals never appear contiguously in this generated source (the
# write pipeline they pass through would otherwise corrupt them).  The compiled
# patterns are identical to the former ``_THINK_RE`` / ``_THINK_UNCLOSED_RE``.
_THINK_OPEN = chr(60) + "think" + chr(62)
_THINK_CLOSE = chr(60) + "/think" + chr(62)
#  closed / unclosed think blocks
THINK_RE = re.compile(_THINK_OPEN + r".*?" + _THINK_CLOSE, re.DOTALL)
THINK_UNCLOSED_RE = re.compile(
    _THINK_OPEN + r"(?:(?!" + _THINK_CLOSE + r").)*$", re.DOTALL
)


def new_tool_call_id() -> str:
    """Allocate a fresh tool-call id (short, collision-resistant)."""
    return str(uuid.uuid4())[:9]


def is_degenerate_output(text: str) -> bool:
    """Detect degenerate repetitive output from local models.

    Returns True when the output is dominated by a single repeated
    character, which indicates context exhaustion or model collapse.
    Only triggers on outputs longer than 100 characters to avoid
    false positives on short valid responses.
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


def strip_prose_markers(
    content: str, *, family_regexes: list[re.Pattern] | tuple[re.Pattern, ...]
) -> str:
    """Strip common chat-template residue and every family's tool-call markers
    from model output.

    Applies (in order, all DOTALL):

    - the generic think-block pair (``THINK_RE`` / ``THINK_UNCLOSED_RE``),
    - every marker regex in ``family_regexes`` (the caller passes the union of
      all families' think blocks, special tokens, and native tool-call syntax).

    The caller is responsible for the *final* step (stripping any recovered
    bare JSON / code-fenced JSON) because whether to run that and what shape
    to demand depends on which family's parser fired.
    """
    content = THINK_RE.sub("", content)
    content = THINK_UNCLOSED_RE.sub("", content)
    for pattern in family_regexes:
        content = pattern.sub("", content)
    return content


# ---------------------------------------------------------------------------
# Message normalization (shared by Mistral and Gemma)
# ---------------------------------------------------------------------------


def normalize_messages_strict_alternation(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize messages for strict role-alternation (Mistral, Gemma 4).

    These models' chat templates require: system? -> (user -> assistant)*.
    Two passes fix violations:

    Pass 1: Fold mid-conversation system messages into the initial system
    message (the template only allows one system message at position 0).

    Pass 2: Merge consecutive same-role messages. This happens when e.g.
    /cmd appends a user message with command output and then the next user
    input appends another user message. Never merge assistant messages that
    carry tool_calls, and never merge or touch tool-role messages.
    """
    # --- Pass 1: fold mid-conversation system messages ---
    pass1: list[dict[str, Any]] = []
    system_extras: list[str] = []

    for msg in messages:
        if msg["role"] == "system" and pass1:
            system_extras.append(msg["content"])
        else:
            pass1.append(msg)

    if system_extras and pass1 and pass1[0]["role"] == "system":
        merged = pass1[0]["content"] + "\n\n" + "\n\n".join(system_extras)
        pass1[0] = {**pass1[0], "content": merged}

    # --- Pass 2: merge consecutive same-role messages ---
    result: list[dict[str, Any]] = []
    for msg in pass1:
        role = msg["role"]
        if (
            result
            and role == result[-1]["role"]
            and role in ("user", "assistant")
            and "tool_calls" not in msg
            and "tool_calls" not in result[-1]
        ):
            result[-1] = {
                **result[-1],
                "content": result[-1]["content"] + "\n\n" + msg["content"],
            }
        else:
            result.append(msg)

    return result


def normalize_tool_results_to_user(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert assistant ``tool_calls`` and ``role: "tool"`` result messages
    into plain alternating user/assistant text.

    Shared by the Mistral and Gemma normalizers.  Does not touch any other
    message, and never merges messages that carry ``tool_calls``.
    """
    result: list[dict[str, Any]] = []
    for msg in messages:
        if "tool_calls" in msg:
            result.append({**msg, "tool_calls": msg["tool_calls"]})
        elif msg.get("role") == "tool":
            content = msg.get("content", "")
            tool_id = msg.get("tool_call_id", "")
            result.append({
                "role": "user",
                "content": f"[Tool result ({tool_id})]:\n{content}",
            })
        else:
            result.append(msg)
    return result


# ---------------------------------------------------------------------------
# Shared formatting helpers
# ---------------------------------------------------------------------------


def format_tool_entries(
    tools: list[dict[str, Any]], *, compact: bool = False
) -> list[str]:
    """Format tool entries (shared by the Qwen and Mistral renderers).

    Returns a list of lines describing each tool's name, description, and
    parameters. The caller provides the family-specific header.
    """
    lines: list[str] = []
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"## {name}")
        if compact:
            first_sentence = desc.split(". ")[0]
            if not first_sentence.endswith("."):
                first_sentence += "."
            lines.append(first_sentence)
        else:
            lines.append(desc)

        props = params.get("properties", {})
        required = params.get("required", [])
        if props:
            if not compact:
                lines.append("Parameters:")
            for pname, pdef in props.items():
                req = ", required" if pname in required else ""
                ptype = pdef.get("type", "")
                if compact:
                    enum_vals = pdef.get("enum")
                    if enum_vals:
                        enum_str = "|".join(str(v) for v in enum_vals)
                        lines.append(f"- {pname} ({enum_str}{req})")
                    else:
                        lines.append(f"- {pname} ({ptype}{req})")
                else:
                    pdesc = pdef.get("description", "")
                    lines.append(f"- {pname} ({ptype}{req}): {pdesc}")
        lines.append("")
    return lines


# ---------------------------------------------------------------------------
# The Grammar base class
# ---------------------------------------------------------------------------


@dataclass
class Grammar:
    """A model family's tool-call wire format.

    Each family subclass (``qwen`` / ``mistral`` / ``gemma``) overrides three
    hooks:

    - :meth:`render_tools` — how tool schemas are rendered into the prompt.
    - :meth:`parse_native` — extract this family's native tool-call syntax.
    - :meth:`recover`     — family-specific bare-JSON fallback (may be absent).

    The shared pipeline (structured calls, the universal marker strip, and the
    bare-JSON scrub) is implemented once here in :meth:`parse`.
    """

    family: str = "qwen"

    def render_tools(
        self, tools: list[dict[str, Any]], *, compact: bool = False
    ) -> str:
        raise NotImplementedError

    def parse_native(self, content: str) -> list[ToolCall]:
        """Parse this family's native tool-call syntax out of *content*.

        The default (most families) emits nothing; families override this.
        """
        return []

    def recover(self, content: str) -> tuple[list[ToolCall], bool]:
        """Family-specific bare-JSON recovery.  Returns ``(calls, fired)``.

        The default is a no-op — the family has no bare-JSON fallback.
        """
        return [], False

    def normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rewrite the message history into this family's expected shape.

        The default is the identity (no rewriting).
        """
        return messages

    # ------------------------------------------------------------------
    # The shared pipeline used by every family
    # ------------------------------------------------------------------

    def parse(
        self,
        content: str,
        *,
        structured: list[Any] | None = None,
        family_strip=None,
    ) -> tuple[list[ToolCall], str, bool]:
        """Run the full parse pipeline and return ``(tool_calls, content, fired)``.

        ``fired`` is True only when the family's bare-JSON :meth:`recover`
        fired, so the caller knows to scrub the recovered JSON blob from the
        prose.  ``content`` is the cleaned prose (with think blocks and every
        family's markers stripped).

        When ``family_strip`` is None (the default), the union of *all*
        families' marker regexes is applied — the original local.py stripped
        that union from every response regardless of family, and this
        preserves that behaviour exactly.
        """
        if family_strip is None:
            # Lazy import: grammars/__init__ imports the family modules, which
            # import common — an eager import here would be circular.
            from natshell.inference.grammars import ALL_FAMILY_STRIP

            family_strip = ALL_FAMILY_STRIP
        tool_calls = parse_structured_tool_calls(structured)
        if not tool_calls:
            tool_calls = self.parse_native(content)
        fired = False
        if not tool_calls:
            tool_calls, fired = self.recover(content)
        content = strip_prose_markers(
            content, family_regexes=tuple(family_strip)
        )
        if fired:
            content = self.scrub_recovered(content)
        return tool_calls, content, fired

    def scrub_recovered(self, content: str) -> str:
        """After a bare-JSON recovery, remove the JSON blob from the prose.

        The default strips code-fenced JSON and clears any remaining leading
        bare tool-call JSON.  Families that need a different shape (e.g.
        requiring ``"arguments"`` on every call) override this.
        """
        content = CODE_FENCE_JSON_RE.sub("", content)
        remaining = content.strip()
        if is_bare_tool_json(remaining):
            content = ""
        return content
