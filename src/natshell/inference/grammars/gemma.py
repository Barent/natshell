"""Gemma-family tool-call grammar.

Gemma 4 emits tool calls as

    <|tool_call>call:NAME{key:<|\"|>val<|\"|>,num:42}<tool_call|>

— string values wrapped in ``<|\"|>`` delimiters, numeric/boolean values bare,
arguments separated by commas.  Its thinking channel uses ``<|channel>...<channel|>``,
and a handful of special tokens (``<eos>``, ``<unusedNN>``, …) can leak into
prose output.

When the model emits plain JSON instead of the native format, the parser still
recovers the calls from bare JSON (at the start of the content, or wrapped in
markdown code fences) via :func:`common.try_bare_json_recovery`.

Message histories are converted to the native text form (assistant
``tool_calls`` → Gemma tool-call text, ``role: "tool"`` → labelled user
messages) and then normalized to strict role alternation, exactly as the
former ``_convert_gemma_tool_messages`` + ``_normalize_messages_strict_alternation``
pair did in ``inference/local.py``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from natshell.inference.engine import ToolCall
from natshell.inference.grammars.common import (
    Grammar,
    is_bare_tool_json,
    new_tool_call_id,
    normalize_messages_strict_alternation,
    try_bare_json_recovery,
)

logger = logging.getLogger(__name__)

__all__ = [
    "GEMMA_TOOL_CALL_RE",
    "GEMMA_THINK_RE",
    "GEMMA_THINK_UNCLOSED_RE",
    "GEMMA_SPECIAL_TOKEN_RE",
    "parse_gemma_tool_args",
    "format_tools_for_prompt_gemma",
    "format_gemma_tool_call_text",
    "GRAMMAR",
]

# Gemma 4 tool call: <|tool_call>call:NAME{...}<tool_call|>
# NAME may contain hyphens (skills like "web-research" are invoked by name).
GEMMA_TOOL_CALL_RE = re.compile(
    r"<\|tool_call>call:([\w-]+)\{(.*?)\}<tool_call\|>", re.DOTALL
)
# Gemma 4 think blocks: <|channel>thought...<channel|>
GEMMA_THINK_RE = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)
GEMMA_THINK_UNCLOSED_RE = re.compile(
    r"<\|channel>(?:(?!<channel\|>).)*$", re.DOTALL
)
# Gemma special tokens that leak into text output
GEMMA_SPECIAL_TOKEN_RE = re.compile(
    r"<\|tool_response>.*?<tool_response\|>"  # tool response markers
    r"|<\|tool_response>"                      # unclosed tool response
    r"|<eos>"                                  # end-of-sequence
    r"|<unused\d+>"                            # unused vocabulary tokens
    r"|<\|eos\|>"                              # alternate EOS format
    r"|<\|eot\|>",                             # end-of-turn
    re.DOTALL,
)

# Gemma's string-value delimiter, assembled from chr() fragments so the raw
# ``<|\"|>`` literal never appears in a way the write pipeline could mangle.
# The string value is byte-for-byte identical to the former ``<|"|>`` literal.
_STR_DELIM = chr(60) + "|" + chr(34) + "|" + chr(62)  # 5 chars


def _gemma_escape(value: str) -> str:
    """Wrap a string value with Gemma's ``<|\"|>`` delimiter."""
    return f"{_STR_DELIM}{value}{_STR_DELIM}"


def parse_gemma_tool_args(text: str) -> dict[str, Any]:
    """Parse Gemma 4 tool call arguments from the native ``key:val`` format.

    String values are wrapped with the ``<|\"|>`` delimiter; numeric and
    boolean values are bare.
    """
    args: dict[str, Any] = {}
    if not text.strip():
        return args

    # Split on top-level commas that are outside <|\"|>...<|\"|> delimiters
    parts: list[str] = []
    current: list[str] = []
    inside_string = False
    i = 0
    while i < len(text):
        if text[i:].startswith(_STR_DELIM):
            inside_string = not inside_string
            current.append(_STR_DELIM)
            i += 5
        elif text[i] == "," and not inside_string:
            parts.append("".join(current))
            current = []
            i += 1
        else:
            current.append(text[i])
            i += 1
    if current:
        parts.append("".join(current))

    for part in parts:
        colon = part.find(":")
        if colon == -1:
            continue
        key = part[:colon].strip()
        val_raw = part[colon + 1:].strip()

        # String value: strip <|\"|> delimiters
        if _STR_DELIM in val_raw:
            args[key] = val_raw.replace(_STR_DELIM, "")
        else:
            # Numeric or boolean
            if val_raw.lower() == "true":
                args[key] = True
            elif val_raw.lower() == "false":
                args[key] = False
            else:
                try:
                    args[key] = int(val_raw)
                except ValueError:
                    try:
                        args[key] = float(val_raw)
                    except ValueError:
                        args[key] = val_raw
    return args


def format_gemma_tool_call_text(name: str, arguments: dict[str, Any]) -> str:
    """Format a tool call as Gemma-native text for message history.

    Converts ``{"command": "ls -la", "timeout": 60}`` into the native
    ``<|tool_call>call:execute_shell{...}<tool_call|>`` form.
    """
    parts: list[str] = []
    for key, value in arguments.items():
        if isinstance(value, str):
            parts.append(f"{key}:{_gemma_escape(value)}")
        elif isinstance(value, bool):
            parts.append(f"{key}:{'true' if value else 'false'}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}:{value}")
        else:
            # Fallback: serialize as escaped string
            parts.append(f"{key}:{_gemma_escape(str(value))}")
    args_str = ",".join(parts)
    return f"<|tool_call>call:{name}{{{args_str}}}<tool_call|>"


def format_tools_for_prompt_gemma(
    tools: list[dict[str, Any]], *, compact: bool = False
) -> str:
    """Format tool schemas for Gemma 4 models.

    Gemma 4 uses ``<|tool>declaration:NAME{...}<tool|>`` blocks with
    ``<|\"|>`` string delimiters instead of JSON.

    Args:
        tools: Tool schemas in OpenAI-compatible format.
        compact: When True, use abbreviated descriptions.
    """
    esc = _gemma_escape

    if compact:
        header = [
            "# Available Tools",
            "You MUST call tools using the exact format below.",
            "NEVER use JSON, Python dicts, or pseudocode for tool calls.",
            "<|tool_call>call:tool_name{param:" + esc("value") + "}<tool_call|>",
            "",
        ]
    else:
        header = [
            "# Available Tools",
            "",
            "You MUST use tools to perform actions. To call a tool, output:",
            "",
            "<|tool_call>call:tool_name{param:" + esc("value") + "}<tool_call|>",
            "",
            "String values MUST be wrapped with <|\"|> delimiters.",
            "NEVER use JSON, Python dicts, or pseudocode for tool calls.",
            "You can call multiple tools by outputting multiple <|tool_call> blocks.",
            "",
            "Example — run a shell command:",
            "<|tool_call>call:execute_shell{command:" + esc("ls -la /tmp") + "}<tool_call|>",
            "",
        ]

    lines: list[str] = list(header)
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        if compact:
            first_sentence = desc.split(". ")[0]
            if not first_sentence.endswith("."):
                first_sentence += "."
            desc = first_sentence

        # Build the <|tool>declaration:...<tool|> block
        props = params.get("properties", {})
        required = params.get("required", [])

        prop_parts: list[str] = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "string")
            pdesc = pdef.get("description", "")
            if compact:
                prop_parts.append(
                    f"{pname}:{{type:{esc(ptype)}}}"
                )
            else:
                prop_parts.append(
                    f"{pname}:{{type:{esc(ptype)},description:{esc(pdesc)}}}"
                )

        req_parts = ",".join(esc(r) for r in required)
        props_str = ",".join(prop_parts)

        decl = (
            f"<|tool>declaration:{name}{{"
            f"description:{esc(desc)},"
            f"parameters:{{properties:{{{props_str}}},"
            f"required:[{req_parts}],"
            f"type:{esc('object')}}}"
            f"}}<tool|>"
        )
        lines.append(decl)

    return "\n".join(lines)


class GemmaGrammar(Grammar):
    """The Gemma 4 ``<|tool_call>call:NAME{...}`` wire format."""

    family = "gemma"

    def render_tools(self, tools, *, compact: bool = False) -> str:
        return format_tools_for_prompt_gemma(tools, compact=compact)

    def parse_native(self, content: str) -> list[ToolCall]:
        """Parse <|tool_call>call:NAME{...}<tool_call|> blocks (Gemma 4 style)."""
        tool_calls: list[ToolCall] = []
        for match in GEMMA_TOOL_CALL_RE.finditer(content):
            name = match.group(1)
            args_text = match.group(2)
            try:
                arguments = parse_gemma_tool_args(args_text)
            except Exception:
                logger.warning(
                    "Failed to parse Gemma tool_call args: %s", match.group(0)
                )
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=new_tool_call_id(),
                    name=name,
                    arguments=arguments,
                )
            )
        return tool_calls

    def recover(self, content: str) -> tuple[list[ToolCall], bool]:
        """Gemma emitted JSON instead of the native format.

        Recovers {"name": ..., "arguments": {...}} (or a flat
        {"name": ..., "command": ...}) — the three standard shapes handled by
        :func:`common.try_bare_json_recovery`.  Special tokens are stripped
        before the bare-JSON check, exactly as the former parser did.
        """
        # Strip Gemma special tokens before looking for JSON
        cleaned = GEMMA_SPECIAL_TOKEN_RE.sub("", content).strip()
        json_text = None
        if cleaned.startswith(("{", "[")):
            json_text = cleaned
        if json_text is None:
            from natshell.inference.grammars.common import CODE_FENCE_JSON_RE

            fence_match = CODE_FENCE_JSON_RE.search(cleaned)
            if fence_match:
                json_text = fence_match.group(1)

        calls, fired = try_bare_json_recovery(json_text)
        if fired:
            logger.debug(
                "Recovered %d bare-JSON Gemma tool call(s) "
                "(expected native format)",
                len(calls),
            )
        return calls, fired

    def normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to Gemma-native text, then enforce strict alternation.

        Runs the former ``_convert_gemma_tool_messages`` stage followed by
        ``_normalize_messages_strict_alternation`` (that order matters — the
        template folding assumes native text is already in place).
        """
        converted = self._convert_tool_messages(messages)
        return normalize_messages_strict_alternation(converted)

    @staticmethod
    def _convert_tool_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format tool messages to Gemma-native text format.

        The Gemma 4 chat template expects tool call arguments as dicts (to
        format with ``<|\"|>`` delimiters) and tool results via a
        ``tool_responses`` key — neither of which matches the OpenAI format
        NatShell uses internally.  So we convert both message types to plain
        text that preserves the Gemma tool call syntax the model was trained
        on:

        - Assistant messages with ``tool_calls`` → assistant messages with
          Gemma-format ``<|tool_call>call:NAME{...}<tool_call|>`` text.
        - ``role: "tool"`` result messages → ``role: "user"`` messages with a
          labelled result block.
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("tool_calls"):
                # Convert assistant tool_calls to Gemma-native text
                content_parts: list[str] = []
                if msg.get("content"):
                    content_parts.append(msg["content"])
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    raw_args = func.get("arguments", {})
                    if isinstance(raw_args, str):
                        try:
                            raw_args = json.loads(raw_args)
                        except (json.JSONDecodeError, TypeError):
                            raw_args = {}
                    content_parts.append(
                        format_gemma_tool_call_text(name, raw_args)
                    )
                result.append({
                    "role": "assistant",
                    "content": "\n".join(content_parts),
                })
            elif msg.get("role") == "tool":
                # Convert tool result to a user message
                content = msg.get("content", "")
                tool_id = msg.get("tool_call_id", "")
                # Include a clear label so the model knows this is a tool result
                result.append({
                    "role": "user",
                    "content": f"[Tool result ({tool_id})]:\n{content}",
                })
            else:
                result.append(msg)
        return result


GRAMMAR = GemmaGrammar()
