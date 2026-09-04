"""Qwen-family tool-call grammar (the default for unknown model families).

Qwen3 models ignore llama-cpp-python's native tool handling and instead emit
``
`` XML blocks in the content stream:

     {"name": "execute_shell", "arguments": {"command": "ls"}}

There is no bare-JSON recovery for this family and no message rewriting —
the engine's universal marker strip takes care of leftovers.

NOTE: the tool-call markers are assembled from ``chr()`` fragments (see
``_TC_OPEN`` / ``_TC_CLOSE``) because the agent write pipeline we generate
this source through mangles the raw literal markers.  The compiled regex is
byte-for-byte identical to the former ``_TOOL_CALL_RE`` in
``inference/local.py``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from natshell.inference.engine import ToolCall
from natshell.inference.grammars.common import (
    Grammar,
    format_tool_entries,
    new_tool_call_id,
)

logger = logging.getLogger(__name__)

__all__ = ["TOOL_CALL_RE", "format_tools_for_prompt", "GRAMMAR"]

#  {"name": ..., "arguments": ...} blocks (assembled — see module note)
_TC_OPEN = chr(60) + "tool_call" + chr(62)
_TC_CLOSE = chr(60) + "/tool_call" + chr(62)
TOOL_CALL_RE = re.compile(_TC_OPEN + r"\s*(\{.*?\})\s*" + _TC_CLOSE, re.DOTALL)


def format_tools_for_prompt(
    tools: list[dict[str, Any]], *, compact: bool = False
) -> str:
    """Format tool schemas as plain text for injection into the system prompt.

    This is used instead of llama-cpp-python's built-in tool handling because
    Qwen3 models don't follow the chatml-function-calling response format.

    Args:
        tools: Tool schemas in OpenAI-compatible format.
        compact: When True, use abbreviated descriptions and parameter lists
            to reduce token usage on small context windows (≤16K).
    """
    # The example block renders the qwen markers as assembled in TOOL_CALL_RE
    # (the agent write pipeline that generates this file mangles raw markers).
    example = (
        _TC_OPEN
        + '{"name": "tool_name", "arguments": {...}}'
        + _TC_CLOSE
    )
    if compact:
        header = [
            "# Available Tools",
            "You MUST use the " + _TC_OPEN + " format to call tools.",
            "NEVER substitute pseudocode or Python scripts for tool calls.",
            example,
            "",
        ]
    else:
        header = [
            "# Available Tools",
            "",
            "You MUST use tools to perform actions. To call a tool, output:",
            "",
            _TC_OPEN,
            '{"name": "tool_name", "arguments": {"param": "value"}}',
            _TC_CLOSE,
            "",
        ]
    return "\n".join(header + format_tool_entries(tools, compact=compact))


class QwenGrammar(Grammar):
    """The Qwen3 wire format (and the fallback for unknown families)."""

    family = "qwen"

    def render_tools(self, tools, *, compact: bool = False) -> str:
        return format_tools_for_prompt(tools, compact=compact)

    def parse_native(self, content: str) -> list[ToolCall]:
        """Parse  blocks from *content* (Qwen3 style)."""
        tool_calls: list[ToolCall] = []
        for match in TOOL_CALL_RE.finditer(content):
            try:
                parsed = json.loads(match.group(1))
                name = parsed.get("name", "")
                arguments = parsed.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                tool_calls.append(
                    ToolCall(
                        id=new_tool_call_id(),
                        name=name,
                        arguments=arguments,
                    )
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning(
                    "Failed to parse tool_call from content: %s", match.group(0)
                )
        return tool_calls


GRAMMAR = QwenGrammar()
