"""Mistral-family tool-call grammar.

Mistral models emit ``[TOOL_CALLS]`` followed by a JSON array of calls
instead of XML tags:

    [TOOL_CALLS] [{"name": "execute_shell", "arguments": {"command": "ls"}}]

When the ``[TOOL_CALLS]`` prefix is forgotten the parser still recovers the
calls from the bare JSON (at the start of the content or inside a markdown
code fence).  Message histories are normalized to strict role alternation.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from natshell.inference.engine import ToolCall
from natshell.inference.grammars import common
from natshell.inference.grammars.common import (
    CODE_FENCE_JSON_RE,
    Grammar,
    format_tool_entries,
    is_bare_tool_json,
    new_tool_call_id,
)

logger = logging.getLogger(__name__)

__all__ = ["MISTRAL_TOOL_CALLS_RE", "format_tools_for_prompt_mistral", "GRAMMAR"]

# [TOOL_CALLS] JSON array (Mistral style)
MISTRAL_TOOL_CALLS_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)


def format_tools_for_prompt_mistral(
    tools: list[dict[str, Any]], *, compact: bool = False
) -> str:
    """Format tool schemas for Mistral models.

    Mistral models use [TOOL_CALLS] followed by a JSON array instead of XML tags.

    Args:
        tools: Tool schemas in OpenAI-compatible format.
        compact: When True, use abbreviated descriptions and parameter lists.
    """
    if compact:
        header = [
            "# Available Tools",
            "You MUST call tools using the exact format below.",
            "Do NOT describe commands in prose — call the tool directly.",
            "NEVER wrap tool calls in markdown code fences.",
            '[TOOL_CALLS] [{"name": "tool_name", "arguments": {...}}]',
            "",
        ]
    else:
        header = [
            "# Available Tools",
            "",
            "You MUST use tools to perform actions. To call a tool, output:",
            "",
            '[TOOL_CALLS] [{"name": "tool_name", "arguments": {"param": "value"}}]',
            "",
            "Do NOT describe commands in prose — call the tool directly.",
            "NEVER wrap tool calls in markdown code fences.",
            "You can call multiple tools at once by including multiple objects in the array.",
            "",
        ]
    return "\n".join(header + format_tool_entries(tools, compact=compact))


class MistralGrammar(Grammar):
    """The Mistral ``[TOOL_CALLS]`` wire format."""

    family = "mistral"

    def render_tools(self, tools, *, compact: bool = False) -> str:
        return format_tools_for_prompt_mistral(tools, compact=compact)

    def parse_native(self, content: str) -> list[ToolCall]:
        """Parse a ``[TOOL_CALLS]`` JSON array from *content*."""
        match = MISTRAL_TOOL_CALLS_RE.search(content)
        if not match:
            return []
        try:
            calls = json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse [TOOL_CALLS] from content: %s", match.group(0))
            return []
        tool_calls: list[ToolCall] = []
        for call in calls:
            name = call.get("name", "")
            if "arguments" in call:
                arguments = call.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
            else:
                # Flat format: args at top level alongside "name"
                arguments = {k: v for k, v in call.items() if k != "name"}
            tool_calls.append(
                ToolCall(id=new_tool_call_id(), name=name, arguments=arguments)
            )
        return tool_calls

    def recover(self, content: str) -> tuple[list[ToolCall], bool]:
        """Mistral forgot the [TOOL_CALLS] prefix but emitted valid JSON.

        Accept bare JSON at the start of content, or inside markdown code
        fences (the three standard shapes are handled by
        :func:`common.try_bare_json_recovery`).
        """
        json_text = None

        # Strategy A: bare JSON at start of content
        stripped = content.strip()
        if stripped.startswith(("{", "[")):
            json_text = stripped

        # Strategy B: JSON inside markdown code fences
        if json_text is None:
            fence_match = CODE_FENCE_JSON_RE.search(content)
            if fence_match:
                json_text = fence_match.group(1)

        calls, fired = common.try_bare_json_recovery(json_text)
        if fired:
            logger.debug(
                "Recovered %d bare-JSON Mistral tool call(s) (missing [TOOL_CALLS] prefix)",
                len(calls),
            )
        return calls, fired

    def scrub_recovered(self, content: str) -> str:
        """Mistral's scrub demands a stricter JSON shape than the default.

        The recovered blob is only cleared when every candidate carries an
        explicit ``"arguments"`` key (matching the original parser).
        """
        content = CODE_FENCE_JSON_RE.sub("", content)
        if is_bare_tool_json(content.strip(), require_arguments=True):
            content = ""
        return content

    def normalize_messages(self, messages):
        """Mistral templates require strict user/assistant alternation."""
        return common.normalize_messages_strict_alternation(messages)


GRAMMAR = MistralGrammar()
