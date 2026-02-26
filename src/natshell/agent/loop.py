"""Core ReAct agent loop — plans, executes tools, observes, repeats."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator

from natshell.agent.context import SystemContext
from natshell.agent.system_prompt import build_system_prompt
from natshell.config import AgentConfig
from natshell.inference.engine import CompletionResult, InferenceEngine, ToolCall
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class EventType(Enum):
    THINKING = "thinking"
    PLANNING = "planning"       # Model's text before tool calls
    EXECUTING = "executing"     # About to run a tool
    TOOL_RESULT = "tool_result" # Result from a tool
    CONFIRM_NEEDED = "confirm_needed"  # Awaiting user confirmation
    BLOCKED = "blocked"         # Command was blocked
    RESPONSE = "response"       # Final text response from model
    ERROR = "error"             # Something went wrong


@dataclass
class AgentEvent:
    """An event yielded by the agent loop for the TUI to render."""

    type: EventType
    data: Any = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None


class AgentLoop:
    """The ReAct agent loop — the brain of NatShell."""

    def __init__(
        self,
        engine: InferenceEngine,
        tools: ToolRegistry,
        safety: SafetyClassifier,
        config: AgentConfig,
    ) -> None:
        self.engine = engine
        self.tools = tools
        self.safety = safety
        self.config = config
        self.messages: list[dict[str, Any]] = []

    def initialize(self, system_context: SystemContext) -> None:
        """Build the system prompt and initialize conversation."""
        system_prompt = build_system_prompt(system_context)
        self.messages = [{"role": "system", "content": system_prompt}]

    async def handle_user_message(
        self,
        user_input: str,
        confirm_callback=None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Process a user message through the full agent loop.

        Yields AgentEvent objects for the TUI to render.

        Args:
            user_input: The user's natural language request.
            confirm_callback: An async callable that takes a ToolCall and returns
                            True (confirmed) or False (declined). Required when
                            safety mode is 'confirm'.
        """
        self.messages.append({"role": "user", "content": user_input})

        for step in range(self.config.max_steps):
            # Signal that the model is thinking
            yield AgentEvent(type=EventType.THINKING)

            # Get model response
            try:
                result = await self.engine.chat_completion(
                    messages=self.messages,
                    tools=self.tools.get_tool_schemas(),
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
            except Exception as e:
                logger.exception("Inference error")
                yield AgentEvent(type=EventType.ERROR, data=f"Inference error: {e}")
                return

            # Handle truncated responses (thinking consumed all tokens)
            if result.finish_reason == "length" and not result.tool_calls:
                raw = result.content or ""
                # Check if content is only <think> residue or empty
                stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
                if not stripped:
                    yield AgentEvent(
                        type=EventType.ERROR,
                        data="Response was truncated — the model used all available tokens "
                        "without producing a complete response. Try a simpler request.",
                    )
                    return

            # Case 1: Model wants to call tools
            if result.tool_calls:
                # If the model also provided text (planning/reasoning), emit it
                if result.content:
                    yield AgentEvent(type=EventType.PLANNING, data=result.content)

                for tool_call in result.tool_calls:
                    # Safety classification
                    risk = self.safety.classify_tool_call(
                        tool_call.name, tool_call.arguments
                    )

                    if risk == Risk.BLOCKED:
                        yield AgentEvent(
                            type=EventType.BLOCKED, tool_call=tool_call
                        )
                        self._append_tool_exchange(
                            tool_call,
                            "BLOCKED: This command was blocked by the safety classifier. "
                            "Try an alternative approach.",
                        )
                        continue

                    if risk == Risk.CONFIRM and confirm_callback:
                        yield AgentEvent(
                            type=EventType.CONFIRM_NEEDED, tool_call=tool_call
                        )
                        confirmed = await confirm_callback(tool_call)
                        if not confirmed:
                            self._append_tool_exchange(
                                tool_call,
                                "DECLINED: The user declined to execute this command.",
                            )
                            continue

                    # Execute the tool
                    yield AgentEvent(type=EventType.EXECUTING, tool_call=tool_call)

                    tool_result = await self.tools.execute(
                        tool_call.name, tool_call.arguments
                    )

                    yield AgentEvent(
                        type=EventType.TOOL_RESULT,
                        tool_call=tool_call,
                        tool_result=tool_result,
                    )

                    # Append exchange to conversation history
                    self._append_tool_exchange(
                        tool_call, tool_result.to_message_content()
                    )

                # Continue the loop — model will see tool results and decide next step
                continue

            # Case 2: Model responded with text only (task complete or needs info)
            if result.content:
                self.messages.append({
                    "role": "assistant",
                    "content": result.content,
                })
                yield AgentEvent(type=EventType.RESPONSE, data=result.content)
                return

            # Case 3: Empty response (shouldn't happen, but handle gracefully)
            yield AgentEvent(
                type=EventType.ERROR,
                data="Model returned an empty response.",
            )
            return

        # Hit max steps
        yield AgentEvent(
            type=EventType.RESPONSE,
            data=f"I've reached the maximum number of steps ({self.config.max_steps}). "
            f"Here's what I've done so far. You can continue with a follow-up request.",
        )

    def _append_tool_exchange(self, tool_call: ToolCall, result_content: str) -> None:
        """Append a tool call + result pair to the message history."""
        # Assistant message with tool call
        self.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments),
                },
            }],
        })
        # Tool result message
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result_content,
        })

    def clear_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
