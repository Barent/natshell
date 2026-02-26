"""Local inference backend using llama-cpp-python."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any

from natshell.inference.engine import CompletionResult, EngineInfo, ToolCall

logger = logging.getLogger(__name__)

# Regex to match <tool_call>{"name": ..., "arguments": ...}</tool_call> blocks
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
# Regex to match <think>...</think> blocks (including empty ones)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Regex to match unclosed <think> blocks (truncated responses)
_THINK_UNCLOSED_RE = re.compile(r"<think>(?:(?!</think>).)*$", re.DOTALL)


class LocalEngine:
    """LLM inference via bundled llama.cpp (llama-cpp-python)."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 8192,
        n_threads: int = 0,
        n_gpu_layers: int = 0,
    ) -> None:
        from llama_cpp import Llama

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count() or 4,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        logger.info(f"Loaded model: {model_path} (ctx={n_ctx}, threads={n_threads})")

    def engine_info(self) -> EngineInfo:
        return EngineInfo(
            engine_type="local",
            model_name=Path(self.model_path).name,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
        )

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> CompletionResult:
        """Run chat completion via llama.cpp. Runs in thread to avoid blocking."""
        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # llama-cpp-python's create_chat_completion is synchronous
        response = await asyncio.to_thread(self.llm.create_chat_completion, **kwargs)

        return self._parse_response(response)

    def _parse_response(self, response: dict) -> CompletionResult:
        """Parse llama-cpp-python response into our CompletionResult.

        Qwen3 models output <tool_call> XML tags in the content field rather
        than using the structured tool_calls field. This method extracts those
        tool calls and strips <think> blocks from content.
        """
        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        content = message.get("content") or ""
        tool_calls: list[ToolCall] = []

        # First, check for structured tool_calls (standard OpenAI format)
        if message.get("tool_calls"):
            for tc in message["tool_calls"]:
                func = tc.get("function", {})
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                tool_calls.append(ToolCall(
                    id=tc.get("id", str(uuid.uuid4())[:8]),
                    name=func.get("name", ""),
                    arguments=args,
                ))

        # Parse <tool_call> XML tags from content (Qwen3 style)
        if not tool_calls:
            for match in _TOOL_CALL_RE.finditer(content):
                try:
                    parsed = json.loads(match.group(1))
                    name = parsed.get("name", "")
                    arguments = parsed.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    tool_calls.append(ToolCall(
                        id=str(uuid.uuid4())[:8],
                        name=name,
                        arguments=arguments,
                    ))
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Failed to parse tool_call from content: %s", match.group(0))

        # Strip <think> and <tool_call> tags from content
        content = _THINK_RE.sub("", content)
        content = _THINK_UNCLOSED_RE.sub("", content)  # handle truncated think blocks
        content = _TOOL_CALL_RE.sub("", content)
        content = content.strip() or None

        return CompletionResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
