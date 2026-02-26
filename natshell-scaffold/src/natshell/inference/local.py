"""Local inference backend using llama-cpp-python."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any

from natshell.inference.engine import CompletionResult, ToolCall

logger = logging.getLogger(__name__)


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

        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count() or 4,
            n_gpu_layers=n_gpu_layers,
            chat_format="chatml-function-calling",
            verbose=False,
        )
        logger.info(f"Loaded model: {model_path} (ctx={n_ctx}, threads={n_threads})")

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
        """Parse llama-cpp-python response into our CompletionResult."""
        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        content = message.get("content")
        tool_calls: list[ToolCall] = []

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

        return CompletionResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
        )
