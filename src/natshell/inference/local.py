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


def _infer_context_size(model_path: str) -> int:
    """Infer an appropriate context size from the model filename.

    Looks for a parameter-count pattern like '4B', '8B', '1.7B' in the
    filename.  Models with ≤4B parameters get 4096; larger models get 8192.
    Falls back to 4096 if no pattern is found.
    """
    name = Path(model_path).name.lower()
    match = re.search(r"(\d+(?:\.\d+)?)b", name)
    if match:
        param_billions = float(match.group(1))
        if param_billions > 4:
            return 8192
    return 4096


class LocalEngine:
    """LLM inference via bundled llama.cpp (llama-cpp-python)."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 0,
        n_threads: int = 0,
        n_gpu_layers: int = 0,
        main_gpu: int = -1,
    ) -> None:
        from llama_cpp import Llama

        if n_ctx <= 0:
            n_ctx = _infer_context_size(model_path)

        # Resolve main_gpu: -1 means auto-detect best GPU
        from natshell.gpu import best_gpu_index, gpu_backend_available

        resolved_gpu = main_gpu
        if main_gpu == -1 and n_gpu_layers != 0 and gpu_backend_available():
            resolved_gpu = best_gpu_index()
            if resolved_gpu != 0:
                logger.info(f"Auto-selected GPU device {resolved_gpu}")

        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.main_gpu = resolved_gpu

        llama_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads or os.cpu_count() or 4,
            "n_gpu_layers": n_gpu_layers,
            "chat_format": "chatml-function-calling",
            "verbose": False,
        }
        if resolved_gpu > 0 and gpu_backend_available():
            llama_kwargs["main_gpu"] = resolved_gpu

        self.llm = Llama(**llama_kwargs)

        if n_gpu_layers != 0:
            try:
                from llama_cpp import llama_supports_gpu_offload
                if not llama_supports_gpu_offload():
                    logger.warning("GPU layers requested but llama-cpp-python has no GPU support — running on CPU")
            except ImportError:
                pass
        logger.info(f"Loaded model: {model_path} (ctx={n_ctx}, threads={n_threads}, main_gpu={resolved_gpu})")

    def engine_info(self) -> EngineInfo:
        return EngineInfo(
            engine_type="local",
            model_name=Path(self.model_path).name,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            main_gpu=self.main_gpu,
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
