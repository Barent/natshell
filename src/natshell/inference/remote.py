"""Remote inference backend — connects to any OpenAI-compatible API (Ollama, vLLM, etc.)."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from natshell.inference.engine import CompletionResult, EngineInfo, ToolCall

logger = logging.getLogger(__name__)


class RemoteEngine:
    """LLM inference via a remote OpenAI-compatible API."""

    def __init__(self, base_url: str, model: str, api_key: str = "", n_ctx: int = 0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.n_ctx = n_ctx
        # Use a short connect timeout but generous read timeout;
        # per-request read timeout is scaled in chat_completion() based on max_tokens.
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
        )
        logger.info(f"Remote engine: {base_url} model={model}")

        # Warn if sending API key over plaintext HTTP to a non-localhost host
        if api_key:
            parsed = urlparse(self.base_url)
            if (parsed.scheme == "http"
                    and parsed.hostname not in ("localhost", "127.0.0.1", "::1")):
                logger.warning(
                    "API key configured over HTTP (not HTTPS) to %s — "
                    "credentials will be sent in plaintext.", base_url
                )

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> CompletionResult:
        """Send a chat completion request to the remote API."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Scale read timeout for large generations: assume ≥10 tok/s + 60s overhead
        read_timeout = max(300.0, max_tokens / 10.0 + 60.0)

        url = f"{self.base_url}/chat/completions"
        response = await self.client.post(
            url, json=payload, headers=headers,
            timeout=httpx.Timeout(connect=30.0, read=read_timeout, write=30.0, pool=30.0),
        )
        response.raise_for_status()

        data = response.json()
        return self._parse_response(data)

    def _parse_response(self, data: dict) -> CompletionResult:
        """Parse OpenAI-format response."""
        choice = data["choices"][0]
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

        # Strip <think> tags from content (Qwen3 models produce these)
        if content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip() or None

        usage = data.get("usage", {})
        return CompletionResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def engine_info(self) -> EngineInfo:
        return EngineInfo(
            engine_type="remote",
            model_name=self.model,
            base_url=self.base_url,
            n_ctx=self.n_ctx,
        )

    async def close(self) -> None:
        await self.client.aclose()
