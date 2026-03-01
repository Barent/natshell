"""Remote inference backend — connects to any OpenAI-compatible API (Ollama, vLLM, etc.)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from typing import Any
from urllib.parse import urlparse

import httpx

from natshell.inference.engine import CompletionResult, EngineInfo, ToolCall

logger = logging.getLogger(__name__)

# Retry config for transient failures
_MAX_RETRIES = 2
_RETRY_BACKOFF = 1.0  # seconds; doubles each retry

# Known error strings that indicate the prompt exceeded the model's context window.
# Covers Ollama, OpenAI, vLLM, and other OpenAI-compatible servers.
_CONTEXT_OVERFLOW_PATTERNS = (
    "context length",
    "context_length",
    "maximum context",
    "token limit",
    "too many tokens",
    "prompt is too long",
    "num_ctx",
    "request too large",
)


class ContextOverflowError(ConnectionError):
    """The remote API rejected the request because the prompt exceeds the model's context window."""


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
            if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
                logger.warning(
                    "API key configured over HTTP (not HTTPS) to %s — "
                    "credentials will be sent in plaintext.",
                    base_url,
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

        # Scale read timeout for large generations: assume ≥10 tok/s + 60s overhead.
        # Pool timeout must match — the server may be busy processing before it
        # starts sending the response.
        read_timeout = max(300.0, max_tokens / 10.0 + 60.0)
        req_timeout = httpx.Timeout(
            connect=30.0, read=read_timeout, write=30.0, pool=read_timeout,
        )

        url = f"{self.base_url}/chat/completions"

        # Retry on transient connection failures with exponential backoff
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = await self.client.post(
                    url, json=payload, headers=headers, timeout=req_timeout,
                )
                response.raise_for_status()
                break  # success
            except httpx.ConnectError as e:
                last_exc = ConnectionError(
                    f"Cannot connect to {self.base_url} — is the server running?"
                )
                last_exc.__cause__ = e
            except httpx.ConnectTimeout as e:
                last_exc = ConnectionError(
                    f"Connection to {self.base_url} timed out."
                )
                last_exc.__cause__ = e
            except (httpx.ReadTimeout, httpx.PoolTimeout) as e:
                last_exc = ConnectionError(
                    f"Request to {self.base_url} timed out waiting for response "
                    f"({type(e).__name__})."
                )
                last_exc.__cause__ = e
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                body = e.response.text[:200] if e.response else ""
                # Detect context window overflow (400/413 with known patterns)
                if status in (400, 413):
                    body_lower = body.lower()
                    if any(pat in body_lower for pat in _CONTEXT_OVERFLOW_PATTERNS):
                        raise ContextOverflowError(
                            f"Prompt exceeds model context window (HTTP {status}): {body}"
                        ) from e
                # Only retry on transient server errors (502/503/504)
                if status not in (502, 503, 504):
                    raise ConnectionError(
                        f"Remote API error {status}: {body}"
                    ) from e
                last_exc = ConnectionError(
                    f"Remote API error {status}: {body}"
                )
                last_exc.__cause__ = e

            # If we have retries left, back off and retry
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Remote request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, _MAX_RETRIES + 1, last_exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                raise last_exc  # type: ignore[misc]

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

                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", str(uuid.uuid4())[:8]),
                        name=func.get("name", ""),
                        arguments=args,
                    )
                )

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
