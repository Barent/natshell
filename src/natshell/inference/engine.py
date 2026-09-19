"""Inference engine abstraction — protocol and shared types."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class StreamChunk:
    """One streamed delta from an incremental completion.

    ``text`` is raw model text for this chunk — prose, thinking residue or
    tool-call markup; the caller decides which it is.  The *final*
    :class:`CompletionResult` (parsed tool calls, etc.) still comes from the
    engine's buffered/full parse, never from piecing chunks together.
    """

    text: str = ""


@dataclass
class CompletionResult:
    """Result from a chat completion request."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    degenerate: bool = False


@dataclass
class EngineInfo:
    """Metadata about the current inference engine."""

    engine_type: str  # "local" or "remote"
    model_name: str = ""
    base_url: str = ""
    n_ctx: int = 0
    n_gpu_layers: int = 0
    main_gpu: int | None = None          # configured value: -1 = auto-detect
    resolved_main_gpu: int | None = None  # actual device index used for inference


class InferenceEngine(Protocol):
    """Protocol for LLM inference backends."""

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> CompletionResult: ...

    def engine_info(self) -> EngineInfo: ...


@runtime_checkable
class StreamingEngine(Protocol):
    """An :class:`InferenceEngine` that can stream tokens (R2-1).

    ``stream_completion`` is an async generator: it yields zero or more
    :class:`StreamChunk` items (raw model text deltas, in arrival order)
    and finally yields the :class:`CompletionResult` — the same result a
    blocking ``chat_completion`` would have produced for identical inputs
    (the engine runs its normal parse pipeline over the *buffered* content,
    never over pieced-together deltas).  Callers feature-detect streaming
    support with ``isinstance(engine, StreamingEngine)``; engines that do
    not support it (e.g. a bare RemoteEngine) are not conforming.
    """

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> CompletionResult: ...

    def engine_info(self) -> EngineInfo: ...

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncIterator[StreamChunk | CompletionResult]: ...
