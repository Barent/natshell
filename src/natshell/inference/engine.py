"""Inference engine abstraction — protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """A tool call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class CompletionResult:
    """Result from a chat completion request."""

    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


@dataclass
class EngineInfo:
    """Metadata about the current inference engine."""

    engine_type: str  # "local" or "remote"
    model_name: str = ""
    base_url: str = ""
    n_ctx: int = 0
    n_gpu_layers: int = 0


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
