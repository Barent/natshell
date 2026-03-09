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
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Regex to match [TOOL_CALLS] JSON array (Mistral style)
_MISTRAL_TOOL_CALLS_RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)
# Regex to match <think>...</think> blocks (including empty ones)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# Regex to match unclosed <think> blocks (truncated responses)
_THINK_UNCLOSED_RE = re.compile(r"<think>(?:(?!</think>).)*$", re.DOTALL)


def _detect_model_family(model_path: str) -> str:
    """Detect the model family from the filename.

    Returns "mistral" for Mistral models, "qwen" for everything else.
    """
    name = Path(model_path).name.lower()
    if "mistral" in name:
        return "mistral"
    return "qwen"


def _infer_context_size(model_path: str) -> int:
    """Infer an appropriate context size from the model filename.

    Looks for a parameter-count pattern like '4B', '8B', '1.7B' in the
    filename and maps it to a reasonable context size.
    Falls back to 4096 if no pattern is found.
    """
    name = Path(model_path).name.lower()
    # Mistral Nemo supports 128K; 8K default for 8 GB VRAM
    if "mistral" in name and "nemo" in name:
        return 8192
    match = re.search(r"(\d+(?:\.\d+)?)b", name)
    if match:
        param_billions = float(match.group(1))
        if param_billions <= 1:
            return 2048
        elif param_billions <= 4:
            return 4096
        elif param_billions <= 8:
            return 8192
        elif param_billions <= 14:
            return 16384
        else:
            return 32768
    return 4096


def _format_tools_for_prompt(
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
    if compact:
        lines = [
            "# Available Tools",
            'Call a tool: <tool_call>{"name": "...", "arguments": {...}}</tool_call>',
            "",
        ]
    else:
        lines = [
            "# Available Tools",
            "",
            "You MUST use tools to perform actions. To call a tool, output:",
            "",
            "<tool_call>",
            '{"name": "tool_name", "arguments": {"param": "value"}}',
            "</tool_call>",
            "",
        ]

    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"## {name}")
        if compact:
            # First sentence only
            first_sentence = desc.split(". ")[0]
            if not first_sentence.endswith("."):
                first_sentence += "."
            lines.append(first_sentence)
        else:
            lines.append(desc)

        props = params.get("properties", {})
        required = params.get("required", [])
        if props:
            if not compact:
                lines.append("Parameters:")
            for pname, pdef in props.items():
                req = ", required" if pname in required else ""
                ptype = pdef.get("type", "")
                if compact:
                    enum_vals = pdef.get("enum")
                    if enum_vals:
                        enum_str = "|".join(str(v) for v in enum_vals)
                        lines.append(f"- {pname} ({enum_str}{req})")
                    else:
                        lines.append(f"- {pname} ({ptype}{req})")
                else:
                    pdesc = pdef.get("description", "")
                    lines.append(f"- {pname} ({ptype}{req}): {pdesc}")
        lines.append("")

    return "\n".join(lines)


def _format_tools_for_prompt_mistral(
    tools: list[dict[str, Any]], *, compact: bool = False
) -> str:
    """Format tool schemas for Mistral models.

    Mistral models use [TOOL_CALLS] followed by a JSON array instead of XML tags.

    Args:
        tools: Tool schemas in OpenAI-compatible format.
        compact: When True, use abbreviated descriptions and parameter lists.
    """
    if compact:
        lines = [
            "# Available Tools",
            'Call a tool: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]',
            "",
        ]
    else:
        lines = [
            "# Available Tools",
            "",
            "You MUST use tools to perform actions. To call a tool, output:",
            "",
            '[TOOL_CALLS] [{"name": "tool_name", "arguments": {"param": "value"}}]',
            "",
            "You can call multiple tools at once by including multiple objects in the array.",
            "",
        ]

    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        lines.append(f"## {name}")
        if compact:
            first_sentence = desc.split(". ")[0]
            if not first_sentence.endswith("."):
                first_sentence += "."
            lines.append(first_sentence)
        else:
            lines.append(desc)

        props = params.get("properties", {})
        required = params.get("required", [])
        if props:
            if not compact:
                lines.append("Parameters:")
            for pname, pdef in props.items():
                req = ", required" if pname in required else ""
                ptype = pdef.get("type", "")
                if compact:
                    enum_vals = pdef.get("enum")
                    if enum_vals:
                        enum_str = "|".join(str(v) for v in enum_vals)
                        lines.append(f"- {pname} ({enum_str}{req})")
                    else:
                        lines.append(f"- {pname} ({ptype}{req})")
                else:
                    pdesc = pdef.get("description", "")
                    lines.append(f"- {pname} ({ptype}{req}): {pdesc}")
        lines.append("")

    return "\n".join(lines)


class LocalEngine:
    """LLM inference via bundled llama.cpp (llama-cpp-python)."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 0,
        n_threads: int = 0,
        n_gpu_layers: int = 0,
        main_gpu: int = -1,
        prompt_cache: bool = True,
        prompt_cache_mb: int = 256,
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
        self.model_family = _detect_model_family(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.main_gpu = resolved_gpu

        llama_kwargs: dict[str, Any] = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_threads": n_threads or os.cpu_count() or 4,
            "n_gpu_layers": n_gpu_layers,
            "verbose": False,
        }
        if resolved_gpu > 0 and gpu_backend_available():
            llama_kwargs["main_gpu"] = resolved_gpu

        self.llm = Llama(**llama_kwargs)

        # Enable RAM-based prompt cache for faster repeated prefixes
        if prompt_cache:
            try:
                from llama_cpp import LlamaRAMCache

                self.llm.set_cache(
                    LlamaRAMCache(capacity_bytes=prompt_cache_mb * 1024 * 1024)
                )
                logger.info("Prompt cache enabled (%d MB)", prompt_cache_mb)
            except (ImportError, AttributeError, Exception) as exc:
                logger.debug("Prompt cache unavailable: %s", exc)

        if n_gpu_layers != 0:
            try:
                from llama_cpp import llama_supports_gpu_offload

                if not llama_supports_gpu_offload():
                    logger.warning(
                        "GPU layers requested but llama-cpp-python"
                        " has no GPU support — running on CPU"
                    )
            except ImportError:
                pass
        logger.info(
            "Loaded model: %s (ctx=%d, threads=%d, main_gpu=%d)",
            model_path, n_ctx, n_threads, resolved_gpu,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.llm.tokenize(text.encode("utf-8")))

    def engine_info(self) -> EngineInfo:
        return EngineInfo(
            engine_type="local",
            model_name=Path(self.model_path).name,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            main_gpu=self.main_gpu,
        )

    def _normalize_messages_for_mistral(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge mid-conversation system messages into the initial system message.

        Mistral's chat template only allows a single system message at position 0.
        Any subsequent system messages (e.g. planning/analysis mode hints injected
        by the agent loop) must be folded into the initial system message to avoid
        the "Only user and assistant roles are supported" template error.
        """
        result: list[dict[str, Any]] = []
        system_extras: list[str] = []

        for msg in messages:
            if msg["role"] == "system" and result:
                # Subsequent system message — collect content to merge later
                system_extras.append(msg["content"])
            else:
                result.append(msg)

        if system_extras and result and result[0]["role"] == "system":
            merged = result[0]["content"] + "\n\n" + "\n\n".join(system_extras)
            result[0] = {**result[0], "content": merged}

        return result

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> CompletionResult:
        """Run chat completion via llama.cpp. Runs in thread to avoid blocking.

        Tool definitions are injected as plain text into the system prompt
        rather than relying on llama-cpp-python's tool handling, which doesn't
        work correctly with Qwen3 models.
        """
        if self.model_family == "mistral":
            messages = self._normalize_messages_for_mistral(messages)

        if tools:
            messages = self._inject_tools(messages, tools)

        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # llama-cpp-python's create_chat_completion is synchronous
        try:
            response = await asyncio.to_thread(self.llm.create_chat_completion, **kwargs)
        except ValueError as e:
            err_str = str(e).lower()
            if "context window" in err_str or "exceed" in err_str:
                from natshell.inference.remote import ContextOverflowError

                raise ContextOverflowError(
                    f"Prompt exceeds local model context window ({self.n_ctx} tokens): {e}"
                ) from e
            raise

        return self._parse_response(response)

    def _inject_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Inject tool definitions into the system message as plain text."""
        compact = self.n_ctx <= 16384
        if self.model_family == "mistral":
            tool_text = _format_tools_for_prompt_mistral(tools, compact=compact)
        else:
            tool_text = _format_tools_for_prompt(tools, compact=compact)

        # Shallow-copy the list and deep-copy only the system message
        messages = list(messages)
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                messages[i] = {**msg, "content": msg["content"] + "\n\n" + tool_text}
                break

        return messages

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

                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", str(uuid.uuid4())[:8]),
                        name=func.get("name", ""),
                        arguments=args,
                    )
                )

        # Parse <tool_call> XML tags from content (Qwen3 style)
        if not tool_calls:
            for match in _TOOL_CALL_RE.finditer(content):
                try:
                    parsed = json.loads(match.group(1))
                    name = parsed.get("name", "")
                    arguments = parsed.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    tool_calls.append(
                        ToolCall(
                            id=str(uuid.uuid4())[:8],
                            name=name,
                            arguments=arguments,
                        )
                    )
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Failed to parse tool_call from content: %s", match.group(0))

        # Parse [TOOL_CALLS] JSON array from content (Mistral style)
        if not tool_calls:
            mistral_match = _MISTRAL_TOOL_CALLS_RE.search(content)
            if mistral_match:
                try:
                    calls = json.loads(mistral_match.group(1))
                    for call in calls:
                        name = call.get("name", "")
                        arguments = call.get("arguments", {})
                        if isinstance(arguments, str):
                            arguments = json.loads(arguments)
                        tool_calls.append(
                            ToolCall(
                                id=str(uuid.uuid4())[:8],
                                name=name,
                                arguments=arguments,
                            )
                        )
                except (json.JSONDecodeError, KeyError):
                    logger.warning(
                        "Failed to parse [TOOL_CALLS] from content: %s",
                        mistral_match.group(0),
                    )

        # Strip <think>, <tool_call>, and [TOOL_CALLS] blocks from content
        content = _THINK_RE.sub("", content)
        content = _THINK_UNCLOSED_RE.sub("", content)  # handle truncated think blocks
        content = _TOOL_CALL_RE.sub("", content)
        content = _MISTRAL_TOOL_CALLS_RE.sub("", content)
        content = content.strip() or None

        usage = response.get("usage", {})
        return CompletionResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )
