"""Local inference backend using llama-cpp-python.

The model family's tool-call wire format (prompt rendering, native parsing,
bare-JSON recovery, message normalization) lives in the
``natshell.inference.grammars`` package — one module per family (``qwen``,
``mistral``, ``gemma``) sharing the pipeline primitives in
``grammars.common``.  This module owns only the engine: model loading, GPU
selection, context sizing, chat completion, and the thin delegations that
keep the historical API surface intact (tests and ``agent/loop.py`` import
the underscored helpers from here).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from natshell.inference.engine import (
    CompletionResult,
    EngineInfo,
    StreamChunk,
)

# ---------------------------------------------------------------------------
# Per-family modules
# ---------------------------------------------------------------------------
from natshell.inference.grammars import gemma, mistral, qwen  # noqa: E402

# ---------------------------------------------------------------------------
# Family-agnostic helpers (shared pipeline primitives)
# ---------------------------------------------------------------------------
# Re-exported here so the historical import surface
# (``from natshell.inference.local import _THINK_RE, ...``) keeps working.
from natshell.inference.grammars.common import (  # noqa: E402
    CODE_FENCE_JSON_RE,
    THINK_RE,
    THINK_UNCLOSED_RE,
    is_degenerate_output,
)

# ---------------------------------------------------------------------------
# Backwards-compatible aliases (historical names, now implemented in grammars/)
# ---------------------------------------------------------------------------

# Qwen family (the default wire format)
_TOOL_CALL_RE = qwen.TOOL_CALL_RE
_format_tools_for_prompt = qwen.format_tools_for_prompt

# Mistral family
_MISTRAL_TOOL_CALLS_RE = mistral.MISTRAL_TOOL_CALLS_RE
_format_tools_for_prompt_mistral = mistral.format_tools_for_prompt_mistral

# Gemma family
_GEMMA_TOOL_CALL_RE = gemma.GEMMA_TOOL_CALL_RE
_GEMMA_THINK_RE = gemma.GEMMA_THINK_RE
_GEMMA_THINK_UNCLOSED_RE = gemma.GEMMA_THINK_UNCLOSED_RE
_GEMMA_SPECIAL_TOKEN_RE = gemma.GEMMA_SPECIAL_TOKEN_RE
_parse_gemma_tool_args = gemma.parse_gemma_tool_args
_format_gemma_tool_call_text = gemma.format_gemma_tool_call_text
_format_tools_for_prompt_gemma = gemma.format_tools_for_prompt_gemma

# Shared regexes
_CODE_FENCE_JSON_RE = CODE_FENCE_JSON_RE
_THINK_RE = THINK_RE
_THINK_UNCLOSED_RE = THINK_UNCLOSED_RE


def _is_degenerate_output(text: str) -> bool:
    """Detect degenerate repetitive output (re-export, see grammars.common)."""
    return is_degenerate_output(text)


def _detect_model_family(model_path: str) -> str:
    """Detect the model family from the filename.

    Returns "mistral" for Mistral models, "gemma" for Gemma models,
    "qwen" for everything else.
    """
    name = Path(model_path).name.lower()
    if "mistral" in name:
        return "mistral"
    if "gemma" in name:
        return "gemma"
    if "qwen" not in name:
        logger = logging.getLogger(__name__)
        logger.warning(
            "Unknown model family for %r — defaulting to qwen tool format. "
            "If tool calls fail, the model may need a custom parser.",
            Path(model_path).name,
        )
    return "qwen"


def _infer_context_size(model_path: str) -> int:
    """Infer an appropriate context size from the model filename.

    Looks for a parameter-count pattern like '4B', '8B', '1.7B' in the
    filename and maps it to a reasonable context size.
    Falls back to 4096 if no pattern is found.
    """
    name = Path(model_path).name.lower()
    # Mistral Nemo supports 128K; 32K default fits ~11 GB VRAM (integrated GPUs)
    if "mistral" in name and "nemo" in name:
        return 32768
    # Gemma 4: E2B/E4B support 128K, 12B supports 128K, 26B-A4B/31B support 256K.
    # 32K fits comfortably on integrated GPUs (16+ GB shared RAM).
    if "gemma" in name:
        if "e2b" in name or "e4b" in name:
            return 32768
        if "12b" in name:
            return 32768
        if "26b" in name or "31b" in name:
            return 65536
        return 32768
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


logger = logging.getLogger(__name__)


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
        self._configured_main_gpu = main_gpu  # preserve -1 for "auto" display

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

    # ------------------------------------------------------------------
    # Grammar access
    # ------------------------------------------------------------------

    @property
    def grammar(self):
        """This engine's tool-call grammar (chosen by model family)."""
        from natshell.inference.grammars import get_grammar

        return get_grammar(self.model_family)

    # ------------------------------------------------------------------
    # Counting / info
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer."""
        return len(self.llm.tokenize(text.encode("utf-8")))

    def engine_info(self) -> EngineInfo:
        return EngineInfo(
            engine_type="local",
            model_name=Path(self.model_path).name,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            main_gpu=self._configured_main_gpu,  # -1 = auto; resolved value used for llama
            resolved_main_gpu=self.main_gpu,
        )

    # ------------------------------------------------------------------
    # Message normalization (delegated to the family grammar)
    # ------------------------------------------------------------------

    def _convert_gemma_tool_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-format tool messages to Gemma-native text format.

        Delegates to the Gemma grammar's message conversion stage (kept as a
        method for the historical ``engine._convert_gemma_tool_messages``
        call site in tests).  Deliberately does not route through
        ``self.grammar`` so it works on a bare instance (no ``model_family``).
        """
        return gemma.GRAMMAR._convert_tool_messages(messages)

    def _normalize_messages_strict_alternation(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Normalize messages for strict role-alternation (Mistral, Gemma 4).

        Delegates to the shared implementation in ``grammars.common``.
        """
        from natshell.inference.grammars.common import (
            normalize_messages_strict_alternation,
        )

        return normalize_messages_strict_alternation(messages)

    # ------------------------------------------------------------------
    # Chat completion
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> CompletionResult:
        """Run chat completion via llama.cpp. Runs in thread to avoid blocking.

        Tool definitions are rendered by the model family's grammar and
        injected as plain text into the system prompt.
        """
        from natshell.inference.grammars import get_grammar

        grammar = get_grammar(self.model_family)
        # Family-specific message normalization (strict alternation for
        # Mistral and Gemma; identity otherwise)
        messages = grammar.normalize_messages(messages)

        if tools:
            messages = self._inject_tools(messages, tools)

        kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "repeat_penalty": 1.1,
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

    # ── Streaming (R2-1) ─────────────────────────────────────────────────
    # The grammar's parse() pipeline runs once, over the *buffered* full
    # content, exactly as _parse_response does for the blocking path —
    # StreamChunks are only raw text deltas for the TUI to render live.

    async def stream_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        """Stream raw text deltas, then yield the parsed :class:`CompletionResult`.

        llama-cpp-python's ``stream=True`` mode returns an iterator of
        chunk dicts (``choices[0].delta.content`` per chunk); that iterator
        is consumed inside a worker thread (synchronous, like the blocking
        path) so the async generator only awaits one thread hop and yields
        each chunk as it arrives.  The terminal result is produced by the
        identical parse pipeline the blocking ``chat_completion`` uses, run
        on the concatenated content — tool parsing, think-residue stripping
        and degenerate-suppression all behave exactly as before.
        """
        from natshell.inference.grammars import get_grammar

        grammar = get_grammar(self.model_family)
        # Same message normalization + tool injection as chat_completion,
        # so both paths see the same prompt.
        messages = grammar.normalize_messages(messages)
        if tools:
            messages = self._inject_tools(messages, tools)

        def _run_stream():
            chunks: list[str] = []
            finish_reason = "stop"
            usage: dict[str, Any] = {}
            # Bound through an Any-typed callable: the blocking-return
            # TypedDict overload of create_chat_completion does not describe
            # stream=True (a generator of chunk dicts), so annotate via Any
            # to iterate it cleanly.
            stream: Any = self.llm.create_chat_completion
            for item in stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                repeat_penalty=1.1,
                stream=True,
            ):
                if not isinstance(item, dict):
                    continue
                choice = (item.get("choices") or [{}])[0]
                if not isinstance(choice, dict):
                    continue
                text = (choice.get("delta") or {}).get("content")
                if text is None:
                    delta = choice.get("message")
                    if isinstance(delta, dict):
                        text = delta.get("content")
                if text:
                    chunks.append(text)
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
                if isinstance(item.get("usage"), dict) and item["usage"]:
                    usage = item["usage"]
            return chunks, finish_reason, usage

        try:
            chunks, finish_reason, usage = await asyncio.to_thread(_run_stream)
        except ValueError as e:
            err_str = str(e).lower()
            if "context window" in err_str or "exceed" in err_str:
                from natshell.inference.remote import ContextOverflowError

                raise ContextOverflowError(
                    f"Prompt exceeds local model context window ({self.n_ctx} tokens): {e}"
                ) from e
            raise

        for text in chunks:
            yield StreamChunk(text=text)
        # Reuse the full parse pipeline over the buffered content so tool
        # parsing, think-residue stripping and degenerate detection are
        # identical to the non-streaming path.
        yield self._parse_response(
            {
                "choices": [
                    {
                        "message": {"content": "".join(chunks)},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }
        )

    def _inject_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Inject tool definitions into the system message as plain text."""
        from natshell.inference.grammars import get_grammar

        compact = self.n_ctx < 16384
        tool_text = get_grammar(self.model_family).render_tools(tools, compact=compact)

        # Shallow-copy the list and deep-copy only the system message
        messages = list(messages)
        for i, msg in enumerate(messages):
            if msg["role"] == "system":
                messages[i] = {**msg, "content": msg["content"] + "\n\n" + tool_text}
                break

        return messages

    def _parse_response(self, response: dict) -> CompletionResult:
        """Parse llama-cpp-python response into our CompletionResult.

        Delegates to the model family's grammar: structured tool calls first,
        then this family's native syntax, then the family's bare-JSON
        recovery.  Content is cleaned of every family's think blocks, special
        tokens, and tool-call markers; recovered JSON blobs are scrubbed;
        degenerate output (character-repetition collapse) is suppressed.
        """
        choice = response["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        content = message.get("content") or ""
        from natshell.inference.grammars import get_grammar

        grammar = get_grammar(self.model_family)

        tool_calls, content, _fired = grammar.parse(
            content,
            structured=message.get("tool_calls"),
        )

        content = content.strip() or None

        degenerate = False
        if content and is_degenerate_output(content):
            logger.warning(
                "Degenerate output detected (%d chars, dominated by "
                "repeated characters) — suppressing garbage output",
                len(content),
            )
            content = None
            degenerate = True

        usage = response.get("usage", {})
        return CompletionResult(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            degenerate=degenerate,
        )
