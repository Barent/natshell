"""Core ReAct agent loop — plans, executes tools, observes, repeats."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from natshell.agent.context import SystemContext
from natshell.agent.context_manager import ContextManager
from natshell.agent.system_prompt import build_system_prompt
from natshell.config import AgentConfig, MemoryConfig, ModelConfig, PromptConfig
from natshell.inference.engine import CompletionResult, InferenceEngine, ToolCall
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools import edit_file as _edit_file_mod
from natshell.tools import execute_shell as _exec_shell_mod
from natshell.tools import read_file as _read_file_mod
from natshell.tools.execute_shell import needs_sudo_password as _needs_sudo_password
from natshell.tools.file_tracker import reset_tracker
from natshell.tools.limits import ToolLimits
from natshell.tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

_PLAN_REQUEST_RE = re.compile(
    r"\b(?:create|write|make|draft|update|build)\b.{0,30}\bplan\b"
    r"|\bplan\b.{0,30}\b(?:for|how|what)\b"
    r"|\bplan\s+to\s+(?:update|fix|refactor|migrate|implement|add|remove|change|deploy|install|configure|set\s*up|build|create|upgrade)\b",
    re.IGNORECASE,
)


def _is_plan_request(text: str) -> bool:
    """Detect if the user is asking the model to create/write a plan."""
    return bool(_PLAN_REQUEST_RE.search(text))


_ANALYSIS_REQUEST_RE = re.compile(
    r"\b(?:review|audit|analyze|examine|inspect)\b.{0,40}\b(?:code|codebase|security|module|implementation|PR|pull\s*request|diff|repository|repo)\b"
    r"|\b(?:code|security|codebase)\s+(?:review|audit|analysis)\b",
    re.IGNORECASE,
)


def _is_analysis_request(text: str) -> bool:
    """Detect if the user is asking for a code review, audit, or analysis."""
    return bool(_ANALYSIS_REQUEST_RE.search(text))


class EventType(Enum):
    THINKING = "thinking"
    PLANNING = "planning"  # Model's text before tool calls
    EXECUTING = "executing"  # About to run a tool
    TOOL_RESULT = "tool_result"  # Result from a tool
    CONFIRM_NEEDED = "confirm_needed"  # Awaiting user confirmation
    BLOCKED = "blocked"  # Command was blocked
    RESPONSE = "response"  # Final text response from model
    ERROR = "error"  # Something went wrong
    RUN_STATS = "run_stats"  # Cumulative stats for the full agent run
    QUEUED_MESSAGE = "queued_message"  # User message injected mid-run
    PLAN_STEP = "plan_step"  # Plan step divider (start/update)
    PLAN_COMPLETE = "plan_complete"  # Entire plan finished


@dataclass
class AgentEvent:
    """An event yielded by the agent loop for the TUI to render."""

    type: EventType
    data: Any = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    metrics: dict[str, Any] | None = None


def _build_metrics(result: CompletionResult, elapsed_ms: int) -> dict[str, Any]:
    """Build a metrics dict from inference result and timing."""
    metrics: dict[str, Any] = {"response_time_ms": elapsed_ms}
    if result.completion_tokens:
        metrics["completion_tokens"] = result.completion_tokens
        if elapsed_ms > 0:
            metrics["tokens_per_sec"] = result.completion_tokens / (elapsed_ms / 1000)
    if result.prompt_tokens:
        metrics["prompt_tokens"] = result.prompt_tokens
    return metrics


def _build_run_stats(
    steps: int,
    total_wall_ms: int,
    total_inference_ms: int,
    total_prompt_tokens: int,
    total_completion_tokens: int,
) -> dict[str, Any]:
    """Build cumulative stats for an entire agent run."""
    stats: dict[str, Any] = {
        "steps": steps,
        "total_wall_ms": total_wall_ms,
        "total_inference_ms": total_inference_ms,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
    }
    total_tokens = total_prompt_tokens + total_completion_tokens
    if total_tokens:
        stats["total_tokens"] = total_tokens
    if total_inference_ms > 0 and total_completion_tokens:
        stats["avg_tokens_per_sec"] = total_completion_tokens / (total_inference_ms / 1000)
    return stats


class AgentLoop:
    """The ReAct agent loop — the brain of NatShell."""

    def __init__(
        self,
        engine: InferenceEngine,
        tools: ToolRegistry,
        safety: SafetyClassifier,
        config: AgentConfig,
        fallback_config: ModelConfig | None = None,
        prompt_config: PromptConfig | None = None,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        self.engine = engine
        self.tools = tools
        self.safety = safety
        self.config = config
        self.fallback_config = fallback_config
        self._prompt_config = prompt_config
        self._memory_config = memory_config or MemoryConfig()
        self._system_context: SystemContext | None = None
        self.messages: list[dict[str, Any]] = []
        self._context_manager: ContextManager | None = None
        self._max_tokens: int = config.max_tokens
        # Edit failure tracking
        self._edit_failures: int = 0
        self._edit_successes: int = 0
        self._completion_warning_sent: bool = False
        # Repetitive read detection
        self._read_counts: dict[str, int] = {}
        # Repetitive URL fetch detection (cumulative, not just consecutive)
        self._fetch_url_counts: dict[str, int] = {}
        # Command-family repetition detection (e.g., 10+ `du` calls)
        self._cmd_family_counts: dict[str, int] = {}
        # Duplicate tool call detection
        self._last_tool_key: str = ""
        self._consecutive_dupes: int = 0
        # Similar-command detection (strips flags/pipes to find semantic dupes)
        self._similar_cmd_key: str = ""
        self._consecutive_similar: int = 0
        # Context overflow recovery guard
        self._context_recovery_attempted: bool = False
        # Message queue for mid-run user input
        self._message_queue: asyncio.Queue[str] = asyncio.Queue()
        # Context-window-based tool filter (set in _setup_context_manager)
        self._context_tool_filter: set[str] | None = None

    def initialize(self, system_context: SystemContext) -> None:
        """Build the system prompt and initialize conversation."""
        self._system_context = system_context
        try:
            n_ctx = self.engine.engine_info().n_ctx or 4096
        except (AttributeError, TypeError):
            n_ctx = 4096
        compact = n_ctx < 16384

        # Working memory injection
        working_memory: str | None = None
        memory_path_str = ""
        effective_chars = self._memory_config.max_chars
        if self._memory_config.enabled:
            from natshell.agent.working_memory import (
                effective_memory_chars,
                load_working_memory,
                memory_file_path,
                should_inject_memory,
            )

            mem_path = memory_file_path(Path.cwd())
            memory_path_str = str(mem_path)
            if should_inject_memory(n_ctx, self._memory_config.min_ctx):
                effective_chars = effective_memory_chars(n_ctx, self._memory_config.max_chars)
                mem = load_working_memory(Path.cwd(), effective_chars)
                if mem is not None:
                    working_memory = mem.content

        system_prompt = build_system_prompt(
            system_context,
            compact=compact,
            prompt_config=self._prompt_config,
            working_memory=working_memory,
            memory_path=memory_path_str,
            max_memory_chars=effective_chars,
        )
        self.messages = [{"role": "system", "content": system_prompt}]
        self._setup_context_manager()

    def reload_working_memory(self) -> str | None:
        """Re-read agents.md and update the system prompt in-place.

        Returns the memory content if loaded, or None.
        """
        if not self.messages or not self._system_context:
            return None

        from natshell.agent.working_memory import (
            load_working_memory,
            memory_file_path,
            should_inject_memory,
        )

        try:
            n_ctx = self.engine.engine_info().n_ctx or 4096
        except (AttributeError, TypeError):
            n_ctx = 4096

        if not should_inject_memory(n_ctx, self._memory_config.min_ctx):
            return None

        from natshell.agent.working_memory import effective_memory_chars

        effective_chars = effective_memory_chars(n_ctx, self._memory_config.max_chars)
        mem = load_working_memory(Path.cwd(), effective_chars)
        content = mem.content if mem else None
        mem_path = str(memory_file_path(Path.cwd()))

        compact = n_ctx < 16384
        system_prompt = build_system_prompt(
            self._system_context,
            compact=compact,
            prompt_config=self._prompt_config,
            working_memory=content,
            memory_path=mem_path,
            max_memory_chars=effective_chars,
        )
        self.messages[0] = {"role": "system", "content": system_prompt}
        return content

    async def swap_engine(self, new_engine: InferenceEngine) -> None:
        """Replace the inference engine at runtime. Clears conversation history."""
        old_engine = self.engine
        self.engine = new_engine
        self.clear_history()
        if self._system_context:
            self.initialize(self._system_context)
        if hasattr(old_engine, "close"):
            await old_engine.close()

    def _effective_max_tokens(self, n_ctx: int) -> int:
        """Scale max_tokens based on context window size.

        Uses 25% of the context window (capped at 65536).  For small context
        windows (≤16K), always uses the 25% scaling to prevent the config
        default (8192) from consuming the entire context and starving the
        prompt budget.  For larger contexts, the configured value is used as
        a minimum floor.
        """
        scaled = min(n_ctx // 4, 65536)
        if n_ctx <= 16384:
            # Small local models: enforce 25% cap regardless of config
            return scaled
        return max(self.config.max_tokens, scaled)

    _DEFAULT_MAX_STEPS = 15
    _CONTEXT_PRESSURE_THRESHOLD = 0.75

    def _effective_max_steps(self, n_ctx: int) -> int:
        """Scale max_steps based on context window size.

        Larger models with bigger context windows handle more complex multi-step
        tasks.  Only auto-scales when the configured value is the default (15);
        an explicit user override is respected as-is.
        """
        if self.config.max_steps != self._DEFAULT_MAX_STEPS:
            return self.config.max_steps
        if n_ctx >= 1048576:
            return 200
        elif n_ctx >= 524288:
            return 150
        elif n_ctx >= 262144:
            return 120
        elif n_ctx >= 131072:
            return 60
        elif n_ctx >= 32768:
            return 50
        elif n_ctx >= 16384:
            return 35
        elif n_ctx >= 8192:
            return 25
        return self._DEFAULT_MAX_STEPS

    def _effective_max_output_chars(self, n_ctx: int) -> int:
        """Scale shell output truncation with context window."""
        if n_ctx >= 1048576:
            return 128000
        elif n_ctx >= 524288:
            return 96000
        elif n_ctx >= 262144:
            return 64000
        elif n_ctx >= 131072:
            return 32000
        elif n_ctx >= 65536:
            return 16000
        elif n_ctx >= 32768:
            return 12000
        elif n_ctx >= 16384:
            return 8000
        return 4000

    def _effective_read_file_lines(self, n_ctx: int) -> int:
        """Scale read_file default line count with context window."""
        if n_ctx >= 1048576:
            return 8000
        elif n_ctx >= 524288:
            return 6000
        elif n_ctx >= 262144:
            return 4000
        elif n_ctx >= 131072:
            return 3000
        elif n_ctx >= 65536:
            return 2000
        elif n_ctx >= 32768:
            return 1000
        elif n_ctx >= 16384:
            return 500
        return 200

    def enqueue_message(self, text: str) -> None:
        """Queue a user message for injection between agent steps."""
        self._message_queue.put_nowait(text)

    def _drain_queued_messages(self) -> list[str]:
        """Drain all queued messages, returning them in order."""
        messages: list[str] = []
        while True:
            try:
                messages.append(self._message_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return messages

    @staticmethod
    def _resolve_path(path: str) -> str:
        """Canonicalize a path for tracking (mirrors file_tracker)."""
        return str(Path(path).expanduser().resolve())

    @staticmethod
    def _normalize_shell_cmd(cmd: str) -> str:
        """Strip flags and pipes to detect semantically duplicate commands.

        ``grep -o "Check" file | head -1`` and ``grep "Check" file``
        both normalise to ``grep Check file``, so the similar-command
        detector can spot the pattern even when the model tweaks flags.
        """
        base = cmd.split("|")[0].strip()
        try:
            tokens = shlex.split(base)
        except ValueError:
            tokens = base.split()
        core = [t for t in tokens if not t.startswith("-")]
        return " ".join(core)

    def _setup_context_manager(self) -> None:
        """Create a ContextManager sized to the current engine's context window."""
        try:
            info = self.engine.engine_info()
            n_ctx = info.n_ctx or 4096
        except (AttributeError, TypeError):
            # Gracefully handle mock engines or engines without engine_info
            n_ctx = 4096

        self._max_tokens = self._effective_max_tokens(n_ctx)
        self._max_steps = self._effective_max_steps(n_ctx)

        # Limit tool set for small context windows to reduce token overhead
        # and improve tool selection accuracy for smaller models
        if n_ctx <= 8192:
            from natshell.tools.registry import SMALL_CONTEXT_TOOLS

            self._context_tool_filter = SMALL_CONTEXT_TOOLS
            logger.info(
                "Small context window (%d tokens) — limiting to %d core tools",
                n_ctx,
                len(SMALL_CONTEXT_TOOLS),
            )
        else:
            self._context_tool_filter = None

        # Scale tool limits with context window
        max_output = self._effective_max_output_chars(n_ctx)
        read_lines = self._effective_read_file_lines(n_ctx)
        _exec_shell_mod.configure_limits(max_output)
        _read_file_mod.configure_limits(read_lines)
        edit_limits = ToolLimits(max_output_chars=max_output, read_file_lines=read_lines)
        _edit_file_mod.set_limits(edit_limits)
        self.tools.limits.max_output_chars = max_output
        self.tools.limits.read_file_lines = read_lines

        if self._max_tokens != self.config.max_tokens:
            logger.info(
                "Scaled max_tokens %d → %d for %d-token context window",
                self.config.max_tokens,
                self._max_tokens,
                n_ctx,
            )
        if self._max_steps != self.config.max_steps:
            logger.info(
                "Scaled max_steps %d → %d for %d-token context window",
                self.config.max_steps,
                self._max_steps,
                n_ctx,
            )

        tokenizer_fn = None
        # Only use the tokenizer if it's explicitly defined (not auto-generated by mocks)
        if "count_tokens" in dir(type(self.engine)):
            tokenizer_fn = self.engine.count_tokens

        # Estimate tool definition token overhead (injected into system prompt by engine)
        tool_schemas = self.tools.get_tool_schemas(allowed=self._context_tool_filter)
        if tool_schemas:
            from natshell.inference.local import _format_tools_for_prompt

            compact = n_ctx < 16384
            tool_text = _format_tools_for_prompt(tool_schemas, compact=compact)
            if tokenizer_fn:
                try:
                    self._tool_token_overhead = tokenizer_fn(tool_text)
                except Exception:
                    self._tool_token_overhead = len(tool_text) // 4
            else:
                self._tool_token_overhead = len(tool_text) // 4
        else:
            self._tool_token_overhead = 0

        response_reserve = self._max_tokens
        tool_reserve = max(
            self.config.context_reserve or 800,
            self._tool_token_overhead + 200,
        )
        budget = n_ctx - response_reserve - tool_reserve
        budget = max(budget, 1024)  # minimum viable budget

        self._context_manager = ContextManager(
            context_budget=budget,
            tokenizer_fn=tokenizer_fn,
        )

    async def handle_user_message(
        self,
        user_input: str,
        confirm_callback=None,
        password_callback=None,
        tool_filter: set[str] | None = None,
        skip_intent_detection: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """
        Process a user message through the full agent loop.

        Yields AgentEvent objects for the TUI to render.

        Args:
            user_input: The user's natural language request.
            confirm_callback: An async callable that takes a ToolCall and returns
                            True (confirmed) or False (declined). Required when
                            safety mode is 'confirm'.
            password_callback: An async callable that takes a ToolCall and returns
                            the sudo password (str) or None if cancelled.
            tool_filter: If provided, only expose these tools to the model.
            skip_intent_detection: If True, skip plan/analysis intent injection.
                            Used by /plan generation to prevent the planning-mode
                            system message from conflicting with the plan prompt.
        """
        self.messages.append({"role": "user", "content": user_input})

        if not skip_intent_detection:
            # Inject planning mode reminder when user asks for a plan
            if _is_plan_request(user_input):
                self.messages.append({
                    "role": "system",
                    "content": (
                        "[Planning mode] The user is asking you to plan. "
                        "Describe your approach in text FIRST. Do not modify files "
                        "or run commands until the user approves the plan."
                    ),
                })

            # Inject analysis guidance when user asks for a review/audit/analysis
            if _is_analysis_request(user_input):
                self.messages.append({
                    "role": "system",
                    "content": (
                        "[Analysis mode] The user is asking for a code review or analysis. "
                        "Read configuration and safety-critical files first. "
                        "Trace data flows — do not stop at function signatures. "
                        "Verify every finding against actual code before reporting it. "
                        "Use your full step budget for thorough analysis."
                    ),
                })

        # Merge persistent context filter with per-call tool_filter
        if self._context_tool_filter is not None and tool_filter is not None:
            effective_filter: set[str] | None = self._context_tool_filter & tool_filter
        elif self._context_tool_filter is not None:
            effective_filter = self._context_tool_filter
        else:
            effective_filter = tool_filter

        # Reset edit failure tracking for this run
        self._edit_failures = 0
        self._edit_successes = 0
        self._completion_warning_sent = False
        self._read_counts = {}
        self._fetch_url_counts = {}
        self._cmd_family_counts = {}
        self._last_tool_key = ""
        self._consecutive_dupes = 0
        self._similar_cmd_key = ""
        self._consecutive_similar = 0
        self._context_recovery_attempted = False

        # Cumulative stats for this run
        run_t0 = time.monotonic()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_inference_ms = 0
        steps_used = 0

        max_steps = getattr(self, "_max_steps", self.config.max_steps)
        for step in range(max_steps):
            steps_used = step + 1

            # Drain queued messages from the user
            queued = self._drain_queued_messages()
            if queued:
                combined = "\n\n".join(queued)
                guidance_msg = (
                    "[IMPORTANT — USER GUIDANCE RECEIVED MID-TASK]\n"
                    "The user has sent the following message while you were "
                    "working. Read it carefully and adjust your approach "
                    "accordingly. This takes priority over your current plan.\n\n"
                    f"{combined}"
                )
                self.messages.append({"role": "user", "content": guidance_msg})
                for queued_text in queued:
                    yield AgentEvent(type=EventType.QUEUED_MESSAGE, data=queued_text)

            # Signal that the model is thinking
            yield AgentEvent(type=EventType.THINKING)

            # Progressively tighten output truncation as steps are consumed
            _exec_shell_mod.configure_step_scaling(step, max_steps)

            # Compress old tool exchanges every 3 steps, then trim context
            if step % 3 == 0:
                self._compress_old_messages()
            if self._context_manager:
                self.messages = self._context_manager.trim_messages(self.messages)

            # Pre-flight check: force compaction if context pressure is high
            if self._context_manager:
                estimated = self._context_manager.estimate_tokens(self.messages)
                try:
                    n_ctx_pf = self.engine.engine_info().n_ctx or 0
                except (AttributeError, TypeError):
                    n_ctx_pf = 0
                if n_ctx_pf > 0 and (
                    estimated + self._max_tokens > n_ctx_pf
                    or estimated / n_ctx_pf > self._CONTEXT_PRESSURE_THRESHOLD
                ):
                    stats = self.compact_history()
                    if stats.get("compacted"):
                        self.messages = self._context_manager.trim_messages(self.messages)
                        yield AgentEvent(
                            type=EventType.ERROR,
                            data="Context nearing capacity — automatically compacted conversation.",
                        )

            # Get model response
            try:
                t0 = time.monotonic()
                result = await self.engine.chat_completion(
                    messages=self.messages,
                    tools=self.tools.get_tool_schemas(allowed=effective_filter),
                    temperature=self.config.temperature,
                    max_tokens=self._max_tokens,
                )
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                total_inference_ms += elapsed_ms
                total_prompt_tokens += result.prompt_tokens or 0
                total_completion_tokens += result.completion_tokens or 0

                # Calibrate token budget from actual API usage
                if result.prompt_tokens > 0 and self._context_manager:
                    estimated = self._context_manager.estimate_tokens(self.messages)
                    estimated += getattr(self, '_tool_token_overhead', 0)
                    self._context_manager.calibrate_from_actual(estimated, result.prompt_tokens)

                # Proactive compaction when context pressure is high
                try:
                    n_ctx = self.engine.engine_info().n_ctx or 0
                except (AttributeError, TypeError):
                    n_ctx = 0
                if (
                    n_ctx > 0
                    and result.prompt_tokens > 0
                    and result.prompt_tokens / n_ctx > self._CONTEXT_PRESSURE_THRESHOLD
                ):
                    compact_stats = self.compact_history()
                    if compact_stats.get("compacted"):
                        yield AgentEvent(
                            type=EventType.ERROR,
                            data="Context nearing capacity — automatically compacted conversation.",
                        )
            except Exception as e:
                logger.exception("Inference error")
                # Handle context overflow: compact and retry on same engine
                from natshell.inference.remote import ContextOverflowError

                if isinstance(e, ContextOverflowError):
                    if self._context_recovery_attempted:
                        yield AgentEvent(
                            type=EventType.ERROR,
                            data="Context window still full after compaction. "
                            "Use /clear to reset the conversation.",
                        )
                        return
                    stats = self.compact_history()
                    if stats.get("compacted"):
                        self._context_recovery_attempted = True
                        yield AgentEvent(
                            type=EventType.ERROR,
                            data="Context window full — automatically compacted "
                            "conversation. Retrying…",
                        )
                        continue  # retry this step with compacted context
                    else:
                        yield AgentEvent(
                            type=EventType.ERROR,
                            data="Context window full and conversation is too "
                            "short to compact. Use /clear to reset.",
                        )
                        return

                if self._can_fallback(e):
                    # --- Phase 1: try compaction + retry if server is alive ---
                    if (
                        not self._context_recovery_attempted
                        and len(self.messages) > 3
                    ):
                        from natshell.inference.ollama import ping_server

                        server_alive = await ping_server(self.engine.base_url)
                        if server_alive:
                            stats = self.compact_history()
                            if stats.get("compacted"):
                                self._context_recovery_attempted = True
                                yield AgentEvent(
                                    type=EventType.ERROR,
                                    data="Remote server timed out — compacted "
                                    "conversation and retrying\u2026",
                                )
                                continue  # retry on same remote engine

                    # --- Phase 2: fallback with preserved context ---
                    # Compact if not already done, then save non-system messages
                    if (
                        not self._context_recovery_attempted
                        and len(self.messages) > 3
                    ):
                        self.compact_history()
                    preserved = (
                        self.messages[1:] if len(self.messages) > 1 else []
                    )

                    fell_back = await self._try_local_fallback()
                    if fell_back:
                        # Inject preserved context if it fits in local budget
                        context_restored = False
                        if preserved:
                            try:
                                n_ctx = (
                                    self.engine.engine_info().n_ctx or 4096
                                )
                                max_tok = self._effective_max_tokens(n_ctx)
                                reserve = self.config.context_reserve or 800
                                budget = n_ctx - max_tok - reserve
                                if self._context_manager:
                                    current = (
                                        self._context_manager.estimate_tokens(
                                            self.messages
                                        )
                                    )
                                    needed = (
                                        self._context_manager.estimate_tokens(
                                            preserved
                                        )
                                    )
                                    if current + needed < budget:
                                        self.messages.extend(preserved)
                                        context_restored = True
                            except Exception:
                                logger.debug(
                                    "Could not restore context after fallback",
                                    exc_info=True,
                                )

                        msg = (
                            "Remote server unreachable."
                            " Switched to local model."
                        )
                        if context_restored:
                            msg += (
                                " Previous conversation context preserved."
                            )
                        else:
                            msg += " History cleared."
                        # Warn if fallback is CPU-only
                        try:
                            from llama_cpp import llama_supports_gpu_offload

                            if not llama_supports_gpu_offload():
                                msg += (
                                    " Note: local model is running on CPU"
                                    " (llama-cpp-python has no GPU support)."
                                )
                        except ImportError:
                            pass
                        yield AgentEvent(type=EventType.ERROR, data=msg)
                        return
                yield AgentEvent(type=EventType.ERROR, data=f"Inference error: {e}")
                return

            # Handle degenerate output (repetitive garbage from local models)
            if result.degenerate:
                compact_stats = self.compact_history()
                if compact_stats.get("compacted"):
                    yield AgentEvent(
                        type=EventType.ERROR,
                        data=(
                            "Model produced degenerate output "
                            "(repeated characters). Automatically "
                            "compacted conversation — retrying."
                        ),
                    )
                    continue
                yield AgentEvent(
                    type=EventType.ERROR,
                    data=(
                        "Model produced degenerate output "
                        "(repeated characters). The context window "
                        "may be full. Try /clear to reset."
                    ),
                )
                return

            # Handle truncated responses (thinking consumed all tokens)
            if result.finish_reason == "length" and not result.tool_calls:
                raw = result.content or ""
                # Check if content is only <think> residue or empty
                # Strip both closed and unclosed <think> blocks
                stripped = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
                stripped = re.sub(
                    r"<think>(?:(?!</think>).)*$", "", stripped, flags=re.DOTALL
                ).strip()
                if stripped:
                    # Partial response — show it but warn the user
                    self.messages.append({"role": "assistant", "content": stripped})
                    yield AgentEvent(
                        type=EventType.RESPONSE,
                        data=stripped,
                        metrics=_build_metrics(result, elapsed_ms),
                    )
                    yield AgentEvent(
                        type=EventType.ERROR,
                        data="Response was truncated (hit token limit). "
                        "The context window may be full. Try /clear to reset.",
                    )
                    if steps_used > 1:
                        run_wall_ms = int((time.monotonic() - run_t0) * 1000)
                        yield AgentEvent(
                            type=EventType.RUN_STATS,
                            metrics=_build_run_stats(
                                steps_used,
                                run_wall_ms,
                                total_inference_ms,
                                total_prompt_tokens,
                                total_completion_tokens,
                            ),
                        )
                    return
                else:
                    yield AgentEvent(
                        type=EventType.ERROR,
                        data="Response was truncated — the model used all"
                        " available tokens without producing a complete"
                        " response. Try a simpler request or /clear to reset.",
                    )
                    return

            # Case 1: Model wants to call tools
            if result.tool_calls:
                logger.debug(
                    "Step %d: tool calls = %s",
                    steps_used,
                    [tc.name for tc in result.tool_calls],
                )
                # If the model also provided text (planning/reasoning), emit it
                if result.content:
                    yield AgentEvent(type=EventType.PLANNING, data=result.content)

                for tool_call in result.tool_calls:
                    # Safety classification
                    risk = self.safety.classify_tool_call(tool_call.name, tool_call.arguments)
                    logger.debug(
                        "Tool %s classified as %s", tool_call.name, risk.name,
                    )

                    if risk == Risk.BLOCKED:
                        yield AgentEvent(type=EventType.BLOCKED, tool_call=tool_call)
                        self._append_tool_exchange(
                            tool_call,
                            "BLOCKED: This command was blocked by the safety classifier. "
                            "Try an alternative approach.",
                        )
                        continue

                    if risk == Risk.CONFIRM and confirm_callback:
                        yield AgentEvent(type=EventType.CONFIRM_NEEDED, tool_call=tool_call)
                        confirmed = await confirm_callback(tool_call)
                        if not confirmed:
                            self._append_tool_exchange(
                                tool_call,
                                "DECLINED: The user declined to execute this command.",
                            )
                            continue

                    # Restart thinking animation before execution
                    yield AgentEvent(type=EventType.THINKING)
                    # Execute the tool
                    yield AgentEvent(type=EventType.EXECUTING, tool_call=tool_call)

                    tool_result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    logger.debug(
                        "Tool %s → exit_code=%s, output_len=%d",
                        tool_call.name, tool_result.exit_code,
                        len(tool_result.output or ""),
                    )

                    # If sudo needs a password, prompt the user and retry
                    if (
                        tool_call.name == "execute_shell"
                        and password_callback
                        and _needs_sudo_password(tool_result)
                    ):
                        password = await password_callback(tool_call)
                        if password:
                            from natshell.tools.execute_shell import (
                                _has_sudo_invocation,
                                set_sudo_password,
                            )

                            set_sudo_password(password)
                            # If the command doesn't contain sudo at a command
                            # position (e.g. "apt install" which internally
                            # invokes sudo), prepend it so the password
                            # injection in execute_shell kicks in.
                            retry_args = dict(tool_call.arguments)
                            cmd = retry_args.get("command", "")
                            if cmd and not _has_sudo_invocation(cmd):
                                retry_args["command"] = f"sudo {cmd}"
                            # Re-classify the modified command — prepending
                            # sudo may change the risk level.
                            retry_risk = self.safety.classify_tool_call(
                                tool_call.name, retry_args
                            )
                            if retry_risk == Risk.BLOCKED:
                                yield AgentEvent(
                                    type=EventType.BLOCKED, tool_call=tool_call
                                )
                                self._append_tool_exchange(
                                    tool_call,
                                    "BLOCKED: The retried command with sudo was "
                                    "blocked by the safety classifier.",
                                )
                                continue
                            if retry_risk == Risk.CONFIRM and confirm_callback:
                                yield AgentEvent(
                                    type=EventType.CONFIRM_NEEDED,
                                    tool_call=tool_call,
                                )
                                confirmed = await confirm_callback(tool_call)
                                if not confirmed:
                                    self._append_tool_exchange(
                                        tool_call,
                                        "DECLINED: The user declined the retried "
                                        "command with sudo.",
                                    )
                                    continue
                            yield AgentEvent(type=EventType.THINKING)
                            tool_result = await self.tools.execute(
                                tool_call.name, retry_args
                            )

                    # Track edit_file/write_file results
                    if tool_call.name == "edit_file":
                        if tool_result.exit_code != 0:
                            self._edit_failures += 1
                        else:
                            self._edit_successes += 1
                    elif tool_call.name == "write_file" and tool_result.exit_code == 0:
                        self._edit_successes += 1

                    yield AgentEvent(
                        type=EventType.TOOL_RESULT,
                        tool_call=tool_call,
                        tool_result=tool_result,
                    )

                    # Build result content with warnings
                    result_content = tool_result.to_message_content()

                    # Duplicate tool call detection — catch infinite retry loops
                    tool_key = f"{tool_call.name}:{json.dumps(tool_call.arguments, sort_keys=True)}"
                    if tool_key == self._last_tool_key:
                        self._consecutive_dupes += 1
                    else:
                        self._last_tool_key = tool_key
                        self._consecutive_dupes = 1

                    _DUPE_WARN_THRESHOLD = 3
                    _DUPE_ABORT_THRESHOLD = 5
                    if self._consecutive_dupes >= _DUPE_ABORT_THRESHOLD:
                        result_content += (
                            f"\n\n\u26a0 CRITICAL: You have called {tool_call.name} "
                            f"with identical arguments {self._consecutive_dupes} "
                            "times in a row. The output will not change. "
                            "STOP making this tool call. Complete the task using "
                            "your existing knowledge and information already gathered."
                        )
                        self._append_tool_exchange(tool_call, result_content)
                        # Reset counter so the LLM can use other tools freely
                        self._last_tool_key = ""
                        self._consecutive_dupes = 0
                        # Break out of the tool-dispatch inner loop.
                        # The outer step loop will call the LLM again — it will see
                        # the warning in history and produce a final response.
                        break
                    elif self._consecutive_dupes >= _DUPE_WARN_THRESHOLD:
                        result_content += (
                            f"\n\n\u26a0 You have called {tool_call.name} with "
                            f"identical arguments {self._consecutive_dupes} times "
                            "in a row and gotten the same result each time. "
                            "Try a DIFFERENT approach — change the arguments, "
                            "use a different tool, or fix the underlying issue "
                            "before retrying."
                        )

                    # Repetitive read detection
                    if tool_call.name == "read_file":
                        read_path = tool_call.arguments.get("path", "")
                        if read_path:
                            resolved = self._resolve_path(read_path)
                            self._read_counts[resolved] = self._read_counts.get(resolved, 0) + 1
                            count = self._read_counts[resolved]
                            if count >= 3:
                                result_content += (
                                    f"\n\n\u26a0 You have read this file {count} times "
                                    "without modifying it. Stop re-reading and use the "
                                    "information you already have to make changes. "
                                    "If edit_file is failing, use write_file instead."
                                )

                    # Repetitive URL fetch detection
                    # (cumulative across all calls, not just consecutive)
                    if tool_call.name == "fetch_url":
                        url = tool_call.arguments.get("url", "")
                        if url:
                            self._fetch_url_counts[url] = (
                                self._fetch_url_counts.get(url, 0) + 1
                            )
                            count = self._fetch_url_counts[url]
                            if count == 2:
                                result_content += (
                                    f"\n\n\u26a0 You have fetched this URL {count} times. "
                                    "The content will not change. Do NOT fetch it again — "
                                    "use the information already in your context."
                                )
                            elif count >= 3:
                                result_content += (
                                    f"\n\n\u26a0 CRITICAL: You have fetched this URL"
                                    f" {count} times. "
                                    "STOP fetching it. The result is already in your"
                                    " conversation history. Use your existing knowledge"
                                    " to complete the task."
                                )

                    # Command-family repetition detection for execute_shell
                    if tool_call.name == "execute_shell":
                        cmd = tool_call.arguments.get("command", "")
                        # Extract the first token as the command family
                        family = cmd.strip().split()[0] if cmd.strip() else ""
                        # Normalise common prefixes (sudo X → X)
                        if family == "sudo" and len(cmd.strip().split()) > 1:
                            family = cmd.strip().split()[1]
                        if family:
                            self._cmd_family_counts[family] = (
                                self._cmd_family_counts.get(family, 0) + 1
                            )
                            fam_count = self._cmd_family_counts[family]
                            _FAM_WARN = 4
                            _FAM_CRITICAL = 7
                            _FAM_HARD_STOP = 12
                            if fam_count >= _FAM_HARD_STOP:
                                result_content += (
                                    f"\n\n\u26a0 HARD STOP: You have run"
                                    f" `{family}` {fam_count} times."
                                    " You MUST use a completely"
                                    " different approach or tool."
                                    " Further `{family}` calls"
                                    " are blocked."
                                )
                                self._append_tool_exchange(
                                    tool_call, result_content,
                                )
                                break
                            elif fam_count >= _FAM_CRITICAL:
                                result_content += (
                                    f"\n\n\u26a0 CRITICAL: You have run"
                                    f" `{family}` {fam_count} times in"
                                    " this session. STOP running more"
                                    f" `{family}` commands. Synthesize"
                                    " your findings from the output"
                                    " already gathered and give the"
                                    " user a complete answer NOW."
                                )
                            elif fam_count >= _FAM_WARN:
                                result_content += (
                                    f"\n\n\u26a0 You have run `{family}`"
                                    f" {fam_count} times. Consolidate"
                                    " your findings and answer with"
                                    " what you have. Avoid further"
                                    f" `{family}` calls unless"
                                    " absolutely necessary."
                                )

                        # Similar-command detection — catches near-duplicate
                        # commands that differ only in flags or pipes
                        # (e.g. grep -o vs grep -n on the same pattern/file)
                        norm_key = self._normalize_shell_cmd(cmd)
                        if norm_key and len(norm_key.split()) > 1:
                            if norm_key == self._similar_cmd_key:
                                self._consecutive_similar += 1
                            else:
                                self._similar_cmd_key = norm_key
                                self._consecutive_similar = 1

                            _SIM_WARN = 3
                            _SIM_ABORT = 5
                            if self._consecutive_similar >= _SIM_ABORT:
                                result_content += (
                                    f"\n\n\u26a0 CRITICAL: You have run"
                                    f" {self._consecutive_similar}"
                                    " near-identical commands in a"
                                    " row (same target, different"
                                    " flags). The result will not"
                                    " change. STOP and try a"
                                    " completely different approach."
                                )
                                self._append_tool_exchange(
                                    tool_call, result_content,
                                )
                                self._similar_cmd_key = ""
                                self._consecutive_similar = 0
                                break
                            elif (
                                self._consecutive_similar >= _SIM_WARN
                            ):
                                result_content += (
                                    f"\n\n\u26a0 You have run"
                                    f" {self._consecutive_similar}"
                                    " near-identical commands."
                                    " Changing flags or adding"
                                    " pipes will not produce"
                                    " different results. Try a"
                                    " different approach."
                                )

                    # Reset read count when write/edit succeeds on a path
                    if tool_call.name in ("edit_file", "write_file") and tool_result.exit_code == 0:
                        write_path = tool_call.arguments.get("path", "")
                        if write_path:
                            resolved = self._resolve_path(write_path)
                            self._read_counts.pop(resolved, None)

                    # Escalating warnings on repeated edit failures
                    if (
                        tool_call.name == "edit_file"
                        and tool_result.exit_code != 0
                    ):
                        if self._edit_failures >= 3:
                            result_content += (
                                "\n\n\u26a0 REPEATED EDIT FAILURES (3+). "
                                "STOP using edit_file for this file. "
                                "Use write_file to rewrite the entire file instead."
                            )
                        elif self._edit_failures >= 2:
                            result_content += (
                                "\n\n\u26a0 Multiple edit failures. Try: "
                                "(1) use the closest match shown above as your old_text, or "
                                "(2) use write_file to rewrite the entire file instead."
                            )

                    # Step budget awareness
                    pct_used = steps_used / max_steps
                    if pct_used >= 0.90:
                        remaining = max_steps - steps_used
                        result_content += (
                            f"\n\n\u26a0 URGENT: [{steps_used}/{max_steps} steps used"
                            f" \u2014 only {remaining} steps left. Finish NOW.]"
                        )
                    elif pct_used >= 0.75:
                        result_content += (
                            f"\n\n\u26a0 [{steps_used}/{max_steps} steps used"
                            " \u2014 wrap up soon]"
                        )
                    elif pct_used >= 0.50:
                        result_content += (
                            f"\n\n[{steps_used}/{max_steps} steps used]"
                        )
                    elif steps_used == 1:
                        result_content += (
                            f"\n\n[Budget: {max_steps} steps available"
                            " \u2014 plan your approach before diving in]"
                        )

                    # Append exchange to conversation history
                    self._append_tool_exchange(tool_call, result_content)

                # Continue the loop — model will see tool results and decide next step
                continue

            # Case 2: Model responded with text only (task complete or needs info)
            if result.content:
                # Completion guard: warn if all edits failed
                if (
                    self._edit_failures > 0
                    and self._edit_successes == 0
                    and not self._completion_warning_sent
                ):
                    self._completion_warning_sent = True
                    self.messages.append(
                        {"role": "assistant", "content": result.content}
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "[SYSTEM] Warning: All edit_file calls failed. "
                                "Verify changes were applied before declaring "
                                "the task complete."
                            ),
                        }
                    )
                    continue

                self.messages.append(
                    {
                        "role": "assistant",
                        "content": result.content,
                    }
                )
                yield AgentEvent(
                    type=EventType.RESPONSE,
                    data=result.content,
                    metrics=_build_metrics(result, elapsed_ms),
                )
                if steps_used > 1:
                    run_wall_ms = int((time.monotonic() - run_t0) * 1000)
                    yield AgentEvent(
                        type=EventType.RUN_STATS,
                        metrics=_build_run_stats(
                            steps_used,
                            run_wall_ms,
                            total_inference_ms,
                            total_prompt_tokens,
                            total_completion_tokens,
                        ),
                    )
                return

            # Case 3: Empty response (shouldn't happen, but handle gracefully)
            logger.warning(
                "Empty response from model: finish_reason=%s, "
                "prompt_tokens=%s, completion_tokens=%s",
                result.finish_reason, result.prompt_tokens,
                result.completion_tokens,
            )
            yield AgentEvent(
                type=EventType.ERROR,
                data="Model returned an empty response.",
            )
            return

        # Hit max steps
        run_wall_ms = int((time.monotonic() - run_t0) * 1000)
        yield AgentEvent(
            type=EventType.RESPONSE,
            data=f"I've reached the maximum number of steps ({max_steps}). "
            f"Here's what I've done so far. You can continue with a follow-up request.",
        )
        yield AgentEvent(
            type=EventType.RUN_STATS,
            metrics=_build_run_stats(
                steps_used,
                run_wall_ms,
                total_inference_ms,
                total_prompt_tokens,
                total_completion_tokens,
            ),
        )

    def _can_fallback(self, error: Exception) -> bool:
        """Check if we should attempt fallback to local model."""
        import httpx

        from natshell.inference.remote import (
            AuthenticationError,
            ContextOverflowError,
            RemoteEngine,
        )

        # Context overflow is not a connectivity issue — don't swap engines
        if isinstance(error, ContextOverflowError):
            return False
        # Auth errors mean the server is reachable but the key is wrong — don't swap
        if isinstance(error, AuthenticationError):
            return False
        if not isinstance(self.engine, RemoteEngine):
            return False
        if self.fallback_config is None:
            return False
        return isinstance(
            error,
            (
                httpx.ConnectError,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
                httpx.PoolTimeout,
                httpx.RemoteProtocolError,
                ConnectionError,
                OSError,
            ),
        )

    async def _try_local_fallback(self) -> bool:
        """Attempt to load and swap to the local model. Returns True on success."""
        if self.fallback_config is None:
            return False

        # Resolve model path
        model_path = self.fallback_config.path
        if model_path == "auto":
            from natshell.platform import data_dir

            model_dir = data_dir() / "models"
            model_path = str(model_dir / self.fallback_config.hf_file)

        if not Path(model_path).exists():
            logger.warning("Local model not found at %s — cannot fall back", model_path)
            return False

        try:
            from natshell.inference.local import LocalEngine

            engine = await asyncio.to_thread(
                LocalEngine,
                model_path=model_path,
                n_ctx=self.fallback_config.n_ctx,
                n_threads=self.fallback_config.n_threads,
                n_gpu_layers=self.fallback_config.n_gpu_layers,
                main_gpu=self.fallback_config.main_gpu,
            )
            await self.swap_engine(engine)

            # Warn if GPU offload was requested but unavailable
            if self.fallback_config.n_gpu_layers != 0:
                try:
                    from llama_cpp import llama_supports_gpu_offload

                    if not llama_supports_gpu_offload():
                        logger.warning(
                            "Fallback model running on CPU — llama-cpp-python"
                            " was built without GPU support. Reinstall with"
                            ' CMAKE_ARGS="-DGGML_VULKAN=on" for GPU acceleration.'
                        )
                except ImportError:
                    pass

            return True
        except Exception:
            logger.exception("Failed to load local model for fallback")
            return False

    # Number of most-recent messages to keep uncompressed (3 tool exchanges)
    _COMPRESS_PRESERVE_RECENT = 6

    def _compress_old_messages(self) -> None:
        """Compress old tool exchanges to save context tokens.

        Replaces full file content in write_file arguments and truncates
        long tool results for messages older than the last few exchanges.
        Safe because the model has already processed these results.
        """
        if len(self.messages) <= self._COMPRESS_PRESERVE_RECENT + 1:
            return

        cutoff = len(self.messages) - self._COMPRESS_PRESERVE_RECENT

        for i in range(1, cutoff):  # skip system prompt
            msg = self.messages[i]

            # Compress write_file arguments (elide full file content)
            if msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args_str = func.get("arguments", "")

                    if name == "write_file" and len(args_str) > 300:
                        try:
                            args = json.loads(args_str)
                            content = args.get("content", "")
                            if len(content) > 100:
                                args["content"] = f"[{len(content)} chars elided]"
                                func["arguments"] = json.dumps(args)
                        except (json.JSONDecodeError, TypeError):
                            pass

            # Compress long tool results
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if len(content) > 800:
                    lines = content.split("\n")
                    if len(lines) > 8:
                        head = "\n".join(lines[:4])
                        tail = "\n".join(lines[-3:])
                        msg["content"] = (
                            f"{head}\n"
                            f"... [{len(lines) - 7} lines elided] ...\n"
                            f"{tail}"
                        )

    def _append_tool_exchange(self, tool_call: ToolCall, result_content: str) -> None:
        """Append a tool call + result pair to the message history."""
        # Assistant message with tool call
        # Use "" instead of None — llama-cpp-python may iterate content and choke on None
        self.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments),
                        },
                    }
                ],
            }
        )
        # Tool result message
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_content,
            }
        )

    def set_step_limit(self, limit: int) -> None:
        """Override the step limit for the next handle_user_message call.

        Used by plan execution to enforce per-step budgets that differ
        from the context-scaled default.
        """
        self._max_steps = limit

    def clear_history(self) -> None:
        """Clear conversation history, keeping only the system prompt."""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
        reset_tracker()
        self._read_counts = {}
        self._fetch_url_counts = {}
        self._cmd_family_counts = {}
        self._last_tool_key = ""
        self._consecutive_dupes = 0
        self._similar_cmd_key = ""
        self._consecutive_similar = 0
        # Drain any pending queued messages
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def compact_history(self, dry_run: bool = False) -> dict[str, Any]:
        """Compact conversation history, keeping system prompt and last 2 messages.

        Args:
            dry_run: If True, compute and return stats without mutating messages.

        Returns a stats dict with compaction results.
        """
        if len(self.messages) <= 3:
            return {"compacted": False}

        cm = self._context_manager
        before_msgs = len(self.messages)
        before_tokens = cm.estimate_tokens(self.messages) if cm else 0

        system = self.messages[0]

        # Collect non-system messages, keep last 2
        rest = self.messages[1:]
        last_2 = rest[-2:]
        dropped = rest[:-2]

        # Build extractive summary
        summary = ""
        if cm and dropped:
            summary = cm.build_summary(dropped)

        summary_msg: dict[str, Any] = {
            "role": "system",
            "content": (
                f"[Context compacted: {len(dropped)} messages replaced with summary.\n"
                f"{summary}\n"
                "Recent context follows.]"
            ),
        }

        new_messages = [system, summary_msg] + last_2
        after_msgs = len(new_messages)
        after_tokens = cm.estimate_tokens(new_messages) if cm else 0

        if not dry_run:
            self.messages = new_messages

        return {
            "compacted": True,
            "before_msgs": before_msgs,
            "after_msgs": after_msgs,
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "summary": summary,
        }
