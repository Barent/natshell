"""Core ReAct agent loop — plans, executes tools, observes, repeats."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from natshell.agent.context import SystemContext
from natshell.agent.context_manager import ContextManager
from natshell.agent.events import AgentEvent, EventType
from natshell.agent.intent import is_analysis_request, is_plan_request
from natshell.agent.repetition_guard import RepetitionGuard
from natshell.agent.system_prompt import build_system_prompt
from natshell.config import AgentConfig, MemoryConfig, ModelConfig, PromptConfig
from natshell.inference.engine import CompletionResult, InferenceEngine, ToolCall
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.scaling import (
    MAX_OUTPUT_CHARS_TABLE,
    MAX_STEPS_TABLE,
    READ_FILE_LINES_TABLE,
    scale_for_context,
)
from natshell.tools import edit_file as _edit_file_mod
from natshell.tools import execute_shell as _exec_shell_mod
from natshell.tools import read_file as _read_file_mod
from natshell.tools.execute_shell import needs_sudo_password as _needs_sudo_password
from natshell.tools.file_tracker import reset_tracker
from natshell.tools.limits import ToolLimits
from natshell.tools.registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Intent heuristics now live in natshell.agent.intent.  Re-keep the module-level
# names as thin aliases so existing imports (tests, callers) keep working.
from natshell.agent.intent import (  # noqa: E402
    _ANALYSIS_REQUEST_RE as _ANALYSIS_REQUEST_RE,
    _PLAN_REQUEST_RE as _PLAN_REQUEST_RE,
    is_analysis_request as _is_analysis_request,
    is_plan_request as _is_plan_request,
)


# ─────────────────────────────────────────────────────────────────────────────
# AgentEvent / EventType live in natshell.agent.events (imported above); they
# are re-exported here for backward-compatible imports.
# ─────────────────────────────────────────────────────────────────────────────


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
        skills: list | None = None,
        inject_skills_in_compact: bool = False,
    ) -> None:
        self.engine = engine
        self.tools = tools
        self.safety = safety
        self.config = config
        self.fallback_config = fallback_config
        self._prompt_config = prompt_config
        self._memory_config = memory_config or MemoryConfig()
        self._skills = skills or []
        self._inject_skills_in_compact = inject_skills_in_compact
        self._system_context: SystemContext | None = None
        self.messages: list[dict[str, Any]] = []
        self._context_manager: ContextManager | None = None
        self._max_tokens: int = config.max_tokens
        # Repetition / edit-failure guards (state + detectors live in
        # natshell.agent.repetition_guard)
        self._repetition_guard = RepetitionGuard()
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
            skills=self._skills or None,
            inject_skills_in_compact=self._inject_skills_in_compact,
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
            skills=self._skills or None,
            inject_skills_in_compact=self._inject_skills_in_compact,
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
        return scale_for_context(n_ctx, MAX_STEPS_TABLE, self._DEFAULT_MAX_STEPS)

    def _effective_max_output_chars(self, n_ctx: int) -> int:
        """Scale shell output truncation with context window."""
        return scale_for_context(n_ctx, MAX_OUTPUT_CHARS_TABLE, 4000)

    def _effective_read_file_lines(self, n_ctx: int) -> int:
        """Scale read_file default line count with context window."""
        return scale_for_context(n_ctx, READ_FILE_LINES_TABLE, 200)

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

        # Reset repetition/edit-failure tracking for this run (lives in
        # natshell.agent.repetition_guard — thresholds & state in one place)
        self._repetition_guard.reset()
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
                                reason = self._describe_remote_error(e)
                                yield AgentEvent(
                                    type=EventType.ERROR,
                                    data=f"{reason} — compacted "
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
                    # Bind the arguments to their parameter names *before*
                    # classifying, so the classifier judges the call that will
                    # actually run.  execute() repairs misnamed parameters by
                    # position, which used to happen after this point: a model
                    # that wrote "cmd" instead of "command" -- a mistake the
                    # remap exists because small models make constantly -- got
                    # classified against an absent key and ran unconfirmed.
                    #
                    # Assigning back onto the call also makes the confirmation
                    # dialog show the arguments the tool will receive.
                    normalized = self.tools.normalize_arguments(
                        tool_call.name, tool_call.arguments
                    )
                    if normalized is not None:
                        tool_call.arguments = normalized

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


                    # If sudo needs a password, prompt the user and retry.
                    # (Event order + re-classification preserved via
                    #  natshell.agent.sudo_retry — see that module.)
                    from natshell.agent.sudo_retry import run as _sudo_retry_run

                    if (
                        tool_call.name == "execute_shell"
                        and password_callback
                        and _needs_sudo_password(tool_result)
                    ):
                        buffered: list[AgentEvent] = []
                        outcome = await _sudo_retry_run(
                            tool_call,
                            tools=self.tools,
                            safety=self.safety,
                            password_callback=password_callback,
                            confirm_callback=confirm_callback,
                            on_event=buffered.append,
                        )
                        for ev in buffered:
                            yield ev
                        if outcome.status == "retried":
                            tool_result = outcome.tool_result
                        else:
                            # blocked / declined_password / declined_confirm
                            if outcome.status == "blocked":
                                self._append_tool_exchange(
                                    tool_call,
                                    "BLOCKED: The retried command with sudo was "
                                    "blocked by the safety classifier.",
                                )
                            elif outcome.status == "declined_confirm":
                                self._append_tool_exchange(
                                    tool_call,
                                    "DECLINED: The user declined the retried "
                                    "command with sudo.",
                                )
                            elif outcome.status == "declined_password":
                                self._append_tool_exchange(
                                    tool_call,
                                    "DECLINED: The user cancelled the sudo "
                                    "password prompt.",
                                )
                            continue

                    # Record the (possibly retried) result + ask the guard
                    # whether it should warn the model.  The guard also owns
                    # the edit-succeed/fail bookkeeping and the read-count
                    # reset, in the same order the inline version had.
                    self._repetition_guard.register_outcome(
                        tool_call.name, tool_result.exit_code
                    )

                    # Yield the tool result event.  (This matches the
                    # original order: TOOL_RESULT fires *before* the guard
                    # checks so the TUI sees the raw result either way.)
                    yield AgentEvent(
                        type=EventType.TOOL_RESULT,
                        tool_call=tool_call,
                        tool_result=tool_result,
                    )

                    # Observations (repetition warnings + stop signal).
                    # NOTE: we must skip the register_outcome path inside
                    # .observe() here — already done above — otherwise the
                    # counters would double-count.
                    observation = self._repetition_guard.observe(
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        exit_code=tool_result.exit_code,
                        record=False,
                    )
                    result_content = (
                        tool_result.to_message_content() + observation.suffix
                    )

                    # Step budget awareness (unchanged semantics; kept in the
                    # loop so it can see max_steps from the current config.)
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

                    # Append exchange to conversation history.  A ``stop``
                    # observation (duplicate-abort, command-family hard stop,
                    # similar-cmd abort) ends the tool-dispatch inner loop
                    # the same way the inline version did with an explicit
                    # ``break``.
                    self._append_tool_exchange(tool_call, result_content)
                    if observation.stop:
                        break

                # Continue the loop — model will see tool results and decide next step
                continue

            # Case 2: Model responded with text only (task complete or needs info)
            if result.content:
                # Completion guard: warn if all edits failed (state now lives
                # in the repetition guard)
                if self._repetition_guard.completion_guard_due:
                    self._repetition_guard.mark_completion_guard_sent()
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

    def _describe_remote_error(self, error: Exception) -> str:
        """Build a short user-facing label for a remote inference failure."""
        from natshell.agent.fallback import describe_remote_error

        return describe_remote_error(error)

    def _can_fallback(self, error: Exception) -> bool:
        """Check if we should attempt fallback to local model."""
        from natshell.agent.fallback import can_fallback

        return can_fallback(error, self.engine, self.fallback_config)

    async def _try_local_fallback(self) -> bool:
        """Attempt to load and swap to the local model. Returns True on success."""
        from natshell.agent.fallback import load_fallback_engine

        engine = await load_fallback_engine(self.fallback_config)
        if engine is None:
            return False
        await self.swap_engine(engine)
        return True

    def _compress_old_messages(self) -> None:
        """Compress old tool exchanges to save context tokens.

        Delegates to :meth:`natshell.agent.context_manager.ContextManager.
        compress_artifacts`, which rewrites the most expensive bytes
        (write_file contents, long tool results) in place for the older
        half of the conversation.  Recent messages are left untouched.
        """
        if self._context_manager is None:
            return
        self._context_manager.compress_artifacts(self.messages)

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
        self._repetition_guard.reset()
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
