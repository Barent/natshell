"""Inference-failure recovery — the ordered recovery ladder.

Extracted from ``AgentLoop.handle_user_message`` (R1-6).  When a
``chat_completion`` call fails mid-run, the loop used to inline a
multi-phase ladder:

1. **Context overflow** (``ContextOverflowError``) — compact the
   conversation once and retry on the same engine; a second overflow
   (or a conversation too short to compact) ends the run with a
   ``/clear`` hint.
2. **Connectivity failure** (see ``fallback.can_fallback``) —
   *Phase 1*: if the remote server is still alive, compact and retry;
   *Phase 2*: load the local model, swap engines, and re-inject the
   preserved context if it fits the local model's budget.
3. **Anything else** — surface the raw inference error.

The ladder is stateful per run: the ``attempted`` flag (formerly
``AgentLoop._context_recovery_attempted``) ensures compaction is only
attempted once before the ladder escalates.  ``AgentLoop`` keeps
``_context_recovery_attempted`` as a property delegating to
``RecoveryCoordinator.attempted`` so existing tests and callers keep
working.

``RecoveryStrategy`` is an alias kept for the plan's wording ("model as
an ordered RecoveryStrategy object"); they are the same class.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Awaitable, Callable

from natshell.agent.events import AgentEvent, EventType
from natshell.agent.fallback import can_fallback, describe_remote_error
from natshell.inference.remote import ContextOverflowError

logger = logging.getLogger(__name__)


class RecoveryOutcome(Enum):
    """What the agent loop should do after a recovery attempt."""

    RETRY = "retry"  # retry the current step on the current engine
    STOP = "stop"  # yield the terminal banner(s) and end the run


class RecoveryCoordinator:
    """Stateful owner of the ordered recovery ladder.

    Receives callbacks bound to the owning :class:`AgentLoop` so it can
    observe and mutate loop state (current engine, messages,
    compaction) without importing the loop (which would be a cycle).

    All user-facing event strings are byte-identical to the
    pre-extraction inline code in ``loop.py`` — the test suites
    (``TestContextOverflow``, ``TestRuntimeFallback``,
    ``TestCompactionBeforeFallback``) pin them.
    """

    def __init__(
        self,
        *,
        engine_ref: Callable[[], Any],
        fallback_config: Any,
        compact: Callable[[], dict[str, Any]],
        effective_max_tokens: Callable[[int], int],
        context_reserve: int,
        messages_ref: Callable[[], list[dict[str, Any]]],
        load_fallback: Callable[[Any], Awaitable[Any]],
        swap_engine: Callable[[Any], Awaitable[None]],
        context_manager_ref: Callable[[], Any | None],
    ) -> None:
        self._engine = engine_ref
        self._fallback_config = fallback_config
        self._compact_history = compact
        self._effective_max_tokens = effective_max_tokens
        self._context_reserve = context_reserve
        self._messages = messages_ref
        self._load_local_engine = load_fallback
        self._swap_engine = swap_engine
        self._context_manager = context_manager_ref
        # Compaction attempted at least once this run (the ladder
        # escalates after this latches).
        self.attempted: bool = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset per-run state (called at the start of each run)."""
        self.attempted = False

    # ── entry point ──────────────────────────────────────────────────────

    async def handle(
        self, error: Exception
    ) -> tuple[RecoveryOutcome, list[AgentEvent]]:
        """Run the ordered recovery ladder for *error*.

        Returns the outcome (retry the step / stop the run) and the list
        of events the loop should yield.
        """
        events: list[AgentEvent] = []

        # ── Ladder 1: context overflow — compact once, then give up ────
        if isinstance(error, ContextOverflowError):
            if self.attempted:
                events.append(
                    AgentEvent(
                        type=EventType.ERROR,
                        data="Context window still full after compaction. "
                        "Use /clear to reset the conversation.",
                    )
                )
                return RecoveryOutcome.STOP, events
            stats = self._compact_history()
            if stats.get("compacted"):
                self.attempted = True
                events.append(
                    AgentEvent(
                        type=EventType.ERROR,
                        data="Context window full — automatically compacted "
                        "conversation. Retrying…",
                    )
                )
                return RecoveryOutcome.RETRY, events
            events.append(
                AgentEvent(
                    type=EventType.ERROR,
                    data="Context window full and conversation is too "
                    "short to compact. Use /clear to reset.",
                )
            )
            return RecoveryOutcome.STOP, events

        # ── Ladder 2: connectivity failure — ping, compact, fall back ──
        if self._can_fallback(error):
            engine = self._engine()

            # --- Phase 1: try compaction + retry if the server is alive ---
            if not self.attempted and len(self._messages()) > 3:
                from natshell.inference.ollama import ping_server

                server_alive = await ping_server(engine.base_url)
                if server_alive:
                    stats = self._compact_history()
                    if stats.get("compacted"):
                        self.attempted = True
                        reason = describe_remote_error(error)
                        events.append(
                            AgentEvent(
                                type=EventType.ERROR,
                                data=f"{reason} — compacted "
                                "conversation and retrying\u2026",
                            )
                        )
                        return RecoveryOutcome.RETRY, events

            # --- Phase 2: fallback with preserved context ---
            # Compact if not already done, then save non-system messages.
            if not self.attempted and len(self._messages()) > 3:
                self._compact_history()
            msgs = self._messages()
            preserved = msgs[1:] if len(msgs) > 1 else []

            if not await self._local_fallback():
                events.append(
                    AgentEvent(type=EventType.ERROR, data=f"Inference error: {error}")
                )
                return RecoveryOutcome.STOP, events

            # Inject preserved context if it fits in the local budget
            context_restored = False
            if preserved:
                try:
                    n_ctx = self._engine().engine_info().n_ctx or 4096
                    max_tok = self._effective_max_tokens(n_ctx)
                    budget = n_ctx - max_tok - self._context_reserve
                    cm = self._context_manager()
                    if cm is not None:
                        current = cm.estimate_tokens(self._messages())
                        needed = cm.estimate_tokens(preserved)
                        if current + needed < budget:
                            self._messages().extend(preserved)
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
                msg += " Previous conversation context preserved."
            else:
                msg += " History cleared."
            # Warn if the fallback model is CPU-only
            try:
                from llama_cpp import llama_supports_gpu_offload

                if not llama_supports_gpu_offload():
                    msg += (
                        " Note: local model is running on CPU"
                        " (llama-cpp-python has no GPU support)."
                    )
            except ImportError:
                pass
            events.append(AgentEvent(type=EventType.ERROR, data=msg))
            return RecoveryOutcome.STOP, events

        # ── Ladder 3: anything else — surface the raw error ────────────
        events.append(AgentEvent(type=EventType.ERROR, data=f"Inference error: {error}"))
        return RecoveryOutcome.STOP, events

    # ── internals ────────────────────────────────────────────────────────

    def _can_fallback(self, error: Exception) -> bool:
        """Check if we should attempt fallback to the local model."""
        return can_fallback(error, self._engine(), self._fallback_config)

    async def _local_fallback(self) -> bool:
        """Attempt to load and swap to the local model. True on success."""
        engine = await self._load_local_engine(self._fallback_config)
        if engine is None:
            return False
        await self._swap_engine(engine)
        return True


# Alias for the plan's wording — "model as an ordered RecoveryStrategy".
RecoveryStrategy = RecoveryCoordinator
