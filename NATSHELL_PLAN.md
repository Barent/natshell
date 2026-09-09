# NatShell — Improvement Plan (working status)

Single source of truth for the recurring improvement job. **The job reads and
updates this file on every run.** It encodes *current ground truth* (checked
against the live working tree), not the original 2026-09-01 review — several of
the review's items are already done. The original two-part analysis lives in
`REVIEW_NAT_SHELL.md`.

## Standing rules (non-negotiable)

- **Branch:** work on `refactor/simplify-core`, rebased on `origin/main`.
  Push every completed unit. When a coherent group of units is ready, open
  **one PR from Barent** and report its URL.
- **Security:** work ONLY off `origin/main` / Barent's own PRs. **Never** fetch
  or apply code from other forks/contributors or non-Barent-authored PRs. Do
  not merge/close issues you did not fix yourself.
- **Quality gate:** `.venv/bin/python -m pytest tests/ -q` must be **fully
  green** before every commit. Do not commit a red tree.
- **Proportionality:** small targeted changes that scale to the change they
  address. No monolithic rewrites. Each unit below is intended to ship as one
  focused commit/PR.
- One session = finish as much green, committable, *shipped* work as the budget
  allows; if a unit won't finish cleanly, stop at a clean boundary, leave an
  `IN PROGRESS` note below, and let the next tick resume.
- Update the **Status** column + the **Changelog** on every run.

## Status

| # | Unit | Review | Status | Notes |
|---|------|--------|--------|-------|
| R1-1 | Split 790-line `handle_user_message`; extract RepetitionGuard | R1§1 | ✅ DONE | `agent/repetition_guard.py` landed; 6 detectors own their state. Method now ~600 lines (natural remainder). |
| R1-2 | Unify 3 context managers into one `ContextStore` | R1§2 | ✅ ~DONE | `_compress_old_messages` delegates to `ContextManager.compress_artifacts`; both summaries use shared `build_summary`. Residual: `compact_history` (loop.py:1053) still builds its summary-marker inline → extract a shared `context_marker(count, summary)` helper it and `trim_messages` share. |
| R1-3 | ToolCallGrammar per-family (qwen/mistral/gemma) + delete if-ladder | R1§3 | ✅ DONE | `inference/grammars/{common,qwen,mistral,gemma,__init__}.py`; `local.py` 927→378 lines; `get_grammar()` registry. Suite 1598 green. |
| R1-4 | app.py: dedupe 3× confirm/password into one factory | R1§4 | ✅ DONE | `_gated_confirm_callback()` + `_password_callback` factory used by run_agent/run_plan/run_plan_generation. |
| R1-5 | intent.py / events.py / SudoRetry extraction | R1§5 | ✅ DONE | `agent/intent.py`, `agent/events.py`, `agent/sudo_retry.py` landed. |
| R1-6 | Extraction of recovery state machine to `recovery.py` | R1§1.3 | ✅ DONE | `agent/recovery.py` owns the ordered ladder + per-run `attempted` latch; loop delegates via `RecoveryCoordinator.handle`; `_context_recovery_attempted` now a read-only property; 16 new tests in `tests/test_recovery.py`. Suite 1614 green. |
| R1-7 | Slim `handle_user_message` to ~250 lines (orchestrator only) | R1§1 | 🛠 IN PROGRESS | 491→437 lines (`0873256`): extracted `_inject_intent`, `_preflight_compaction`, `_apply_inference_feedback`. Remaining candidates (each carries `continue`/`return` flow, harder): degenerate-output block, `length`-truncation block, the tool-call dispatch loop. Resume next tick. |
| R1-8 | `plan_executor.py`: split prompt-templates from pure helpers | R1§5 | ⏳ PENDING | `_build_step_prompt` instruction strings → greppable/testable data. |
| — | Small: group `execute_shell` sudo helpers into one `SudoHandler` | R1§5 | ⏳ OPTIONAL | `_inject_sudo_dash_s`/`_has_sudo_invocation`/`needs_sudo_password`/`configure_limits`. |

## Review 2 — performance additions (the real remaining plan)

Ordered per the review's own "simplify first, then the seams appear" ranking.
Each is a self-contained unit: implement → new/extended tests → green → commit →
push. **Verify the seam R1 created before building on it.**

| # | Unit | Review | Status | Verification seam / notes |
|---|------|--------|--------|---------------------------|
| R2-1 | **Streaming**: `InferenceEngine.stream_completion(...) -> AsyncIterator[Chunk]` (llama-cpp `stream=True`), route tokens into the existing `THINKING` event; parser still runs on *final* buffered content via `grammar.parse`. Add to `engine.py` protocol + `local.py` + TUI/headless render. | R2§1 | ⏳ NEXT | `local.py` count_tokens/to_thread present; **no `stream` verb yet**. Grammar registry (R1-3) is the seam that decides chunk=tool-call vs prose. |
| R2-2 | **Parallel read-only tool execution**: group SAFE/read-only `_READ_ONLY_TOOLS` calls → `asyncio.gather`; keep SAME-PATH + mutating calls serial. `classifier.py:_READ_ONLY_TOOLS` is the source of the safe set. | R2§2 | ⏳ PENDING | loop.py `for tool_call in` is serial today (no `asyncio.gather`). ~40 lines + a same-path guard. |
| R2-3 | **Cache-stable tool prefix**: freeze the rendered tool-definition block once per engine (keyed by tool-filter); append conversation as pure suffix. | R2§3 | ⏳ PENDING | `_inject_tools` (local.py:317) re-renders every call today. |
| R2-4 | `execute_shell`: **stream** stdout to the TUI as it arrives + **background** handle/`tail`. | R2§6 | ⏳ PENDING | after R2-1's stream primitive exists. |
| R2-5 | **LLM compaction tier**: optional summarizer pass (cheapest local engine / fallback) to summarize a large dropped window, replacing pure extractive glue. | R2§5 | ⏳ PENDING | Gate on R2-1/R2-2 measured first. |
| R2-6 | **Autotune**: persist `_build_metrics`/`_build_run_stats` to `~/.local/share/natshell/metrics/`, feed back into `n_ctx`/`max_tokens` selection (`scaling.py`, `_infer_context_size`). | R2§7 | ⏳ PENDING | Long-term; removes guesswork tables. |
| — | R2§4 "real tokenizer in budget" | R2§4 | ✅ DONE | `tokenizer_fn = self.engine.count_tokens` already wired into `ContextManager`. |

## Changelog (newest first)

- **2026-09-08** R1-7 (first sub-step, `0873256`): extracted
  `_inject_intent` / `_preflight_compaction` / `_apply_inference_feedback`
  out of `handle_user_message` (491→437 lines); behaviour byte-identical,
  covered by existing intent + proactive-compaction tests. Suite **1614 green**.
- **2026-09-08** R1-6 DONE: `agent/recovery.py` — ordered recovery ladder +
  per-run latch extracted from `handle_user_message` (except block 130→11
  lines); `RecoveryCoordinator.handle` + injected collaborators; 16 new
  tests in `tests/test_recovery.py`; push `8707db9`; suite **1614 green**.
- **2026-09-08** R1-3 finalized: gemma family + registries landed (`350523a`)
  on `refactor/simplify-core`; local.py 927→378 lines; full suite **1598 green**.
  Plan file created; standing rules recorded.
- **2026-09-04** R1-5 (repetition guard/sudo retry/plan), common grammar helpers,
  qwen+mistral family modules landed.
