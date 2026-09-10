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
| R1-7 | Slim `handle_user_message` to ~250 lines (orchestrator only) | R1§1 | ✅ DONE | `handle_user_message` 790 → **252 lines** (`0873256` → `920ece4` → `627d9fc`; loop.py file 1048 → 824 lines). Extracted: intent/pre-flight/feedback (method helpers), degenerate + length-truncation outcomes + run stats → `agent/step_metrics.py` (22 tests), tool-call dispatch lifecycle → `agent/tool_dispatch.py` (15 tests). Event order/side effects byte-identical throughout; the loop stays the sole owner of message mutation and flow (continue/return/break). |
| R1-8 | `plan_executor.py`: split prompt-templates from pure helpers | R1§5 | ✅ DONE | `agent/plan_prompts.py` owns `_build_plan_prompt`/`_build_step_prompt`/`_build_verify_fix_prompt`/`VERIFY_FIX_BUDGET`/`_shallow_tree` (400 lines of greppable templates); `plan_executor.py` 429→75 lines keeps `_effective_plan_max_steps` + `validate_plan` and re-exports the moved names (`06381d6`). Prompts byte-identical vs pre-move snapshot; suite 1651 green. |
| — | Small: group `execute_shell` sudo helpers into one `SudoHandler` | R1§5 | ⏳ OPTIONAL | `_inject_sudo_dash_s`/`_has_sudo_invocation`/`needs_sudo_password`/`configure_limits`. |

## Review 2 — performance additions (the real remaining plan)

Ordered per the review's own "simplify first, then the seams appear" ranking.
Each is a self-contained unit: implement → new/extended tests → green → commit →
push. **Verify the seam R1 created before building on it.**

| # | Unit | Review | Status | Verification seam / notes |
|---|------|--------|--------|---------------------------|
| R2-1 | **Streaming**: `InferenceEngine.stream_completion(...) -> AsyncIterator[Chunk]` (llama-cpp `stream=True`), route tokens into the existing `THINKING` event; parser still runs on *final* buffered content via `grammar.parse`. Add to `engine.py` protocol + `local.py` + TUI/headless render. | R2§1 | ✅ CORE DONE | Engine-level streaming shipped (`9fcb768`): `StreamChunk` + `StreamingEngine` protocol (feature-detect), `LocalEngine.stream_completion` async generator (one `to_thread` hop; terminal `CompletionResult` via the *same* `_parse_response` pipeline — tool parse/think-strip/degenerate/overflow all identical). 7 tests in `tests/test_streaming_local.py`. **Follow-up remaining:** route chunks into the TUI `THINKING` widget / headless render (loop.py/app.py) — small, needs an event-thread hop in the Textual app. |
| R2-2 | **Parallel read-only tool execution**: group SAFE/read-only `_READ_ONLY_TOOLS` calls → `asyncio.gather`; keep SAME-PATH + mutating calls serial. `classifier.py:_READ_ONLY_TOOLS` is the source of the safe set. | R2§2 | ✅ DONE | `tool_dispatch.dispatch_tool_batch` ships segment-aware batching: runs of `PARALLEL_SAFE_TOOLS` (= `_READ_ONLY_TOOLS`, all 5 verified concurrency-free) go via `asyncio.gather`; mutating/guard-stateful calls keep the serial path. Event/exchange order = in-batch concatenation (byte-identical to serial). Guard stop halts later segments like the old `break`; guard still fires under concurrency (observe() is sync per-coroutine). 12 new tests; suite 1667 green. |
| R2-3 | **Cache-stable tool prefix**: freeze the rendered tool-definition block once per engine (keyed by tool-filter); append conversation as pure suffix. | R2§3 | ✅ DONE | `_inject_tools` memoizes the rendered block per (family, compact-tier, sha256 of canonical tool JSON) in a 32-entry LRU on the engine. Byte-identical output (pinned vs `grammar.render_tools`); 6 new tests in tests/test_tool_prefix_cache.py; suite 1673 green. |
| R2-4 | `execute_shell`: **stream** stdout to the TUI as it arrives + **background** handle/`tail`. | R2§6 | ⏳ PENDING | after R2-1's stream primitive exists. |
| R2-5 | **LLM compaction tier**: optional summarizer pass (cheapest local engine / fallback) to summarize a large dropped window, replacing pure extractive glue. | R2§5 | ⏳ PENDING | Gate on R2-1/R2-2 measured first. |
| R2-6 | **Autotune**: persist `_build_metrics`/`_build_run_stats` to `~/.local/share/natshell/metrics/`, feed back into `n_ctx`/`max_tokens` selection (`scaling.py`, `_infer_context_size`). | R2§7 | ⏳ PENDING | Long-term; removes guesswork tables. |
| — | R2§4 "real tokenizer in budget" | R2§4 | ✅ DONE | `tokenizer_fn = self.engine.count_tokens` already wired into `ContextManager`. |

## Changelog (newest first)

- **2026-09-10** R2-3 DONE (`a23ebaf`): cache-stable tool prefix —
  `LocalEngine._inject_tools` now memoizes the rendered tool-definition
  block per (family, compact-tier, canonical-tools hash) in a bounded 32-
  entry LRU on the engine instance; output byte-identical to
  `grammar.render_tools`, only the redundant per-call re-render is
  removed. 6 new tests in tests/test_tool_prefix_cache.py. Suite **1673
  green**.
- **2026-09-10** R2-2 DONE (`1ab9651`): parallel read-only tool
  execution — `dispatch_tool_batch` groups a response's tool calls into
  segments; consecutive runs of `PARALLEL_SAFE_TOOLS` (list_directory,
  natshell_help, skill, fetch_url, kiwix_search) execute concurrently via
  `asyncio.gather`, everything else keeps the serial one-at-a-time path.
  Event/exchange ordering remains the in-batch concatenation (serial-
  identical); guard `stop` still halts the later segments exactly as the
  old inline `break`; dupe-abort still fires under concurrency. 12 new
  tests in tests/test_tool_dispatch.py. Suite **1667 green**.
- **2026-09-09** R2-1 CORE DONE (`9fcb768`): engine-level streaming
  shipped — `StreamChunk` + `StreamingEngine` (runtime-checkable) protocol
  in `inference/engine.py`, `LocalEngine.stream_completion` async
  generator (drains llama-cpp `stream=True` in one worker thread, yields
  raw text deltas, then the terminal `CompletionResult` from the *same*
  parse pipeline as the blocking path), 7 new tests in
  `tests/test_streaming_local.py`. Suite **1658 green**. TUI/headless token
  routing is the recorded follow-up.
- **2026-09-09** R1-8 DONE (`06381d6`): plan prompt-templates split out of
  `plan_executor.py` into `agent/plan_prompts.py` (429→75 lines of pure
  helpers + re-exports; rendered prompts byte-identical against a
  pre-move snapshot of 10 tiers/cases). Suite **1651 green**. R1 is now
  fully DONE. Next up: R2-1 (streaming).
- **2026-09-09** R1-7 DONE (`627d9fc`): tool-call dispatch lifecycle
  (normalize → classify → confirm → execute → sudo retry → guard observe →
  budget hint) extracted from `handle_user_message` into
  `agent/tool_dispatch.py`; method is now **252 lines** (started at 790),
  hitting the ~250-line orchestrator target; **15 new tests** in
  `tests/test_tool_dispatch.py`. Suite **1651 green**.
- **2026-09-09** R1-7 sub-step 2 (`920ece4`): degenerate-output +
  length-truncation outcome blocks and the per-run stat counters moved out of
  `handle_user_message` into `agent/step_metrics.py` (437→390 lines); event
  text + ordering byte-identical, think-strip now reuses the shared grammar
  regexes; **22 new tests** in `tests/test_step_metrics.py`. Suite **1636
  green** (was 1614). Pushed to `refactor/simplify-core`.
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
