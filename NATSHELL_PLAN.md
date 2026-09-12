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
| R1-2 | Unify 3 context managers into one `ContextStore` | R1§2 | ✅ ~DONE | `_compress_old_messages` delegates to `ContextManager.compress_artifacts`; both summaries use shared `build_summary`. **R2-5a (`7660723`) further resolved the residual:** the two inline summary-marker constructions (`trim_messages` budget path + `compact_history` manual path) are now unified on a shared `ContextManager.context_marker()` helper (both keep their distinct preambles). Only residual left: the two paths still pass different preambles (intentional — different user-facing cases). |
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
| R2-1 | **Streaming**: `InferenceEngine.stream_completion(...) -> AsyncIterator[Chunk]` (llama-cpp `stream=True`), route tokens into the existing `THINKING` event; parser still runs on *final* buffered content via `grammar.parse`. Add to `engine.py` protocol + `local.py` + TUI/headless render. | R2§1 | ✅ DONE | Engine-level streaming (`9fcb768`) + TUI/headless token routing (`26c32a8`): `THINKING_TOKEN` event drained by the loop between THINKING and the first outcome (stream-failure → blocking fallback → recovery ladder), `ThinkingBlock` promotes the indicator on first token and is superseded by the terminal PLANNING/RESPONSE message (no duplication), headless no-ops the type. 16 new tests in `tests/test_r21_tui_streaming.py`. |
| R2-2 | **Parallel read-only tool execution**: group SAFE/read-only `_READ_ONLY_TOOLS` calls → `asyncio.gather`; keep SAME-PATH + mutating calls serial. `classifier.py:_READ_ONLY_TOOLS` is the source of the safe set. | R2§2 | ✅ DONE | `tool_dispatch.dispatch_tool_batch` ships segment-aware batching: runs of `PARALLEL_SAFE_TOOLS` (= `_READ_ONLY_TOOLS`, all 5 verified concurrency-free) go via `asyncio.gather`; mutating/guard-stateful calls keep the serial path. Event/exchange order = in-batch concatenation (byte-identical to serial). Guard stop halts later segments like the old `break`; guard still fires under concurrency (observe() is sync per-coroutine). 12 new tests; suite 1667 green. |
| R2-3 | **Cache-stable tool prefix**: freeze the rendered tool-definition block once per engine (keyed by tool-filter); append conversation as pure suffix. | R2§3 | ✅ DONE | `_inject_tools` memoizes the rendered block per (family, compact-tier, sha256 of canonical tool JSON) in a 32-entry LRU on the engine. Byte-identical output (pinned vs `grammar.render_tools`); 6 new tests in tests/test_tool_prefix_cache.py; suite 1673 green. |
| R2-4 | `execute_shell`: **stream** stdout to the TUI as it arrives + **background** handle/`tail`. | R2§6 | ✅ DONE | **Streaming half shipped (`8348e46` + `6d4c03d`, 2026-09-10):** `stream_execute_shell` (asyncio subprocess, 4 KiB stdout pump, concurrent stderr drain, `asyncio.timeout` → kill + exit-124 parity, missing-shell 127 parity) reuses the *same* `_effective_timeout` / `_filtered_env` / `_prepare_sudo` / `_scrub_sudo_prompt` helpers the blocking path now uses (refactored onto them, no behaviour change). `ToolRegistry.register_streaming`/`execute_streaming` own "which tools stream"; `dispatch_tool_call/batch` gained a `stream_output` switch (default **False** ⇒ every existing caller, headless, plans and the mocked `tools.execute` tests keep the historical path) that surfaces each chunk as `EventType.TOOL_OUTPUT` in arrival order between EXECUTING and TOOL_RESULT. TUI: `run_agent`/`run_plan` opt in; `app.py` routes TOOL_OUTPUT → `CommandBlock.set_partial` (new `set_partial`/final-`set_result` pair, copyable mid-stream), verified end-to-end via a Textual pilot. Headless: TOOL_OUTPUT is a documented no-op. 16 new tests (`tests/test_stream_execute_shell.py`): chunk order, byte-parity (truncation + sudo scrub), timeout-124 shape, sudo stdin transport + pkg `y\n`, 127, loop event ordering + default-off, TUI partial→final. **Background half shipped (`bd6865e`, 2026-09-11):** `shell_bg` tool (`tools/shell_bg.py`) — three SAFE-ish actions over a `0o700` `data_dir()/bg` handle dir: `launch` (detached `start_new_session` child, stdout+stderr → `<id>.log`, 16-hex atomic handle JSON, daemon reaper, session-scoped Popen registry, post-restart pid fallback, `killpg` SIGTERM→SIGKILL, `clean_orphans()` >24h at startup). `_RW_LOCK` serializes reaper/kill read-modify-write (found+fixed a real clobber race in testing). Classifier: launch classifies exactly like execute_shell (BLOCKED stays BLOCKED even "detached"); tail/kill CONFIRM→SAFE-in-danger. Out of `SMALL_CONTEXT_TOOLS`+`PLAN_SAFE_TOOLS`. 33 new tests (`tests/test_shell_bg.py`). Suite **1722 green** (was 1689). |
| R2-5 | **LLM compaction tier**: optional summarizer pass (cheapest local engine / fallback) to summarize a large dropped window, replacing pure extractive glue. | R2§5 | 🔄 IN PROGRESS | **Seam shipped (`7660723`, 2026-09-12):** `ContextManager.summarizer` callback + `summarize()` (custom return replaces the extractive `build_summary`; empty/raising falls back to extractive) + `context_marker()` shared by both `trim_messages` (budget) and `compact_history` (manual/forced) paths — kills the R1-2 residual (inline marker) and gives the LLM tier one extension point. Zero behaviour change by default; 8 new tests; suite **1745 green**. **Remaining (core):** the actual async summarizer pass — a `summarize_messages` on the local engine, wired into `ContextManager` via the seam, with a **timeout + extractive fallback** so a slow/failed local model never hangs a compaction; config toggle `[compaction] llm = true` persisted; verify with Qwen3-4B on a real dropped window. The R2-1/R2-2 gate is now satisfiable (R2-1 fully shipped, engines present locally). |
| R2-6 | **Autotune**: persist `_build_metrics`/`_build_run_stats` to `~/.local/share/natshell/metrics/`, feed back into `n_ctx`/`max_tokens` selection (`scaling.py`, `_infer_context_size`). | R2§7 | ⏳ PENDING | Long-term; removes guesswork tables. |
| — | R2§4 "real tokenizer in budget" | R2§4 | ✅ DONE | `tokenizer_fn = self.engine.count_tokens` already wired into `ContextManager`. |

## Changelog (newest first)

- **2026-09-12** R2-5a SEAM DONE (`7660723`) — the LLM-compaction-tier
  extension point lands with zero default behaviour change. `ContextManager`
  gains an optional `summarizer` (constructor arg or `cm.summarizer`);
  `summarize()` honours it and falls back to the extractive `build_summary`
  on an empty/raising return (an LLM tier can never break or empty a
  compaction). The two duplicated marker constructions (budget `trim_messages`
  + manual `compact_history`) unify on `context_marker()` — also resolving
  the R1-2 inline-marker residual — with both marker shapes byte-identical.
  `compact_history` now flows through the same seam and reuses the resolved
  summary in stats (summarizer invoked once). 8 new tests; **1745 green**.
  Core R2-5 (the async LLM pass + config toggle + local-engine wiring) is
  recorded IN PROGRESS for the next tick.
- **2026-09-12** R2-1 TUI-TOKEN FOLLOW-UP DONE (`26c32a8`) — streamed model
  tokens now reach the TUI. The loop drains `StreamingEngine` conformers via
  a new `_stream_response()` helper: every delta surfaces as a
  `THINKING_TOKEN` event between THINKING and the first outcome, the
  terminal `CompletionResult` (same parse pipeline as the blocking call)
  feeds every downstream branch unchanged; a failing stream falls back to
  `chat_completion` *before* the recovery ladder is consulted, and a
  failure surviving both paths still reaches that ladder. The TUI's
  `_render_agent_event` grows a new `ThinkingBlock` on the first token
  (promoting the `ThinkingIndicator` with its running clock so the timer
  doesn't reset), keeps the partial text copyable, and removes the
  placeholder when the terminal PLANNING/RESPONSE/… event mounts the
  canonical message — no duplication. Headless documents
  `THINKING_TOKEN` as a no-op (no per-token stderr spam). 16 new tests in
  `tests/test_r21_tui_streaming.py` (loop ordering, both-fail → recovery,
  mid-stream crash, non-streaming engines, widget body/elapsed/copy, TUI
  promotion + supersession via a Textual pilot, headless no-op). Suite
  **1738 green** (was 1722), ruff clean. R2-1 is now fully DONE.
- **2026-09-11** R2-4 BACKGROUND HALF DONE (`bd6865e`) — the `shell_bg`
  background-process tool lands, completing R2-4. `launch` spawns a detached
  `bash -c` child (`start_new_session`, so `killpg` takes the whole tree),
  appends stdout+stderr to `<id>.log`, and writes a 16-hex atomic handle JSON
  into a `0o700` `data_dir()/bg` dir; a daemon reaper reaps the child and a
  session-scoped Popen registry keeps it addressable (post-restart status/kill
  fall back to the pid), `tail` reports status + log tail, `kill` does
  SIGTERM→SIGKILL after 2 s, and `clean_orphans()` sweeps handles >24 h at
  startup. Testing surfaced + fixed a real reaper/kill read-modify-write
  clobber race (new `_RW_LOCK` + `_update_handle` merge). The classifier
  classifies `launch` *exactly* like `execute_shell` (a BLOCKED command stays
  BLOCKED even when "detached"); `tail`/`kill` are CONFIRM, SAFE under
  `--danger-fast`. Deliberately kept out of `SMALL_CONTEXT_TOOLS`/
  `PLAN_SAFE_TOOLS`. 33 new tests in `tests/test_shell_bg.py`; ruff clean.
  Suite **1722 green** (was 1689). Remaining plan: R2-1 TUI-token follow-up,
  R2-5, R2-6.
- **2026-09-10** R2-4 STREAMING HALF DONE (`8348e46`, `6d4c03d`) — `execute_shell`
  now streams its stdout into the TUI chunk-by-chunk. `stream_execute_shell`
  reuses the blocking path's timeout/env/sudo/scrub helpers (refactored
  onto them, no behaviour change); `ToolRegistry.register_streaming` owns
  "which tools stream"; a new `stream_output` switch (default **False**) in
  `dispatch_tool_*` surfaces each chunk as `EventType.TOOL_OUTPUT` between
  EXECUTING and TOOL_RESULT; the TUI's `run_agent`/`run_plan` opt in and
  render into `CommandBlock.set_partial` (new partial/final+copy pair),
  verified end-to-end through a Textual pilot; headless no-ops the type.
  16 new tests in `tests/test_stream_execute_shell.py`; suite **1689 green**
  (was 1673), ruff clean. The background handle/`tail` sub-step is designed
  (recorded in the R2-4 row) but not yet built; R2-1's TUI-token follow-up
  and R2-5/R2-6 remain pending.

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
