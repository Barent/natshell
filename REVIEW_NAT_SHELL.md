# NatShell — Two Reviews

Reviewed against the working tree in `/home/barent/natshell` (v1.0.47, commit `920fed5`).
Footnotes cite `file:line` so every claim is checkable. I read the full hot path, not just the signatures.

**Shape of what we have**

| Area | File | Size | Role |
|------|------|------|------|
| Agent brain | `agent/loop.py` | 1398 | ReAct loop, *all* of it |
| Local inference | `inference/local.py` | 927 | llama-cpp, 3 tool-call grammars |
| Context mgmt | `agent/context_manager.py` (+ 2 more in loop.py) | 270 | trimming / compaction (×3) |
| TUI | `app.py` | 1573 | god object |
| Safety | `safety/classifier.py` | 333 | regex risk tiers |
| Shell tool | `tools/execute_shell.py` | 411 | `bash -c`, all-or-nothing |
| **Totals** | | **~15,700 src / ~17,600 tests** | tests exceed source — a real asset |

The test suite (45 files, 17.6k lines) is larger than the codebase. That is the single most important fact in this document: it means aggressive refactoring is *cheap and safe* here in a way it wouldn't be anywhere else. Almost every recommendation below can be made with the tests as the safety net.

---

# REVIEW 1 — Reduce complexity, simplify the program

## 1. `handle_user_message` is a 790-line god-method. Split it. — *highest leverage*

`loop.py:428–1218` is one `async` generator that, in a single body:

- injects intent (planning/analysis) system messages (`455–478`),
- runs the step loop, compression, pre-flight compaction (`534–556`),
- calls inference + calibrates + proactive-compacts (`559–593`),
- handles context overflow (`594–622`), remote fallback phase 1+2 with context restoration and GPU warnings (`624–712`),
- handles degenerate output (`714–735`), truncated output (`737–779`),
- executes every tool call: sudo-password retry (`846–899`), then **six** separate repetition guards appended inline as warning strings — duplicate tool (`919–953`), re-read (`955–968`), re-fetch URL (`970–992`), command-family (`994–1041`), near-identical command (`1043–1083`) — plus completion guard (`1141–1161`).

This is where bugs live and where every feature (Review 2) will have to slot in. Right now adding anything means editing a 790-line function with no seams.

**Extract into objects, keep behaviour byte-identical:**

1. `RepetitionGuard` — one dataclass holding the six counters (`loop.py:152–167`) and a single `observe(tool_call, result) -> Observation` that returns `(suffix_to_append: str, stop_running: bool)`. Today those six blocks are copy-paste string-building with thresholds buried in the method body. One test file (`tests/` already has `test_message_queue`, `test_agent`) becomes the owner. **Delete ~200 lines of method body.**
2. `RecoveryStrategy` — the overflow + fallback + context-restore + GPU-warning ladder (`594–712`). It's an ordered state machine ("attempt compact → retry → fallback → restore") dressed as a nested `if/elif`. Model it as steps. This is the highest-risk code in the file (it touches engine swap and conversation state).
3. `SudoRetry` — `846–899` is a self-contained "run, detect sudo prompt, prompt, re-classify, confirm, rerun" with an awkward `continue` mid-loop. It's already conceptually a helper; make it one and the main loop just reads "did it succeed."

**Migration is safe** because each block has clear in/out types and the tests already exercise the surrounding behaviour. Do it in three PRs — `RepetitionGuard` first (purest, lowest risk), `SudoRetry` second, `RecoveryStrategy` last.

## 2. Context management exists in three overlapping implementations.

- `ContextManager.trim_messages` + `build_summary` (`context_manager.py:111–270`)
- `loop._compress_old_messages` (`loop.py:1244–1288`) — a *second* compressor for write_file arg elision + long-tool-result truncation
- `loop.compact_history` (`loop.py:1348–1398`) — a *third* summary builder (also extractive)

Two of the three hand-rolled a summary format and all three re-implement "keep system + recent N, drop oldest, never split a tool-call pair." They are subtly different, which means they give *different* context for the same situation depending on which path fires — that's a correctness landmine, not just duplication.

**Unify into one `ContextStore`:** single owner of the message list, one `fits_within(n_ctx)` predicate, one `summarizer` (see Review 2 §6 for upgrading it), and two public verbs — `compact(budget)` and `compress_artifacts()`. `_compress_old_messages` and `compact_history` become methods on it. `app.py`'s `/compact` and `/undo` then have one target to call.

## 3. The 5-way fallback parse ladder is a registry waiting to happen.

`local.py::_parse_response` (`670–927`) is a 260-line `if not tool_calls:` chain:
structured → Qwen `<tool_call>` XML → Gemma native → Mistral `[TOOL_CALLS]` → bare-JSON-Gemma → bare-JSON-Mistral, plus a parallel *formatter* triad (`_format_tools_for_prompt` / `_mistral` / `_gemma`, `236–410`) and two message normalizers (`_convert_gemma_tool_messages`, `_normalize_messages_strict_alternation`, `501–604`).

The Gemma and Mistral "bare JSON recovery" blocks (`770–866`) are near-identical — the only difference is the family guard. That's a smell: **there's no `ToolCallGrammar` abstraction**, so each family is a copy-paste branch.

Introduce a `ToolCallGrammar` protocol: `render_definitions(tools) -> str`, `parse(content) -> list[ToolCall]`, `normalize(messages)`. One per family (Qwen, Mistral, Gemma). `LocalEngine` picks its grammar from `_detect_model_family` (`133`). This deletes the if-ladder, de-duplicates the two bare-JSON branches, and (bonus) means adding a *fourth* model family is 50 lines in an isolated file instead of a 260-line edit.

**Note the proportionality caveat:** three grammar classes are not over-engineering — there are three real, incompatible wire formats and the code already *is* three formats. The refactor just says what it already says. That's simplification, not growth.

## 4. `app.py` is a 1573-line god object, and you already know this.

40+ methods on `NatShellApp`, and the same two closures are **defined three times** (confirmed: `def confirm_callback` ×3, `def password_callback` ×3 — at `app.py:452/648/778`). Each is a `run_agent`, `run_plan_generation`, `run_plan` — three entry paths that each re-wire confirm + password, then each re-implement the render loop.

Extract:
- A single `RunSession` that owns `(confirm_callback, password_callback, render(event))`. The three `run_*` methods become thin wrappers choosing a `ToolFilter` + step budget and delegating.
- The slash-command table (`SLASH_COMMANDS`, `108`) is already a list — pair it with a dispatcher map instead of the growing `if/elif` in `_handle_slash_command` (`479`).

This is the same "god-method → named objects" move as §1, applied to the TUI.

## 5. Smaller, but each is a real cut

- **Module-level mutable state** is the recurring wart: `execute_shell.configure_limits` / `set_sudo_password` (`tools/execute_shell.py:38–160`) and the loop's global-ish counters are state threaded through globals. Where a tool owns a setting, keep it on the tool instance. This is what makes tests order-dependent and is the reason `_setup_context_manager` has to re-touch `_exec_shell_mod`, `_read_file_mod`, `_edit_file_mod` (`loop.py:368–376`) every time.
- **`_is_plan_request` / `_is_analysis_request`** (`loop.py:46–60`) are two regexes doing NLP-shaped heuristics, living in the agent module, with their own tests. They belong in a `natshell/intent.py`. Pure move, zero risk.
- **`plan_executor.py`** embeds long instruction strings (`_build_step_prompt`) alongside tree-building and budget scaling. Split "prompt templates" from "pure helpers" so the templates are greppable/testable as data.
- **`_inject_sudo_dash_s` + `_has_sudo_invocation` + `needs_sudo_password`** (`execute_shell.py:91–194`) are four free functions doing one thing (sudo plumbing). Group into a `SudoHandler` class — it removes the `from natshell.tools.execute_shell import ...` reach-ins scattered through `loop.py` (`854–857`).

## Suggested end state

```
agent/
  loop.py            # thin: step loop + event stream only (~250 lines)
  repetition_guard.py#  the 6 detectors, one dataclass
  recovery.py        # overflow → fallback → restore state machine
  context_store.py   # the ONE context manager (absorbs context_manager.py)
  intent.py          # _is_plan_request / _is_analysis_request
inference/
  grammars/
    __init__.py      # ToolCallGrammar protocol + registry
    qwen.py  mistral.py  gemma.py
  local.py           # engine wiring only (~300 lines)
app.py               # TUI shell + widget wiring
  run_session.py     # confirm/password/render, shared by all 3 runs
```

---

# REVIEW 2 — What to ADD to improve performance

The honest constraint first: **the hot path is latency-to-first-token**, and today the engine has **no streaming** (confirmed: `inference/` has no `stream` method; one blocking `to_thread` per completion). On a local model every step is "wait for the whole response, then show it." Every other speedup is downstream of that.

Ranked by (visible speedup per unit of added complexity):

## 1. Streaming — *add it; biggest single win*

Give `InferenceEngine` a second verb, `stream_completion(...) -> AsyncIterator[Chunk]`. `llama-cpp-python` already supports `stream=True` (`local.py:636`). The `THINKING` event (`loop.py:528`) is already a placeholder — turn it into "tokens as they arrive."

- **Why first:** it's the difference between "spinner for 12s → text" and "text typing out for 12s." Perceived latency drops by an order of magnitude with zero change to correctness, and it's the natural home for the token events.
- **Risk:** low. `CompletionResult` already carries `finish_reason`/token counts; streaming just yields partials and assembles the same result. The *parser* (Review 1 §3) must be able to parse on the *final* buffered content, not per-token — which is a small design point, not a rewrite.
- **Interaction with Review 1:** you want the grammar abstraction *first*, because a stream has to decide "is this chunk a tool-call or prose?" — that decision is exactly `grammar.parse`.

## 2. Parallel tool execution — *free, and the model already asks for it*

`result.tool_calls` is a **list** (`engine.py:24`), and the loop runs them **serially** (`loop.py:792` `for tool_call in result.tool_calls:`). When the model emits `read_file(a), read_file(b), list_directory(c)` in one turn, you pay 3× latency for work that's 3 independent reads.

Group tool calls by "independence": read-only tools (`_READ_ONLY_TOOLS`, `classifier.py:41`) can run with `asyncio.gather`; anything `SAFE→CONFIRM`-mutating stays serial. **~2–3× faster on multi-file tasks, ~40 lines.** Guard: if two calls touch the same path, keep them serial.

## 3. Make the prompt-cache actually hit — *cheap, and you already pay for the cache*

`local.py:459–469` allocates a `LlamaRAMCache(256 MB)`, but `chat_completion` re-builds the system message with tool definitions **every call** (`_inject_tools`, `649–668`) and re-serializes a growing message list. The llama.cpp prefix cache only helps if the *prefix is byte-stable*.

- Freeze the tool-definition block once per engine (it only changes when the tool filter does), and append conversation as a pure suffix.
- **Effect:** repeated steps re-prefill far less. Free speed on every multi-step run. Low risk.

## 4. Use the real tokenizer in the budget, not `len//4`

`ContextManager._count` falls back to `len//4` unless `tokenizer_fn` is present (`context_manager.py:74`). `LocalEngine.count_tokens` (`local.py:487`) is *there*. Force it on for local engines so `trim_messages` / `calibrate_from_actual` operate on true token counts. Smaller models overshoot less; you trim later (i.e. keep more context — the model does better). Near-zero cost.

## 5. Upgrade "compaction" from extraction to a call on your own model

The summarizers (`build_summary`, `context_manager.py:210`) are extractive string-glue. For *long* tasks, one `chat_completion` to a small model to summarize the dropped window beats line-snipping — and now that you have a fallback engine (`loop._try_local_fallback`), a "summarizer tier" (cheapest local model) can do it. Bigger win, more complexity; do this only after §2–§4.

## 6. `execute_shell`: stream + background for long commands

Today a 300s `make` is all-or-nothing blocking (`execute_shell.py:289–411`), and mid-command you get nothing. Two additions:
- **Stream** stdout to the TUI as it arrives (same `stream` primitive as §1), so the user watches builds progress.
- **Background** option: return a handle immediately, `tail` later. This is what turns "agent can't wait" cases into "agent waits cheaply." Medium complexity, high perceived value for a *system-admin* tool.

## 7. Wire the metrics you already compute into autotune

`_build_metrics` / `_build_run_stats` (`loop.py:89–121`) already produce `tokens_per_sec`, per-step ms, token counts — and then discard them into events. Log them to `~/.local/share/natshell/` and let `n_ctx`/`max_tokens` selection (`gpu.py`, `_infer_context_size`) be *measured* rather than filename-guessed (`local.py:153`). Long-term this removes the guesswork tables in `scaling.py`. Low complexity, compounding value.

## What I'd deliberately NOT add

- **More tool-call fallback regexes.** The ladder exists because three families are messy; adding a fourth parser is solving a non-problem until a real fourth model ships. The §1 grammar registry is the escape hatch that makes "add a family" a 1-file PR — that's enough.
- **A second repetition-detection system.** There are already six (`loop.py:152–167`). More heuristic warnings is exactly the kind of accretion Review 1 is removing. If you want to *add* detection, fold it into the `RepetitionGuard` object, not the loop.
- **Subprocesses-per-call with fresh env** — the env filter in `execute_shell.py:316–321` is correct, but caching it is a micro-win to leave for later.

---

# Analysis — do the two ideas sit together?

They do, and the order matters. They are **complementary, and one is the enabler for the other** — but they pull opposite directions, so you must not do them in the wrong sequence.

**The tension.** Review 2 *adds* surface (streaming, parallelism, autotune). Review 1 *removes* surface (god-method, triple context mgmt, parse ladder). Every performance feature in Review 2 has to plug into a specific seam — and **right now several of those seams don't exist**:

| Feature (R2) | Seam it needs | Exists today? |
|---|---|---|
| Streaming (§1) | a place that decides "chunk = tool call vs prose" | ❌ buried in 260-line `_parse_response` |
| Parallel tools (§2) | a clean iteration point over `result.tool_calls` | ⚠️ serial `for` inside a 790-line method |
| Prompt-cache (§3) | engine owned state (not re-built per call) | ⚠️ rebuilt in `chat_completion` |
| Real tokenizer (§4) | single `ContextManager` | ⚠️ three competing managers |
| Autotune (§7) | a logged, typed metrics stream | ⚠️ computed then discarded |

If you bolt streaming + parallelism onto the current `handle_user_message`, you will be editing that 790-line generator *and* its six inline guards at the same time — which is precisely where new bugs are born. **Simplify first, then the perf seams appear for free.**

**The proportionality read** (which is the right standard here):

- The **first 20% of both reviews** — `RepetitionGuard` extraction, one `ContextStore`, and `ToolCallGrammar` (R1) + streaming, parallel read-only tools, cache-stable prefix (R2) — give **~80% of the felt improvement**. Everything in R2 §5–7 is real but lower-yield *relative* to that, and each one should be gated on "did the previous one measurably help" (which §7's telemetry lets you verify, per the "honest blocking over spinning wheels" principle).
- **Order, concretely:**
  1. R1§1 `RepetitionGuard` + R1§2 `ContextStore` (pure moves, tests as net)
  2. R1§3 `ToolCallGrammar` (creates the streaming seam + kills the ladder)
  3. R2§1 streaming → R2§2 parallel read-only → R2§3/§4 cache-stable + real tokenizer
  4. Then, and only then, R2§6 shell streaming and R2§5/§7.

**Net assessment:** this is a *well-built* harness — the safety layer is genuinely good (fail-closed classifier, `classify_command` splitting on the same tokenizer as the executor, `classifier.py:183`), and a 17.6k-line test suite means the code is *designed* to be refactored. Its one structural problem is **concentration**: three of the five riskiest concerns (loop, parse, TUI) live in single ~800–1400-line methods with god objects around them. Review 1 is about *spreading* that concentration into named, tested units. Review 2 is about *using* the seams that spreading creates. Do R1 to earn R2, and the two are one coherent program, not a choice between two.

*— D.*
