# Codebase Consolidation — Reduce Size and Complexity, Keep All Functionality

This plan consolidates duplicated patterns in the NatShell codebase. It removes no
features. Every step must leave the full test suite green.

**Rules for the executing agent — read before every step:**

1. Follow each step EXACTLY as written. Do not redesign, rename, or "improve" beyond
   what the step specifies. Where exact code is given, use it verbatim.
2. Modify ONLY the files listed in each step. Do not reformat unrelated code.
   Ruff line-length limit is 100.
3. After each step, run the step's Verify command. If it fails, fix ONLY the failure.
   If still failing after 2 fix attempts, run `git checkout -- .` to discard the
   step's changes, record the step as failed, and move to the next step.
4. Commit after each passing step with the message given in the step.
   Never run `git push`.
5. Run tests with `python3 -m pytest` (no venv in this repo; pytest is on PATH at
   `~/.local/bin/pytest` if `python3 -m pytest` fails).
6. Do not edit test files unless a step explicitly says to.

## 1. Centralize the context-window scaling ladders into one module

The "n_ctx → scaled value" if/elif ladder is duplicated in five places. Replace all
five with lookups into one shared table module.

READ `src/natshell/agent/loop.py` (only lines 260–345)
READ `src/natshell/agent/working_memory.py` (only lines 110–140)
READ `src/natshell/agent/plan_executor.py` (only lines 1–40)
CREATE `src/natshell/scaling.py`
MODIFY `src/natshell/agent/loop.py`
MODIFY `src/natshell/agent/working_memory.py`
MODIFY `src/natshell/agent/plan_executor.py`

First, create a working branch: `git checkout -b refactor/consolidation`

Then create `src/natshell/scaling.py` with EXACTLY this content:

```python
"""Context-window tier scaling — single source of truth for n_ctx → value ladders.

Each table maps a minimum context size to a scaled value, sorted descending.
``scale_for_context`` returns the value for the largest threshold that *n_ctx*
meets or exceeds, or *default* when none match.
"""

from __future__ import annotations

ScaleTable = tuple[tuple[int, int], ...]


def scale_for_context(n_ctx: int, table: ScaleTable, default: int) -> int:
    """Return the scaled value for *n_ctx* from *table*, else *default*."""
    for threshold, value in table:
        if n_ctx >= threshold:
            return value
    return default


# Agent step budget (loop.py _effective_max_steps)
MAX_STEPS_TABLE: ScaleTable = (
    (1048576, 200),
    (524288, 150),
    (262144, 120),
    (131072, 60),
    (32768, 50),
    (16384, 35),
    (8192, 25),
)

# Shell output truncation (loop.py _effective_max_output_chars)
MAX_OUTPUT_CHARS_TABLE: ScaleTable = (
    (1048576, 128000),
    (524288, 96000),
    (262144, 64000),
    (131072, 32000),
    (65536, 16000),
    (32768, 12000),
    (16384, 8000),
)

# read_file default line count (loop.py _effective_read_file_lines)
READ_FILE_LINES_TABLE: ScaleTable = (
    (1048576, 8000),
    (524288, 6000),
    (262144, 4000),
    (131072, 3000),
    (65536, 2000),
    (32768, 1000),
    (16384, 500),
)

# Working-memory character budget (working_memory.py effective_memory_chars)
MEMORY_CHARS_TABLE: ScaleTable = (
    (524288, 32000),
    (262144, 24000),
    (131072, 16000),
    (65536, 12000),
    (32768, 8000),
)

# Per-plan-step agent budget (plan_executor.py _effective_plan_max_steps)
PLAN_MAX_STEPS_TABLE: ScaleTable = (
    (1048576, 120),
    (524288, 90),
    (131072, 65),
    (65536, 55),
    (32768, 45),
    (16384, 35),
    (8192, 30),
)
```

IMPORTANT: the thresholds and values above were transcribed from the current code.
Before replacing anything, compare each table against the existing ladder it
replaces. If any number differs from the current code, THE CURRENT CODE WINS —
fix the table, not the ladder. Behavior must be bit-for-bit identical.

Then make these replacements. Keep every method/function signature, docstring, and
config-override check exactly as it is — only replace the if/elif ladder bodies:

In `src/natshell/agent/loop.py`, add this import near the other natshell imports at
the top of the file:

```python
from natshell.scaling import (
    MAX_OUTPUT_CHARS_TABLE,
    MAX_STEPS_TABLE,
    READ_FILE_LINES_TABLE,
    scale_for_context,
)
```

Replace the body of `_effective_max_steps` (currently around line 282) so the method
reads (keep the existing docstring):

```python
    def _effective_max_steps(self, n_ctx: int) -> int:
        if self.config.max_steps != self._DEFAULT_MAX_STEPS:
            return self.config.max_steps
        return scale_for_context(n_ctx, MAX_STEPS_TABLE, self._DEFAULT_MAX_STEPS)
```

Replace the body of `_effective_max_output_chars` (keep docstring):

```python
    def _effective_max_output_chars(self, n_ctx: int) -> int:
        return scale_for_context(n_ctx, MAX_OUTPUT_CHARS_TABLE, 4000)
```

Replace the body of `_effective_read_file_lines` (keep docstring):

```python
    def _effective_read_file_lines(self, n_ctx: int) -> int:
        return scale_for_context(n_ctx, READ_FILE_LINES_TABLE, 200)
```

DO NOT touch `_effective_max_tokens` — it is formula-based, not a ladder.

In `src/natshell/agent/working_memory.py`, add the import
`from natshell.scaling import MEMORY_CHARS_TABLE, scale_for_context` and replace the
ladder in `effective_memory_chars` so it reads (keep signature and docstring):

```python
def effective_memory_chars(n_ctx: int, base_chars: int = 4000) -> int:
    return scale_for_context(n_ctx, MEMORY_CHARS_TABLE, base_chars)
```

In `src/natshell/agent/plan_executor.py`, add the import
`from natshell.scaling import PLAN_MAX_STEPS_TABLE, scale_for_context` and replace
the ladder in `_effective_plan_max_steps` so it reads (keep signature, docstring,
and the `configured != _DEFAULT_PLAN_MAX_STEPS` early return):

```python
def _effective_plan_max_steps(n_ctx: int, configured: int = _DEFAULT_PLAN_MAX_STEPS) -> int:
    if configured != _DEFAULT_PLAN_MAX_STEPS:
        return configured
    return scale_for_context(n_ctx, PLAN_MAX_STEPS_TABLE, 20)
```

Do NOT modify any test file in this step. The existing tests pin the tier values and
must pass unchanged.

Commit message: `refactor: centralize context scaling ladders in scaling.py`

Verify: python3 -m pytest tests/test_agent.py tests/test_working_memory.py tests/test_plan_execution.py tests/test_tools.py -q

## 2. Consolidate the three headless event-consumption loops

`src/natshell/headless.py` has three near-identical `async for event ... match`
blocks and three identical `_confirm_callback` definitions. Replace them with one
shared consumer and one callback factory. Output written to stdout vs stderr must
not change: in `run_headless` the RESPONSE text goes to STDOUT; everywhere else all
event logging goes to stderr.

READ `src/natshell/headless.py` (entire file, 473 lines)
MODIFY `src/natshell/headless.py`

Add these two shared helpers to `headless.py`, directly below the imports and above
`run_headless` (use EXACTLY this code):

```python
def make_confirm_callback(auto_approve: bool):
    """Build the standard headless confirmation callback."""

    async def _confirm_callback(tool_call: Any) -> bool:
        if auto_approve:
            _err(f"[auto-approved] {tool_call.name}: {tool_call.arguments}")
            return True
        _err(f"[declined — use --danger-fast to auto-approve] {tool_call.name}")
        return False

    return _confirm_callback


async def consume_events(
    stream,
    *,
    emit=None,
    on_response=None,
    on_tool_result=None,
    on_executing=None,
) -> bool:
    """Shared event loop for all headless runners.

    emit: line sink (defaults to _err; exeplan passes its step logger).
    on_response: handler for the final RESPONSE text (required).
    on_tool_result: optional extra hook, called after standard output logging.
    on_executing: optional replacement for the default "[executing] name" line.
    Returns True if any ERROR event was seen.
    """
    if emit is None:
        emit = _err
    had_error = False
    async for event in stream:
        match event.type:
            case EventType.RESPONSE:
                on_response(event.data)
            case EventType.TOOL_RESULT:
                if event.tool_result and event.tool_result.output:
                    emit(event.tool_result.output)
                if event.tool_result and event.tool_result.error:
                    emit(f"[stderr] {event.tool_result.error}")
                if on_tool_result:
                    on_tool_result(event)
            case EventType.PLANNING:
                emit(f"[thinking] {event.data}")
            case EventType.EXECUTING:
                if event.tool_call:
                    if on_executing:
                        on_executing(event)
                    else:
                        emit(f"[executing] {event.tool_call.name}")
            case EventType.BLOCKED:
                if event.tool_call:
                    emit(f"[BLOCKED] {event.tool_call.name}: {event.tool_call.arguments}")
            case EventType.ERROR:
                emit(f"[error] {event.data}")
                had_error = True
            case EventType.RUN_STATS:
                if event.metrics:
                    steps = event.metrics.get("steps", "?")
                    wall = event.metrics.get("total_wall_ms", 0)
                    emit(f"[stats] {steps} steps in {wall}ms")
    return had_error
```

Then convert the three runners:

In `run_headless`: delete the nested `_confirm_callback` and the whole
`async for event ... match` block. Replace with:

```python
    had_error = await consume_events(
        agent.handle_user_message(
            prompt,
            confirm_callback=make_confirm_callback(auto_approve),
        ),
        on_response=lambda data: print(data, flush=True),
    )
```

Keep the existing `return 1 if had_error else 0` logic.

In `run_headless_plan`: delete the nested `_confirm_callback` and the match block.
Keep the surrounding `try/except Exception` that returns 1 on failure. Inside the
try, replace the loop with:

```python
        had_error = await consume_events(
            agent.handle_user_message(
                prompt,
                confirm_callback=make_confirm_callback(auto_approve),
                tool_filter=PLAN_SAFE_TOOLS,
                skip_intent_detection=True,
            ),
            on_response=lambda data: _err(f"[response] {data}"),
        )
```

Everything after the loop (the `had_error` check and the PLAN.md existence check)
stays as-is.

In `run_headless_exeplan`: delete the nested `_confirm_callback` (call
`make_confirm_callback(auto_approve)` at the same place it was defined, storing it
in a variable used inside the step loop). Inside the per-step `try:` block, replace
the `async for event ... match` block with hook closures plus one call. The local
variables `hit_max_steps`, `step_files`, `step_metrics`, `step_tool_count` are
mutated from closures, so declare them `nonlocal` where assigned:

```python
            def _on_response(data: str) -> None:
                nonlocal hit_max_steps
                _log(f"[response] {data}")
                if data and "maximum number of steps" in data:
                    hit_max_steps = True

            def _on_tool_result(event: Any) -> None:
                if (
                    event.tool_call
                    and event.tool_call.name in ("write_file", "edit_file")
                    and event.tool_result
                    and event.tool_result.exit_code == 0
                ):
                    path = event.tool_call.arguments.get("path", "")
                    if path:
                        action = (
                            "created"
                            if event.tool_call.name == "write_file"
                            else "modified"
                        )
                        step_files.append(f"{path} ({action} in step {step.number})")

            def _on_executing(event: Any) -> None:
                nonlocal step_tool_count
                step_tool_count += 1
                elapsed = int(time.monotonic() - step_t0)
                _log(
                    f"[executing] {event.tool_call.name} "
                    f"({step_tool_count}/{effective_max} calls, {elapsed}s)"
                )

            def _on_stats_capture(event: Any) -> None:
                pass  # placeholder removed below if unused

            await consume_events(
                agent.handle_user_message(prompt, confirm_callback=confirm),
                emit=_log,
                on_response=_on_response,
                on_tool_result=_on_tool_result,
                on_executing=_on_executing,
            )
```

CAREFUL — one behavioral detail: the old exeplan loop stored `event.metrics` into
`step_metrics` on RUN_STATS. `consume_events` only logs stats. To preserve this,
extend `consume_events` with one more optional hook `on_stats=None` called with
`event.metrics` inside the RUN_STATS case (after the emit), and in exeplan pass:

```python
            def _on_stats(metrics: dict[str, Any]) -> None:
                nonlocal step_metrics
                step_metrics = metrics
```

Delete the unused `_on_stats_capture` placeholder if you added it. Everything after
the event loop in exeplan (status update, state persistence, summary printing) stays
exactly as-is.

Note: `run_headless_plan` previously ignored BLOCKED events; the shared consumer
logs them to stderr. This is an accepted, strictly-additive logging difference —
plan mode uses PLAN_SAFE_TOOLS so BLOCKED effectively cannot occur.

After the refactor, `headless.py` must contain exactly one `match event.type:`
block (inside `consume_events`) and exactly one `async def _confirm_callback`
(inside `make_confirm_callback`). Check with:
`grep -c "match event.type" src/natshell/headless.py` → must print 1.

Commit message: `refactor: single shared event consumer for headless runners`

Verify: python3 -m pytest tests/test_headless.py tests/test_plan_execution.py tests/test_plan_state.py -q

## 3. Move natshell_help static topics from Python strings to markdown files

`src/natshell/tools/natshell_help.py` embeds ~400 lines of documentation as Python
string literals in `_STATIC_TOPICS`. Move each topic to a markdown data file and
load on demand. The 3 dynamic topics (`config`, `config_reference`, `safety`) stay
in Python unchanged.

READ `src/natshell/tools/natshell_help.py` (entire file, 506 lines)
CREATE `src/natshell/tools/help_topics/` (directory of 17 .md files)
MODIFY `src/natshell/tools/natshell_help.py`
MODIFY `pyproject.toml`

Step 3a — export the existing text VERBATIM (do not retype it by hand). Run this
one-off script from the repo root, then delete it:

```bash
python3 - <<'EOF'
import sys
sys.path.insert(0, "src")
from pathlib import Path
from natshell.tools.natshell_help import _STATIC_TOPICS
out = Path("src/natshell/tools/help_topics")
out.mkdir(exist_ok=True)
for topic, text in _STATIC_TOPICS.items():
    (out / f"{topic}.md").write_text(text, encoding="utf-8")
print(sorted(p.name for p in out.iterdir()))
EOF
```

Expected: 17 files — overview.md, commands.md, tools.md, models.md,
troubleshooting.md, kiwix.md, getting_started.md, profiles.md, sessions.md,
plans.md, plugins.md, headless.md, mcp.md, backup.md, prompt_customization.md,
memory.md, keyboard_shortcuts.md. If the count differs, list `_STATIC_TOPICS`
keys and reconcile before continuing.

Step 3b — in `natshell_help.py`, delete the entire `_STATIC_TOPICS = { ... }` dict
and add these loaders in its place:

```python
from importlib import resources


def _topics_dir():
    return resources.files("natshell.tools").joinpath("help_topics")


def _load_static_topic(topic: str) -> str | None:
    """Read a static help topic from the bundled markdown files."""
    try:
        return _topics_dir().joinpath(f"{topic}.md").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None


def _static_topic_names() -> list[str]:
    try:
        return sorted(
            entry.name[:-3]
            for entry in _topics_dir().iterdir()
            if entry.name.endswith(".md")
        )
    except (FileNotFoundError, OSError):
        return []
```

Update the two usages:

- `VALID_TOPICS = sorted(list(_STATIC_TOPICS.keys()) + list(_DYNAMIC_TOPICS.keys()))`
  becomes `VALID_TOPICS = sorted(_static_topic_names() + list(_DYNAMIC_TOPICS.keys()))`
- In the `natshell_help(topic)` handler, replace the
  `if topic in _STATIC_TOPICS: return ToolResult(output=_STATIC_TOPICS[topic])`
  branch with:

```python
    static_text = _load_static_topic(topic)
    if static_text is not None:
        return ToolResult(output=static_text)
```

Keep `_DYNAMIC_TOPICS`, `set_safety_config`, and all `_topic_*` functions unchanged.

Step 3c — in `pyproject.toml`, extend the `[tool.setuptools.package-data]` entry so
the markdown ships in the wheel. Change:

```toml
natshell = ["config.default.toml", "ui/styles.tcss", "skills/**/*", "skills/**/.*"]
```

to:

```toml
natshell = [
    "config.default.toml",
    "ui/styles.tcss",
    "skills/**/*",
    "skills/**/.*",
    "tools/help_topics/*.md",
]
```

Step 3d — check for direct references before running tests:
`grep -rn "_STATIC_TOPICS" src/ tests/`. If any test references `_STATIC_TOPICS`,
update ONLY those references to use `_load_static_topic(name)` /
`_static_topic_names()`, keeping every assertion's expected content unchanged.

Commit message: `refactor: load help topics from bundled markdown files`

Verify: python3 -m pytest tests/test_natshell_help.py tests/test_tools.py -q

## 4. Extract engine-fallback helpers from the agent loop

Move the three fallback-related members out of `AgentLoop`
(`_describe_remote_error`, `_can_fallback`, and the engine-construction part of
`_try_local_fallback`) into a new `agent/fallback.py`. Keep thin delegating methods
on `AgentLoop` with the SAME names and signatures so all existing callers and tests
keep working.

READ `src/natshell/agent/loop.py` (only lines 1230–1350)
CREATE `src/natshell/agent/fallback.py`
MODIFY `src/natshell/agent/loop.py`

Create `src/natshell/agent/fallback.py`. Copy the bodies from `loop.py` VERBATIM,
adjusting only `self.X` references to the function parameters:

```python
"""Remote-engine failure classification and local-fallback engine loading."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
```

Then three functions:

- `def describe_remote_error(error: Exception) -> str:` — body copied verbatim from
  `AgentLoop._describe_remote_error` (it uses no self state).
- `def can_fallback(error: Exception, engine: Any, fallback_config: Any) -> bool:` —
  body copied from `AgentLoop._can_fallback`, with `self.engine` →
  `engine` and `self.fallback_config` → `fallback_config`.
- `async def load_fallback_engine(fallback_config: Any):` — the engine-building part
  of `AgentLoop._try_local_fallback`: resolve the model path (the
  `fallback_config.path == "auto"` branch), return `None` if the path does not
  exist, construct `LocalEngine` via `asyncio.to_thread`, do the
  GPU-offload warning check, and return the engine (or `None` on any exception,
  with `logger.exception(...)` preserved). It must NOT call `swap_engine` — that
  stays in the loop.

In `loop.py`, replace the three originals with thin delegates (keep docstrings as
one-liners):

```python
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
```

Note the original `_try_local_fallback` returned False when
`self.fallback_config is None` — `load_fallback_engine` must keep that guard
(`if fallback_config is None: return None`).

Commit message: `refactor: extract engine fallback helpers to agent/fallback.py`

Verify: python3 -m pytest tests/test_agent.py tests/test_engine_swap.py tests/test_engine_preference.py -q

## 5. Register kiwix_search through a built-in skill instead of the core registry

`kiwix_search` is the most niche core tool. The skill system already supports
per-skill `tools.py` registration (`skills/__init__.py` calls
`module.register(tool_registry)` during `load_skills`), and every entry mode —
TUI, headless, and MCP — calls `load_skills` (`__main__.py:431`) before use. Move
kiwix registration into a new built-in `kiwix` skill. The module
`src/natshell/tools/kiwix_search.py` itself does NOT move.

READ `src/natshell/tools/registry.py` (only lines 240–290, `create_default_registry`)
READ `src/natshell/skills/web-research/SKILL.md` (to copy frontmatter style)
CREATE `src/natshell/skills/kiwix/SKILL.md`
CREATE `src/natshell/skills/kiwix/tools.py`
MODIFY `src/natshell/tools/registry.py`

Create `src/natshell/skills/kiwix/SKILL.md` using the same frontmatter format as
the existing built-in skills (read `web-research/SKILL.md` first and match its
header layout exactly — `name: kiwix`, one-line description under 200 chars):

```markdown
---
name: kiwix
description: Offline wiki/documentation search via a local Kiwix server (provides the kiwix_search tool)
---

# Kiwix offline search

The `kiwix_search` tool queries a local Kiwix server (default port 8080) for
offline copies of Wikipedia, Stack Exchange, and other ZIM archives.

- Use `kiwix_search` when the user asks factual/reference questions and no
  internet access is available.
- The server URL comes from `[kiwix]` in config.toml and is auto-discovered at
  startup when unset.
- Results return titles, snippets, and URLs readable with `fetch_url`.
```

Create `src/natshell/skills/kiwix/tools.py`:

```python
"""Registers the kiwix_search tool when the kiwix skill is enabled."""


def register(registry):
    from natshell.tools.kiwix_search import DEFINITION, kiwix_search

    registry.register(DEFINITION, kiwix_search)
```

In `src/natshell/tools/registry.py`, inside `create_default_registry()`:

- Delete the two import lines
  `from natshell.tools.kiwix_search import DEFINITION as KIWIX_DEF` and
  `from natshell.tools.kiwix_search import kiwix_search`.
- Delete the line `registry.register(KIWIX_DEF, kiwix_search)`.

Do NOT remove `"kiwix_search"` from `PLAN_SAFE_TOOLS` or `SMALL_CONTEXT_TOOLS` —
those are name allowlists and must keep working when the skill registers the tool.
Do NOT touch the kiwix URL wiring in `__main__.py` (lines ~452–456) — it configures
module state, independent of registration.

Then check for tests that assert kiwix is in the DEFAULT registry:
`grep -rln "kiwix" tests/`. In any test that builds `create_default_registry()` and
expects `kiwix_search` in `tool_names()`, update the expectation: the default
registry no longer contains it. `tests/test_kiwix_search.py` imports the module
directly and needs no changes.

Add one new test at the end of `tests/test_skills.py` (adapt imports/fixtures to
match that file's existing style after reading it):

```python
def test_kiwix_skill_registers_tool():
    """The built-in kiwix skill's tools.py registers kiwix_search."""
    from natshell.skills.kiwix import tools as kiwix_tools  # noqa: F401  (path check)


def test_kiwix_tools_register():
    from natshell.tools.registry import ToolRegistry
    import importlib.util
    from pathlib import Path

    tools_py = (
        Path(__file__).parent.parent
        / "src" / "natshell" / "skills" / "kiwix" / "tools.py"
    )
    spec = importlib.util.spec_from_file_location("kiwix_skill_tools", tools_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    registry = ToolRegistry()
    module.register(registry)
    assert "kiwix_search" in registry.tool_names()
```

If `src/natshell/skills/kiwix/` has no `__init__.py` and the first test's import
fails for that reason, delete the first test and keep only `test_kiwix_tools_register`
(built-in skills are data directories, not packages).

Also check whether `tests/test_skills.py` asserts a built-in skill COUNT (search
for `10`): the built-in count is now 11 — update any such assertion.

Commit message: `refactor: ship kiwix_search via built-in kiwix skill`

Verify: python3 -m pytest tests/test_kiwix_search.py tests/test_skills.py tests/test_tools.py tests/test_mcp_server.py -q

## 6. Update CLAUDE.md to match the refactored code

Documentation-only step. NOTE: an earlier idea to delete the Mistral
`[TOOL_CALLS]` parser from `inference/local.py` was investigated and REJECTED —
Mistral Nemo 12B is still the bundled `enhanced` tier (`model_manager.py:46`), so
the parser is live code. Do not remove it.

READ `CLAUDE.md`
MODIFY `CLAUDE.md`

Make exactly these edits:

1. In the `### Utilities` section, add:
   `- \`scaling.py\` — shared n_ctx→value tier tables (\`scale_for_context\`); used by loop, working_memory, plan_executor`
2. In the `### Agent` section, add:
   `- \`agent/fallback.py\` — remote-error classification and local fallback engine loading`
   and add lines for `agent/working_memory.py` and `agent/plan_state.py` if absent
   (one-line descriptions based on their module docstrings).
3. In the `### Tools` section: add `- \`tools/kiwix_search.py\` — offline Kiwix
   wiki search (registered via the built-in kiwix skill, not the default registry)`
   and `- \`tools/update_config.py\`` with a one-line description from its
   docstring. Note under `natshell_help.py` that static topics live in
   `tools/help_topics/*.md`.
4. In the `### Core` section, add a line for `setup_wizard.py` (one-line
   description from its module docstring).
5. Change "ships 10 built-ins" to "ships 11 built-ins" in the skill system bullet
   (key design decision #17).
6. In the `## Testing` section, update the test count: run
   `python3 -m pytest --collect-only -q 2>/dev/null | tail -1` and use the real
   number, and update the file count (`ls tests/test_*.py | wc -l`).

Commit message: `docs: sync CLAUDE.md with consolidation refactor`

Verify: grep -q "scaling.py" CLAUDE.md && grep -q "11 built-ins" CLAUDE.md && echo CLAUDE-OK

## 7. Full-suite verification and summary

No source changes in this step (fixes only if something fails).

1. Run the entire test suite: `python3 -m pytest -q`. Every test must pass. If any
   fail, identify which earlier step caused it (`git log --oneline` shows one
   commit per step) and fix within that step's file set only.
2. Run the linter: `python3 -m ruff check src/ tests/` (or `~/.local/bin/ruff
   check src/ tests/`). Fix any new violations in files this plan touched.
3. Confirm the size reduction: `git diff main --stat | tail -3` and report the
   total insertions/deletions.
4. Confirm no feature loss with these spot checks (all must succeed):
   - `python3 -c "import sys; sys.path.insert(0,'src'); from natshell.scaling import scale_for_context, MAX_STEPS_TABLE; assert scale_for_context(262144, MAX_STEPS_TABLE, 15) == 120"`
   - `python3 -c "import sys; sys.path.insert(0,'src'); from natshell.tools.natshell_help import VALID_TOPICS; assert 'overview' in VALID_TOPICS and 'safety' in VALID_TOPICS, VALID_TOPICS"`
   - `grep -c "match event.type" src/natshell/headless.py` prints `1`
5. Write a short summary of: steps completed, steps failed/skipped (if any), test
   count before/after (must be equal or higher), and net line delta.

Commit message (only if fixes were made): `fix: post-refactor cleanups from full-suite run`

Verify: python3 -m pytest -q
