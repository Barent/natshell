# NatShell Skill System — Implementation Plan

Status: ready for implementation.
Target implementer: Claude Sonnet (or any capable agent) given this document plus the natshell repo.

---

## 1. Goals

Replace NatShell's plugin system (`src/natshell/plugins.py`) with a **skill** system modeled on Claude Code / Anthropic skills.

A **skill** bundles:
- Model-facing instructions (a markdown body),
- Optional reference docs (read on demand),
- Optional helper scripts (executed on demand),
- Optional tool registration (subsumes the old plugin contract).

Every discovered skill's `name + description` is listed in the system prompt. The model invokes a new `skill` tool to load the full body when a user task matches.

Ship **10 starter skills** inside the package covering common task classes.

The plugin loader is removed. Power users who relied on `~/.config/natshell/plugins/*.py` migrate by wrapping their `register()` function inside a skill's `tools.py`.

Design choices already locked in (do not relitigate):
- Skills subsume plugins (no coexistence).
- Activation is model-driven via descriptions.
- Built-in skills ship inside the package at `src/natshell/skills/`.

---

## 2. Skill anatomy

Each skill is a directory:

```
<skill-name>/
  SKILL.md            REQUIRED  YAML frontmatter + instructions body
  references/         OPTIONAL  longer docs cited from SKILL.md, read on demand via read_file
    *.md
  scripts/            OPTIONAL  helper Python/shell scripts the skill body may run via run_code
    *.py / *.sh
  tools.py            OPTIONAL  `def register(registry): ...` for custom ToolDefinition registration
```

`SKILL.md` format:

```markdown
---
name: spreadsheet
description: Read, write, and transform CSV/XLSX files. Use when the user asks about Excel, CSV, tabular data, or bulk row operations.
---

# Spreadsheet skill
<body>
```

Hard rules (validated at load time, violations skipped with a stderr warning):
- `name` must match `^[a-z][a-z0-9-]{1,31}$` and equal the directory name.
- `description` must be a single line, ≤ 200 chars, non-empty.
- Frontmatter delimiters are `---` on their own lines at the very start of the file.
- Body has no length limit; only the description is injected into the system prompt.

---

## 3. Frontmatter parser (no PyYAML)

Add a small private helper in `src/natshell/skills.py`. Do **not** add a PyYAML dependency.

```python
import re
from pathlib import Path

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?\n)---\s*\n(.*)$", re.DOTALL)
_KV_RE = re.compile(r"^([a-z_][a-z0-9_]*):\s*(.*?)\s*$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")

def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (metadata, body). Raises ValueError on malformed frontmatter."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError("missing or malformed frontmatter")
    meta_block, body = m.group(1), m.group(2)
    meta: dict[str, str] = {}
    for line in meta_block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        kv = _KV_RE.match(line)
        if not kv:
            raise ValueError(f"bad frontmatter line: {line!r}")
        key, val = kv.group(1), kv.group(2)
        # Strip surrounding quotes if present
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        meta[key] = val
    return meta, body
```

Only `name` and `description` are honored. Other keys are tolerated for forward compatibility.

---

## 4. `src/natshell/skills.py` — full module spec

Replace `src/natshell/plugins.py` with a new module:

```python
"""Skill discovery, registration, and access."""
from __future__ import annotations

import importlib
import importlib.resources
import importlib.util
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from natshell.config import NatShellConfig
from natshell.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

USER_SKILLS_DIR = Path.home() / ".config" / "natshell" / "skills"
PROJECT_SKILLS_DIRNAME = ".natshell/skills"

@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path        # directory containing SKILL.md
    source: str       # "builtin" | "user" | "project"

class SkillRegistry:
    def __init__(self, skills: list[Skill], disabled: set[str]):
        self._skills = {s.name: s for s in skills}
        self._disabled = disabled

    def all(self) -> list[Skill]:
        return list(self._skills.values())

    def enabled(self) -> list[Skill]:
        return [s for s in self._skills.values() if s.name not in self._disabled]

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def load_body(self, name: str) -> str | None:
        s = self.get(name)
        if s is None:
            return None
        text = (s.path / "SKILL.md").read_text(encoding="utf-8", errors="replace")
        _meta, body = _parse_frontmatter(text)
        return body.strip()

    def list_assets(self, name: str) -> dict[str, list[Path]]:
        """Return {'references': [...], 'scripts': [...]} for the named skill."""
        s = self.get(name)
        if s is None:
            return {"references": [], "scripts": []}
        out = {"references": [], "scripts": []}
        for sub in ("references", "scripts"):
            d = s.path / sub
            if d.is_dir():
                out[sub] = sorted(p for p in d.iterdir() if p.is_file())
        return out


def _iter_skill_dirs() -> Iterable[tuple[Path, str]]:
    """Yield (skill_dir, source_label). Later sources override earlier on name collision."""
    # 1. builtin
    try:
        builtin_root = importlib.resources.files("natshell.skills")
        for entry in builtin_root.iterdir():
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                yield Path(str(entry)), "builtin"
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    # 2. user
    if USER_SKILLS_DIR.is_dir():
        for entry in USER_SKILLS_DIR.iterdir():
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                yield entry, "user"
    # 3. project
    project_dir = Path.cwd() / PROJECT_SKILLS_DIRNAME
    if project_dir.is_dir():
        for entry in project_dir.iterdir():
            if entry.is_dir() and (entry / "SKILL.md").is_file():
                yield entry, "project"


def load_skills(tool_registry: ToolRegistry, config: NatShellConfig) -> SkillRegistry:
    seen: dict[str, Skill] = {}
    disabled: set[str] = set(getattr(config.skills, "disabled", []) or [])
    if not getattr(config.skills, "enabled", True):
        return SkillRegistry([], disabled)

    for skill_dir, source in _iter_skill_dirs():
        try:
            text = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="replace")
            meta, _body = _parse_frontmatter(text)
        except Exception as e:
            logger.warning("skipping skill at %s: %s", skill_dir, e)
            continue

        name = meta.get("name", "").strip()
        desc = meta.get("description", "").strip()
        if not _NAME_RE.match(name):
            logger.warning("skipping skill at %s: invalid name %r", skill_dir, name)
            continue
        if name != skill_dir.name:
            logger.warning("skipping skill at %s: name %r != dir %r", skill_dir, name, skill_dir.name)
            continue
        if not desc or len(desc) > 200:
            logger.warning("skipping skill %s: description missing or > 200 chars", name)
            continue
        if "\n" in desc:
            logger.warning("skipping skill %s: description must be one line", name)
            continue

        if name in seen:
            logger.info("skill %s: %s overrides %s", name, source, seen[name].source)
        seen[name] = Skill(name=name, description=desc, path=skill_dir, source=source)

        # Optional tools.py
        tools_py = skill_dir / "tools.py"
        if tools_py.is_file():
            try:
                spec = importlib.util.spec_from_file_location(
                    f"natshell_skill_tools_{name.replace('-', '_')}", tools_py
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                if hasattr(module, "register") and callable(module.register):
                    module.register(tool_registry)
                else:
                    logger.warning("skill %s tools.py has no register() callable", name)
            except Exception as e:
                logger.warning("skill %s tools.py load failed: %s", name, e)

    return SkillRegistry(list(seen.values()), disabled)
```

Notes:
- Errors **never** crash startup; mirror the warn-and-continue pattern from the deleted plugin loader.
- Project-source skills require an explicit opt-in directory (`.natshell/skills/`), so cwd never leaks unrelated files.
- `_parse_frontmatter` and `_NAME_RE` live in this same module.

---

## 5. New tool: `src/natshell/tools/skill.py`

```python
from natshell.tools.registry import ToolDefinition, ToolResult

# Module-level handle injected by __main__ wiring (see §10).
_SKILL_REGISTRY = None

def set_skill_registry(registry):
    global _SKILL_REGISTRY
    _SKILL_REGISTRY = registry


SKILL_DEFINITION = ToolDefinition(
    name="skill",
    description=(
        "Load the full instructions for a named skill. "
        "Call this when the user's task matches a skill listed in <available_skills>."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the skill to load (e.g. 'spreadsheet').",
            }
        },
        "required": ["name"],
    },
    requires_confirmation=False,
)


async def skill_handler(name: str) -> ToolResult:
    if _SKILL_REGISTRY is None:
        return ToolResult(output="", error="skill registry not initialized", exit_code=1)
    s = _SKILL_REGISTRY.get(name)
    if s is None or s.name in _SKILL_REGISTRY._disabled:
        available = ", ".join(x.name for x in _SKILL_REGISTRY.enabled()) or "(none)"
        return ToolResult(
            output="",
            error=f"unknown or disabled skill: {name!r}. Available: {available}",
            exit_code=1,
        )
    body = _SKILL_REGISTRY.load_body(name) or ""
    assets = _SKILL_REGISTRY.list_assets(name)

    parts = [f"# Skill: {name}", body]
    if assets["references"] or assets["scripts"]:
        parts.append("\n---\n## Bundled assets\nUse read_file or run_code on these as needed.\n")
        for ref in assets["references"]:
            parts.append(f"- reference: {ref}")
        for sc in assets["scripts"]:
            parts.append(f"- script: {sc}")
    return ToolResult(output="\n".join(parts), error=None, exit_code=0)
```

Add `"skill"` to **both** `PLAN_SAFE_TOOLS` and `SMALL_CONTEXT_TOOLS` in `src/natshell/tools/registry.py` so skills work in plan mode and on small models.

The output respects existing truncation in `tools/limits.py`; if a SKILL.md body would exceed the cap, the registry's `load_body()` returns the first chunk and the model can `read_file` the file directly for the rest. (Standard truncation marker behavior.)

---

## 6. System prompt injection — `src/natshell/agent/system_prompt.py`

`build_system_prompt` gains a new keyword arg:

```python
def build_system_prompt(
    context: SystemContext,
    *,
    compact: bool = False,
    prompt_config: PromptConfig | None = None,
    working_memory: str | None = None,
    memory_path: str = "",
    max_memory_chars: int = 4000,
    skills: list["Skill"] | None = None,
    inject_skills_in_compact: bool = False,
) -> str:
```

Place the new section **after the `<system_info>` block** and before the "Efficient Exploration" section. In compact mode, omit unless `inject_skills_in_compact=True`.

```python
if skills and (not compact or inject_skills_in_compact):
    lines = ["<available_skills>"]
    lines.append(
        "The following skills are available. Call the `skill` tool with a name to load full "
        "instructions when the user's task matches a skill's description."
    )
    for s in sorted(skills, key=lambda s: s.name):
        lines.append(f"- {s.name}: {s.description}")
    lines.append("</available_skills>")
    sections.append("\n".join(lines))
```

Token budget: ~30–50 tokens per entry × 10 skills ≈ 400 tokens. Acceptable in non-compact mode (n_ctx ≥ 16K).

---

## 7. Config — `src/natshell/config.py` + `config.default.toml`

Add a dataclass:

```python
@dataclass
class SkillsConfig:
    enabled: bool = True
    disabled: list[str] = field(default_factory=list)
    inject_in_compact: bool = False
```

- Add `skills: SkillsConfig = field(default_factory=SkillsConfig)` to `NatShellConfig`.
- Extend `VALID_CONFIG_KEYS` (`config.py:142–200`) with `"skills": {"enabled", "disabled", "inject_in_compact"}`.
- Wire loading in the existing TOML reader and saving in `save_config_value()`.

Default config (`config.default.toml`):

```toml
[skills]
enabled = true
disabled = []
inject_in_compact = false
```

---

## 8. Slash commands — `commands.py` + `app.py`

Mirror the existing `/memory` subcommand pattern (`commands.py:129-160`). Add to `commands.py`:

```python
async def show_skills(skill_registry, args: str = "") -> str:
    """Handle /skills, /skills show <n>, /skills enable <n>, /skills disable <n>."""
    args = args.strip()
    if not args:
        rows = []
        for s in skill_registry.all():
            state = "disabled" if s.name in skill_registry._disabled else "enabled"
            rows.append(f"  {s.name:<16} [{s.source:<7}] {state:<8} {s.description}")
        return "Available skills:\n" + ("\n".join(rows) if rows else "  (none)")

    parts = args.split(maxsplit=1)
    sub = parts[0].lower()
    name = parts[1].strip() if len(parts) > 1 else ""

    if sub == "show" and name:
        body = skill_registry.load_body(name)
        return body or f"unknown skill: {name}"
    if sub in {"enable", "disable"} and name:
        # Mutate config + persist via save_config_value("skills", "disabled", new_list)
        ...
        return f"skill {name} {sub}d"
    return "usage: /skills | /skills show <name> | /skills enable <name> | /skills disable <name>"
```

In `src/natshell/app.py`:
- Add `"/skills"` to the `SLASH_COMMANDS` list around line 107.
- Add a `case "/skills":` branch in `_handle_slash_command` around line 473.
- Use `conversation.mount(SystemMessage(...))` to render output (same pattern as `/memory`).

---

## 9. Plugin removal & migration

- **Delete** `src/natshell/plugins.py`.
- **Delete** the plugin import + invocation in `src/natshell/__main__.py:428-432`:
  ```python
  from natshell.plugins import load_plugins
  loaded = load_plugins(tools)
  if loaded:
      print(f"Loaded {loaded} plugin(s)")
  ```
  Replace with the skill loader (see §10).
- **One-time warning**: if `~/.config/natshell/plugins/` exists and contains any `.py` file, print to stderr at startup:
  ```
  ~/.config/natshell/plugins/ is no longer loaded. Wrap your register() function in a
  skill: ~/.config/natshell/skills/<name>/tools.py. See SKILL_AUTHORING.md.
  ```
  Suppress after first warning per session — no config flag needed; just print and move on.
- **Tests**: replace `tests/test_plugins.py` entirely with `tests/test_skills.py`. Delete the old file.
- **Docs**: add a short migration section to README and a new `SKILL_AUTHORING.md` (see §13).

---

## 10. Wiring in `src/natshell/__main__.py`

Replace plugin loading with skill loading. Approximate sequence:

```python
from natshell.skills import load_skills
from natshell.tools.skill import SKILL_DEFINITION, skill_handler, set_skill_registry

# After `tools = ToolRegistry(...)` is created:
skill_registry = load_skills(tools, config)
set_skill_registry(skill_registry)
tools.register(SKILL_DEFINITION, skill_handler)

# When constructing AgentLoop, pass skills into the system-prompt builder:
agent = AgentLoop(
    ...,
    skills=skill_registry.enabled(),
    inject_skills_in_compact=config.skills.inject_in_compact,
)
```

Update `AgentLoop.__init__` (in `src/natshell/agent/loop.py`) to accept `skills` and `inject_skills_in_compact` and forward them to `build_system_prompt`. Pin them on the instance so a `/skills enable` triggered re-render uses the current state.

A `/skills enable|disable` command should rebuild the system prompt on next agent invocation. Simplest approach: store `skill_registry` on the app; when its `_disabled` set changes, invalidate the cached system prompt (call whatever method already invalidates it for `/memory reload`).

---

## 11. The 10 shipped skills

Create `src/natshell/skills/<name>/SKILL.md` for each. Descriptions below are the literal `description:` frontmatter values.

Each `SKILL.md` body should be 60–150 lines and follow this **mandatory template**:

```markdown
---
name: <name>
description: <one-line description, ≤ 200 chars>
---

# <Title>

## When to use
<2–4 bullets describing the trigger conditions>

## When NOT to use
<2–3 bullets describing tasks that look similar but should not invoke this skill>

## Procedure
1. <numbered step>
2. ...

## Recipes
<short snippets for the most common operations — shell one-liners, run_code blocks, etc.>

## Pitfalls
<bullets — known footguns, platform gotchas, tool-confirmation surprises>

## References
<bullets pointing into references/ or scripts/ if present>
```

Below is the canonical content target for each shipped skill. Implementer fills in bodies following the template; the **descriptions are fixed** (do not paraphrase — they're tuned for token cost and signal):

### 1. `spreadsheet`
> Read, write, and transform CSV and XLSX files. Use for Excel, tabular data, bulk row edits, and column transforms.

Body must cover: stdlib `csv` for plain CSV; `openpyxl` via `run_code` for XLSX (note install fallback to `pip install --user openpyxl` via execute_shell); reading sheet/header before mutating; preserving formulas warning. Include `scripts/csv_inspect.py` (prints dialect, header, first 5 rows, row count) and `scripts/xlsx_to_csv.py`.

### 2. `pdf`
> Extract text, merge, split, and search PDF documents. Use whenever the user mentions a PDF file.

Body must cover: `pypdf` via run_code for extract/merge/split; `pdftotext` (poppler) fallback via execute_shell on Linux/macOS; OCR fallback note (`tesseract` + `ocrmypdf`) if text extraction returns empty. Include `scripts/pdf_text.py` (text extraction with page range) and `scripts/pdf_merge.py`.

### 3. `docx`
> Read and edit Microsoft Word documents (.docx). Use for document editing and content extraction.

Body must cover: `python-docx` via run_code; never edit binary `.doc` (recommend conversion via `libreoffice --headless --convert-to docx`); preserve styles by editing run-level text rather than replacing paragraphs. Include `scripts/docx_text.py` and `scripts/docx_replace.py`.

### 4. `coding`
> Navigate large codebases and make surgical edits while following project conventions. Use for refactors, feature work, and bug fixes spanning multiple files.

Body must cover: search before editing (`search_files`); read full file before `edit_file` (existing tracker constraint); match existing style; never introduce dependencies without justification; small commits. Include `references/python.md`, `references/javascript.md`, `references/go.md`, `references/rust.md` — each ~30 lines of language-idiomatic conventions and common test/build commands.

### 5. `testing`
> Detect the project's test framework (pytest, jest, go test, cargo test, mocha, rspec) and run, parse, and triage failures.

Body must cover: detection cascade (presence of `pyproject.toml`/`pytest.ini` → pytest; `package.json` scripts → jest/vitest/mocha; `go.mod` → `go test ./...`; `Cargo.toml` → `cargo test`; `Gemfile` → `bundle exec rspec`); always run from repo root; capture stderr; parse first failure deeply rather than skim all. Include `references/pytest.md` (markers, `-k`, `--lf`, common flags) and `references/jest.md`.

### 6. `git-workflow`
> Branch, commit, rebase, and resolve conflicts. Use for PR prep, conventional commit messages, and history cleanup.

Body must cover: prefer `git_tool` for safe ops, fall back to `execute_shell` for advanced; never `--force` without confirmation; commit message conventions (Conventional Commits); rebase vs merge guidance; conflict resolution loop (read both sides, decide intent, mark resolved). Include `references/conventional-commits.md`.

### 7. `system-admin`
> Linux/macOS administration: services, processes, package managers, and log triage.

Body must cover: detect package manager via `command -v` cascade; service control (`systemctl` vs `launchctl`); process inspection (`ps`, `top`, `lsof`); log locations (`journalctl`, `/var/log/`, macOS `log show`). Include `references/systemd.md` and `references/macos-launchctl.md`.

### 8. `data-analysis`
> Inspect, filter, and summarize JSON, CSV, and log data using jq, awk, and pandas.

Body must cover: `jq` recipe library (select, map, group_by, has); `awk` recipe library (column sums, distinct counts); pandas via run_code for anything multi-step; sampling (`head -n 1000`) before full-dataset operations on large files. Include `references/jq-recipes.md` and `scripts/csv_summary.py`.

### 9. `web-research`
> Fetch and summarize web pages and offline kiwix docs. Use when the user asks for current information or external references.

Body must cover: prefer `kiwix_search` for offline reference (man pages, docs); `fetch_url` for live web (respecting SSRF guards); cite URLs back to user; summarize rather than dumping raw HTML. No scripts required.

### 10. `markdown-docs`
> Author and edit README, CHANGELOG, and other markdown documentation, following the project's existing style.

Body must cover: read existing docs before writing new ones to match tone/structure; CommonMark compliance; CHANGELOG via Keep a Changelog format; never add badges/emojis unless already present. Include `references/keep-a-changelog.md`.

---

## 12. Tests — `tests/test_skills.py`

Replace `tests/test_plugins.py`. Required test classes/cases:

- **Discovery**:
  - Builtin skills load (mock `importlib.resources` or use a temp package — easiest: parameterize over real shipped dirs and just assert each loads).
  - User-dir skill overrides builtin with same name; warning logged.
  - Project-dir skill overrides user.
  - Invalid frontmatter, missing description, name mismatch, name too long, multi-line description, name regex fail — all skipped with warning, never raise.
- **Disabled skills**: `[skills].disabled = ["pdf"]` hides `pdf` from `enabled()` and from system prompt.
- **`tools.py` integration**: a fixture skill with a valid `tools.py` registers a tool; one with a broken `tools.py` does not crash load.
- **`skill` tool handler**:
  - Returns body sans frontmatter.
  - Lists `references/` and `scripts/` paths when present.
  - Returns `error` when name unknown or disabled.
- **System prompt rendering**: `build_system_prompt(skills=[...])` includes `<available_skills>` in non-compact, omits in compact unless `inject_skills_in_compact=True`.
- **Each shipped skill** (parameterized): frontmatter parses, name matches dir, description ≤ 200 chars and one line, body ≥ 200 chars (sanity).
- **Slash commands**: `/skills`, `/skills show <name>`, `/skills enable/disable <name>` exercised against an in-memory app fixture (mirror existing `test_slash_commands.py` style).

---

## 13. Documentation

Add **`SKILL_AUTHORING.md`** at repo root (~80 lines) covering:
- Skill anatomy (copy from §2).
- The frontmatter spec.
- The `register()` contract for `tools.py` (copy from existing `tests/test_plugins.py:206-222` example, just relocated).
- How to test a skill locally (drop into `~/.config/natshell/skills/<name>/`, run `/skills`).
- Plugin migration: a 10-line before/after showing how an old plugin maps onto a new skill's `tools.py`.

Update **`README.md`**:
- Replace the plugin paragraph with a skill paragraph.
- Add a one-line pointer to `SKILL_AUTHORING.md`.

Update **`CLAUDE.md`** §Modules to remove `plugins.py`, add `skills.py` and `tools/skill.py`, and reference the shipped `src/natshell/skills/` tree.

---

## 14. Packaging — `pyproject.toml`

Built-in skills are package data and must ship in the wheel. Add (or verify) under `[tool.setuptools.package-data]`:

```toml
[tool.setuptools.package-data]
natshell = [
    "config.default.toml",
    "skills/**/*",
]
```

Verify by building a wheel (`python -m build`) and inspecting it (`unzip -l dist/natshell-*.whl | grep skills`) — every `SKILL.md`, `references/*`, and `scripts/*` must be present.

---

## 15. Implementation order (suggested)

1. **Infrastructure first, no skills yet**:
   - Add `SkillsConfig`, default config entry, `VALID_CONFIG_KEYS` update.
   - Create `src/natshell/skills.py` with discovery + frontmatter parsing.
   - Create `src/natshell/tools/skill.py` and add `"skill"` to `PLAN_SAFE_TOOLS` and `SMALL_CONTEXT_TOOLS`.
   - Wire `load_skills()` into `__main__.py`; delete `plugins.py` and update tests.
   - Add system-prompt rendering for `<available_skills>`.
   - Add `/skills` slash command.
2. **Verify with one skill**: write `src/natshell/skills/spreadsheet/SKILL.md` plus its scripts. Run `/skills show spreadsheet` interactively. Headless: `natshell --headless "convert /tmp/x.csv to xlsx"` and confirm the model invokes `skill(name="spreadsheet")` early.
3. **Author remaining 9 skills**: write each `SKILL.md` body following the template in §11. Create `references/` and `scripts/` files where listed.
4. **Tests**: write `tests/test_skills.py` covering everything in §12. Delete `tests/test_plugins.py`. Run full `pytest`.
5. **Docs**: write `SKILL_AUTHORING.md`, update `README.md` and `CLAUDE.md`.
6. **Package data check**: build a wheel and verify shipped skills are inside.

---

## 16. Critical files (full list)

**New files**:
- `src/natshell/skills.py`
- `src/natshell/tools/skill.py`
- `src/natshell/skills/spreadsheet/SKILL.md` (+ scripts)
- `src/natshell/skills/pdf/SKILL.md` (+ scripts)
- `src/natshell/skills/docx/SKILL.md` (+ scripts)
- `src/natshell/skills/coding/SKILL.md` (+ references/)
- `src/natshell/skills/testing/SKILL.md` (+ references/)
- `src/natshell/skills/git-workflow/SKILL.md` (+ references/)
- `src/natshell/skills/system-admin/SKILL.md` (+ references/)
- `src/natshell/skills/data-analysis/SKILL.md` (+ references/, scripts/)
- `src/natshell/skills/web-research/SKILL.md`
- `src/natshell/skills/markdown-docs/SKILL.md` (+ references/)
- `tests/test_skills.py`
- `SKILL_AUTHORING.md`

**Modified files**:
- `src/natshell/__main__.py` — replace plugin call with skill load + register `skill` tool
- `src/natshell/agent/loop.py` — accept `skills`, forward to system prompt
- `src/natshell/agent/system_prompt.py` — render `<available_skills>` section
- `src/natshell/tools/registry.py` — add `"skill"` to `PLAN_SAFE_TOOLS` and `SMALL_CONTEXT_TOOLS`
- `src/natshell/config.py` — `SkillsConfig`, `VALID_CONFIG_KEYS`, loader/saver wiring
- `src/natshell/config.default.toml` — `[skills]` section
- `src/natshell/commands.py` — `show_skills` handler
- `src/natshell/app.py` — `/skills` dispatch + `SLASH_COMMANDS` list
- `pyproject.toml` — package-data entry
- `README.md` — replace plugin section with skill section
- `CLAUDE.md` — module list update

**Deleted files**:
- `src/natshell/plugins.py`
- `tests/test_plugins.py`

---

## 17. Reuse — patterns to lift

- Plugin loader resilience: `plugins.py:18-82` — copy the warn-and-continue try/except shape.
- Registry pattern: `tools/registry.py:66-91` — `SkillRegistry` mirrors `ToolRegistry`'s shape.
- Working-memory injection: `agent/system_prompt.py` memory section — the structural pattern for injecting bounded optional content into the prompt.
- `save_config_value()` (`config.py:218-273`) — directly usable for persisting `[skills].disabled`.
- `/memory` command shape (`commands.py:129-160`) — clone for `/skills` subcommand handling.
- `read_file` truncation conventions (`tools/read_file.py`) — the `skill` tool emits its body using the same truncation marker so the model handles long skills naturally.
- App slash dispatch (`app.py:473-520` match statement) — append `/skills` case alongside existing handlers.

---

## 18. Verification checklist (end-to-end)

- [ ] `pytest` passes (1,175+ tests, no regressions).
- [ ] `natshell --headless "convert /tmp/sample.csv to xlsx"` results in `skill(name="spreadsheet")` being called.
- [ ] `natshell --headless "extract text from /tmp/x.pdf"` results in `skill(name="pdf")` being called.
- [ ] `/skills` lists all 10 shipped skills, marked `[builtin]`, all enabled.
- [ ] `/skills show coding` prints the body.
- [ ] `/skills disable web-research` persists; on next launch, `web-research` is gone from the system prompt and `skill(name="web-research")` returns a clear error.
- [ ] `~/.config/natshell/skills/myskill/SKILL.md` overrides a builtin skill of the same name (warning logged).
- [ ] Stale `~/.config/natshell/plugins/foo.py` triggers the migration warning once and does not load.
- [ ] Built wheel contains `natshell/skills/<name>/SKILL.md` for all 10 names.
- [ ] System prompt token count with skills enabled stays within the existing budget targets for each context tier (measure via existing prompt-cache or context-manager tests).
- [ ] Headless and MCP modes both surface skills (no separate code path needed; they reuse the agent's system prompt).
