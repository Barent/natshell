# Skill Authoring Guide

Skills are directories that bundle model-facing instructions, optional reference docs, optional helper scripts, and optional custom tool registration. NatShell ships 10 built-in skills; you can add your own or override any built-in.

## Anatomy of a skill

```
<skill-name>/
  SKILL.md        REQUIRED — YAML frontmatter + instructions body
  references/     OPTIONAL — longer docs that SKILL.md cites; loaded on demand via read_file
    *.md
  scripts/        OPTIONAL — helper scripts the skill body may run via run_code
    *.py  *.sh
  tools.py        OPTIONAL — def register(registry): ... adds custom tools
```

## SKILL.md format

```markdown
---
name: my-skill
description: One-line description for the model, ≤ 200 characters.
---

# My Skill

Short purpose statement and when to use / when not to use this skill.

## Procedure
1. Step one
2. Step two

## Common pitfalls
- Pitfall A
- Pitfall B
```

### Frontmatter rules

| Field | Requirement |
|-------|-------------|
| `name` | Lowercase letters, digits, hyphens; 2–32 chars; must match the directory name |
| `description` | Single line, ≤ 200 characters; appears in the system prompt so the model knows when to call this skill |

The body is only loaded when the model explicitly calls `skill(name="…")`. Keep descriptions precise — they are the model's cue for when to invoke the skill. Keep bodies practical — numbered procedures and copy-paste recipes work best.

## Where to place skills

| Location | Purpose |
|----------|---------|
| `~/.config/natshell/skills/<name>/` | Personal skills, available in all projects |
| `.natshell/skills/<name>/` | Project-local skills, committed alongside the code |

Built-in skills ship inside the Python package (`src/natshell/skills/`). User and project skills override built-ins with the same name.

## Adding custom tools via tools.py

```python
# ~/.config/natshell/skills/my-skill/tools.py

from natshell.tools.registry import ToolDefinition, ToolResult

DEFINITION = ToolDefinition(
    name="my_tool",
    description="Does something useful.",
    parameters={
        "type": "object",
        "properties": {"arg": {"type": "string"}},
        "required": ["arg"],
    },
    requires_confirmation=False,
)

async def my_tool(arg: str) -> ToolResult:
    return ToolResult(output=f"You said: {arg}")

def register(registry):
    registry.register(DEFINITION, my_tool)
```

Errors in `tools.py` are logged as warnings and never crash startup.

## Managing skills

```
/skills                   list all discovered skills (name, source, enabled/disabled)
/skills show <name>       print the full SKILL.md body
/skills enable <name>     re-enable a disabled skill
/skills disable <name>    hide a skill from the model (persisted in config.toml)
```

Disabled skills are listed in `[skills].disabled` in `~/.config/natshell/config.toml`. The master toggle `[skills].enabled = false` hides all skills.

## Migrating from the old plugin system

If you have files in `~/.config/natshell/plugins/`, NatShell will print a one-time warning on startup. To migrate:

1. Create `~/.config/natshell/skills/<name>/`
2. Write a `SKILL.md` with appropriate `name` and `description` frontmatter
3. Move your `register()` function into `tools.py` inside that directory
4. Delete or archive the old `.py` file from `plugins/`

## Built-in skills

| Skill | Description |
|-------|-------------|
| `spreadsheet` | Read, write, and transform CSV and XLSX files |
| `pdf` | Extract text, merge, split, and search PDF documents |
| `docx` | Read and edit Microsoft Word (.docx) documents |
| `coding` | Navigate codebases and make surgical multi-file edits |
| `testing` | Detect test frameworks, run tests, and triage failures |
| `git-workflow` | Branch, commit, rebase, and resolve conflicts |
| `system-admin` | Linux/macOS services, processes, packages, and logs |
| `data-analysis` | Inspect and summarize JSON, CSV, and log data |
| `web-research` | Fetch and summarize web pages and offline docs |
| `markdown-docs` | Author and edit README, CHANGELOG, and markdown docs |
