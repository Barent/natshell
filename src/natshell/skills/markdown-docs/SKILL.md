---
name: markdown-docs
description: Author and edit README, CHANGELOG, and other markdown documentation, following the project's existing style.
---

# Markdown documentation skill

## When to use
- User asks to write or update a README, CHANGELOG, or other `.md` documentation file
- User asks to fix or improve documentation style/formatting
- User asks to generate API or module documentation in markdown

## When NOT to use
- The user wants a Word document — use docx skill
- The user wants code changes, not documentation — use coding skill
- The user asks to generate code from docs — that is a coding task

## Procedure
1. **Read existing docs first**: read the README and any other `.md` files to understand the project's tone, structure, and conventions before writing.
2. **Match the style**: heading levels, bold usage, code block languages, list punctuation — copy from what's already there.
3. **CommonMark compliance**: use standard markdown; avoid GFM extensions that may not render everywhere unless the project already uses them.
4. **CHANGELOG**: use Keep a Changelog format (see references/).
5. **Never add badges, emojis, or decorations** unless they already appear in the project's docs.
6. **Verify links**: if adding a link to a file or section, confirm the target exists.

## Recipes

**Check existing markdown files:**
```
list_directory(path="/project/docs")
read_file(path="/project/README.md")
```

**CommonMark heading hierarchy:**
```markdown
# Title (H1 — one per document)
## Section (H2)
### Subsection (H3)
#### Detail (H4 — use sparingly)
```

**Code blocks with language hints:**
```markdown
\```python
def greet(name: str) -> str:
    return f"Hello, {name}"
\```
```

**Admonition (GFM callout — GitHub only):**
```markdown
> [!NOTE]
> This is a note.

> [!WARNING]
> This is a warning.
```

**Table:**
```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value A  | Value B  | Value C  |
```

**Changelog entry (Keep a Changelog):**
```markdown
## [1.2.0] - 2024-03-15

### Added
- New spreadsheet skill for CSV/XLSX manipulation

### Fixed
- Crash when config file is missing on first run

### Changed
- Plugin system replaced with skill system
```

**Update a specific section in an existing README:**
1. `read_file` the full README
2. Identify the section boundaries
3. Use `edit_file` with multi-line `old_text` covering the full section to replace

## Pitfalls
- Editing headings with `edit_file` requires matching the exact whitespace and newlines — read the file first.
- GitHub Flavored Markdown extensions (tables, task lists, callouts) may not render on all platforms — check the project's context.
- Trailing spaces create hard line breaks in CommonMark — clean them up unless intentional.
- Don't reformat sections you weren't asked to change — scope edits to what was requested.
- If adding a new section, match the heading level of existing parallel sections.

## References
- `references/keep-a-changelog.md` — CHANGELOG format, versioning conventions
