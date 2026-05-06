---
name: coding
description: Navigate large codebases and make surgical edits while following project conventions. Use for refactors, feature work, and bug fixes spanning multiple files.
---

# Coding skill

## When to use
- User asks to implement a feature, fix a bug, or refactor code across multiple files
- User asks to navigate and understand a large codebase
- User asks for a code review or asks what a piece of code does
- User asks to add tests for existing code

## When NOT to use
- The task is only running a test suite — use testing skill
- The task is only a git operation — use git-workflow skill
- The task is scripting a one-off data transform — use data-analysis skill

## Procedure
1. **Locate the project root**: find build file (pyproject.toml, package.json, Cargo.toml, go.mod, Makefile).
2. **Search before editing**: use `search_files` to find relevant symbols, functions, or patterns.
3. **Read full files before editing**: read_file must see the full file before edit_file is called (tracker enforces this).
4. **Plan the change**: state which files change and why before making edits.
5. **Make surgical edits**: use `edit_file` with multi-line `old_text` for unique matches. Use `write_file` only for new files or full rewrites.
6. **Verify**: run the project's build/check command or tests after editing.
7. **Match existing style**: indentation, naming, comment density — copy from surrounding code, don't introduce new conventions.

## Recipes

**Find where a function is defined:**
```
search_files(pattern="def my_function", path="/project/src")
```

**Find all callers of a function:**
```
search_files(pattern="my_function(", path="/project")
```

**Read a file in sections (large file):**
```
read_file(path="/project/src/module.py", limit=100)           # first 100 lines
read_file(path="/project/src/module.py", offset=100, limit=100)  # next 100
```

**Multi-line edit (unique match):**
Use enough surrounding context so old_text appears exactly once in the file.

**Check for syntax errors (Python):**
```bash
python3 -m py_compile /path/to/file.py && echo OK
```

**Check for syntax errors (Go):**
```bash
go vet ./...
```

**Check for syntax errors (Rust):**
```bash
cargo check
```

**Check for syntax errors (Node):**
```bash
node --check /path/to/file.js
```

## Pitfalls
- Never edit a file you haven't fully read — the FileReadTracker will block edit_file on partially-read files.
- After 2 edit_file failures on the same file, switch to write_file (full rewrite).
- Never guess at what a function does from its name — read the implementation.
- Avoid adding dependencies without user approval.
- Never declare a bug fixed without running a verification step.
- Watch for double-nesting: verify directory paths with list_directory before writing.

## References
- `references/python.md` — Python idioms, test commands, common gotchas
- `references/javascript.md` — JS/TS idioms, test commands, ESM vs CJS
- `references/go.md` — Go idioms, test commands, module layout
- `references/rust.md` — Rust idioms, test commands, cargo workflow
