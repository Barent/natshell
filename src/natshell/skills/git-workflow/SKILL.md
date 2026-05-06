---
name: git-workflow
description: Branch, commit, rebase, and resolve conflicts. Use for PR prep, conventional commit messages, and history cleanup.
---

# Git workflow skill

## When to use
- User asks to commit changes, create branches, or prepare a PR
- User needs to resolve merge conflicts
- User wants to rebase, squash commits, or clean up history
- User asks for help with git operations beyond basic status/diff

## When NOT to use
- The user only wants to read code history — use `git_tool` directly
- The user is doing CI/CD setup — use system-admin or coding skill

## Procedure
1. **Always check status first**: `git_tool(operation="status")` before any mutating operation.
2. **Use `git_tool` for safe read operations**: status, diff, log, branch.
3. **Use `execute_shell` for advanced operations** not covered by git_tool: rebase, push, pull, merge.
4. **Never `--force` push** without explicit user confirmation. Always explain the risk.
5. For commits: write a clear message following Conventional Commits (see references/).
6. For conflicts: read both sides (ours and theirs), understand intent, then resolve.

## Recipes

**Check status and staged changes:**
```
git_tool(operation="status")
git_tool(operation="diff", args="--staged")
```

**Create and switch to a branch:**
```bash
git checkout -b feature/my-feature
```

**Commit with Conventional Commits message:**
```bash
git commit -m "feat: add user authentication endpoint"
git commit -m "fix: handle empty input in CSV parser"
git commit -m "chore: update dependencies"
```

**Amend last commit message (unpushed only):**
```bash
git commit --amend -m "better message"
```

**Interactive rebase (squash last 3 commits):**
```bash
git rebase -i HEAD~3
# In editor: change "pick" to "squash" for commits to merge
```

**View commit history:**
```
git_tool(operation="log", args="-10 --oneline")
```

**Stash uncommitted work:**
```
git_tool(operation="stash", args="push")
git_tool(operation="stash", args="pop")
```

**Resolve merge conflicts:**
1. Identify conflicting files: `git_tool(operation="status")`
2. Read the conflict markers in each file
3. Decide which version (or combination) is correct
4. Edit the file to remove conflict markers and keep the right content
5. Stage: `git add <file>`
6. Continue: `git rebase --continue` or `git merge --continue`

**Push to remote:**
```bash
git push origin feature/my-feature
git push --set-upstream origin feature/my-feature  # first push
```

## Pitfalls
- Never `git push --force` without explaining the rewrite-history risk to the user.
- NatShell blocks `--amend`, `--author=`, `--date=` flags in git_tool for safety — use execute_shell if truly needed.
- Rebase rewrites history — don't rebase shared branches.
- Always stage specific files rather than `git add .` to avoid accidentally committing secrets.
- Conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) must all be removed before committing.

## References
- `references/conventional-commits.md` — commit message format and examples
