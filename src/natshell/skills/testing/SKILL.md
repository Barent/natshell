---
name: testing
description: Detect the project's test framework (pytest, jest, go test, cargo test, mocha, rspec) and run, parse, and triage failures.
---

# Testing skill

## When to use
- User asks to run tests, fix failing tests, or add tests
- User asks why tests are failing or to debug a test failure
- User asks to find the test framework used in a project

## When NOT to use
- The user wants to write production code alongside tests — use coding skill
- The user is working with CI/CD pipeline config — use system-admin or coding skill

## Procedure
1. **Detect the test framework**: check project root files (see detection cascade below).
2. **Run tests from the project root**.
3. **Capture stderr** as well as stdout — most frameworks print errors to stderr.
4. **Parse the first failure deeply**: read the failing test file and the code under test to understand why.
5. **Fix root cause**, not symptoms. Don't suppress assertions — fix the code or the test.
6. **Re-run** to confirm the fix.

## Detection cascade
| Signal | Framework | Command |
|--------|-----------|---------|
| `pyproject.toml` or `pytest.ini` or `setup.cfg [tool:pytest]` | pytest | `pytest` |
| `package.json` scripts with "jest" | Jest | `npx jest` |
| `package.json` scripts with "vitest" | Vitest | `npx vitest run` |
| `package.json` scripts with "mocha" | Mocha | `npx mocha` |
| `go.mod` | go test | `go test ./...` |
| `Cargo.toml` | cargo test | `cargo test` |
| `Gemfile` with `rspec` | RSpec | `bundle exec rspec` |

When uncertain, run `cat package.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('scripts',{}))"` to inspect scripts.

## Recipes

**Run pytest with verbose output on failures:**
```bash
pytest -v --tb=short 2>&1
```

**Run only failed tests:**
```bash
pytest --lf
```

**Run specific test by name:**
```bash
pytest -k "test_my_function"
```

**Run Jest in non-interactive mode:**
```bash
npx jest --no-coverage 2>&1
```

**Run specific Jest test file:**
```bash
npx jest tests/foo.test.js
```

**Run Go tests with verbose output:**
```bash
go test ./... -v 2>&1
```

**Run Cargo tests:**
```bash
cargo test 2>&1
```

**Run RSpec:**
```bash
bundle exec rspec --format documentation 2>&1
```

## Pitfalls
- Always run from the **project root** — relative import paths break if you run from a subdirectory.
- Parse the **first failure** completely before moving to others. One root cause often fixes many failures.
- Don't skip or delete tests to make a suite pass — fix the underlying code.
- `pytest` outputs to both stdout and stderr; capture both with `2>&1`.
- For Node.js tests, `npm test` may run in watch mode — prefer `npx jest --watchAll=false` or `npx vitest run`.
- Some projects have multiple test suites (unit, integration) — check the `scripts` section of `package.json`.

## References
- `references/pytest.md` — pytest markers, common flags, fixture patterns
- `references/jest.md` — Jest matchers, mocking, common patterns
