# Conventional Commits

Format: `<type>(<scope>): <description>`

## Types
| Type | When to use |
|------|-------------|
| `feat` | New feature for the user |
| `fix` | Bug fix for the user |
| `docs` | Documentation only changes |
| `style` | Formatting, missing semi-colons, etc. (no logic change) |
| `refactor` | Code change that neither fixes a bug nor adds a feature |
| `test` | Adding or correcting tests |
| `chore` | Build process, dependency updates, tooling |
| `perf` | Performance improvement |
| `ci` | CI configuration changes |
| `revert` | Reverts a previous commit |

## Scope (optional)
The scope is the component or module affected:
- `feat(auth): add OAuth2 login`
- `fix(parser): handle empty input`
- `chore(deps): update numpy to 2.0`

## Breaking changes
Add `!` after type, or add `BREAKING CHANGE:` footer:
```
feat!: remove deprecated API endpoints

BREAKING CHANGE: The v1 endpoint /api/v1/users has been removed.
Migrate to /api/v2/users.
```

## Examples
```
feat: add skill system to replace plugins
fix: prevent crash when config file is missing
docs: update README with skill authoring guide
refactor: extract frontmatter parser to skills module
test: add parametrized tests for all 10 shipped skills
chore: add skills/** to package data in pyproject.toml
```

## Rules
- Use imperative mood: "add" not "added" or "adds"
- No period at end of description
- Keep description under 72 characters
- Reference issues in body or footer: `Closes #123`
