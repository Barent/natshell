# Keep a Changelog format

Reference: https://keepachangelog.com/

## File structure
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.2.0] - 2024-03-15
### Added
- ...

## [1.1.0] - 2024-02-01
### Fixed
- ...

[Unreleased]: https://github.com/user/repo/compare/v1.2.0...HEAD
[1.2.0]: https://github.com/user/repo/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/user/repo/releases/tag/v1.1.0
```

## Change types (use only what applies)
| Type | Description |
|------|-------------|
| `Added` | New features |
| `Changed` | Changes to existing functionality |
| `Deprecated` | Features that will be removed |
| `Removed` | Features removed in this release |
| `Fixed` | Bug fixes |
| `Security` | Vulnerability fixes |

## Versioning (Semantic Versioning)
- `MAJOR.MINOR.PATCH`
- MAJOR: breaking API change
- MINOR: new backward-compatible feature
- PATCH: backward-compatible bug fix
- Pre-release: `1.0.0-alpha.1`, `1.0.0-beta.2`, `1.0.0-rc.1`

## Rules
- Newest version at the top
- One `[Unreleased]` section at the top for next release
- Date format: `YYYY-MM-DD`
- Each entry is a bullet point (imperative: "Add" not "Added a")
- Link versions to compare diffs at the bottom of the file
