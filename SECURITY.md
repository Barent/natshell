# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in NatShell, please report it responsibly:

**Email**: Open a [GitHub Security Advisory](https://github.com/Barent/natshell/security/advisories/new) (preferred) or email the maintainer directly via the contact on the GitHub profile.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

## Response

- Acknowledgment within 48 hours
- Fix timeline depends on severity — critical issues are prioritized for immediate patch release
- Credit given in the changelog unless you prefer anonymity

## Scope

NatShell executes shell commands by design. The safety classifier (regex-based, in `safety/classifier.py`) is a guardrail, not a sandbox. Vulnerabilities in scope include:
- Bypasses of the blocked command classifier
- Path traversal in session/backup persistence
- SSRF bypasses in `fetch_url`
- Prompt injection that circumvents safety rules
- Information disclosure (API keys, credentials leaking)

Out of scope:
- Users intentionally running in `danger` safety mode
- Commands approved by the user via the confirmation dialog
