"""User-intent heuristics — lightweight regex gates on the user's message.

Extracted from :mod:`natshell.agent.loop` so they can be reused (and tested)
without importing the agent runtime. The old import paths
(``natshell.agent.loop._is_plan_request`` etc.) keep working via re-exports.
"""

from __future__ import annotations

import re

_PLAN_REQUEST_RE = re.compile(
    r"\b(?:create|write|make|draft|update|build)\b.{0,30}\bplan\b"
    r"|\bplan\b.{0,30}\b(?:for|how|what)\b"
    r"|\bplan\s+to\s+(?:update|fix|refactor|migrate|implement|add|remove|change|deploy|install|configure|set\s*up|build|create|upgrade)\b",
    re.IGNORECASE,
)

_ANALYSIS_REQUEST_RE = re.compile(
    r"\b(?:review|audit|analyze|examine|inspect)\b.{0,40}\b(?:code|codebase|security|module|implementation|PR|pull\s*request|diff|repository|repo)\b"
    r"|\b(?:code|security|codebase)\s+(?:review|audit|analysis)\b",
    re.IGNORECASE,
)


def is_plan_request(text: str) -> bool:
    """Detect if the user is asking the model to create/write a plan."""
    return bool(_PLAN_REQUEST_RE.search(text))


def is_analysis_request(text: str) -> bool:
    """Detect if the user is asking for a code review, audit, or analysis."""
    return bool(_ANALYSIS_REQUEST_RE.search(text))
