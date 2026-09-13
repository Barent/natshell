"""Shared pytest configuration for the NatShell test suite."""

from __future__ import annotations

import os

# Keep run-metrics recording (R2-6) from writing into the real
# ~/.local/share/natshell/metrics while agent-loop tests run.  Tests that
# exercise the store itself construct RunMetricsStore with an explicit
# temp dir_path, which bypasses this switch.
os.environ.setdefault("NATSHELL_DISABLE_METRICS", "1")
