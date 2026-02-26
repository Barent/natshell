"""Tests for the clipboard utility module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import subprocess

import pytest

from natshell.ui.clipboard import (
    _reset,
    backend_name,
    copy,
    detect_backend,
)


@pytest.fixture(autouse=True)
def reset_backend():
    """Ensure each test starts with a fresh backend detection."""
    _reset()
    yield
    _reset()


# ─── detect_backend ──────────────────────────────────────────────────────────


class TestDetectBackend:
    def test_xclip_found(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
            assert detect_backend() == "xclip"

    def test_xsel_found_no_xclip(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            def which_side_effect(tool):
                if tool == "xsel":
                    return "/usr/bin/xsel"
                return None
            mock_which.side_effect = which_side_effect
            assert detect_backend() == "xsel"

    def test_wl_copy_found_no_xclip_xsel(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            def which_side_effect(tool):
                if tool == "wl-copy":
                    return "/usr/bin/wl-copy"
                return None
            mock_which.side_effect = which_side_effect
            assert detect_backend() == "wl-copy"

    def test_nothing_found_falls_back_to_osc52(self):
        with patch("natshell.ui.clipboard.shutil.which", return_value=None):
            assert detect_backend() == "osc52"

    def test_result_is_cached(self):
        with patch("natshell.ui.clipboard.shutil.which", return_value=None):
            first = detect_backend()
        # Second call should return cached value without calling which()
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/xclip"
            second = detect_backend()
        assert first == second == "osc52"

    def test_prefers_xclip_over_xsel(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            mock_which.side_effect = lambda t: f"/usr/bin/{t}"
            assert detect_backend() == "xclip"


# ─── copy ─────────────────────────────────────────────────────────────────────


class TestCopy:
    def test_copy_with_xclip_success(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
            detect_backend()

        with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert copy("hello") is True
            mock_run.assert_called_once_with(
                ["xclip", "-selection", "clipboard"],
                input="hello",
                capture_output=True,
                text=True,
                timeout=5,
            )

    def test_copy_with_xsel_success(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            mock_which.side_effect = lambda t: "/usr/bin/xsel" if t == "xsel" else None
            detect_backend()

        with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert copy("hello") is True
            mock_run.assert_called_once_with(
                ["xsel", "--clipboard", "--input"],
                input="hello",
                capture_output=True,
                text=True,
                timeout=5,
            )

    def test_copy_with_wl_copy_success(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            mock_which.side_effect = lambda t: "/usr/bin/wl-copy" if t == "wl-copy" else None
            detect_backend()

        with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert copy("hello") is True
            mock_run.assert_called_once_with(
                ["wl-copy"],
                input="hello",
                capture_output=True,
                text=True,
                timeout=5,
            )

    def test_copy_failure_returns_false(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
            detect_backend()

        with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert copy("hello") is False

    def test_copy_exception_returns_false(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
            detect_backend()

        with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="xclip", timeout=5)
            assert copy("hello") is False

    def test_copy_osc52_with_app(self):
        with patch("natshell.ui.clipboard.shutil.which", return_value=None):
            detect_backend()

        app = MagicMock()
        assert copy("hello", app=app) is True
        app.copy_to_clipboard.assert_called_once_with("hello")

    def test_copy_osc52_without_app(self):
        with patch("natshell.ui.clipboard.shutil.which", return_value=None):
            detect_backend()

        assert copy("hello") is False

    def test_copy_osc52_app_exception(self):
        with patch("natshell.ui.clipboard.shutil.which", return_value=None):
            detect_backend()

        app = MagicMock()
        app.copy_to_clipboard.side_effect = RuntimeError("no terminal")
        assert copy("hello", app=app) is False


# ─── backend_name ─────────────────────────────────────────────────────────────


class TestBackendName:
    def test_xclip_name(self):
        with patch("natshell.ui.clipboard.shutil.which") as mock_which:
            mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
            assert backend_name() == "xclip"

    def test_osc52_name(self):
        with patch("natshell.ui.clipboard.shutil.which", return_value=None):
            name = backend_name()
            assert "OSC52" in name
