"""Tests for the clipboard utility module."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, mock_open, patch

import pytest

from natshell.ui.clipboard import (
    _build_command,
    _build_read_command,
    _reset,
    backend_name,
    copy,
    detect_backend,
    read,
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
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
                assert detect_backend() == "xclip"

    def test_xsel_found_no_xclip(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
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

    def test_prefers_xclip_over_xsel_on_x11(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: f"/usr/bin/{t}"
                assert detect_backend() == "xclip"

    def test_prefers_wl_copy_on_wayland(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: f"/usr/bin/{t}"
                assert detect_backend() == "wl-copy"


# ─── copy ─────────────────────────────────────────────────────────────────────


class TestCopy:
    def test_copy_with_xclip_success(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
                detect_backend()

            with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="hello")
                assert copy("hello") is True
                # First call: write, second call: verify read-back
                write_call = mock_run.call_args_list[0]
                assert write_call.args[0] == ["xclip", "-selection", "clipboard"]
                assert write_call.kwargs["input"] == "hello"
                verify_call = mock_run.call_args_list[1]
                assert verify_call.args[0] == ["xclip", "-selection", "clipboard", "-o"]

    def test_copy_with_xsel_success(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: "/usr/bin/xsel" if t == "xsel" else None
                detect_backend()

            with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="hello")
                assert copy("hello") is True
                write_call = mock_run.call_args_list[0]
                assert write_call.args[0] == ["xsel", "--clipboard", "--input"]
                verify_call = mock_run.call_args_list[1]
                assert verify_call.args[0] == ["xsel", "--clipboard", "--output"]

    def test_copy_with_wl_copy_success(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: (
                    f"/usr/bin/{t}" if t in ("wl-copy", "wl-paste") else None
                )
                detect_backend()

            with patch("natshell.ui.clipboard.shutil.which", return_value="/usr/bin/wl-paste"):
                with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="hello")
                    assert copy("hello") is True
                    write_call = mock_run.call_args_list[0]
                    assert write_call.args[0] == ["wl-copy"]
                    verify_call = mock_run.call_args_list[1]
                    assert verify_call.args[0] == ["wl-paste", "--no-newline"]

    def test_copy_verify_fails_returns_false(self):
        """Write succeeds but read-back is empty — daemon died."""
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
                detect_backend()

            with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                # Write returns 0 but verify read-back returns empty
                mock_run.side_effect = [
                    MagicMock(returncode=0),  # write
                    MagicMock(returncode=0, stdout=""),  # verify: empty
                ]
                assert copy("hello") is False

    def test_copy_wayland_xclip_verify_uses_wl_paste(self):
        """On Wayland with xclip, verify reads via wl-paste not xclip -o."""
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                # Only xclip available at detection time (no wl-copy)
                mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
                detect_backend()

            # But wl-paste IS available for verification
            with patch("natshell.ui.clipboard.shutil.which", return_value="/usr/bin/wl-paste"):
                with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                    mock_run.side_effect = [
                        MagicMock(returncode=0),  # xclip write
                        MagicMock(returncode=0, stdout=""),  # wl-paste verify: empty!
                    ]
                    assert copy("hello") is False
                    verify_call = mock_run.call_args_list[1]
                    assert verify_call.args[0] == ["wl-paste", "--no-newline"]

    def test_copy_failure_returns_false(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
                detect_backend()

            with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1)
                assert copy("hello") is False

    def test_copy_exception_returns_false(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}):
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


# ─── macOS (pbcopy) ─────────────────────────────────────────────────────────


class TestMacOSBackend:
    def test_detect_pbcopy_on_darwin(self):
        with patch("natshell.ui.clipboard.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert detect_backend() == "pbcopy"

    def test_build_command_pbcopy(self):
        assert _build_command("pbcopy") == ["pbcopy"]

    def test_build_read_command_pbpaste(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _build_read_command("pbcopy") == ["pbpaste"]

    def test_copy_with_pbcopy_success(self):
        with patch("natshell.ui.clipboard.sys") as mock_sys:
            mock_sys.platform = "darwin"
            detect_backend()

        with patch.dict("os.environ", {}, clear=True):
            with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="hello")
                assert copy("hello") is True
                write_call = mock_run.call_args_list[0]
                assert write_call.args[0] == ["pbcopy"]
                assert write_call.kwargs["input"] == "hello"
                verify_call = mock_run.call_args_list[1]
                assert verify_call.args[0] == ["pbpaste"]

    def test_backend_name_pbcopy(self):
        with patch("natshell.ui.clipboard.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert backend_name() == "pbcopy"


# ─── WSL (clip.exe) ─────────────────────────────────────────────────────────


class TestWSLBackend:
    def test_detect_clip_exe_on_wsl(self):
        with patch("natshell.ui.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("builtins.open", mock_open(read_data="Linux Microsoft WSL2")):
                with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                    mock_which.side_effect = lambda t: (
                        "/mnt/c/Windows/system32/clip.exe" if t == "clip.exe" else None
                    )
                    assert detect_backend() == "clip.exe"

    def test_wsl_no_clip_exe_falls_through(self):
        """WSL without clip.exe falls through to Linux X11/Wayland detection."""
        with patch("natshell.ui.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("builtins.open", mock_open(read_data="Linux Microsoft WSL2")):
                with patch("natshell.ui.clipboard.shutil.which", return_value=None):
                    assert detect_backend() == "osc52"

    def test_build_command_clip_exe(self):
        assert _build_command("clip.exe") == ["clip.exe"]

    def test_build_read_command_clip_exe_with_powershell(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch(
                "natshell.ui.clipboard.shutil.which",
                return_value="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe",
            ):
                cmd = _build_read_command("clip.exe")
                assert cmd == ["powershell.exe", "-c", "Get-Clipboard"]

    def test_build_read_command_clip_exe_no_powershell(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("natshell.ui.clipboard.shutil.which", return_value=None):
                assert _build_read_command("clip.exe") is None

    def test_copy_with_clip_exe_success(self):
        with patch("natshell.ui.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("builtins.open", mock_open(read_data="Linux Microsoft WSL2")):
                with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                    mock_which.side_effect = lambda t: (
                        "/mnt/c/clip.exe" if t == "clip.exe" else None
                    )
                    detect_backend()

        # clip.exe without powershell — can't verify, assume success
        with patch("natshell.ui.clipboard.shutil.which", return_value=None):
            with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                assert copy("hello") is True
                write_call = mock_run.call_args_list[0]
                assert write_call.args[0] == ["clip.exe"]


# ─── read ─────────────────────────────────────────────────────────────────────


class TestRead:
    def test_read_xclip(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
                detect_backend()

                with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="hello world")
                    result = read()
                    assert result == "hello world"
                    mock_run.assert_called_once_with(
                        ["xclip", "-selection", "clipboard", "-o"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

    def test_read_xsel(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: "/usr/bin/xsel" if t == "xsel" else None
                detect_backend()

                with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="pasted text")
                    result = read()
                    assert result == "pasted text"
                    mock_run.assert_called_once_with(
                        ["xsel", "--clipboard", "--output"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

    def test_read_pbpaste(self):
        with patch("natshell.ui.clipboard.sys") as mock_sys:
            mock_sys.platform = "darwin"
            detect_backend()

        with patch.dict("os.environ", {}, clear=True):
            with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="mac text")
                result = read()
                assert result == "mac text"
                mock_run.assert_called_once_with(
                    ["pbpaste"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

    def test_read_wl_paste(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "wayland"}):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: (
                    f"/usr/bin/{t}" if t in ("wl-copy", "wl-paste") else None
                )
                detect_backend()

        with patch("natshell.ui.clipboard.shutil.which", return_value="/usr/bin/wl-paste"):
            with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="wayland text")
                result = read()
                assert result == "wayland text"
                mock_run.assert_called_once_with(
                    ["wl-paste", "--no-newline"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

    def test_read_powershell_wsl(self):
        with patch("natshell.ui.clipboard.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("builtins.open", mock_open(read_data="Linux Microsoft WSL2")):
                with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                    mock_which.side_effect = lambda t: (
                        "/mnt/c/clip.exe" if t == "clip.exe" else None
                    )
                    detect_backend()

        with patch.dict("os.environ", {}, clear=True):
            with patch("natshell.ui.clipboard.shutil.which", return_value="/mnt/c/powershell.exe"):
                with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="wsl text")
                    result = read()
                    assert result == "wsl text"
                    mock_run.assert_called_once_with(
                        ["powershell.exe", "-c", "Get-Clipboard"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )

    def test_read_osc52_returns_none(self):
        with patch("natshell.ui.clipboard.shutil.which", return_value=None):
            detect_backend()
        assert read() is None

    def test_read_subprocess_failure_returns_none(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
                detect_backend()

                with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=1)
                    assert read() is None

    def test_read_timeout_returns_none(self):
        with patch.dict("os.environ", {"XDG_SESSION_TYPE": "x11"}, clear=False):
            with patch("natshell.ui.clipboard.shutil.which") as mock_which:
                mock_which.side_effect = lambda t: "/usr/bin/xclip" if t == "xclip" else None
                detect_backend()

                with patch("natshell.ui.clipboard.subprocess.run") as mock_run:
                    mock_run.side_effect = subprocess.TimeoutExpired(cmd="xclip", timeout=5)
                    assert read() is None
