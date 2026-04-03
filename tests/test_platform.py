"""Tests for the platform detection module."""

from __future__ import annotations

from unittest.mock import mock_open, patch

from natshell.platform import (
    config_dir,
    current_platform,
    data_dir,
    is_arm64,
    is_linux,
    is_macos,
    is_windows,
    is_wsl,
)


class TestCurrentPlatform:
    """Test current_platform() with mocked sys.platform and /proc/version."""

    def setup_method(self):
        # Clear the lru_cache before each test
        current_platform.cache_clear()

    def teardown_method(self):
        current_platform.cache_clear()

    def test_darwin_returns_macos(self):
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert current_platform() == "macos"

    def test_linux_with_microsoft_proc_version_returns_wsl(self):
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            proc_content = "Linux version 5.15.0 (Microsoft WSL2)"
            with patch("builtins.open", mock_open(read_data=proc_content)):
                assert current_platform() == "wsl"

    def test_linux_without_microsoft_returns_linux(self):
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            proc_content = "Linux version 6.12.69+deb13-amd64"
            with patch("builtins.open", mock_open(read_data=proc_content)):
                assert current_platform() == "linux"

    def test_linux_no_proc_version_returns_linux(self):
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("builtins.open", side_effect=OSError("No such file")):
                assert current_platform() == "linux"

    def test_result_is_cached(self):
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "darwin"
            first = current_platform()

        # Second call should use cache, not re-check sys.platform
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            second = current_platform()

        assert first == second == "macos"


class TestHelpers:
    def setup_method(self):
        current_platform.cache_clear()

    def teardown_method(self):
        current_platform.cache_clear()

    def test_is_macos_true_on_darwin(self):
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "darwin"
            assert is_macos() is True
            assert is_wsl() is False
            assert is_linux() is False

    def test_is_wsl_true_on_wsl(self):
        current_platform.cache_clear()
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("builtins.open", mock_open(read_data="Linux Microsoft WSL2")):
                assert is_wsl() is True
                assert is_macos() is False
                assert is_linux() is True  # WSL counts as Linux

    def test_is_linux_true_on_native_linux(self):
        current_platform.cache_clear()
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("builtins.open", mock_open(read_data="Linux version 6.12")):
                assert is_linux() is True
                assert is_macos() is False
                assert is_wsl() is False

    def test_is_windows_true_on_win32(self):
        current_platform.cache_clear()
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert is_windows() is True
            assert is_macos() is False
            assert is_linux() is False
            assert is_wsl() is False


class TestWin32Platform:
    """Test Windows-specific detection."""

    def setup_method(self):
        current_platform.cache_clear()

    def teardown_method(self):
        current_platform.cache_clear()

    def test_win32_returns_windows(self):
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert current_platform() == "windows"

    def test_win32_checked_before_linux(self):
        """win32 check occurs before Linux/WSL branch."""
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "win32"
            # Even if /proc/version existed, win32 takes precedence
            assert current_platform() == "windows"


class TestIsArm64:
    def setup_method(self):
        is_arm64.cache_clear()

    def teardown_method(self):
        is_arm64.cache_clear()

    def test_arm64_windows(self):
        with patch("natshell.platform._platform.machine", return_value="ARM64"):
            assert is_arm64() is True

    def test_aarch64_linux(self):
        is_arm64.cache_clear()
        with patch("natshell.platform._platform.machine", return_value="aarch64"):
            assert is_arm64() is True

    def test_x86_64(self):
        is_arm64.cache_clear()
        with patch("natshell.platform._platform.machine", return_value="x86_64"):
            assert is_arm64() is False


class TestDirectoryHelpers:
    def setup_method(self):
        current_platform.cache_clear()

    def teardown_method(self):
        current_platform.cache_clear()

    def test_data_dir_unix(self):
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("builtins.open", mock_open(read_data="Linux version 6.12")):
                d = data_dir()
                # Use Path parts to avoid separator issues on Windows
                assert d.parts[-3:] == (".local", "share", "natshell")

    def test_data_dir_windows(self):
        current_platform.cache_clear()
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch.dict("os.environ", {"LOCALAPPDATA": "C:\\Users\\test\\AppData\\Local"}):
                d = data_dir()
                assert "AppData" in str(d)
                assert d.name == "natshell"

    def test_config_dir_unix(self):
        current_platform.cache_clear()
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            with patch("builtins.open", mock_open(read_data="Linux version 6.12")):
                d = config_dir()
                assert d.parts[-2:] == (".config", "natshell")

    def test_config_dir_windows(self):
        current_platform.cache_clear()
        with patch("natshell.platform.sys") as mock_sys:
            mock_sys.platform = "win32"
            with patch.dict("os.environ", {"APPDATA": "C:\\Users\\test\\AppData\\Roaming"}):
                d = config_dir()
                assert "AppData" in str(d)
                assert d.name == "natshell"
