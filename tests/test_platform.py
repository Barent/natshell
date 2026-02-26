"""Tests for the platform detection module."""

from __future__ import annotations

from unittest.mock import mock_open, patch

from natshell.platform import current_platform, is_linux, is_macos, is_wsl


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
