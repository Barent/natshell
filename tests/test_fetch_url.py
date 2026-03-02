"""Tests for the fetch_url tool."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from natshell.config import SafetyConfig
from natshell.safety.classifier import Risk, SafetyClassifier
from natshell.tools.fetch_url import (
    _is_private_ip,
    fetch_url,
    reset_limits,
    set_limits,
)
from natshell.tools.limits import ToolLimits
from natshell.tools.registry import PLAN_SAFE_TOOLS, create_default_registry


@pytest.fixture(autouse=True)
def _reset_tool_limits():
    yield
    reset_limits()


# ─── SSRF protection ─────────────────────────────────────────────────────────


class TestIsPrivateIp:
    def test_loopback_v4(self):
        assert _is_private_ip("127.0.0.1")
        assert _is_private_ip("127.255.255.255")

    def test_rfc1918_10(self):
        assert _is_private_ip("10.0.0.1")
        assert _is_private_ip("10.255.255.255")

    def test_rfc1918_172(self):
        assert _is_private_ip("172.16.0.1")
        assert _is_private_ip("172.31.255.255")

    def test_rfc1918_192(self):
        assert _is_private_ip("192.168.0.1")
        assert _is_private_ip("192.168.1.1")

    def test_link_local_metadata(self):
        assert _is_private_ip("169.254.169.254")

    def test_loopback_v6(self):
        assert _is_private_ip("::1")

    def test_unique_local_v6(self):
        assert _is_private_ip("fc00::1")
        assert _is_private_ip("fd00::1")

    def test_link_local_v6(self):
        assert _is_private_ip("fe80::1")

    def test_public_ip_not_blocked(self):
        assert not _is_private_ip("8.8.8.8")
        assert not _is_private_ip("1.1.1.1")
        assert not _is_private_ip("93.184.216.34")

    def test_unparseable_treated_as_private(self):
        assert _is_private_ip("not-an-ip")


# ─── Scheme validation ───────────────────────────────────────────────────────


class TestSchemeValidation:
    async def test_ftp_rejected(self):
        result = await fetch_url("ftp://example.com/file.txt")
        assert result.exit_code == 1
        assert "http://" in result.error or "https://" in result.error

    async def test_file_rejected(self):
        result = await fetch_url("file:///etc/passwd")
        assert result.exit_code == 1
        assert "http://" in result.error or "https://" in result.error

    async def test_no_scheme_rejected(self):
        result = await fetch_url("example.com")
        assert result.exit_code == 1


# ─── SSRF blocking (mocked DNS) ──────────────────────────────────────────────


def _make_addrinfo(ip: str, family=2):
    """Build a fake getaddrinfo result for a single IP."""
    return [(family, 1, 6, "", (ip, 80))]


class TestSsrfBlocking:
    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("127.0.0.1"))
    async def test_blocks_localhost(self, mock_dns):
        result = await fetch_url("http://localhost/test")
        assert result.exit_code == 1
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("10.0.0.1"))
    async def test_blocks_rfc1918_10(self, mock_dns):
        result = await fetch_url("http://internal.corp/api")
        assert result.exit_code == 1
        assert "private" in result.error.lower() or "blocked" in result.error.lower()

    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("172.16.0.1"))
    async def test_blocks_rfc1918_172(self, mock_dns):
        result = await fetch_url("http://internal.corp/api")
        assert result.exit_code == 1

    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("192.168.1.1"))
    async def test_blocks_rfc1918_192(self, mock_dns):
        result = await fetch_url("http://router.local/admin")
        assert result.exit_code == 1

    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("169.254.169.254"))
    async def test_blocks_metadata_endpoint(self, mock_dns):
        result = await fetch_url("http://169.254.169.254/latest/meta-data/")
        assert result.exit_code == 1

    @patch(
        "natshell.tools.fetch_url.socket.getaddrinfo",
        return_value=[(10, 1, 6, "", ("::1", 80, 0, 0))],
    )
    async def test_blocks_ipv6_loopback(self, mock_dns):
        result = await fetch_url("http://[::1]/test")
        assert result.exit_code == 1


# ─── Successful fetch (mocked HTTP) ──────────────────────────────────────────


def _mock_response(
    status_code: int = 200,
    content_type: str = "text/html",
    text: str = "<html><body>Hello</body></html>",
    content: bytes | None = None,
):
    """Build a mock httpx.Response."""
    if content is None:
        content = text.encode("utf-8")
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.headers = {"content-type": content_type}
    resp.text = text
    resp.content = content
    return resp


class TestFetchSuccess:
    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("93.184.216.34"))
    @patch("natshell.tools.fetch_url.httpx.AsyncClient")
    async def test_fetch_public_url(self, mock_client_cls, mock_dns):
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(
            text="Example Domain", content=b"Example Domain"
        )
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await fetch_url("https://example.com")
        assert result.exit_code == 0
        assert "Example Domain" in result.output
        assert "200" in result.output

    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("93.184.216.34"))
    @patch("natshell.tools.fetch_url.httpx.AsyncClient")
    async def test_json_response(self, mock_client_cls, mock_dns):
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(
            content_type="application/json",
            text='{"key": "value"}',
            content=b'{"key": "value"}',
        )
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await fetch_url("https://api.example.com/data")
        assert result.exit_code == 0
        assert '"key"' in result.output


# ─── Timeout ──────────────────────────────────────────────────────────────────


class TestTimeout:
    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("93.184.216.34"))
    @patch("natshell.tools.fetch_url.httpx.AsyncClient")
    async def test_timeout_error(self, mock_client_cls, mock_dns):
        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.TimeoutException("timed out")
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await fetch_url("https://slow.example.com", timeout=5)
        assert result.exit_code == 1
        assert "timed out" in result.error.lower()

    async def test_timeout_clamped_to_max(self):
        """Timeout values above 60 should be clamped, not rejected."""
        # This will fail at DNS resolution, but the point is it doesn't
        # crash on the timeout value itself.
        with patch(
            "natshell.tools.fetch_url.socket.getaddrinfo",
            side_effect=socket.gaierror("dns fail"),
        ):
            result = await fetch_url("https://example.com", timeout=9999)
            assert result.exit_code == 1  # Failed, but didn't crash


# ─── Response truncation ─────────────────────────────────────────────────────


class TestTruncation:
    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("93.184.216.34"))
    @patch("natshell.tools.fetch_url.httpx.AsyncClient")
    async def test_large_response_truncated(self, mock_client_cls, mock_dns):
        large_text = "x" * 10000
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(
            text=large_text, content=large_text.encode()
        )
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        # Set a small limit
        set_limits(ToolLimits(max_output_chars=500))
        result = await fetch_url("https://example.com/large")
        assert result.truncated
        # Output should be limited (header + 500 chars of body max)
        assert len(result.output) < 1000


# ─── Binary content ──────────────────────────────────────────────────────────


class TestBinaryContent:
    @patch("natshell.tools.fetch_url.socket.getaddrinfo", return_value=_make_addrinfo("93.184.216.34"))
    @patch("natshell.tools.fetch_url.httpx.AsyncClient")
    async def test_binary_not_dumped(self, mock_client_cls, mock_dns):
        mock_client = AsyncMock()
        mock_client.get.return_value = _mock_response(
            content_type="image/png",
            text="",
            content=b"\x89PNG\r\n\x1a\n" + b"\x00" * 1000,
        )
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)

        result = await fetch_url("https://example.com/image.png")
        assert result.exit_code == 0
        assert "Binary content" in result.output
        assert "image/png" in result.output
        # Should NOT contain raw binary data
        assert "\x89PNG" not in result.output


# ─── Registration & classification ────────────────────────────────────────────


class TestRegistration:
    def test_registered_in_default_registry(self):
        registry = create_default_registry()
        assert "fetch_url" in registry.tool_names

    def test_in_plan_safe_tools(self):
        assert "fetch_url" in PLAN_SAFE_TOOLS

    def test_schema_generated(self):
        registry = create_default_registry()
        schemas = registry.get_tool_schemas()
        names = [s["function"]["name"] for s in schemas]
        assert "fetch_url" in names

    def test_schema_in_plan_safe_filter(self):
        registry = create_default_registry()
        schemas = registry.get_tool_schemas(allowed=PLAN_SAFE_TOOLS)
        names = [s["function"]["name"] for s in schemas]
        assert "fetch_url" in names


class TestSafetyClassification:
    def test_classified_as_safe(self):
        config = SafetyConfig(mode="confirm", always_confirm=[], blocked=[])
        classifier = SafetyClassifier(config)
        risk = classifier.classify_tool_call("fetch_url", {"url": "https://example.com"})
        assert risk == Risk.SAFE


# ─── DNS failure ──────────────────────────────────────────────────────────────


class TestDnsFailure:
    @patch(
        "natshell.tools.fetch_url.socket.getaddrinfo",
        side_effect=socket.gaierror("Name or service not known"),
    )
    async def test_dns_failure_returns_error(self, mock_dns):
        result = await fetch_url("https://nonexistent.invalid")
        assert result.exit_code == 1
        assert "dns" in result.error.lower() or "failed" in result.error.lower()
