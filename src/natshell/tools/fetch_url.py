"""Fetch URL contents with SSRF protection."""

from __future__ import annotations

import ipaddress
import logging
import socket

import httpx

from natshell.tools.limits import ToolLimits
from natshell.tools.registry import ToolDefinition, ToolResult

logger = logging.getLogger(__name__)

# ── Shared limits (overwritten by agent loop once n_ctx is known) ─────────

_limits = ToolLimits()


def set_limits(limits: ToolLimits) -> None:
    global _limits
    _limits = limits


def reset_limits() -> None:
    global _limits
    _limits = ToolLimits()


# ── SSRF protection ──────────────────────────────────────────────────────

# Private/reserved IPv4 and IPv6 networks to block
_BLOCKED_NETWORKS = [
    # IPv4
    ipaddress.IPv4Network("127.0.0.0/8"),       # Loopback
    ipaddress.IPv4Network("10.0.0.0/8"),         # RFC 1918
    ipaddress.IPv4Network("172.16.0.0/12"),      # RFC 1918
    ipaddress.IPv4Network("192.168.0.0/16"),     # RFC 1918
    ipaddress.IPv4Network("169.254.0.0/16"),     # Link-local (includes cloud metadata)
    ipaddress.IPv4Network("0.0.0.0/8"),          # "This" network
    # IPv6
    ipaddress.IPv6Network("::1/128"),            # Loopback
    ipaddress.IPv6Network("fc00::/7"),           # Unique local
    ipaddress.IPv6Network("fe80::/10"),          # Link-local
]

_MAX_RESPONSE_BYTES = 1_048_576  # 1 MB
_MAX_TIMEOUT = 60
_DEFAULT_TIMEOUT = 30
_USER_AGENT = "NatShell/0.1"


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP address falls within a private/reserved network."""
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return True  # Unparseable → treat as blocked
    for network in _BLOCKED_NETWORKS:
        if addr in network:
            return True
    return False


# ── Tool definition ──────────────────────────────────────────────────────

DEFINITION = ToolDefinition(
    name="fetch_url",
    description=(
        "Fetch the contents of a URL. Returns the response body as text. "
        "Useful for reading documentation, checking APIs, or downloading "
        "text files. Blocked from accessing private/internal network addresses."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch (must be http:// or https://).",
            },
            "timeout": {
                "type": "integer",
                "description": "Request timeout in seconds (default 30, max 60).",
            },
        },
        "required": ["url"],
    },
    requires_confirmation=False,
)


# ── Handler ──────────────────────────────────────────────────────────────


async def fetch_url(url: str, timeout: int = _DEFAULT_TIMEOUT) -> ToolResult:
    """Fetch URL contents with SSRF protection and size limits."""
    # Validate scheme
    if not url.startswith(("http://", "https://")):
        return ToolResult(
            error=f"Invalid URL scheme — only http:// and https:// are allowed: {url}",
            exit_code=1,
        )

    # Clamp timeout
    timeout = max(1, min(timeout, _MAX_TIMEOUT))

    # Extract hostname for SSRF check
    try:
        parsed = httpx.URL(url)
        hostname = parsed.host
    except Exception:
        return ToolResult(error=f"Could not parse URL: {url}", exit_code=1)

    if not hostname:
        return ToolResult(error=f"No hostname in URL: {url}", exit_code=1)

    # Resolve hostname and check all IPs against blocklist BEFORE connecting
    try:
        addrinfo = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, OSError) as e:
        return ToolResult(error=f"DNS resolution failed for {hostname}: {e}", exit_code=1)

    for family, _, _, _, sockaddr in addrinfo:
        ip = sockaddr[0]
        if _is_private_ip(ip):
            return ToolResult(
                error=(
                    f"Blocked: {hostname} resolves to private/reserved address {ip}. "
                    "fetch_url cannot access internal network addresses."
                ),
                exit_code=1,
            )

    # Fetch
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            max_redirects=5,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            response = await client.get(url)
    except httpx.TimeoutException:
        return ToolResult(error=f"Request timed out after {timeout}s: {url}", exit_code=1)
    except httpx.TooManyRedirects:
        return ToolResult(error=f"Too many redirects (max 5): {url}", exit_code=1)
    except httpx.HTTPError as e:
        return ToolResult(error=f"HTTP error: {e}", exit_code=1)

    # Check for redirect-based SSRF: if the final URL hostname differs from
    # the original, verify the new hostname doesn't resolve to a private IP.
    # Note: there is still a theoretical TOCTOU gap (DNS rebinding on the same
    # hostname) but this is not a practical concern for a CLI tool.
    final_url = response.url
    try:
        final_host = final_url.host
    except Exception:
        final_host = None
    if final_host and final_host != hostname:
        try:
            redirect_addrinfo = socket.getaddrinfo(final_host, None)
            for _, _, _, _, sockaddr in redirect_addrinfo:
                ip = sockaddr[0]
                if _is_private_ip(ip):
                    return ToolResult(
                        error=(
                            f"Blocked: redirect target {final_host} resolves to "
                            f"private/reserved address {ip}. "
                            "fetch_url cannot follow redirects to internal network addresses."
                        ),
                        exit_code=1,
                    )
        except (socket.gaierror, OSError):
            pass  # DNS failure on redirect target — response already received

    content_type = response.headers.get("content-type", "")
    status = response.status_code

    # For binary content types, return metadata only
    is_text = any(
        t in content_type.lower()
        for t in ("text/", "json", "xml", "javascript", "html", "yaml", "toml")
    )
    if not is_text and content_type:
        return ToolResult(
            output=(
                f"Status: {status}\n"
                f"Content-Type: {content_type}\n"
                f"Size: {len(response.content)} bytes\n\n"
                "(Binary content — body not shown)"
            ),
        )

    # Read text body with size cap
    body = response.text
    truncated = False
    if len(response.content) > _MAX_RESPONSE_BYTES:
        body = response.content[:_MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")
        truncated = True

    # Apply context-adaptive output truncation
    max_chars = _limits.max_output_chars
    if len(body) > max_chars:
        body = body[:max_chars]
        truncated = True

    header = f"Status: {status}\nContent-Type: {content_type}\n\n"
    return ToolResult(
        output=header + body,
        truncated=truncated,
    )
